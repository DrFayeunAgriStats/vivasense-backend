#' VivaSense Genetics – Trait Relationships Engine
#' Phase 2 / Phase 1 scope: phenotypic correlation only.
#'
#' Dependencies: base R + jsonlite (already present in the Render build).
#' No additional packages are required for phenotypic correlation.
#'
#' Entry point called by trait_relationships_routes.py:
#'   run_correlation_analysis(data, trait_cols, method)


# ============================================================================
# run_correlation_analysis
# ============================================================================

#' Compute phenotypic correlation coefficients and p-values for all trait pairs.
#'
#' Correlations are computed on per-genotype means (averaged across reps and,
#' when present, environments).  This is the standard agronomic approach for
#' phenotypic correlation: it separates genetic signal from rep-level noise and
#' gives one observation per genotype, which is the natural unit of interest.
#'
#' @param data       data.frame with columns: genotype, rep, [environment],
#'                   and all columns named in trait_cols.
#' @param trait_cols character vector of trait column names (>= 2).
#' @param method     "pearson" (default) or "spearman".
#'
#' @return list with fields:
#'   trait_names    — character vector of trait names (same order as input)
#'   n_observations — integer, number of unique genotype means used
#'   method         — character, the method used
#'   r_matrix       — unnamed numeric matrix (n x n), diagonal = 1.0
#'   p_matrix       — unnamed numeric matrix (n x n), diagonal = 0.0,
#'                    NA when pair has < 3 complete genotype means
#'   interpretation — character, plain-English summary
#'   warnings       — list of character strings (may be empty)

run_correlation_analysis <- function(data, trait_cols, method = "pearson") {

  # ── Input validation ──────────────────────────────────────────────────────
  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame")
  }
  if (length(trait_cols) < 2) {
    stop("At least 2 trait columns required for correlation")
  }
  if (!method %in% c("pearson", "spearman")) {
    stop("'method' must be 'pearson' or 'spearman'")
  }
  if (!"genotype" %in% names(data)) {
    stop("Column 'genotype' not found in data")
  }

  # ── Coerce trait columns to numeric ──────────────────────────────────────
  for (col in trait_cols) {
    if (!col %in% names(data)) {
      stop(paste0("Trait column not found in data: ", col))
    }
    data[[col]] <- suppressWarnings(as.numeric(as.character(data[[col]])))
  }

  # ── Compute per-genotype means ────────────────────────────────────────────
  # aggregate() prepends the 'by' column; select only trait columns afterwards.
  means_df <- aggregate(
    data[, trait_cols, drop = FALSE],
    by  = list(genotype = data[["genotype"]]),
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  means_df <- means_df[, trait_cols, drop = FALSE]

  # Replace NaN produced by mean() on all-NA vectors with explicit NA
  for (col in trait_cols) {
    means_df[[col]][is.nan(means_df[[col]])] <- NA_real_
  }

  n <- nrow(means_df)

  if (n < 3) {
    stop(paste0(
      "Only ", n, " unique genotype(s) found — ",
      "minimum 3 required for correlation analysis"
    ))
  }

  # ── Warnings about missing data ───────────────────────────────────────────
  warnings_vec <- character(0)

  for (col in trait_cols) {
    pct_missing <- mean(is.na(means_df[[col]])) * 100
    if (pct_missing > 20) {
      warnings_vec <- c(
        warnings_vec,
        paste0(col, ": ", round(pct_missing, 0),
               "% of genotype means are missing — correlation may be unreliable")
      )
    }
  }

  # ── Correlation matrix (pairwise complete obs) ────────────────────────────
  r_mat <- cor(means_df, method = method, use = "pairwise.complete.obs")

  # ── P-value matrix ────────────────────────────────────────────────────────
  nt    <- length(trait_cols)
  p_mat <- matrix(0.0, nrow = nt, ncol = nt)

  for (i in seq_len(nt)) {
    for (j in seq_len(nt)) {
      if (i == j) next  # diagonal stays 0.0

      x    <- means_df[[trait_cols[i]]]
      y    <- means_df[[trait_cols[j]]]
      ok   <- complete.cases(x, y)
      n_ok <- sum(ok)

      if (n_ok < 3) {
        p_mat[i, j] <- NA_real_
        msg <- paste0(
          "Insufficient paired observations for ",
          trait_cols[i], " vs ", trait_cols[j],
          " (n = ", n_ok, ") — p-value set to null"
        )
        if (!msg %in% warnings_vec) {
          warnings_vec <- c(warnings_vec, msg)
        }
      } else {
        p_mat[i, j] <- tryCatch(
          cor.test(x[ok], y[ok], method = method)$p.value,
          error = function(e) NA_real_
        )
      }
    }
  }

  # ── Interpretation text ───────────────────────────────────────────────────
  method_label <- if (method == "spearman") "Spearman rank" else "Pearson"

  sig_pairs <- list()
  for (i in seq_len(nt - 1)) {
    for (j in seq(i + 1, nt)) {
      r_val <- r_mat[i, j]
      p_val <- p_mat[i, j]
      if (!is.na(r_val) && !is.na(p_val) && p_val < 0.05) {
        sig_pairs <- c(sig_pairs, list(list(
          a         = trait_cols[i],
          b         = trait_cols[j],
          r         = round(r_val, 3),
          direction = if (r_val > 0) "positive" else "negative"
        )))
      }
    }
  }

  if (length(sig_pairs) == 0) {
    interp <- paste0(
      method_label, " correlation analysis on ", n, " genotype means ",
      "revealed no statistically significant trait pairs (p < 0.05). ",
      "Consider increasing genotype sample size or checking trait variability."
    )
  } else {
    pair_strs <- vapply(sig_pairs, function(p) {
      paste0(p$a, " \u2013 ", p$b, " (r = ", p$r, ")")
    }, character(1))

    r_vals    <- sapply(sig_pairs, `[[`, "r")
    strongest <- sig_pairs[[which.max(abs(r_vals))]]

    pos_count <- sum(sapply(sig_pairs, function(p) p$direction == "positive"))
    neg_count <- length(sig_pairs) - pos_count

    pos_note <- if (pos_count > 0)
      "Positive correlations indicate traits that tend to improve together, facilitating indirect selection. "
    else ""

    neg_note <- if (neg_count > 0)
      "Negative correlations suggest trade-offs that may complicate simultaneous improvement."
    else ""

    interp <- paste0(
      method_label, " correlation analysis on ", n, " genotype means ",
      "identified ", length(sig_pairs), " significant pair",
      if (length(sig_pairs) > 1) "s" else "",
      ": ", paste(pair_strs, collapse = "; "), ". ",
      "The strongest association was between ", strongest$a,
      " and ", strongest$b, " (r = ", strongest$r, "). ",
      pos_note, neg_note
    )
  }

  # ── Return ────────────────────────────────────────────────────────────────
  # unname() strips row/column names from matrices so jsonlite serialises
  # them as plain arrays-of-arrays (not named objects).
  list(
    trait_names    = trait_cols,
    n_observations = n,
    method         = method,
    r_matrix       = unname(r_mat),
    p_matrix       = unname(p_mat),
    interpretation = interp,
    warnings       = warnings_vec   # character vector — always serialised as JSON array by jsonlite
  )
}
