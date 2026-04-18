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

  # ── Dual-Mode Computation Helper ──────────────────────────────────────────
  compute_mode_stats <- function(df_in) {
    nt <- length(trait_cols)
    n <- nrow(df_in)
    df_deg <- if(n >= 3) n - 2 else NA
    crit_r <- if(n >= 3) qt(0.975, df_deg) / sqrt(df_deg + qt(0.975, df_deg)^2) else NA

    r_mat <- matrix(NA_real_, nrow = nt, ncol = nt)
    p_mat <- matrix(NA_real_, nrow = nt, ncol = nt)
    ci_lower <- matrix(NA_real_, nrow = nt, ncol = nt)
    ci_upper <- matrix(NA_real_, nrow = nt, ncol = nt)

    for (i in seq_len(nt)) {
      for (j in seq_len(nt)) {
        if (i == j) {
          r_mat[i, j] <- 1.0
          p_mat[i, j] <- 0.0
          next
        }
        x <- df_in[[trait_cols[i]]]
        y <- df_in[[trait_cols[j]]]
        ok <- complete.cases(x, y)
        n_ok <- sum(ok)

        if (n_ok >= 3) {
          ct <- tryCatch(cor.test(x[ok], y[ok], method = method), error = function(e) NULL)
          if (!is.null(ct)) {
            r_mat[i, j] <- ct$estimate
            p_mat[i, j] <- ct$p.value
            if (!is.null(ct$conf.int)) {
              ci_lower[i, j] <- ct$conf.int[1]
              ci_upper[i, j] <- ct$conf.int[2]
            }
          }
        }
      }
    }

    p_adj_mat <- matrix(NA_real_, nrow = nt, ncol = nt)
    p_vals <- numeric(0)
    if (nt > 1) {
      for(i in seq_len(nt - 1)) {
        for(j in seq(i + 1, nt)) {
          p_vals <- c(p_vals, p_mat[i, j])
        }
      }
      p_adj <- p.adjust(p_vals, method="fdr")
      idx <- 1
      for(i in seq_len(nt - 1)) {
        for(j in seq(i + 1, nt)) {
          p_adj_mat[i, j] <- p_adj[idx]
          p_adj_mat[j, i] <- p_adj[idx]
          idx <- idx + 1
        }
      }
    }

    list(
      n_observations = n,
      df = if(is.na(df_deg)) NULL else df_deg,
      critical_r = if(is.na(crit_r)) NULL else crit_r,
      r_matrix = unname(r_mat),
      p_matrix = unname(p_mat),
      p_adj_matrix = unname(p_adj_mat),
      ci_lower_matrix = unname(ci_lower),
      ci_upper_matrix = unname(ci_upper)
    )
  }

  phenotypic <- compute_mode_stats(data)
  genotypic  <- compute_mode_stats(means_df)

  # ── Return ────────────────────────────────────────────────────────────────
  # unname() strips row/column names from matrices so jsonlite serialises
  # them as plain arrays-of-arrays (not named objects).
  list(
    trait_names    = trait_cols,
    method         = method,
    phenotypic     = phenotypic,
    genotypic      = genotypic,
    warnings       = warnings_vec   # character vector — always serialised as JSON array by jsonlite
  )
}
