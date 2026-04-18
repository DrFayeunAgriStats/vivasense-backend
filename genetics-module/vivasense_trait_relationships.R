#' VivaSense Genetics – Trait Relationships Engine
#'
#' Computes three correlation modes:
#'   1. Phenotypic   — all observations (plot-level co-variation)
#'   2. Between-genotype — genotype means (between-genotype association; NOT a genetic parameter)
#'   3. Genotypic    — variance-component based via bivariate REML (sommer)
#'
#' Dependencies: base R + jsonlite + sommer (for genotypic VC mode).
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

  phenotypic       <- compute_mode_stats(data)
  between_genotype <- compute_mode_stats(means_df)

  genotypic_vc <- tryCatch(
    compute_genotypic_vc_correlation(data, trait_cols),
    error = function(e) {
      message("Genotypic VC correlation failed: ", e$message)
      NULL
    }
  )

  if (!is.null(genotypic_vc) && length(genotypic_vc$warnings) > 0) {
    warnings_vec <- c(warnings_vec, genotypic_vc$warnings)
  }

  # ── Return ────────────────────────────────────────────────────────────────
  list(
    trait_names      = trait_cols,
    method           = method,
    phenotypic       = phenotypic,
    between_genotype = between_genotype,
    genotypic        = genotypic_vc,   # NULL when sommer unavailable or all pairs failed
    warnings         = warnings_vec
  )
}


# ============================================================================
# compute_genotypic_vc_correlation
# ============================================================================

#' Estimate genotypic correlations via bivariate REML using sommer.
#'
#' For each trait pair, fits a bivariate mixed model:
#'   cbind(t1, t2) ~ [rep] + vs(genotype, Gtc = unsm(2))
#' and extracts the genetic variance-covariance matrix to compute:
#'   rg = Cov_g(t1, t2) / sqrt(Vg(t1) * Vg(t2))
#'
#' Inference uses Fisher's z-transform on n_genotypes.
#' Returns NULL if sommer is unavailable or no pairs converge.
compute_genotypic_vc_correlation <- function(data, trait_cols) {

  if (!requireNamespace("sommer", quietly = TRUE)) {
    message("sommer not available — genotypic VC correlation skipped")
    return(NULL)
  }

  nt      <- length(trait_cols)
  geno_v  <- unique(data[["genotype"]])
  n_geno  <- length(geno_v)
  has_rep <- "rep" %in% names(data) && length(unique(data[["rep"]])) > 1

  r_mat    <- matrix(NA_real_, nrow = nt, ncol = nt)
  p_mat    <- matrix(NA_real_, nrow = nt, ncol = nt)
  ci_lower <- matrix(NA_real_, nrow = nt, ncol = nt)
  ci_upper <- matrix(NA_real_, nrow = nt, ncol = nt)
  diag(r_mat) <- 1.0
  diag(p_mat) <- 0.0

  vc_warnings <- character(0)
  any_success <- FALSE

  for (i in seq_len(nt - 1)) {
    for (j in seq(i + 1, nt)) {
      t1 <- trait_cols[i]
      t2 <- trait_cols[j]

      x1 <- suppressWarnings(as.numeric(as.character(data[[t1]])))
      x2 <- suppressWarnings(as.numeric(as.character(data[[t2]])))
      df_pair <- data.frame(genotype = data[["genotype"]], x1 = x1, x2 = x2)
      if (has_rep) df_pair$rep <- data[["rep"]]
      df_pair <- df_pair[complete.cases(df_pair), , drop = FALSE]

      n_ok <- length(unique(df_pair$genotype))
      if (n_ok < 3 || nrow(df_pair) < 6) {
        vc_warnings <- c(vc_warnings, paste0(t1, " x ", t2, ": insufficient data for bivariate model"))
        next
      }

      fixed_frm <- if (has_rep) cbind(x1, x2) ~ rep else cbind(x1, x2) ~ 1

      fit <- tryCatch(
        suppressWarnings(suppressMessages(
          sommer::mmer(
            fixed   = fixed_frm,
            random  = ~sommer::vs(genotype, Gtc = sommer::unsm(2)),
            data    = df_pair,
            verbose = FALSE
          )
        )),
        error = function(e) NULL
      )

      if (is.null(fit)) {
        vc_warnings <- c(vc_warnings, paste0(t1, " x ", t2, ": bivariate model did not converge"))
        next
      }

      # Locate the genetic variance-covariance component in fit$sigma
      sigma_names    <- names(fit$sigma)
      geno_sig_name  <- sigma_names[grepl("genotype", sigma_names, fixed = TRUE)]
      if (length(geno_sig_name) == 0) {
        vc_warnings <- c(vc_warnings, paste0(t1, " x ", t2, ": genetic VC not found in model output"))
        next
      }

      Vg <- fit$sigma[[geno_sig_name[1]]]
      if (!is.matrix(Vg) || nrow(Vg) < 2 || ncol(Vg) < 2) {
        vc_warnings <- c(vc_warnings, paste0(t1, " x ", t2, ": unexpected variance matrix structure"))
        next
      }

      vg1  <- Vg[1, 1]
      vg2  <- Vg[2, 2]
      covg <- Vg[1, 2]

      if (is.na(vg1) || is.na(vg2) || vg1 <= 0 || vg2 <= 0) {
        vc_warnings <- c(vc_warnings, paste0(t1, " x ", t2, ": non-positive genetic variance"))
        next
      }

      rg <- max(-1, min(1, covg / sqrt(vg1 * vg2)))
      r_mat[i, j] <- rg
      r_mat[j, i] <- rg
      any_success  <- TRUE

      # Approximate Fisher z-based inference.
      # IMPORTANT: Fisher z SE = 1/sqrt(n-3) treats n_genotypes as independent
      # observations of the correlation, which is an approximation — the true
      # standard error of a REML-estimated rg depends on the information matrix
      # of the bivariate model and is not computed here.  p-values and CIs are
      # therefore conservative approximations only.
      # Suppressed when n_ok < 5 (too few genotypes) or |rg| = 1 (boundary case).
      if (n_ok >= 5 && abs(rg) < 1.0) {
        z     <- atanh(rg)
        se_z  <- 1.0 / sqrt(n_ok - 3)
        p_val <- 2 * (1 - pnorm(abs(z / se_z)))
        z_crit <- qnorm(0.975)
        p_mat[i, j]    <- p_val;  p_mat[j, i]    <- p_val
        ci_lower[i, j] <- tanh(z - z_crit * se_z); ci_lower[j, i] <- ci_lower[i, j]
        ci_upper[i, j] <- tanh(z + z_crit * se_z); ci_upper[j, i] <- ci_upper[i, j]
      } else {
        # Record why inference was suppressed
        reason <- if (n_ok < 5) paste0(t1, " x ", t2, ": too few genotypes (n=", n_ok, ") for reliable inference")
                  else paste0(t1, " x ", t2, ": |rg| = 1 — boundary value, inference suppressed")
        vc_warnings <- c(vc_warnings, reason)
      }
    }
  }

  if (!any_success) return(NULL)

  # FDR-adjusted p-values
  p_adj_mat <- matrix(NA_real_, nrow = nt, ncol = nt)
  if (nt > 1) {
    p_vals <- numeric(0)
    for (i in seq_len(nt - 1))
      for (j in seq(i + 1, nt))
        p_vals <- c(p_vals, p_mat[i, j])
    p_adj <- p.adjust(p_vals, method = "fdr")
    idx <- 1
    for (i in seq_len(nt - 1)) {
      for (j in seq(i + 1, nt)) {
        p_adj_mat[i, j] <- p_adj[idx]; p_adj_mat[j, i] <- p_adj[idx]
        idx <- idx + 1
      }
    }
  }

  df_deg <- if (n_geno >= 3) n_geno - 2L else NA_integer_
  crit_r  <- if (!is.na(df_deg)) qt(0.975, df_deg) / sqrt(df_deg + qt(0.975, df_deg)^2) else NA_real_

  list(
    n_observations       = n_geno,
    df                   = if (is.na(df_deg)) NULL else df_deg,
    critical_r           = if (is.na(crit_r)) NULL else crit_r,
    r_matrix             = unname(r_mat),
    p_matrix             = unname(p_mat),
    p_adj_matrix         = unname(p_adj_mat),
    ci_lower_matrix      = unname(ci_lower),
    ci_upper_matrix      = unname(ci_upper),
    # Signal to consumers that all inference values (p, CI) are approximations
    inference_approximate = TRUE,
    inference_note        = paste0(
      "Genotypic correlation is estimated from variance components using bivariate REML ",
      "(sommer). p-values and confidence intervals use a Fisher z approximation with ",
      "n_genotypes as the effective sample size — this understates uncertainty relative ",
      "to the true REML information matrix. Interpret cautiously, especially with small ",
      "genotype panels or when convergence warnings are present."
    ),
    warnings             = vc_warnings
  )
}
