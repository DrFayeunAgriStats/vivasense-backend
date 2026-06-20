# VivaSense Genetics Engine - R Implementation
# Three-layer architecture: Computation | Validation | Interpretation
# Supports single-environment and multi-environment analysis
# Returns structured JSON + interpretation text
# BUILD_TRIGGER: 2026-06-20 20:35 - Assumption diagnostics deployment

suppressPackageStartupMessages({
  library(jsonlite)
  library(agricolae)
  library(dplyr)
  library(tidyr)
})

HAS_CAR <- requireNamespace("car", quietly = TRUE)

# Guard against near-zero error variance causing F-ratio overflow
safe_f_ratio <- function(ms_effect, ms_error, min_error_ms = 1e-10) {
  if (is.null(ms_error) || is.na(ms_error) || ms_error < min_error_ms) {
    return(NA_real_)
  }
  return(ms_effect / ms_error)
}

sanitize_anova_f_values <- function(anova_table) {
  if (is.null(anova_table) || !is.data.frame(anova_table)) {
    return(anova_table)
  }
  if (!("Mean Sq" %in% names(anova_table)) || !("F value" %in% names(anova_table))) {
    return(anova_table)
  }

  rn <- rownames(anova_table)

  # Subplot error: "Error B" (split-plot) or last "Residuals"/"Within"
  sub_error_rows <- which(rn == "Error B")
  if (length(sub_error_rows) == 0) {
    sub_error_rows <- grep("Residuals|Within", rn, ignore.case = TRUE)
  }
  residual_ms <- if (length(sub_error_rows) > 0) {
    suppressWarnings(as.numeric(anova_table[sub_error_rows[length(sub_error_rows)], "Mean Sq"]))
  } else {
    NA_real_
  }
  residual_df <- if (length(sub_error_rows) > 0) {
    suppressWarnings(as.integer(anova_table[sub_error_rows[length(sub_error_rows)], "Df"]))
  } else {
    NA_integer_
  }

  # Whole-plot error: "Error A" (split-plot) or "whole_plot_error" (legacy)
  wp_error_ms <- if ("Error A" %in% rn) {
    suppressWarnings(as.numeric(anova_table["Error A", "Mean Sq"]))
  } else if ("whole_plot_error" %in% rn) {
    suppressWarnings(as.numeric(anova_table["whole_plot_error", "Mean Sq"]))
  } else {
    NA_real_
  }
  wp_error_df <- if ("Error A" %in% rn) {
    suppressWarnings(as.integer(anova_table["Error A", "Df"]))
  } else if ("whole_plot_error" %in% rn) {
    suppressWarnings(as.integer(anova_table["whole_plot_error", "Df"]))
  } else {
    NA_integer_
  }

  # Row names that should not carry an F-test
  error_pattern <- "^(Error A|Error B|whole_plot_error|Replication|Residuals|Within)$"

  for (term in rn) {
    ms_effect <- suppressWarnings(as.numeric(anova_table[term, "Mean Sq"]))
    df_effect <- suppressWarnings(as.integer(anova_table[term, "Df"]))

    if (grepl(error_pattern, term, ignore.case = TRUE)) {
      anova_table[term, "F value"] <- NA_real_
      if ("Pr(>F)" %in% names(anova_table)) {
        anova_table[term, "Pr(>F)"] <- NA_real_
      }
      next
    }

    # Skip the Type III intercept row — it has no meaningful F test here
    if (grepl("^\\(Intercept\\)$", term, ignore.case = FALSE)) {
      next
    }

    # main_plot is tested against Error A (whole-plot error);
    # all other terms (sub_plot, interactions) are tested against Error B
    if (identical(term, "main_plot")) {
      denom_ms <- wp_error_ms
      denom_df <- wp_error_df
    } else {
      denom_ms <- residual_ms
      denom_df <- residual_df
    }
    
    f_val <- safe_f_ratio(ms_effect, denom_ms)
    anova_table[term, "F value"] <- f_val
    
    # Recalculate p-value with correct error df
    if ("Pr(>F)" %in% names(anova_table)) {
      if (is.na(f_val) || is.na(df_effect) || is.na(denom_df)) {
        anova_table[term, "Pr(>F)"] <- NA_real_
      } else {
        anova_table[term, "Pr(>F)"] <- pf(f_val, df_effect, denom_df, lower.tail = FALSE)
      }
    }
  }

  anova_table
}

# ============================================================================
# MEAN SEPARATION HELPER
# ============================================================================

#' Compute Tukey HSD mean separation for genotypes.
#'
#' @param model   aov or lm model fitted to the data
#' @param trait_name  character, used only for warning messages
#' @param df_error    integer, residual df (optional; extracted from model if NULL)
#' @param ms_error    numeric, residual MS (optional; extracted from model if NULL)
#' @return list(genotype, mean, se, group, test, alpha) or NULL on failure
#'
compute_mean_separation <- function(model, trait_name = "Trait",
                                    df_error = NULL, ms_error = NULL,
                                    trt = "genotype") {
  tukey <- tryCatch({
    if (!is.null(df_error) && !is.null(ms_error)) {
      HSD.test(model, trt,
               DFerror = df_error, MSerror = ms_error,
               group = TRUE, console = FALSE)
    } else {
      HSD.test(model, trt, group = TRUE, console = FALSE)
    }
  }, error = function(e) {
    message(sprintf("[WARN] Tukey HSD failed for %s: %s — trying LSD.test fallback", trait_name, conditionMessage(e)))
    NULL
  })

  # Fallback: LSD.test when Tukey HSD fails (e.g. lm models, singular residuals)
  if (is.null(tukey)) {
    tukey <- tryCatch({
      if (!is.null(df_error) && !is.null(ms_error)) {
        LSD.test(model, trt,
                 DFerror = df_error, MSerror = ms_error,
                 group = TRUE, console = FALSE)
      } else {
        LSD.test(model, trt, group = TRUE, console = FALSE)
      }
    }, error = function(e) {
      message(sprintf("[WARN] LSD.test fallback also failed for %s: %s", trait_name, conditionMessage(e)))
      NULL
    })
    if (!is.null(tukey)) {
      message(sprintf("[INFO] Using LSD.test (Fisher's LSD) for mean separation of %s", trait_name))
      # Override test name below
      attr(tukey, ".test_name") <- "Fisher LSD"
    }
  }

  if (is.null(tukey)) return(NULL)

  test_name <- if (!is.null(attr(tukey, ".test_name"))) attr(tukey, ".test_name") else "Tukey HSD"

  groups_df  <- tukey$groups    # sorted by mean desc: col1=mean, col "groups"=letters
  means_df   <- tukey$means     # alphabetical: col1=mean, se, std, ...
  geno_order <- rownames(groups_df)

  se_vals <- if ("se" %in% names(means_df)) {
    as.numeric(means_df[geno_order, "se"])
  } else {
    rep(NA_real_, length(geno_order))
  }

  # Return atomic vectors so jsonlite never unboxes them with auto_unbox=TRUE
  # Use the first non-"groups" numeric column by name instead of positional [[1]]
  # to guard against agricolae versions that order columns differently.
  mean_col <- setdiff(names(groups_df), "groups")[1]
  list(
    genotype = geno_order,
    mean     = as.numeric(groups_df[[mean_col]]),
    se       = se_vals,
    group    = as.character(groups_df$groups),
    test     = test_name,
    alpha    = 0.05
  )
}

# Standard Falconer & Mackay (1996) selection intensity constants by
# proportion selected (%) used for nearest-match disclosure.
SELECTION_INTENSITY_TABLE <- c(
  "5" = 2.06,
  "7" = 1.755,
  "10" = 1.40,
  "15" = 1.268,
  "20" = 1.11,
  "30" = 0.966
)

normalize_selection_intensity <- function(selection_intensity = 1.40) {
  si <- suppressWarnings(as.numeric(selection_intensity))
  if (is.na(si) || si <= 0) {
    return(1.40)
  }
  si
}

selection_intensity_to_percent <- function(selection_intensity) {
  si <- normalize_selection_intensity(selection_intensity)
  vals <- as.numeric(SELECTION_INTENSITY_TABLE)
  keys <- names(SELECTION_INTENSITY_TABLE)
  idx <- which.min(abs(vals - si))
  as.integer(keys[idx])
}

compute_genetic_advance <- function(h2, sigma_p, i = 1.40) {
  if (is.na(h2) || is.na(sigma_p)) {
    return(NA_real_)
  }
  if (!is.finite(h2) || !is.finite(sigma_p) || sigma_p < 0) {
    return(NA_real_)
  }
  GA <- h2 * normalize_selection_intensity(i) * sigma_p
  return(GA)
}

# ============================================================================
# LAYER 1: COMPUTATION LAYER
# Core mathematical and statistical functions
# ============================================================================

#' Single-Environment Genetics Analysis
#'
#' @param data data frame with columns: genotype, rep, trait_value
#'             (CRD: also optionally "factor" for factorial CRD)
#' @param trait_name character, name of the trait being analyzed
#' @param crd_mode   logical; TRUE = Completely Randomised Design (no blocking).
#'                   FALSE (default) = RCBD (rep is a blocking factor).
#' @return list with variance components and heritability estimates
#'
compute_single_environment <- function(data, trait_name = "Trait",
                                       crd_mode = FALSE,
                                       selection_intensity = 1.40) {

  selection_intensity <- normalize_selection_intensity(selection_intensity)

  has_splitplot <- all(c("main_plot", "sub_plot") %in% colnames(data))

  # For generic split-plot RCBD there is no genotype column in the data.
  # All other designs carry a genotype column.
  has_genotype <- "genotype" %in% colnames(data)
  if (!has_splitplot && has_genotype) {
    data$genotype <- factor(data$genotype)
  }
  data$rep <- factor(data$rep)

  n_genotypes <- if (!has_splitplot && has_genotype) nlevels(data$genotype) else NA_integer_

  # ── Model selection ──────────────────────────────────────────────────────────
  # CRD (no blocking):      trait_value ~ genotype
  # Factorial CRD:          trait_value ~ genotype * factor
  # RCBD (blocking):        trait_value ~ rep + genotype
  # Factorial RCBD:         trait_value ~ rep + genotype * factor
  # Split-plot RCBD:        trait_value ~ main_plot * sub_plot + Error(rep/main_plot)
  #   (rep is NOT a fixed term outside Error(); it enters only through the
  #    whole-plot error stratum.  genotype is NOT part of the generic formula.)
  has_factor <- "factor" %in% colnames(data)
  n_factor_levels <- if (has_factor) {
    data$factor <- factor(data$factor)
    nlevels(data$factor)
  } else {
    1L
  }

  # Guard: catch single-level factors before aov() to produce clear errors
  if (!has_splitplot && has_genotype && !is.na(n_genotypes) && n_genotypes < 2) {
    stop(sprintf(
      "Genotype column must have \u22652 levels for analysis. Got %d level(s). Check your data.",
      n_genotypes
    ))
  }
  if (!crd_mode && !has_splitplot && nlevels(data$rep) < 2) {
    stop(sprintf(
      "Replication column must have \u22652 levels for RCBD. Got %d level(s). Check your data.",
      nlevels(data$rep)
    ))
  }
  if (has_factor && n_factor_levels < 2) {
    stop(sprintf(
      "Factor column must have \u22652 levels for factorial analysis. Got %d level(s). Check your data.",
      n_factor_levels
    ))
  }

  if (has_splitplot) {
    data$main_plot <- factor(data$main_plot)
    data$sub_plot  <- factor(data$sub_plot)
    n_reps <- nlevels(data$rep)
    message(sprintf(
      "[INFO] Split-plot RCBD model for %s: trait_value ~ main_plot * sub_plot + Error(rep/main_plot)",
      trait_name
    ))
    model <- aov(trait_value ~ main_plot * sub_plot + Error(rep/main_plot), data = data)
  } else if (crd_mode) {
    # n_reps = average observations per genotype (inferred from data)
    obs_per_geno <- table(data$genotype)
    n_reps <- round(mean(obs_per_geno))

    if (has_factor) {
      message(sprintf(
        "[INFO] Factorial CRD model for %s: trait_value ~ genotype * factor (%d factor levels)",
        trait_name, n_factor_levels
      ))
      model <- aov(trait_value ~ genotype * factor, data = data)
    } else {
      message(sprintf(
        "[INFO] CRD model for %s: trait_value ~ genotype (n_reps inferred = %d)",
        trait_name, n_reps
      ))
      model <- aov(trait_value ~ genotype, data = data)
    }
  } else {
    # RCBD: rep is a fixed blocking factor
    n_reps <- nlevels(data$rep)
    if (has_factor) {
      message(sprintf(
        "[INFO] Factorial RCBD model for %s: trait_value ~ rep + genotype * factor (%d factor levels)",
        trait_name, n_factor_levels
      ))
      model <- aov(trait_value ~ rep + genotype * factor, data = data)
    } else {
      model <- aov(trait_value ~ rep + genotype, data = data)
    }
  }

  # ── ANOVA table extraction ───────────────────────────────────────────────────
  # Split-plot models fitted with aov(...+Error(...)) require summary() to
  # get correct per-stratum F-tests.  All other models use anova() directly.
  if (has_splitplot) {
    sp_summ <- summary(model)
    strata_names <- names(sp_summ)

    # Identify strata by name (primary) — deterministic for Error(rep/main_plot):
    #   "Error: rep:main_plot" → wp_table  (whole-plot treatment stratum)
    #   "Error: Within"        → sub_table (subplot stratum)
    #   other (only "Error: rep") → rep_stratum_tbl (block/replication stratum)
    wp_table <- NULL
    sub_table <- NULL
    rep_stratum_tbl <- NULL
    for (nm in strata_names) {
      # Guard: sp_summ[[nm]][[1]] may return a non-data-frame on some R builds
      tbl <- tryCatch(sp_summ[[nm]][[1]], error = function(e) {
        message(sprintf("[WARN] stratum '%s' access failed: %s", nm, conditionMessage(e)))
        NULL
      })
      if (is.null(tbl)) next
      rn <- rownames(tbl)
      message(sprintf("[DEBUG] split-plot stratum '%s': rows = %s",
                      nm, paste(rn, collapse = ", ")))

      # Primary: name-based detection — deterministic for Error(rep/main_plot)
      if (grepl("main_plot", nm, fixed = TRUE)) {
        # "Error: rep:main_plot" → whole-plot treatment stratum
        wp_table <- tbl
      } else if (grepl("Within", nm, fixed = TRUE)) {
        # "Error: Within" → subplot stratum
        sub_table <- tbl
      } else {
        # Fallback: rep-only stratum ("Error: rep") — block variation, Residuals only
        if (!is.null(rn) && length(rn) == 1L && "Residuals" %in% rn) {
          rep_stratum_tbl <- tbl
        }
      }
    }
    if (is.null(wp_table) || is.null(sub_table)) {
      stop(paste(
        "Split-plot summary missing expected strata. Available:",
        paste(strata_names, collapse = ", ")
      ))
    }

    # ── Build ANOVA table: Replication | Main plot (A) | Error A | Sub plot (B) | A×B | Error B
    # 1. Replication row (from the rep-only stratum's Residuals)
    # Use last-row fallback consistent with wp/sub: rep stratum table has only Df/Sum Sq/Mean Sq
    # (no F value or Pr(>F) columns in some R builds), so always add both NA columns to match.
    rep_row <- NULL
    if (!is.null(rep_stratum_tbl)) {
      rep_row <- rep_stratum_tbl[nrow(rep_stratum_tbl), , drop = FALSE]
      rownames(rep_row) <- "Replication"
      rep_row[, "F value"] <- NA_real_
      rep_row[, "Pr(>F)"]  <- NA_real_  # always add — rep stratum table omits this column
    }

    # 2. Main-plot treatment rows + Error A
    # Use last-row fallback: R places Residuals last; some builds use integer rownames.
    wp_error_mask <- rownames(wp_table) %in% c("Residuals", "Within")
    if (!any(wp_error_mask)) wp_error_mask[nrow(wp_table)] <- TRUE
    wp_treat <- wp_table[!wp_error_mask, , drop = FALSE]
    wp_error  <- wp_table[ wp_error_mask, , drop = FALSE]
    if (nrow(wp_error) == 1L) rownames(wp_error) <- "Error A"
    wp_error[, "F value"] <- NA_real_
    if ("Pr(>F)" %in% names(wp_error)) wp_error[, "Pr(>F)"] <- NA_real_

    # 4. Subplot treatment and interaction rows + Error B
    sub_error_mask <- rownames(sub_table) %in% c("Residuals", "Within")
    if (!any(sub_error_mask)) sub_error_mask[nrow(sub_table)] <- TRUE
    sub_treat <- sub_table[!sub_error_mask, , drop = FALSE]
    sub_error  <- sub_table[ sub_error_mask, , drop = FALSE]
    if (nrow(sub_error) == 1L) rownames(sub_error) <- "Error B"
    sub_error[, "F value"] <- NA_real_
    if ("Pr(>F)" %in% names(sub_error)) sub_error[, "Pr(>F)"] <- NA_real_

    # 6. Assemble full table — normalise all sub-tables to the same 5-column schema
    # before rbind so column-count mismatches don't produce row.names errors.
    .norm_cols <- function(tbl) {
      if (!"F value" %in% names(tbl)) tbl[, "F value"] <- NA_real_
      if (!"Pr(>F)"  %in% names(tbl)) tbl[, "Pr(>F)"]  <- NA_real_
      tbl[, c("Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)"), drop = FALSE]
    }
    wp_treat  <- .norm_cols(wp_treat)
    wp_error  <- .norm_cols(wp_error)
    sub_treat <- .norm_cols(sub_treat)
    sub_error <- .norm_cols(sub_error)

    anova_table <- tryCatch({
      if (!is.null(rep_row)) {
        rep_row <- .norm_cols(rep_row)
        rbind(rep_row, wp_treat, wp_error, sub_treat, sub_error)
      } else {
        rbind(wp_treat, wp_error, sub_treat, sub_error)
      }
    }, error = function(e) {
      message(sprintf("[WARN] split-plot ANOVA table rbind failed (%s) — excluding rep row", conditionMessage(e)))
      rbind(wp_treat, wp_error, sub_treat, sub_error)
    })

    ms_genotype <- NA_real_
    ms_error_A  <- as.numeric(wp_error[1L, "Mean Sq"])
    ms_error_B  <- as.numeric(sub_error[1L, "Mean Sq"])
    df_error_A  <- as.integer(wp_error[1L, "Df"])
    df_error_B  <- as.integer(sub_error[1L, "Df"])
    ms_error    <- ms_error_B   # subplot error is canonical for backward-compat CV
    n_main      <- nlevels(data$main_plot)

    # Stamp correct F-values directly using known strata error terms.
    # sanitize_anova_f_values uses row-name detection which can pick the wrong
    # residual in edge cases; here we use the already-computed ms/df values so
    # the denominator is unambiguous regardless of assembled row names.
    #   wp_treat rows (whole-plot stratum)  → tested against Error A
    #   sub_treat rows (sub-plot + A×B)     → tested against Error B
    for (rn_sp in rownames(wp_treat)) {
      ms_eff <- suppressWarnings(as.numeric(anova_table[rn_sp, "Mean Sq"]))
      df_eff <- suppressWarnings(as.integer(anova_table[rn_sp, "Df"]))
      fv     <- safe_f_ratio(ms_eff, ms_error_A)
      anova_table[rn_sp, "F value"] <- fv
      anova_table[rn_sp, "Pr(>F)"]  <- if (is.na(fv) || is.na(df_eff) || is.na(df_error_A)) NA_real_ else
        pf(fv, df_eff, df_error_A, lower.tail = FALSE)
    }
    for (rn_sp in rownames(sub_treat)) {
      ms_eff <- suppressWarnings(as.numeric(anova_table[rn_sp, "Mean Sq"]))
      df_eff <- suppressWarnings(as.integer(anova_table[rn_sp, "Df"]))
      fv     <- safe_f_ratio(ms_eff, ms_error_B)
      anova_table[rn_sp, "F value"] <- fv
      anova_table[rn_sp, "Pr(>F)"]  <- if (is.na(fv) || is.na(df_eff) || is.na(df_error_B)) NA_real_ else
        pf(fv, df_eff, df_error_B, lower.tail = FALSE)
    }
    # cv_A / cv_B computed in the variance_components block below (grand_mean available there)

  } else {
    if (HAS_CAR) {
      anova_raw <- car::Anova(model, type = "III")
      anova_raw[["Mean Sq"]] <- anova_raw[["Sum Sq"]] / anova_raw[["Df"]]
    } else {
      warning("car package unavailable - falling back to Type I SS (base anova). Results may differ from Type III SS for unbalanced designs.")
      anova_raw <- anova(model)
    }
    anova_table <- anova_raw

    ms_genotype <- if ("genotype" %in% rownames(anova_table)) {
      anova_table["genotype", "Mean Sq"]
    } else {
      NA_real_
    }

    if ("Residuals" %in% rownames(anova_table)) {
      ms_error <- anova_table["Residuals", "Mean Sq"]
    } else {
      residual_rows <- grep("Residuals|Within", rownames(anova_table), ignore.case = TRUE)
      if (length(residual_rows) > 0) {
        ms_error <- anova_table[residual_rows[length(residual_rows)], "Mean Sq"]
      } else {
        stop("ANOVA table missing residual term")
      }
    }
  }

  anova_table <- sanitize_anova_f_values(anova_table)

  # ── Observation-Level Diagnostics From Fitted ANOVA Model ─────────────────
  diag_model <- tryCatch({
    if (has_splitplot) model[["Error: Within"]] else model
  }, error = function(e) model)

  diag_frame <- tryCatch(model.frame(diag_model), error = function(e) NULL)

  observed_full <- tryCatch({
    if (!is.null(diag_frame)) {
      as.numeric(model.response(diag_frame))
    } else {
      as.numeric(data$trait_value)
    }
  }, error = function(e) NULL)

  resid_full <- tryCatch({
    as.numeric(residuals(diag_model))
  }, error = function(e) NULL)

  fitted_full <- tryCatch({
    as.numeric(fitted(diag_model))
  }, error = function(e) NULL)

  standardized_resid_full <- tryCatch({
    as.numeric(rstandard(diag_model))
  }, error = function(e) {
    if (!is.null(resid_full)) as.numeric(scale(resid_full)) else NULL
  })

  cooks_distance_full <- tryCatch({
    as.numeric(cooks.distance(diag_model))
  }, error = function(e) NULL)

  available_lengths <- c(
    length(observed_full),
    length(fitted_full),
    length(resid_full),
    length(standardized_resid_full),
    length(cooks_distance_full)
  )
  available_lengths <- available_lengths[available_lengths > 0]
  diag_n <- if (length(available_lengths) > 0) min(available_lengths) else 0

  if (diag_n > 0) {
    observed_full <- observed_full[seq_len(diag_n)]
    fitted_full <- fitted_full[seq_len(diag_n)]
    resid_full <- resid_full[seq_len(diag_n)]
    standardized_resid_full <- standardized_resid_full[seq_len(diag_n)]
    cooks_distance_full <- cooks_distance_full[seq_len(diag_n)]
  } else {
    observed_full <- NULL
    fitted_full <- NULL
    resid_full <- NULL
    standardized_resid_full <- NULL
    cooks_distance_full <- NULL
  }

  treatment_labels <- NULL
  block_labels <- NULL
  if (diag_n > 0) {
    treatment_labels <- tryCatch({
      if (!is.null(diag_frame) && "genotype" %in% names(diag_frame)) {
        as.character(diag_frame$genotype)
      } else if (!is.null(diag_frame) && all(c("main_plot", "sub_plot") %in% names(diag_frame))) {
        paste0(as.character(diag_frame$main_plot), " x ", as.character(diag_frame$sub_plot))
      } else if (!is.null(diag_frame) && "sub_plot" %in% names(diag_frame)) {
        as.character(diag_frame$sub_plot)
      } else {
        rep("Observation", diag_n)
      }
    }, error = function(e) rep("Observation", diag_n))

    block_labels <- tryCatch({
      if (!is.null(diag_frame) && "rep" %in% names(diag_frame)) {
        as.character(diag_frame$rep)
      } else {
        rep(NA_character_, diag_n)
      }
    }, error = function(e) rep(NA_character_, diag_n))

    treatment_labels <- treatment_labels[seq_len(min(length(treatment_labels), diag_n))]
    if (length(treatment_labels) < diag_n) {
      treatment_labels <- c(treatment_labels, rep("Observation", diag_n - length(treatment_labels)))
    }

    block_labels <- block_labels[seq_len(min(length(block_labels), diag_n))]
    if (length(block_labels) < diag_n) {
      block_labels <- c(block_labels, rep(NA_character_, diag_n - length(block_labels)))
    }
  }

  cook_threshold <- if (diag_n > 0) (4 / diag_n) else NA_real_
  extreme_flags <- if (!is.null(standardized_resid_full)) abs(standardized_resid_full) > 3 else logical(0)
  influential_flags <- if (!is.null(cooks_distance_full) && is.finite(cook_threshold)) {
    cooks_distance_full > cook_threshold
  } else {
    logical(length(extreme_flags))
  }
  if (length(influential_flags) == 0 && length(extreme_flags) > 0) {
    influential_flags <- rep(FALSE, length(extreme_flags))
  }

  n_extreme_outliers <- if (length(extreme_flags) > 0) sum(extreme_flags, na.rm = TRUE) else 0
  n_influential_obs <- if (length(influential_flags) > 0) sum(influential_flags, na.rm = TRUE) else 0

  diagnostic_observations <- if (diag_n > 0) {
    lapply(seq_len(diag_n), function(i) {
      list(
        observation = i,
        treatment = treatment_labels[i],
        block = block_labels[i],
        observed = observed_full[i],
        fitted = fitted_full[i],
        residual = resid_full[i],
        standardized_residual = standardized_resid_full[i],
        cooks_distance = cooks_distance_full[i],
        extreme_outlier = isTRUE(extreme_flags[i]),
        influential = isTRUE(influential_flags[i])
      )
    })
  } else NULL

  residuals_vs_treatment <- if (!is.null(diagnostic_observations)) {
    lapply(diagnostic_observations, function(x) {
      list(
        treatment = x$treatment,
        residual = x$residual,
        standardized_residual = x$standardized_residual
      )
    })
  } else NULL

  scale_location_points <- if (!is.null(diagnostic_observations)) {
    lapply(diagnostic_observations, function(x) {
      list(
        fitted = x$fitted,
        sqrt_abs_standardized_residual = if (is.null(x$standardized_residual) || is.na(x$standardized_residual)) NA_real_ else sqrt(abs(x$standardized_residual))
      )
    })
  } else NULL

  cooks_distance_points <- if (!is.null(diagnostic_observations)) {
    lapply(diagnostic_observations, function(x) {
      list(
        observation = x$observation,
        treatment = x$treatment,
        cooks_distance = x$cooks_distance
      )
    })
  } else NULL

  standardized_residual_points <- if (!is.null(diagnostic_observations)) {
    lapply(diagnostic_observations, function(x) {
      list(
        observation = x$observation,
        treatment = x$treatment,
        standardized_residual = x$standardized_residual
      )
    })
  } else NULL

  flagged_observations <- if (!is.null(diagnostic_observations)) {
    Filter(function(x) isTRUE(x$extreme_outlier) || isTRUE(x$influential), diagnostic_observations)
  } else NULL

  # ── Assumption Tests (Shapiro-Wilk + Levene/Bartlett) ───────────────────────
  assumption_tests_out <- tryCatch({
    # Split-plot: use within-stratum (Error B) residuals for Shapiro-Wilk
    resid_vec <- if (!is.null(resid_full)) resid_full else {
      tryCatch(
        if (has_splitplot) residuals(model[["Error: Within"]]) else residuals(model),
        error = function(e) residuals(model)
      )
    }
    resid_vec <- resid_vec[is.finite(resid_vec)]

    normality_result <- NULL
    if (length(resid_vec) >= 3 && length(resid_vec) <= 5000) {
      sw <- tryCatch(shapiro.test(resid_vec), error = function(e) NULL)
      if (!is.null(sw)) {
        normality_result <- list(
          test           = "Shapiro-Wilk",
          statistic      = as.numeric(sw$statistic),
          p_value        = as.numeric(sw$p.value),
          passed         = sw$p.value >= 0.05,
          interpretation = if (sw$p.value >= 0.05)
            sprintf(
              "Normality assumption supported (W = %.4f, p = %.4f). Residuals are consistent with a normal distribution.",
              as.numeric(sw$statistic), as.numeric(sw$p.value))
            else
            sprintf(
              "Normality assumption may be violated (W = %.4f, p = %.4f). Consider data transformation or non-parametric alternatives.",
              as.numeric(sw$statistic), as.numeric(sw$p.value))
        )
      }
    }

    homogeneity_result <- NULL
    grp_col <- if (has_splitplot) data$main_plot else if (has_genotype) data$genotype else NULL
    if (!is.null(grp_col) && nlevels(as.factor(grp_col)) >= 2) {
      if (HAS_CAR) {
        lv <- tryCatch(
          car::leveneTest(data$trait_value ~ as.factor(grp_col)),
          error = function(e) NULL
        )
        if (!is.null(lv)) {
          lv_stat <- lv[1, "F value"]
          lv_p    <- lv[1, "Pr(>F)"]
          homogeneity_result <- list(
            test           = "Levene",
            statistic      = as.numeric(lv_stat),
            p_value        = as.numeric(lv_p),
            passed         = lv_p >= 0.05,
            interpretation = if (lv_p >= 0.05)
              sprintf(
                "Homogeneity of variance supported (F = %.4f, p = %.4f). Equal variance assumption is not violated.",
                as.numeric(lv_stat), as.numeric(lv_p))
              else
              sprintf(
                "Heterogeneity of variance detected (F = %.4f, p = %.4f). Variance differs across treatment groups.",
                as.numeric(lv_stat), as.numeric(lv_p))
          )
        }
      }
      if (is.null(homogeneity_result)) {
        bt <- tryCatch(bartlett.test(data$trait_value ~ grp_col), error = function(e) NULL)
        if (!is.null(bt)) {
          homogeneity_result <- list(
            test           = "Bartlett",
            statistic      = as.numeric(bt$statistic),
            p_value        = as.numeric(bt$p.value),
            passed         = bt$p.value >= 0.05,
            interpretation = if (bt$p.value >= 0.05)
              sprintf(
                "Homogeneity of variance supported (K² = %.4f, p = %.4f). Equal variance assumption is not violated.",
                as.numeric(bt$statistic), as.numeric(bt$p.value))
              else
              sprintf(
                "Heterogeneity of variance detected (K² = %.4f, p = %.4f). Variance differs across treatment groups.",
                as.numeric(bt$statistic), as.numeric(bt$p.value))
          )
        }
      }
    }

    norm_pass <- if (!is.null(normality_result))   normality_result$passed   else NA
    homo_pass <- if (!is.null(homogeneity_result)) homogeneity_result$passed else NA
    overall_passed <- isTRUE(norm_pass) && isTRUE(homo_pass)
    overall_interp <- if (is.na(norm_pass) && is.na(homo_pass)) {
      "Assumption tests could not be computed for this dataset."
    } else if (overall_passed) {
      "Both normality and homogeneity of variance assumptions are supported. ANOVA results are reliable under standard parametric assumptions."
    } else if (isFALSE(norm_pass) && isFALSE(homo_pass)) {
      "Both assumptions may be violated. ANOVA results should be interpreted with caution; consider data transformation."
    } else if (isFALSE(norm_pass)) {
      "The normality assumption may be violated. ANOVA is generally robust to mild non-normality with balanced designs, but results should be verified."
    } else {
      "The homogeneity of variance assumption may be violated. Results should be interpreted cautiously, particularly for unbalanced designs."
    }

    reviewer_status <- if (isTRUE(norm_pass) && isTRUE(homo_pass) && n_influential_obs == 0) {
      "PASS"
    } else {
      "WARN"
    }
    reviewer_summary <- if (reviewer_status == "PASS") {
      "✓ Normality satisfied; ✓ Homogeneity satisfied; ✓ No influential outliers detected"
    } else {
      "⚠ Assumptions violated"
    }

    out <- list(overall = list(
      passed = overall_passed,
      interpretation = overall_interp,
      reviewer_summary = reviewer_summary
    ))
    if (!is.null(normality_result))   out$normality   <- normality_result
    if (!is.null(homogeneity_result)) out$homogeneity <- homogeneity_result
    out$outlier_detection <- list(
      standardized_residual_threshold = 3,
      cooks_distance_threshold = cook_threshold,
      n_extreme_outliers = n_extreme_outliers,
      n_influential_observations = n_influential_obs,
      flagged_observations = flagged_observations,
      interpretation = if (n_extreme_outliers == 0 && n_influential_obs == 0) {
        "No extreme outliers or influential observations were detected under the configured thresholds."
      } else {
        "Extreme residual outliers and/or influential observations were detected; inspect flagged records before inference."
      }
    )
    out$reviewer_mode <- list(
      status = reviewer_status,
      normality_satisfied = isTRUE(norm_pass),
      homogeneity_satisfied = isTRUE(homo_pass),
      no_influential_outliers = n_influential_obs == 0,
      summary = reviewer_summary
    )
    if (length(out) <= 1) NULL else out

  }, error = function(e) {
    message(sprintf("[WARN] Assumption tests failed for %s: %s", trait_name, conditionMessage(e)))
    NULL
  })

  # Grand mean — needed for both split-plot and genotype-based paths
  grand_mean    <- mean(data$trait_value, na.rm = TRUE)
  mean_is_valid <- !is.na(grand_mean) && grand_mean != 0

  # Variance components, heritability, and genetic parameters are only
  # meaningful for genotype-based designs.  For generic split-plot RCBD the
  # only relevant error terms are the subplot error and the whole-plot error;
  # genetics-specific blocks are explicitly omitted.
  if (has_splitplot) {
    # Dual CV — computed here where grand_mean is available
    cv_A <- if (!is.na(ms_error_A) && ms_error_A >= 0 && mean_is_valid) {
      (sqrt(ms_error_A) / grand_mean) * 100
    } else NA_real_
    cv_B <- if (!is.na(ms_error_B) && ms_error_B >= 0 && mean_is_valid) {
      (sqrt(ms_error_B) / grand_mean) * 100
    } else NA_real_

    variance_components_out <- list(
      sigma2_subplot_error    = ms_error_B,
      sigma2_whole_plot_error = ms_error_A,
      cv_A                    = cv_A,
      cv_B                    = cv_B,
      n_main_plot_levels      = n_main,
      n_sub_plot_levels       = nlevels(data$sub_plot)
    )
    heritability_out   <- list(not_applicable = TRUE)
    genetic_params_out <- list(not_applicable = TRUE)
    flags_out <- list(
      design_valid = TRUE,
      mean_valid   = mean_is_valid
    )
    negative_sigma2g    <- FALSE
    sigma2_genotype_raw <- NA_real_
  } else {
    # Genotype-based designs: full variance decomposition
    # σ²e = MSE
    # σ²g = (MS_g - MS_e) / (n_reps * n_factor_levels)
    # For simple designs n_factor_levels == 1 so the formula reduces to
    # the standard (MS_g - MS_e) / n_reps.
    sigma2_error        <- ms_error
    sigma2_genotype_raw <- (ms_genotype - ms_error) / (n_reps * n_factor_levels)
    sigma2_genotype     <- max(0, sigma2_genotype_raw)
    negative_sigma2g    <- sigma2_genotype_raw < -0.001

    # Phenotypic variance (entry-mean basis)
    sigma2_phenotypic <- sigma2_genotype + (sigma2_error / n_reps)

    # Broad-sense heritability (h²) on entry-mean basis — clamp to [0, 1]
    if (is.na(sigma2_phenotypic) || sigma2_phenotypic <= 0 || is.na(sigma2_genotype)) {
      h2          <- NA_real_
      h2_is_valid <- FALSE
    } else {
      h2          <- max(0, min(1, sigma2_genotype / sigma2_phenotypic))
      h2_is_valid <- TRUE
    }

    # Genotypic CV, Phenotypic CV, and Genetic Advance at Mean
    gcv <- NA_real_; pcv <- NA_real_; gam <- NA_real_; gam_percent <- NA_real_
    if (mean_is_valid && !is.na(sigma2_genotype) && !is.na(sigma2_phenotypic) && !is.na(h2)) {
      gcv         <- (sqrt(max(0, sigma2_genotype)) / grand_mean) * 100
      pcv         <- (sqrt(sigma2_phenotypic) / grand_mean) * 100
      # Genetic Advance using Falconer form: GA = H² × i × σp
      gam         <- compute_genetic_advance(
        h2 = h2,
        sigma_p = sqrt(max(0, sigma2_phenotypic)),
        i = selection_intensity
      )
      gam_percent <- ifelse(is.na(gam), NA_real_, (gam / grand_mean) * 100)
    }

    variance_components_out <- list(
      sigma2_genotype   = sigma2_genotype,
      sigma2_error      = sigma2_error,
      sigma2_phenotypic = sigma2_phenotypic
    )
    heritability_out <- list(
      h2_broad_sense       = h2,
      h2_is_valid          = h2_is_valid,
      interpretation_basis = "entry-mean (single environment)"
    )
    genetic_params_out <- list(
      GCV                = gcv,
      PCV                = pcv,
      GAM                = gam,
      GAM_percent        = gam_percent,
      selection_intensity = selection_intensity,
      selection_percent = selection_intensity_to_percent(selection_intensity)
    )
    flags_out <- list(
      negative_sigma2_genotype = negative_sigma2g,
      sigma2_g_raw             = sigma2_genotype_raw,
      mean_valid               = mean_is_valid,
      crd_mode                 = crd_mode,
      factorial                = has_factor,
      n_factor_levels          = n_factor_levels
    )
  }

  # Mean separation
  if (has_splitplot) {
    # Helper to format an agricolae LSD/HSD result as a flat list.
    # Returns NULL (not a partial list) on any structural problem so that
    # jsonlite drops the field entirely and Pydantic uses the default None.
    format_lsd_result <- function(sep_obj) {
      if (is.null(sep_obj)) return(NULL)
      tryCatch({
        groups_df <- sep_obj$groups
        means_df  <- sep_obj$means
        if (is.null(groups_df) || !is.data.frame(groups_df) || nrow(groups_df) == 0) return(NULL)
        lvl_order <- rownames(groups_df)
        if (is.null(lvl_order) || length(lvl_order) == 0) return(NULL)
        mean_col <- setdiff(names(groups_df), "groups")[1]
        if (is.na(mean_col) || !(mean_col %in% names(groups_df))) return(NULL)
        se_vals <- if (!is.null(means_df) && "se" %in% names(means_df)) {
          as.numeric(means_df[lvl_order, "se"])
        } else {
          rep(NA_real_, length(lvl_order))
        }
        list(
          genotype = as.character(lvl_order),
          mean     = as.numeric(groups_df[[mean_col]]),
          se       = se_vals,
          group    = as.character(groups_df$groups),
          test     = "Fisher LSD",
          alpha    = 0.05
        )
      }, error = function(e) {
        message(sprintf("[WARN] format_lsd_result failed: %s", conditionMessage(e)))
        NULL
      })
    }

    # Main-plot mean separation (Error A: df = (r-1)(a-1), MS = MSEA)
    message(sprintf("[SPLITPLOT] Starting mean separation for %s", trait_name))
    message("[SPLITPLOT] Aggregating main-plot data...")
    mp_sep_raw <- tryCatch({
      mp_agg <- aggregate(trait_value ~ rep + main_plot, data = data, FUN = mean, na.rm = TRUE)
      message(sprintf("[SPLITPLOT] Main-plot agg: %d rows", nrow(mp_agg)))
      message(sprintf("[SPLITPLOT] Main-plot agg head: %s", paste(capture.output(head(mp_agg, 3)), collapse = "; ")))
      message("[SPLITPLOT] Fitting main-plot model...")
      mp_agg_model <- aov(trait_value ~ rep + main_plot, data = mp_agg)
      message(sprintf("[SPLITPLOT] Main-plot model df.residual: %d", df.residual(mp_agg_model)))
      message("[SPLITPLOT] Running main-plot LSD.test...")
      LSD.test(mp_agg_model, "main_plot", group = TRUE, console = FALSE)
    }, error = function(e) {
      message(sprintf("[ERROR] Main-plot LSD failed for %s: %s", trait_name, conditionMessage(e)))
      NULL
    })
    message(sprintf("[SPLITPLOT] Main-plot LSD result is.null: %s", is.null(mp_sep_raw)))
    main_plot_mean_sep <- format_lsd_result(mp_sep_raw)
    if (!is.null(main_plot_mean_sep))
      main_plot_mean_sep$treatment_label <- "main_plot"
    message(sprintf("[SPLITPLOT] Formatted main_plot_mean_sep is.null: %s", is.null(main_plot_mean_sep)))

    # Sub-plot mean separation (Error B)
    # Mirror the main-plot path: fit a simple aov() on the FULL raw data and call
    # LSD.test in model mode with Error B injected.  Data-frame mode was fragile
    # because its $groups structure differs subtly from model mode, and aggregating
    # to cell means used r̄ = n_main instead of the correct r̄ = n_reps × n_main.
    # With all r*a observations per sub_plot level, LSD.test uses r̄ = r*a, giving
    # SE = sqrt(MS_ErrorB / (r*a)) — the correct formula for sub-plot marginal means.
    sub_sep_raw <- tryCatch({
      if (is.na(df_error_B) || is.na(ms_error_B) || ms_error_B <= 0) {
        message(sprintf("[WARN] Error B invalid (df=%s ms=%s) — skipping sub-plot LSD",
                        df_error_B, ms_error_B))
        NULL
      } else {
        sp_all <- data.frame(
          trait_value = as.numeric(data$trait_value),
          sub_plot    = as.character(data$sub_plot),
          rep         = as.character(data$rep)
        )
        sp_all <- sp_all[!is.na(sp_all$trait_value), , drop = FALSE]
        sp_sub_model <- aov(trait_value ~ sub_plot + rep, data = sp_all)
        message(sprintf("[SPLITPLOT] Sub-plot LSD.test (model mode): df_B=%d ms_B=%f n=%d",
                        df_error_B, ms_error_B, nrow(sp_all)))
        LSD.test(sp_sub_model, "sub_plot",
                 DFerror = df_error_B, MSerror = ms_error_B,
                 group = TRUE, console = FALSE)
      }
    }, error = function(e) {
      message(sprintf("[ERROR] Sub-plot LSD failed for %s: %s", trait_name, conditionMessage(e)))
      NULL
    })
    message(sprintf("[SPLITPLOT] Sub-plot LSD result is.null: %s", is.null(sub_sep_raw)))
    sub_plot_mean_sep <- format_lsd_result(sub_sep_raw)
    if (!is.null(sub_plot_mean_sep)) sub_plot_mean_sep$treatment_label <- "sub_plot"
    message(sprintf("[SPLITPLOT] Formatted mean_sep is.null: %s", is.null(sub_plot_mean_sep)))

    # Interaction means (A×B cell means for interaction plot)
    interaction_means <- tryCatch({
      # Compute cell means across all main_plot × sub_plot combinations
      cell_means <- aggregate(trait_value ~ main_plot + sub_plot, data = data, FUN = mean, na.rm = TRUE)
      # Reshape to wide format for interaction plot: rows = main_plot, cols = sub_plot
      cell_wide <- tidyr::pivot_wider(
        cell_means,
        names_from = sub_plot,
        values_from = trait_value
      )
      # Convert to list structure for JSON: main_plot_levels, sub_plot_levels, means_matrix
      list(
        main_plot_levels = as.character(levels(data$main_plot)),
        sub_plot_levels = as.character(levels(data$sub_plot)),
        cell_means = as.list(cell_means),  # Keep long format too for table display
        means_matrix = as.list(cell_wide), # Wide format for plotting
        cell_se = tryCatch(sqrt(ms_error_B / n_reps), error = function(e) NA_real_)
      )
    }, error = function(e) {
      message(sprintf("[WARN] Interaction means computation failed for %s: %s", trait_name, conditionMessage(e)))
      NULL
    })

    df_err   <- df_error_B
    mean_sep <- sub_plot_mean_sep
    mean_sep_b <- NULL
    factorial_interaction_means <- NULL
  } else {
    df_err   <- anova_table["Residuals", "Df"]
    mean_sep <- compute_mean_separation(model, trait_name,
                                        df_error = df_err, ms_error = ms_error,
                                        trt = "genotype")
    mean_sep_b <- if (has_factor) {
      compute_mean_separation(model, trait_name,
                              df_error = df_err, ms_error = ms_error,
                              trt = "factor")
    } else NULL
    factorial_interaction_means <- if (has_factor && has_genotype) {
      tryCatch({
        int_sep  <- HSD.test(model, c("genotype", "factor"),
                             DFerror = as.integer(df_err), MSerror = ms_error,
                             group = TRUE, console = FALSE)
        rn       <- rownames(int_sep$groups)
        mean_col <- setdiff(names(int_sep$groups), "groups")[1]
        list(
          genotype = sub(":.*",      "", rn),
          factor   = sub("^[^:]*:", "", rn),
          mean     = as.numeric(int_sep$groups[[mean_col]]),
          se       = rep(tryCatch(sqrt(ms_error / n_reps), error = function(e) NA_real_), length(rn)),
          group    = as.character(int_sep$groups$groups),
          test     = "Tukey HSD",
          alpha    = 0.05
        )
      }, error = function(e) {
        message(sprintf("[WARN] Interaction HSD failed for %s (%s) — using cell means",
                        trait_name, conditionMessage(e)))
        cell <- aggregate(trait_value ~ genotype + factor, data = data, FUN = mean)
        cell <- cell[order(-cell$trait_value), ]
        list(
          genotype = as.character(cell$genotype),
          factor   = as.character(cell$factor),
          mean     = cell$trait_value,
          se       = rep(tryCatch(sqrt(ms_error / n_reps), error = function(e) NA_real_), nrow(cell)),
          group    = rep("—", nrow(cell)),
          test     = "Cell Means",
          alpha    = 0.05
        )
      })
    } else NULL
  }

  design_label <- if (has_splitplot) {
    "split_plot_rcbd"
  } else if (crd_mode) {
    if (has_factor) "factorial_crd" else "crd"
  } else {
    if (has_factor) "factorial_rcbd" else "rcbd"
  }

  list(
    environment_mode           = "single_environment",
    design                     = design_label,
    n_genotypes                = n_genotypes,
    n_reps                     = n_reps,
    grand_mean                 = grand_mean,
    variance_components        = variance_components_out,
    heritability               = heritability_out,
    genetic_parameters         = genetic_params_out,
    flags                      = flags_out,
    anova_table                = as.data.frame(anova_table),
    mean_separation            = mean_sep,
    mean_separation_b          = mean_sep_b,
    interaction_separation     = factorial_interaction_means,
    main_plot_mean_separation  = if (has_splitplot) main_plot_mean_sep else NULL,
    interaction_means          = if (has_splitplot) interaction_means else NULL,
    assumption_tests           = assumption_tests_out,
    diagnostic_observations    = diagnostic_observations,
    diagnostic_plots           = list(
      residuals_vs_treatment = residuals_vs_treatment,
      scale_location = scale_location_points,
      cooks_distance = cooks_distance_points,
      standardized_residual = standardized_residual_points
    ),
    residuals                  = resid_full,
    fitted_values              = fitted_full,
    standardized_residuals     = standardized_resid_full,
    cooks_distance             = cooks_distance_full,
    outlier_summary            = list(
      standardized_residual_threshold = 3,
      cooks_distance_threshold = cook_threshold,
      n_extreme_outliers = n_extreme_outliers,
      n_influential_observations = n_influential_obs,
      flagged_observations = flagged_observations
    ),
    ms_genotype                = ms_genotype,
    ms_error                   = ms_error
  )
}


#' Multi-Environment Genetics Analysis
#'
#' @param data data frame with columns: genotype, environment, rep, trait_value
#' @param trait_name character, name of the trait being analyzed
#' @param random_environment logical, if TRUE, include E in denominator of h2
#' @return list with variance components for combined analysis
#'
compute_multi_environment <- function(data, trait_name = "Trait",
                                       random_environment = FALSE,
                                       selection_intensity = 1.40) {

  selection_intensity <- normalize_selection_intensity(selection_intensity)

  # Ensure factors — handles both lowercase (from Python) and Title Case
  data$genotype    <- factor(data$genotype)
  data$environment <- factor(data$environment)
  data$rep         <- factor(data$rep)

  # ── Debug: log factor levels so failures are diagnosable in Render logs ──
  message(sprintf("[DEBUG] Trait: %s", trait_name))
  message(sprintf("[DEBUG] Levels Genotype (%d): %s",
              nlevels(data$genotype),
              paste(levels(data$genotype), collapse = ", ")))
  message(sprintf("[DEBUG] Levels Environment (%d): %s",
              nlevels(data$environment),
              paste(levels(data$environment), collapse = ", ")))
  message(sprintf("[DEBUG] Levels Rep/Block (%d): %s",
              nlevels(data$rep),
              paste(levels(data$rep), collapse = ", ")))
  message(sprintf("[DEBUG] Rows: %d", nrow(data)))

  n_genotypes <- nlevels(data$genotype)
  n_envs      <- nlevels(data$environment)
  n_reps      <- nlevels(data$rep)

  # Guard: catch model errors and return them as structured failures
  model_result <- tryCatch({

    # ANOVA: Main effects + G×E interaction
    # Use lm() with explicit environment:rep (= rep nested in environment) so
    # that the anova table term names are fully predictable.  aov() with %in%
    # can silently alias or drop the genotype:environment term on some R builds.
    model <- lm(trait_value ~ environment + environment:rep + genotype +
                  genotype:environment, data = data)
    anova_table <- anova(model)

    message(sprintf("[DEBUG] anova rownames: %s", paste(rownames(anova_table), collapse = " | ")))

    # Locate the G×E term flexibly — R may order as "genotype:environment"
    # or "environment:genotype" depending on formula parsing.
    ge_term <- rownames(anova_table)[
      grepl("genotype", rownames(anova_table), fixed = TRUE) &
      grepl("environment", rownames(anova_table), fixed = TRUE)
    ]
    if (length(ge_term) == 0) {
      stop(paste(
        "ANOVA table missing G\u00d7E term. Available terms:",
        paste(rownames(anova_table), collapse = ", ")
      ))
    }
    ge_term <- ge_term[1]   # use first match

    list(ok = TRUE, anova_table = anova_table, ge_term = ge_term)

  }, error = function(e) {
    message(sprintf("[ERROR] Trait %s — model failed: %s", trait_name, conditionMessage(e)))
    list(ok = FALSE, message = conditionMessage(e))
  })

  if (!model_result$ok) {
    stop(model_result$message)   # propagates to genetics_analysis tryCatch
  }

  anova_table <- model_result$anova_table
  ge_term     <- model_result$ge_term

  # Extract mean squares
  ms_genotype <- anova_table["genotype", "Mean Sq"]
  ms_ge       <- anova_table[ge_term,    "Mean Sq"]
  ms_error    <- anova_table["Residuals","Mean Sq"]

  anova_table <- sanitize_anova_f_values(anova_table)
  
  # Variance components (fixed genotype, fixed environment, fixed reps)
  sigma2_error <- ms_error
  
  # CRITICAL FIX: Clamp negative variances to zero
  sigma2_genotype_raw <- (ms_genotype - ms_ge) / (n_envs * n_reps)
  sigma2_genotype <- max(0, sigma2_genotype_raw)
  negative_sigma2g <- sigma2_genotype_raw < -0.001
  
  sigma2_ge_raw <- (ms_ge - ms_error) / n_reps
  sigma2_ge <- max(0, sigma2_ge_raw)
  negative_sigma2_ge <- sigma2_ge_raw < -0.001
  
  # Detect weak genotype signal (MSG ≤ MSGE)
  weak_genotype_signal <- ms_genotype <= ms_ge
  
  # Detect weak or negligible G×E (MSGE ≤ MSE)
  weak_ge_signal <- ms_ge <= ms_error
  
  # STRICT: Phenotypic variance on ENTRY-MEAN basis (across environments)
  # Does NOT include σ²E in denominator unless random_environment = TRUE
  if (random_environment) {
    # Advanced mode: treat environment as random
    ms_env <- anova_table["environment", "Mean Sq"]
    sigma2_environment_raw <- (ms_env - ms_error) / (n_genotypes * n_reps)
    sigma2_environment <- max(0, sigma2_environment_raw)
    
    sigma2_phenotypic <- sigma2_genotype + 
                         (sigma2_environment / n_envs) + 
                         (sigma2_ge / n_envs) + 
                         (sigma2_error / (n_envs * n_reps))
  } else {
    # Standard mode: phenotypic variance = G + GE/e + Error/(re)
    sigma2_environment <- NA_real_
    sigma2_phenotypic <- sigma2_genotype + 
                         (sigma2_ge / n_envs) + 
                         (sigma2_error / (n_envs * n_reps))
  }
  
  # CRITICAL FIX: Handle edge case where phenotypic variance ≤ 0
  if (sigma2_phenotypic <= 0) {
    h2 <- NA_real_
    h2_is_valid <- FALSE
  } else {
    # Broad-sense heritability (entry-mean basis)
    h2 <- sigma2_genotype / sigma2_phenotypic
    h2 <- max(0, min(1, h2)) # Clamp to [0, 1]
    h2_is_valid <- TRUE
  }
  
  # Genotypic and Phenotypic CV
  grand_mean <- mean(data$trait_value, na.rm = TRUE)
  
  gcv <- NA_real_
  pcv <- NA_real_
  gam <- NA_real_
  gam_percent <- NA_real_
  mean_is_valid <- !is.na(grand_mean) && grand_mean != 0
  
  if (mean_is_valid) {
    gcv <- (sqrt(max(0, sigma2_genotype)) / grand_mean) * 100
    pcv <- (sqrt(sigma2_phenotypic) / grand_mean) * 100
    
    # Genetic Advance using Falconer form: GA = H² × i × σp
    gam <- compute_genetic_advance(
      h2 = h2,
      sigma_p = sqrt(max(0, sigma2_phenotypic)),
      i = selection_intensity
    )
    gam_percent <- ifelse(is.na(gam), NA_real_, (gam / grand_mean) * 100)
  }
  
  df_err   <- anova_table["Residuals", "Df"]
  mean_sep <- compute_mean_separation(model, trait_name,
                                      df_error = df_err, ms_error = ms_error)

  list(
    environment_mode = "multi_environment",
    n_genotypes = n_genotypes,
    n_environments = n_envs,
    n_reps = n_reps,
    grand_mean = grand_mean,
    variance_components = list(
      sigma2_genotype = sigma2_genotype,
      sigma2_ge = sigma2_ge,
      sigma2_error = sigma2_error,
      sigma2_phenotypic = sigma2_phenotypic,
      heritability_basis = ifelse(random_environment,
                                   "random_environment_model",
                                   "fixed_environment_model")
    ),
    heritability = list(
      h2_broad_sense = h2,
      h2_is_valid = h2_is_valid,
      interpretation_basis = "entry-mean across environments",
      formula = ifelse(random_environment,
                       "σ²p = σ²g + (σ²e / e) + (σ²ge / e) + (σ²error / re)",
                       "σ²p = σ²g + (σ²ge / e) + (σ²error / re)")
    ),
    genetic_parameters = list(
      GCV = gcv,
      PCV = pcv,
      GAM = gam,
      GAM_percent = gam_percent,
      selection_intensity = selection_intensity,
      selection_percent = selection_intensity_to_percent(selection_intensity)
    ),
    flags = list(
      negative_sigma2_genotype = negative_sigma2g,
      sigma2_g_raw = sigma2_genotype_raw,
      negative_sigma2_ge = negative_sigma2_ge,
      sigma2_ge_raw = sigma2_ge_raw,
      weak_genotype_signal = weak_genotype_signal,
      weak_ge_signal = weak_ge_signal,
      mean_valid = mean_is_valid
    ),
    anova_table = as.data.frame(anova_table),
    mean_separation = mean_sep,
    ms_genotype = ms_genotype,
    ms_ge = ms_ge,
    ms_error = ms_error
  )
}


# ============================================================================
# LAYER 2: VALIDATION LAYER
# Check data quality, variance reasonableness, and flag issues
# ============================================================================

validate_input_data <- function(data, env_mode = "single", crd_mode = FALSE) {

  warnings_list <- list()
  is_valid      <- TRUE

  # Required columns vary by design.
  # Generic split-plot RCBD: rep + main_plot + sub_plot + trait_value (no genotype).
  # All other designs: genotype + rep + trait_value.
  # CRD: "rep" is synthetic (present) but not a true blocking factor.
  is_splitplot <- all(c("main_plot", "sub_plot") %in% colnames(data))

  if (is_splitplot) {
    required_cols <- c("rep", "main_plot", "sub_plot", "trait_value")
  } else {
    required_cols <- c("genotype", "rep", "trait_value")
    if (env_mode == "multi") required_cols <- c(required_cols, "environment")
  }

  if (!is_splitplot &&
      xor("main_plot" %in% colnames(data), "sub_plot" %in% colnames(data))) {
    warnings_list$missing_split_plot_columns <-
      "Split-plot design requires both main_plot and sub_plot columns."
    is_valid <- FALSE
  }

  missing_cols <- setdiff(required_cols, colnames(data))
  if (length(missing_cols) > 0) {
    # For CRD mode the absence of "rep" is non-fatal — Python injects a
    # synthetic column, but guard against any edge case where it is absent.
    truly_missing <- if (crd_mode) {
      setdiff(missing_cols, "rep")
    } else {
      missing_cols
    }
    if (length(truly_missing) > 0) {
      warnings_list$missing_columns <- truly_missing
      is_valid <- FALSE
    } else if (length(missing_cols) > 0) {
      warnings_list$note_crd_no_rep <-
        "Replication inferred from repeated observations (CRD assumed)"
    }
  }

  # Check for missing values in key columns
  for (col in required_cols) {
    if (col %in% colnames(data)) {
      na_count <- sum(is.na(data[[col]]))
      if (na_count > 0) {
        warnings_list[[paste0("missing_", col)]] <- na_count
      }
    }
  }

  # Check for NA in trait values
  na_trait <- sum(is.na(data$trait_value))
  if (na_trait > 0) {
    warnings_list$missing_trait_values <- na_trait
  }

  # Check minimum replication (design-aware)
  if (is_splitplot) {
    # For generic split-plot: check that each main_plot × sub_plot cell
    # appears in at least 2 reps.
    min_reps <- data %>%
      group_by(main_plot, sub_plot) %>%
      summarise(n_reps = n_distinct(rep), .groups = "drop") %>%
      pull(n_reps) %>%
      min()
    if (min_reps < 2) {
      warnings_list$insufficient_replication <-
        paste("Minimum reps per main_plot × sub_plot cell:", min_reps,
              "(minimum 2 required)")
      is_valid <- FALSE
    }
  } else if (env_mode == "single") {
    min_reps <- data %>%
      group_by(genotype) %>%
      summarise(n = n(), .groups = "drop") %>%
      pull(n) %>%
      min()
    if (min_reps < 2) {
      warnings_list$insufficient_replication <-
        paste("Minimum observations per genotype:", min_reps, "(minimum 2 required)")
      is_valid <- FALSE
    }
  } else {
    min_reps_per_gxe <- data %>%
      group_by(genotype, environment) %>%
      summarise(n = n(), .groups = "drop") %>%
      pull(n) %>%
      min()
    if (min_reps_per_gxe < 2) {
      warnings_list$insufficient_replication <-
        paste("Minimum reps per G\u00d7E:", min_reps_per_gxe)
      is_valid <- FALSE
    }
  }

  # Check trait variation
  trait_var <- var(data$trait_value, na.rm = TRUE)
  if (is.na(trait_var) || trait_var == 0) {
    warnings_list$no_trait_variation <- "Trait has zero variance"
    is_valid <- FALSE
  }

  list(
    is_valid = is_valid,
    warnings = warnings_list
  )
}


validate_variance_components <- function(result) {
  
  warnings_list <- list()
  is_valid <- TRUE
  
  vc <- result$variance_components
  flags <- result$flags
  
  # Check for negative variance components
  if (isTRUE(flags$negative_sigma2_genotype)) {
    warnings_list$negative_sigma2_genotype <- list(
      value = flags$sigma2_g_raw,
      message = "Genotypic variance was negative and truncated to zero. This indicates weak genotype signal or genotype effects masked by G×E or environmental noise. Heritability estimate may be unreliable."
    )
    is_valid <- FALSE
  }
  
  if (result$environment_mode == "multi_environment" && isTRUE(flags$negative_sigma2_ge)) {
    warnings_list$negative_sigma2_ge <- list(
      value = flags$sigma2_ge_raw,
      message = "G×E variance was negative and truncated to zero. Genotypes interact weakly or not at all across environments; performance may be stable across conditions."
    )
  }
  
  # CRITICAL: Detect weak genotype signal
  if (result$environment_mode == "multi_environment" && isTRUE(flags$weak_genotype_signal)) {
    warnings_list$weak_genotype_signal <- list(
      message = "Genotypic variance is weak relative to G×E. Genotype differentiation may be unreliable; G×E effects dominate the genetic variation."
    )
  }
  
  # CRITICAL: Detect weak or negligible G×E
  if (result$environment_mode == "multi_environment" && isTRUE(flags$weak_ge_signal)) {
    warnings_list$weak_ge_signal <- list(
      message = "G×E variance is negligible or zero. Genotype performance is relatively stable across environments."
    )
  }
  
  # Check heritability validity (skip for designs where h2 is not applicable, e.g. split-plot)
  h2_not_applicable <- isTRUE(result$heritability$not_applicable)
  if (!h2_not_applicable) {
    if (!isTRUE(result$heritability$h2_is_valid)) {
      warnings_list$h2_not_computed <- list(
        message = "Heritability could not be computed. Phenotypic variance is zero or invalid."
      )
      is_valid <- FALSE
    }

    # Check heritability range
    h2 <- result$heritability$h2_broad_sense
    if (length(h2) == 1L && !is.na(h2)) {
      if (h2 < 0.1) {
        warnings_list$low_heritability <- list(
          value = h2,
          message = "Heritability is very low (<0.10), indicating weak genetic control under present conditions. Environmental variation or G×E effects dominate.",
          implication = "Selection response will be minimal. Environmental management may be more effective than selection."
        )
      } else if (h2 < 0.3) {
        warnings_list$moderate_low_heritability <- list(
          value = h2,
          message = "Heritability is low to moderate (0.10–0.30), indicating that environmental factors have substantial influence."
        )
      }
    }
  }

  # Check genetic parameters validity
  gp <- result$genetic_parameters
  gp_not_applicable <- isTRUE(gp$not_applicable)

  if (!isTRUE(flags$mean_valid)) {
    warnings_list$missing_mean_for_cv <- list(
      message = "Grand mean is zero or missing. GCV, PCV, and GAM could not be computed."
    )
  }

  if (!gp_not_applicable && length(gp$GCV) == 1L && length(gp$PCV) == 1L &&
      !is.na(gp$GCV) && !is.na(gp$PCV)) {
    gcv_pcv_ratio <- gp$GCV / gp$PCV

    # Detect weak genetic signal from GCV/PCV ratio
    if (gcv_pcv_ratio < 0.1) {
      warnings_list$weak_genetic_signal_from_cv <- list(
        GCV = gp$GCV,
        PCV = gp$PCV,
        ratio = gcv_pcv_ratio,
        message = "GCV is very low relative to PCV, indicating that genetic variation is small relative to total phenotypic variation. Environmental influence is dominant."
      )
    }
  }
  
  list(
    is_valid = is_valid,
    warnings = warnings_list
  )
}


# ============================================================================
# MAIN ORCHESTRATOR FUNCTION (Layer 3: Interpretation via external engine)
# ============================================================================

#' Orchestrate VivaSense Genetics Analysis
#'
#' @param data              data frame with columns: genotype, rep, trait_value
#'                          (+ environment if multi; + factor if factorial CRD)
#' @param mode              character, "single" or "multi"
#' @param trait_name        character, name of the trait
#' @param random_environment logical (multi-mode only), treat environment as random
#' @param crd_mode          logical; TRUE = CRD (no blocking), FALSE = RCBD
#' @return list with computation result, validation warnings, interpretation
#'
genetics_analysis <- function(data,
                              mode = "single",
                              trait_name = "Trait",
                              random_environment = FALSE,
                              crd_mode = FALSE,
                              selection_intensity = 1.40) {

  selection_intensity <- normalize_selection_intensity(selection_intensity)

  # Validate input data
  data_validation <- validate_input_data(data, env_mode = mode,
                                         crd_mode = crd_mode)

  if (!data_validation$is_valid) {
    return(list(
      status = "ERROR",
      mode = mode,
      errors = data_validation$warnings,
      result = NULL,
      interpretation = NULL
    ))
  }

  # Run computation
  if (mode == "single") {
    result <- tryCatch(
      compute_single_environment(data, trait_name = trait_name,
                                 crd_mode = crd_mode,
                                 selection_intensity = selection_intensity),
      error = function(e) {
        message(sprintf("[ERROR] single-env computation failed for %s: %s",
                        trait_name, conditionMessage(e)))
        return(list(.__error__ = conditionMessage(e)))
      }
    )
  } else if (mode == "multi") {
    result <- tryCatch(
      compute_multi_environment(data, trait_name = trait_name,
                                random_environment = random_environment,
                                selection_intensity = selection_intensity),
      error = function(e) {
        message(sprintf("[ERROR] multi-env computation failed for %s: %s", trait_name, conditionMessage(e)))
        return(list(.__error__ = conditionMessage(e)))
      }
    )
  } else {
    return(list(
      status = "ERROR",
      mode = mode,
      errors = list(invalid_mode = "mode must be 'single' or 'multi'"),
      result = NULL,
      interpretation = NULL
    ))
  }
  
  # Propagate computation errors as structured responses
  if (!is.null(result$`.__error__`)) {
    return(list(
      status = "ERROR",
      mode = mode,
      errors = list(computation_error = result$`.__error__`),
      result = NULL,
      interpretation = paste("Analysis failed:", result$`.__error__`)
    ))
  }

  # Validate variance components
  warnings_vc <- validate_variance_components(result)
  
  # Return structured output
  list(
    status = "SUCCESS",
    mode = mode,
    data_validation = data_validation$warnings,
    variance_warnings = warnings_vc$warnings,
    result = result,
    interpretation = NULL
  )
}


# ============================================================================
# JSON EXPORT HELPER
# ============================================================================

#' Convert result to JSON-serializable list
export_to_json <- function(analysis_result) {

  clean_result <- analysis_result

  # Serialize ANOVA table as named column-arrays (source, df, ss, ms, f_value, p_value).
  # Atomic vectors are never auto_unboxed by jsonlite, so each array stays as an array
  # even for single-row tables.
  if (!is.null(analysis_result$result$anova_table)) {
    at_df <- analysis_result$result$anova_table
    if ("F value" %in% names(at_df)) {
      at_df[["F value"]][!is.finite(at_df[["F value"]])] <- NA_real_
    }
    # Strip the (Intercept) row — tests H0: grand mean = 0, which is trivially
    # false for all biological traits and carries no analytical value for users.
    intercept_rows <- grepl("^\\(Intercept\\)$", rownames(at_df), ignore.case = TRUE)
    if (any(intercept_rows)) {
      at_df <- at_df[!intercept_rows, , drop = FALSE]
    }
    clean_result$result$anova_table <- list(
      source  = rownames(at_df),
      df      = as.integer(at_df[["Df"]]),
      ss      = at_df[["Sum Sq"]],
      ms      = at_df[["Mean Sq"]],
      f_value = at_df[["F value"]],
      p_value = at_df[["Pr(>F)"]]
    )
  }

  # mean_separation is already a plain list of atomic vectors (or NULL).
  # NULL fields are dropped by toJSON; Optional[MeanSeparation] defaults to None in Python.

  json_str <- toJSON(clean_result, pretty = TRUE, auto_unbox = TRUE, na = "null", digits = 10)
  return(json_str)
}
