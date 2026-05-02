# VivaSense Genetics Engine - R Implementation
# Three-layer architecture: Computation | Validation | Interpretation
# Supports single-environment and multi-environment analysis
# Returns structured JSON + interpretation text

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
  residual_rows <- grep("Residuals|Within", rn, ignore.case = TRUE)
  residual_ms <- if (length(residual_rows) > 0) {
    suppressWarnings(as.numeric(anova_table[residual_rows[length(residual_rows)], "Mean Sq"]))
  } else {
    NA_real_
  }

  wp_error_ms <- if ("whole_plot_error" %in% rn) {
    suppressWarnings(as.numeric(anova_table["whole_plot_error", "Mean Sq"]))
  } else {
    NA_real_
  }

  for (i in seq_len(nrow(anova_table))) {
    term <- rn[i]
    ms_effect <- suppressWarnings(as.numeric(anova_table[i, "Mean Sq"]))

    if (grepl("Residuals|Within|whole_plot_error", term, ignore.case = TRUE)) {
      anova_table[i, "F value"] <- NA_real_
      if ("Pr(>F)" %in% names(anova_table)) {
        anova_table[i, "Pr(>F)"] <- NA_real_
      }
      next
    }

    denom <- if (identical(term, "main_plot")) wp_error_ms else residual_ms
    f_val <- safe_f_ratio(ms_effect, denom)
    anova_table[i, "F value"] <- f_val
    if (is.na(f_val) && ("Pr(>F)" %in% names(anova_table))) {
      anova_table[i, "Pr(>F)"] <- NA_real_
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
                                    df_error = NULL, ms_error = NULL) {
  tukey <- tryCatch({
    if (!is.null(df_error) && !is.null(ms_error)) {
      HSD.test(model, "genotype",
               DFerror = df_error, MSerror = ms_error,
               group = TRUE, console = FALSE)
    } else {
      HSD.test(model, "genotype", group = TRUE, console = FALSE)
    }
  }, error = function(e) {
    message(sprintf("[WARN] Tukey HSD failed for %s: %s — trying LSD.test fallback", trait_name, conditionMessage(e)))
    NULL
  })

  # Fallback: LSD.test when Tukey HSD fails (e.g. lm models, singular residuals)
  if (is.null(tukey)) {
    tukey <- tryCatch({
      if (!is.null(df_error) && !is.null(ms_error)) {
        LSD.test(model, "genotype",
                 DFerror = df_error, MSerror = ms_error,
                 group = TRUE, console = FALSE)
      } else {
        LSD.test(model, "genotype", group = TRUE, console = FALSE)
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
  list(
    genotype = geno_order,
    mean     = as.numeric(groups_df[[1]]),
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
  if (!has_splitplot) {
    data$genotype <- factor(data$genotype)
  }
  data$rep <- factor(data$rep)

  n_genotypes <- if (!has_splitplot) nlevels(data$genotype) else NA_integer_

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
  if (!has_splitplot && n_genotypes < 2) {
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

    # Find the whole-plot stratum (contains "main_plot" as a treatment row)
    wp_table <- NULL
    # Find the subplot stratum (contains "sub_plot" as a treatment row)
    sub_table <- NULL
    for (nm in strata_names) {
      tbl <- sp_summ[[nm]][[1]]
      if ("main_plot" %in% rownames(tbl)) wp_table  <- tbl
      if ("sub_plot"  %in% rownames(tbl)) sub_table <- tbl
    }
    if (is.null(wp_table) || is.null(sub_table)) {
      stop(paste(
        "Split-plot summary missing expected strata. Available:",
        paste(strata_names, collapse = ", ")
      ))
    }

    # Rename the whole-plot Residuals row so the combined table is unambiguous
    wp_res <- wp_table[rownames(wp_table) == "Residuals", , drop = FALSE]
    rownames(wp_res) <- "whole_plot_error"
    wp_treat <- wp_table[rownames(wp_table) != "Residuals", , drop = FALSE]

    anova_table <- rbind(wp_treat, wp_res, sub_table)

    ms_genotype <- NA_real_   # not applicable for generic split-plot
    # Subplot residual is the canonical error term for CV and mean separation
    ms_error <- sub_table["Residuals", "Mean Sq"]
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

  # Grand mean — needed for both split-plot and genotype-based paths
  grand_mean    <- mean(data$trait_value, na.rm = TRUE)
  mean_is_valid <- !is.na(grand_mean) && grand_mean != 0

  # Variance components, heritability, and genetic parameters are only
  # meaningful for genotype-based designs.  For generic split-plot RCBD the
  # only relevant error terms are the subplot error and the whole-plot error;
  # genetics-specific blocks are explicitly omitted.
  if (has_splitplot) {
    # Whole-plot error MS is in the renamed "whole_plot_error" row
    wp_error_ms <- tryCatch(
      as.numeric(anova_table["whole_plot_error", "Mean Sq"]),
      error = function(e) NA_real_
    )
    variance_components_out <- list(
      sigma2_subplot_error    = ms_error,
      sigma2_whole_plot_error = wp_error_ms
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

  # Mean separation — only available for genotype-based designs
  if (has_splitplot) {
    df_err   <- NA_integer_
    mean_sep <- NULL
  } else {
    df_err   <- anova_table["Residuals", "Df"]
    mean_sep <- compute_mean_separation(model, trait_name,
                                        df_error = df_err, ms_error = ms_error)
  }

  design_label <- if (has_splitplot) {
    "split_plot_rcbd"
  } else if (crd_mode) {
    if (has_factor) "factorial_crd" else "crd"
  } else {
    if (has_factor) "factorial_rcbd" else "rcbd"
  }

  list(
    environment_mode    = "single_environment",
    design              = design_label,
    n_genotypes         = n_genotypes,
    n_reps              = n_reps,
    grand_mean          = grand_mean,
    variance_components = variance_components_out,
    heritability        = heritability_out,
    genetic_parameters  = genetic_params_out,
    flags               = flags_out,
    anova_table         = as.data.frame(anova_table),
    mean_separation     = mean_sep,
    ms_genotype         = ms_genotype,
    ms_error            = ms_error
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
  
  # Check heritability validity
  if (!isTRUE(result$heritability$h2_is_valid)) {
    warnings_list$h2_not_computed <- list(
      message = "Heritability could not be computed. Phenotypic variance is zero or invalid."
    )
    is_valid <- FALSE
  }
  
  # Check heritability range
  h2 <- result$heritability$h2_broad_sense
  if (!is.na(h2)) {
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
  
  # Check genetic parameters validity
  gp <- result$genetic_parameters
  
  if (!isTRUE(flags$mean_valid)) {
    warnings_list$missing_mean_for_cv <- list(
      message = "Grand mean is zero or missing. GCV, PCV, and GAM could not be computed."
    )
  }
  
  if (!is.na(gp$GCV) && !is.na(gp$PCV)) {
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
