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

# Load the strict interpretation engine
source("vivasense_interpretation_engine.R")

# ============================================================================
# LAYER 1: COMPUTATION LAYER
# Core mathematical and statistical functions
# ============================================================================

#' Single-Environment Genetics Analysis
#' 
#' @param data data frame with columns: genotype, rep, trait_value
#' @param trait_name character, name of the trait being analyzed
#' @return list with variance components and heritability estimates
#'
compute_single_environment <- function(data, trait_name = "Trait") {
  
  # Ensure factors
  data$genotype <- factor(data$genotype)
  data$rep <- factor(data$rep)
  
  n_genotypes <- nlevels(data$genotype)
  n_reps <- nlevels(data$rep)
  
  # ANOVA under CRD or RCBD (assuming RCBD: genotype as fixed, rep as fixed block)
  model <- aov(trait_value ~ rep + genotype, data = data)
  anova_table <- anova(model)
  
  # Extract mean squares
  ms_genotype <- anova_table["genotype", "Mean Sq"]
  ms_error <- anova_table["Residuals", "Mean Sq"]
  
  # Variance components (assuming Model 1: fixed genotype)
  # σ²e = MSE
  # σ²g = (MS_g - MS_e) / n_reps
  sigma2_error <- ms_error
  
  # CRITICAL FIX: Clamp negative variance to zero + flag warning
  sigma2_genotype_raw <- (ms_genotype - ms_error) / n_reps
  sigma2_genotype <- max(0, sigma2_genotype_raw)
  negative_sigma2g <- sigma2_genotype_raw < -0.001
  
  # Phenotypic variance (under RCBD, expected phenotypic variance among entries)
  sigma2_phenotypic <- sigma2_genotype + (sigma2_error / n_reps)
  
  # CRITICAL FIX: Handle edge case where phenotypic variance ≤ 0
  if (sigma2_phenotypic <= 0) {
    h2 <- NA_real_
    h2_is_valid <- FALSE
  } else {
    # Broad-sense heritability (h²) on entry-mean basis
    h2 <- sigma2_genotype / sigma2_phenotypic
    h2 <- max(0, min(1, h2)) # Clamp to [0, 1]
    h2_is_valid <- TRUE
  }
  
  # Genotypic CV and Phenotypic CV
  grand_mean <- mean(data$trait_value, na.rm = TRUE)
  
  gcv <- NA_real_
  pcv <- NA_real_
  gam <- NA_real_
  gam_percent <- NA_real_
  mean_is_valid <- !is.na(grand_mean) && grand_mean != 0
  
  if (mean_is_valid) {
    gcv <- (sqrt(max(0, sigma2_genotype)) / grand_mean) * 100
    pcv <- (sqrt(sigma2_phenotypic) / grand_mean) * 100
    
    # Genetic Advance at Mean (assuming selection intensity i = 1.4 for ~30% selection)
    gam <- 1.4 * sqrt(max(0, sigma2_genotype))
    gam_percent <- (gam / grand_mean) * 100
  }
  
  list(
    environment_mode = "single_environment",
    n_genotypes = n_genotypes,
    n_reps = n_reps,
    grand_mean = grand_mean,
    variance_components = list(
      sigma2_genotype = sigma2_genotype,
      sigma2_error = sigma2_error,
      sigma2_phenotypic = sigma2_phenotypic
    ),
    heritability = list(
      h2_broad_sense = h2,
      h2_is_valid = h2_is_valid,
      interpretation_basis = "entry-mean (single environment)"
    ),
    genetic_parameters = list(
      GCV = gcv,
      PCV = pcv,
      GAM = gam,
      GAM_percent = gam_percent,
      selection_intensity = 1.4
    ),
    flags = list(
      negative_sigma2_genotype = negative_sigma2g,
      sigma2_g_raw = sigma2_genotype_raw,
      mean_valid = mean_is_valid
    ),
    anova_table = as.data.frame(anova_table),
    ms_genotype = ms_genotype,
    ms_error = ms_error
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
                                       random_environment = FALSE) {

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
    
    # Genetic Advance
    gam <- 1.4 * sqrt(max(0, sigma2_genotype))
    gam_percent <- (gam / grand_mean) * 100
  }
  
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
      selection_intensity = 1.4
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
    ms_genotype = ms_genotype,
    ms_ge = ms_ge,
    ms_error = ms_error
  )
}


# ============================================================================
# LAYER 2: VALIDATION LAYER
# Check data quality, variance reasonableness, and flag issues
# ============================================================================

validate_input_data <- function(data, env_mode = "single") {
  
  warnings_list <- list()
  is_valid <- TRUE
  
  # Check required columns
  required_cols <- c("genotype", "rep", "trait_value")
  if (env_mode == "multi") required_cols <- c(required_cols, "environment")
  
  missing_cols <- setdiff(required_cols, colnames(data))
  if (length(missing_cols) > 0) {
    warnings_list$missing_columns <- missing_cols
    is_valid <- FALSE
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
  
  # Check minimum replication
  if (env_mode == "single") {
    min_reps <- data %>% group_by(genotype) %>% summarise(n = n(), .groups = "drop") %>% pull(n) %>% min()
    if (min_reps < 2) {
      warnings_list$insufficient_replication <- paste("Minimum reps per genotype:", min_reps)
      is_valid <- FALSE
    }
  } else {
    min_reps_per_gxe <- data %>% 
      group_by(genotype, environment) %>% 
      summarise(n = n(), .groups = "drop") %>% 
      pull(n) %>% 
      min()
    if (min_reps_per_gxe < 2) {
      warnings_list$insufficient_replication <- paste("Minimum reps per G×E:", min_reps_per_gxe)
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
#' @param data data frame with columns: genotype, rep, trait_value (+ environment if multi)
#' @param mode character, "single" or "multi"
#' @param trait_name character, name of the trait
#' @param random_environment logical (multi-mode only), treat environment as random
#' @return list with computation result, validation warnings, interpretation
#'
genetics_analysis <- function(data, 
                              mode = "single",
                              trait_name = "Trait",
                              random_environment = FALSE) {
  
  # Validate input data
  data_validation <- validate_input_data(data, env_mode = mode)
  
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
      compute_single_environment(data, trait_name = trait_name),
      error = function(e) {
        message(sprintf("[ERROR] single-env computation failed for %s: %s", trait_name, conditionMessage(e)))
        return(list(.__error__ = conditionMessage(e)))
      }
    )
  } else if (mode == "multi") {
    result <- tryCatch(
      compute_multi_environment(data, trait_name = trait_name,
                                random_environment = random_environment),
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
  
  # Generate interpretation using new strict engine
  if (mode == "single") {
    interpretation <- interpret_single_environment_strict(result, warnings_vc)
  } else {
    interpretation <- interpret_multi_environment_strict(result, warnings_vc)
  }
  
  # Return structured output
  list(
    status = "SUCCESS",
    mode = mode,
    data_validation = data_validation$warnings,
    variance_warnings = warnings_vc$warnings,
    result = result,
    interpretation = interpretation
  )
}


# ============================================================================
# JSON EXPORT HELPER
# ============================================================================

#' Convert result to JSON-serializable list
export_to_json <- function(analysis_result) {
  
  # Remove problematic objects (data frames, model objects, etc.)
  clean_result <- analysis_result
  
  # Simplify ANOVA table
  if (!is.null(analysis_result$result$anova_table)) {
    clean_result$result$anova_table <- as.list(as.data.frame(
      analysis_result$result$anova_table,
      check.names = FALSE
    ))
  }
  
  # Convert to JSON
  json_str <- toJSON(clean_result, pretty = TRUE, auto_unbox = TRUE, na = "null", digits = 10)
  return(json_str)
}
