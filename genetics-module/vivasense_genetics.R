library(lmerTest)
library(emmeans)
library(jsonlite) # Added for toJSON
library(jsonlite)
library(agricolae)
library(multcompView)

#' Intelligence Engine: Automated Formula & Strata Generator
#' @param metadata A list containing design_type, factors (name, hierarchy), and rep_var
build_intelligent_formula <- function(metadata) {
  factors <- metadata$factors
  factor_names <- sapply(factors, function(f) f$name)
  design <- metadata$design_type
  rep_var <- metadata$rep_variable
  
  # 1. Fixed Effects: Full Factorial Interaction (up to 4-way)
  fixed_part <- paste(factor_names, collapse = " * ")
  
  # 2. Random Effects / Error Strata Inference
  random_part <- ""
  
  if (design == "RCBD") {
    # Traditional RCBD: Block is random
    random_part <- paste0("(1|", rep_var, ")")
  } else if (design == "SplitPlot") {
    # Split-plot: Main plot is nested in Rep
    # Terminology: (1|Rep) + (1|Rep:MainPlot)
    main_f <- factor_names[sapply(factors, function(f) f$hierarchy == "main_plot")]
    random_part <- paste0("(1|", rep_var, ") + (1|", rep_var, ":", main_f, ")")
  } else if (design == "SplitSplitPlot") {
    # Split-split: Hierarchical nesting
    main_f <- factor_names[sapply(factors, function(f) f$hierarchy == "main_plot")]
    sub_f <- factor_names[sapply(factors, function(f) f$hierarchy == "subplot")]
    random_part <- paste0("(1|", rep_var, ") + (1|", rep_var, ":", main_f, ") + (1|", rep_var, ":", main_f, ":", sub_f, ")")
  } else {
    # CRD: No random blocks
    return(as.formula(paste("trait_value ~", fixed_part)))
  }
  
  formula_str <- paste("trait_value ~", fixed_part, "+", random_part)
  return(as.formula(formula_str))
}

#' Structural Balance Validation for Split-Plot Designs
#' Checks for missing whole-plots, duplicated cells, and uneven replication.
validate_split_plot_structure <- function(data, factor_a, factor_b, rep_var) {
  warnings <- c()
  
  # 1. Check for missing whole plots (Rep x FactorA)
  wp_counts <- table(data[[rep_var]], data[[factor_a]])
  if (any(wp_counts == 0)) {
    warnings <- c(warnings, "Structural imbalance: One or more whole-plots (Rep x Factor A) are missing entirely.")
  }
  
  # 2. Check for missing subplots or unequal replication
  sp_counts <- table(data[[rep_var]], data[[factor_a]], data[[factor_b]])
  if (any(sp_counts == 0)) {
    warnings <- c(warnings, "Structural imbalance: One or more subplot combinations are missing.")
  }
  if (any(sp_counts > 1)) {
    warnings <- c(warnings, "Data anomaly: Duplicated subplot cells detected.")
  }

  return(warnings)
}

#' Hardened aov() Stratum Extraction
#' Avoids fragile positional indexing by identifying strata by name pattern.
#' Identifies "Error: Rep:Main" and "Error: Within" explicitly.
extract_anova_strata <- function(aov_obj, factor_a, factor_b, rep_var) {
  s <- summary(aov_obj)
  strata_names <- names(s)
  
  # Identify Whole Plot Error Stratum (Error: block:variety)
  wp_pattern <- paste0("Error:.*", rep_var, ".*", factor_a)
  wp_idx <- grep(wp_pattern, strata_names, ignore.case = TRUE)
  
  # Identify Subplot Error Stratum (Error: Within)
  sp_idx <- grep("Within", strata_names, ignore.case = TRUE)
  
  return(list(
    wp_stratum = if(length(wp_idx) > 0) s[[wp_idx[1]]] else NULL,
    sp_stratum = if(length(sp_idx) > 0) s[[sp_idx[1]]] else NULL
  ))
}

#' Core Split-Plot Engine with Biometric Hardening
run_split_plot <- function(data, response, factor_a, factor_b, rep_var, alpha = 0.05) {
  # Type coercion
  data[[factor_a]] <- as.factor(data[[factor_a]])
  data[[factor_b]] <- as.factor(data[[factor_b]])
  data[[rep_var]] <- as.factor(data[[rep_var]])
  
  # Task 3: Balance Check
  structural_warnings <- validate_split_plot_structure(data, factor_a, factor_b, rep_var)
  
  # Task 6: Correct Strata Formula
  aov_formula <- as.formula(paste(response, "~", factor_a, "*", factor_b, "+ Error(", rep_var, "/", factor_a, ")"))
  fit_aov <- aov(aov_formula, data = data)
  
  # Task 2: Robust extraction logic
  strata <- extract_anova_strata(fit_aov, factor_a, factor_b, rep_var)
  
  # Extraction of MS for correct F-tests: F(A) = MS_A / MS_ErrorA
  # Stratum 1: Whole plot
  ms_a <- strata$wp_stratum[[1]]["Mean Sq"][1,1]
  ms_err_a <- strata$wp_stratum[[1]]["Mean Sq"][2,1] # The Residuals row in WP stratum
  
  # Stratum 2: Subplot (Within)
  ms_b <- strata$sp_stratum[[1]]["Mean Sq"][1,1]
  ms_ab <- strata$sp_stratum[[1]]["Mean Sq"][2,1]
  ms_err_b <- strata$sp_stratum[[1]]["Mean Sq"][3,1] # Error B (Residual)
  
  grand_mean <- mean(data[[response]], na.rm = TRUE)
  
  # Task 5: Warning Logic (Variance Anomalies)
  warnings <- structural_warnings
  if (!is.na(ms_err_a) && !is.na(ms_err_b) && ms_err_a < ms_err_b) {
    warnings <- c(warnings, "Whole-plot variance component truncated to zero (MS_RepA < MS_error).")
  }
  
  # Task 9: Interaction Hierarchy Reasoning
  p_ab <- strata$sp_stratum[[1]]["Pr(>F)"][2,1]
  if (!is.na(p_ab) && p_ab < alpha) {
    warnings <- c(warnings, sprintf("Significant %s x %s interaction: Main effects should be interpreted cautiously as responses are conditional.", factor_a, factor_b))
  }

  # Package results for JSON export
  return(list(
    design_verified = length(structural_warnings) == 0,
    anova_table = list(
      factor_a = list(ms = ms_a, f = ms_a/ms_err_a, p = strata$wp_stratum[[1]]["Pr(>F)"][1,1], den_ms = ms_err_a),
      factor_b = list(ms = ms_b, f = ms_b/ms_err_b, p = strata$sp_stratum[[1]]["Pr(>F)"][1,1], den_ms = ms_err_b),
      interaction = list(ms = ms_ab, f = ms_ab/ms_err_b, p = p_ab, den_ms = ms_err_b)
    ),
    cv = list(
      main_plot = (sqrt(max(0, ms_err_a)) / grand_mean) * 100,
      sub_plot = (sqrt(max(0, ms_err_b)) / grand_mean) * 100
    ),
    warnings = warnings
  ))
}

#' Interpretation Logic: Interaction Slicing
#' Determines if we should analyze simple effects based on p-values
analyze_interactions <- function(model_fit) {
  anova_tab <- as.data.frame(anova(model_fit))
  p_values <- anova_tab$`Pr(>F)`
  names(p_values) <- rownames(anova_tab)
  
  significant_terms <- names(p_values[p_values < 0.05 & !is.na(p_values)])
  
  # Find highest order interaction among significant terms
  interaction_terms <- significant_terms[grep(":", significant_terms)]
  
  if (length(interaction_terms) > 0) {
    interaction_depth <- sapply(strsplit(interaction_terms, ":"), length)
    highest_interaction <- interaction_terms[which.max(interaction_depth)]
    
    return(list(
      status = "interaction_dominant",
      highest_term = highest_interaction,
      message = sprintf("Interaction %s is significant. Main effects are masked; simple effects analysis recommended.", highest_interaction)
    ))
  }
  
  return(list(status = "main_effects_only", message = "No significant interactions detected."))
}

# Subprocess entry point
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  input_json <- jsonlite::fromJSON(args[1])
  df <- as.data.frame(input_json$data)
  res <- run_split_plot(df, input_json$response, input_json$factor_a, input_json$factor_b, input_json$rep, input_json$alpha)
  cat(jsonlite::toJSON(res, auto_unbox = TRUE, digits = 8))
}
}