# ============================================================================
# VIVASENSE INTERPRETATION ENGINE - Plant Breeding Standards
# Implements strict interpretation rules from quantitative genetics literature
# ============================================================================

# String concatenation operator (not built into base R)
`%+%` <- paste0

# Classification thresholds (plant breeding standard)
classify_heritability <- function(h2) {
  if (is.na(h2)) return("not_computed")
  if (h2 < 0.30) return("low")
  if (h2 < 0.60) return("moderate")
  return("high")
}

classify_gam <- function(gam_percent) {
  if (is.na(gam_percent)) return("not_computed")
  if (gam_percent < 5)  return("low")      # < 5%  : low genetic advance
  if (gam_percent <= 10) return("moderate") # 5–10% : moderate genetic advance
  return("high")                           # > 10% : high genetic advance
}

classify_cv <- function(cv) {
  if (is.na(cv)) return("not_computed")
  if (cv < 10) return("low")
  if (cv < 20) return("moderate")
  return("high")
}

classify_gxe_importance <- function(sigma2_g, sigma2_ge) {
  if (is.na(sigma2_g) || is.na(sigma2_ge)) return("not_computed")
  if (sigma2_g == 0) return("large_dominates")
  
  ratio <- sigma2_ge / sigma2_g
  if (ratio < 0.5) return("small")
  if (ratio <= 1.0) return("moderate")
  return("large")
}

#' Interpret single-environment genetics results
#' 
#' Follows strict plant breeding interpretation standards.
#' Never interprets heritability alone.
#' Always interprets h2 jointly with GAM and GCV/PCV.
#'
interpret_single_environment_strict <- function(result, warnings_vc) {
  
  vc <- result$variance_components
  gp <- result$genetic_parameters
  h2 <- result$heritability$h2_broad_sense
  
  # Classify parameters
  h2_class <- classify_heritability(h2)
  gam_class <- classify_gam(gp$GAM_percent)
  gcv_class <- classify_cv(gp$GCV)
  pcv_class <- classify_cv(gp$PCV)
  
  text_parts <- list()
  
  # Header
  text_parts$header <- sprintf(
    "Single-Environment Genetic Analysis\n%s\n",
    paste(rep("=", 60), collapse = "")
  )
  
  # Basic metadata
  text_parts$metadata <- sprintf(
    "\nTrait: %s\nGenotypes: %d | Replicates: %d | Grand mean: %.2f\nAnalysis basis: Entry-mean within single environment\n",
    "(not specified)",
    result$n_genotypes,
    result$n_reps,
    result$grand_mean
  )
  
  # === STEP 1: Report numerical values ===
  h2_str <- if (is.na(h2)) "not computed" else sprintf("%.4f", h2)
  text_parts$numerical <- paste0(
    "\nEstimated Variance Components:\n",
    sprintf("  Genotypic variance (\u03c3\u00b2G):      %.4f\n", vc$sigma2_genotype),
    sprintf("  Phenotypic variance (\u03c3\u00b2P):     %.4f\n", vc$sigma2_phenotypic),
    sprintf("  Environmental variance (\u03c3\u00b2E):  %.4f\n", vc$sigma2_error),
    "  Broad-sense heritability (h\u00b2): ", h2_str, "\n"
  )

  if (!is.na(gp$GCV)) {
    text_parts$numerical <- paste0(
      text_parts$numerical,
      "\nGenetic Parameters:\n",
      sprintf("  Genotypic coefficient of variation (GCV): %.2f%%\n",  gp$GCV),
      sprintf("  Phenotypic coefficient of variation (PCV): %.2f%%\n", gp$PCV),
      sprintf("  Genetic advance (absolute): %.3f\n",                  gp$GAM),
      sprintf("  Genetic advance as percent of mean (GAM%%): %.2f%%\n", gp$GAM_percent)
    )
  }
  
  # === STEP 2: Interpret heritability + Step 3: Joint h2 & GAM interpretation ===
  text_parts$interpretation <- "\nInterpretation:\n"
  
  if (!isTRUE(result$heritability$h2_is_valid)) {
    text_parts$interpretation <- paste0(
      text_parts$interpretation,
      "Heritability could not be computed. Phenotypic variance is zero or invalid.\n"
    )
  } else if (is.na(h2)) {
    text_parts$interpretation <- paste0(
      text_parts$interpretation,
      "Heritability could not be estimated due to data limitations.\n"
    )
  } else {
    # Joint interpretation of h2 and GAM
    interp <- ""
    
    # High h2 + High GAM
    if (h2_class == "high" && gam_class == "high") {
      interp <- paste0(
        sprintf("The estimated broad-sense heritability (h\u00b2 = %.3f) indicates HIGH genetic control of the trait within this environment. ", h2),
        sprintf("The corresponding genetic advance as percent of mean (GAM = %.2f%%) is HIGH, suggesting substantial expected response to direct selection. ", gp$GAM_percent),
        "The joint pattern of high h\u00b2 and high GAM indicates that additive gene effects are likely important; direct phenotypic selection should be effective."
      )
    } else if (h2_class == "high" && gam_class == "moderate") {
      interp <- paste0(
        sprintf("The estimated broad-sense heritability (h\u00b2 = %.3f) indicates HIGH genetic control within this environment. ", h2),
        sprintf("The genetic advance as percent of mean (GAM = %.2f%%) is MODERATE, indicating a meaningful selection response. ", gp$GAM_percent),
        "Direct phenotypic selection should yield steady genetic progress. Both additive and non-additive gene effects likely contribute to trait variation."
      )
    } else if (h2_class == "high" && gam_class == "low") {
      interp <- paste0(
        sprintf("The estimated broad-sense heritability (h\u00b2 = %.3f) indicates HIGH genetic control, yet the genetic advance as percent of mean (GAM = %.2f%%) is LOW. ", h2, gp$GAM_percent),
        "This dissociation suggests that while phenotypic variation is substantially genetic, the expected response to selection is limited. ",
        "Non-additive gene effects, low effective population size, or strong inbreeding depression may be responsible."
      )
    } else if (h2_class == "moderate" && gam_class == "high") {
      interp <- paste0(
        sprintf("The estimated broad-sense heritability (h\u00b2 = %.3f) indicates MODERATE genetic control, with the genetic advance as percent of mean (GAM = %.2f%%) being HIGH. ", h2, gp$GAM_percent),
        "This combination suggests that while environmental influence is substantial, useful progress from direct selection is achievable. ",
        "Both genetic and environmental management should be considered."
      )
    } else if (h2_class == "moderate" && gam_class == "moderate") {
      interp <- paste0(
        sprintf("The estimated broad-sense heritability (h\u00b2 = %.3f) and genetic advance as percent of mean (GAM = %.2f%%) both indicate MODERATE genetic control. ", h2, gp$GAM_percent),
        "Selection may be useful, though environmental factors remain important. Progress should be expected to be steady but not rapid."
      )
    } else if (h2_class == "moderate" && gam_class == "low") {
      interp <- paste0(
        sprintf("The estimated broad-sense heritability (h\u00b2 = %.3f) suggests MODERATE genetic control, but the genetic advance as percent of mean (GAM = %.2f%%) is LOW. ", h2, gp$GAM_percent),
        "Direct phenotypic selection may be slow. Consider investigating additive effects more carefully or combining selection with environmental optimization."
      )
    } else {
      interp <- paste0(
        sprintf("The estimated broad-sense heritability (h\u00b2 = %.3f) indicates LOW genetic control under the present environment. ", h2),
        "Phenotypic variation is dominated by environmental factors and/or measurement variation. ",
        "Direct phenotypic selection is unlikely to be reliable; focus on improving growing conditions and management practices."
      )
    }
    
    text_parts$interpretation <- paste0(text_parts$interpretation, interp, "\n")
  }
  
  # === STEP 4: Interpret GCV vs PCV ===
  if (!is.na(gp$GCV) && !is.na(gp$PCV)) {
    cv_interp <- ""
    
    # Calculate PCV - GCV
    pcv_gcv_diff <- gp$PCV - gp$GCV
    
    if (pcv_gcv_diff <= 2) {
      cv_interp <- paste0(
        sprintf("The genotypic coefficient of variation (GCV = %.2f%%) is only slightly lower than the phenotypic coefficient of variation (PCV = %.2f%%). ", gp$GCV, gp$PCV),
        "Environmental influence on trait expression is limited, and trait variation reflects primarily genetic differences."
      )
    } else if (pcv_gcv_diff <= 7) {
      cv_interp <- paste0(
        sprintf("The genotypic coefficient of variation (GCV = %.2f%%) is moderately lower than the phenotypic coefficient of variation (PCV = %.2f%%). ", gp$GCV, gp$PCV),
        "Environmental factors exert appreciable influence on trait expression, though genetic variation remains substantial."
      )
    } else {
      cv_interp <- paste0(
        sprintf("The genotypic coefficient of variation (GCV = %.2f%%) is substantially lower than the phenotypic coefficient of variation (PCV = %.2f%%). ", gp$GCV, gp$PCV),
        "Environmental factors strongly affect trait expression, and environmental standardization or management may improve selection response."
      )
    }
    
    text_parts$cv_interp <- paste0("\n", cv_interp, "\n")
  }
  
  # === STEP 6: Breeding implication ===
  text_parts$breeding <- "\n"
  breed_body_single <- if (!isTRUE(result$heritability$h2_is_valid) || is.na(h2)) {
    "Heritability could not be reliably estimated from the present data. Redesign or expand the experiment to improve precision."
  } else if (h2_class == "high") {
    paste0(
      "Strong genetic basis for the trait. Direct phenotypic selection for this trait should be effective in this environment. ",
      "Prioritize identification and advancement of high-value individuals for next-generation breeding."
    )
  } else if (h2_class == "moderate") {
    paste0(
      "Moderate genetic basis. Direct selection is possible but should be combined with attention to environmental standardization. ",
      "Consider multi-environment evaluation to assess stability of selected genotypes."
    )
  } else {
    paste0(
      "Weak genetic basis under present conditions. Direct selection will be unreliable. ",
      "Prioritize improvement of growing conditions, management practices, and measurement precision before intensifying selection."
    )
  }
  text_parts$breeding <- paste0(text_parts$breeding, "Breeding Implication:\n", breed_body_single, "\n")
  
  # === Warnings ===
  if (length(warnings_vc$warnings) > 0) {
    text_parts$warnings <- "\n⚠ Cautions and Warnings:\n"
    for (name in names(warnings_vc$warnings)) {
      w <- warnings_vc$warnings[[name]]
      if (is.list(w)) {
        text_parts$warnings <- paste0(text_parts$warnings, "  • ", w$message, "\n")
      } else {
        text_parts$warnings <- paste0(text_parts$warnings, "  • ", w, "\n")
      }
    }
  }
  
  paste(unlist(text_parts), collapse = "")
}


#' Interpret multi-environment genetics results
#' 
#' Implements strict multi-environment interpretation with G×E consideration
#'
interpret_multi_environment_strict <- function(result, warnings_vc) {

  vc <- result$variance_components
  gp <- result$genetic_parameters
  h2 <- result$heritability$h2_broad_sense

  # Classify parameters
  h2_class  <- classify_heritability(h2)
  gam_class <- classify_gam(gp$GAM_percent)
  gxe_class <- classify_gxe_importance(vc$sigma2_genotype, vc$sigma2_ge)

  text_parts <- list()

  # Header
  text_parts$header <- paste0(
    "Multi-Environment Genetic Analysis\n",
    paste(rep("=", 60), collapse = ""), "\n"
  )

  # Metadata
  env_model <- ifelse(
    identical(vc$heritability_basis, "fixed_environment_model"),
    "Fixed (standard)", "Random (advanced)"
  )
  text_parts$metadata <- paste0(
    "\nTrait: (not specified)",
    "\nGenotypes: ", result$n_genotypes,
    " | Environments: ", result$n_environments,
    " | Replicates per G\u00d7E: ", result$n_reps,
    "\nGrand mean: ", round(result$grand_mean, 2),
    "\nEnvironment model: ", env_model,
    "\nAnalysis basis: Entry-mean broad-sense heritability across environments\n"
  )

  # === STEP 1: Report numerical values ===
  h2_str <- if (is.na(h2)) "not computed" else sprintf("%.4f", h2)
  text_parts$numerical <- paste0(
    "\nEstimated Variance Components:\n",
    sprintf("  Genotypic variance (\u03c3\u00b2G):                      %.4f\n", vc$sigma2_genotype),
    sprintf("  Genotype-by-environment variance (\u03c3\u00b2G\u00d7E): %.4f\n", vc$sigma2_ge),
    sprintf("  Phenotypic variance on entry-mean basis (\u03c3\u00b2P): %.4f\n", vc$sigma2_phenotypic),
    "  Broad-sense heritability (h\u00b2):              ", h2_str, "\n",
    "  Heritability formula: ", result$heritability$formula, "\n"
  )

  if (!is.na(gp$GCV)) {
    text_parts$numerical <- paste0(
      text_parts$numerical,
      "\nGenetic Parameters (across-environment entry-means):\n",
      sprintf("  Genotypic coefficient of variation (GCV): %.2f%%\n",  gp$GCV),
      sprintf("  Phenotypic coefficient of variation (PCV): %.2f%%\n", gp$PCV),
      sprintf("  Genetic advance (absolute): %.3f\n",                  gp$GAM),
      sprintf("  Genetic advance as percent of mean (GAM%%): %.2f%%\n", gp$GAM_percent)
    )
  }

  # === STEP 2-3: Interpret h2 and GAM jointly ===
  text_parts$interpretation <- "\nInterpretation:\n"

  if (!isTRUE(result$heritability$h2_is_valid) || is.na(h2)) {
    text_parts$interpretation <- paste0(
      text_parts$interpretation,
      "Heritability could not be computed. Phenotypic variance is zero or invalid.\n"
    )
  } else {
    interp <- paste0(
      sprintf("The estimated broad-sense heritability across environments (h\u00b2 = %.3f) is classified as %s. ", h2, h2_class),
      sprintf("The genetic advance as percent of mean (GAM = %.2f%%) is classified as %s. ", gp$GAM_percent, gam_class)
    )

    if (h2_class == "high" && gam_class == "high") {
      interp <- paste0(interp,
        "The combination of high h\u00b2 and high GAM indicates strong genetic control with substantial expected response to selection. ",
        "Additive gene effects appear important; direct selection across environments should be effective."
      )
    } else if (h2_class == "high" && gam_class == "moderate") {
      interp <- paste0(interp,
        "High h\u00b2 with moderate GAM indicates heritable control with a meaningful selection response across environments. ",
        "Steady genetic progress is expected from direct selection; non-additive effects may limit the rate of improvement."
      )
    } else if (h2_class == "moderate" && gam_class == "high") {
      interp <- paste0(interp,
        "Moderate h\u00b2 with high GAM indicates that useful selection response is achievable despite environmental influence."
      )
    } else if (h2_class == "moderate" && gam_class == "moderate") {
      interp <- paste0(interp,
        "Moderate h\u00b2 and moderate GAM indicate steady but not rapid genetic progress is achievable under selection."
      )
    } else if (h2_class == "low") {
      interp <- paste0(interp,
        "Low h\u00b2 indicates that genetic control is weak under the present environmental conditions. ",
        "Environmental and/or G\u00d7E effects dominate trait expression; direct selection is less reliable."
      )
    }

    text_parts$interpretation <- paste0(text_parts$interpretation, interp, "\n")
  }

  # === STEP 4: Interpret GCV vs PCV ===
  if (!is.na(gp$GCV) && !is.na(gp$PCV)) {
    pcv_gcv_diff <- gp$PCV - gp$GCV
    cv_suffix <- if (pcv_gcv_diff <= 2) {
      paste0(sprintf("only slightly lower than the phenotypic coefficient of variation (PCV = %.2f%%). ", gp$PCV),
             "Environmental influence across environments is limited.")
    } else if (pcv_gcv_diff <= 7) {
      paste0(sprintf("moderately lower than the phenotypic coefficient of variation (PCV = %.2f%%). ", gp$PCV),
             "Environmental variation across environments is appreciable.")
    } else {
      paste0(sprintf("substantially lower than the phenotypic coefficient of variation (PCV = %.2f%%). ", gp$PCV),
             "Environment and/or G\u00d7E effects substantially influence trait expression.")
    }
    text_parts$cv_interp <- paste0(
      "\n",
      sprintf("The genotypic coefficient of variation (GCV = %.2f%%) is ", gp$GCV),
      cv_suffix, "\n"
    )
  }

  # === STEP 5: Interpret G×E Variance ===
  gxe_body <- if (gxe_class == "small") {
    paste0(
      sprintf("The G\u00d7E variance (\u03c3\u00b2G\u00d7E = %.4f) is SMALL relative to genotypic variance (\u03c3\u00b2G = %.4f). ", vc$sigma2_ge, vc$sigma2_genotype),
      "Genotype performance is relatively stable across environments. ",
      "Selection across environments is more reliable; a single good genotype may perform well in multiple locations."
    )
  } else if (gxe_class == "moderate") {
    paste0(
      sprintf("The G\u00d7E variance (\u03c3\u00b2G\u00d7E = %.4f) is MODERATE relative to genotypic variance (\u03c3\u00b2G = %.4f). ", vc$sigma2_ge, vc$sigma2_genotype),
      "Genotype performance shows meaningful environmental sensitivity. ",
      "Across-environment selection is possible, but attention to consistency across environments is warranted."
    )
  } else if (gxe_class == "large") {
    paste0(
      sprintf("The G\u00d7E variance (\u03c3\u00b2G\u00d7E = %.4f) is LARGE relative to genotypic variance (\u03c3\u00b2G = %.4f). ", vc$sigma2_ge, vc$sigma2_genotype),
      "Genotype performance is strongly influenced by environment; genotypes rank differently across locations or seasons. ",
      "Further stability analysis is recommended; environment-specific breeding or targeted deployment may be more effective than single-population selection."
    )
  } else if (gxe_class == "large_dominates") {
    paste0(
      "G\u00d7E variance dominates genotypic variance. Genotype ranks change substantially across environments. ",
      "This pattern suggests strong environment-specific adaptation; recommend environment-stratified breeding strategies and stability analysis."
    )
  } else {
    ""
  }
  text_parts$gxe_interp <- paste0("\nGenotype-by-Environment Interaction:\n", gxe_body, "\n")

  # === STEP 6: Breeding implication ===
  breed_body <- if (!isTRUE(result$heritability$h2_is_valid) || is.na(h2)) {
    "Heritability could not be reliably estimated. Reconsider experimental design to improve precision."
  } else if (h2_class == "high") {
    impl <- "Strong, stable genetic basis across environments. Direct across-environment selection should be effective."
    if (gxe_class == "small") {
      impl <- paste0(impl, " Single high-performing genotype likely suitable for broad deployment.")
    } else if (gxe_class %in% c("moderate", "large")) {
      impl <- paste0(impl, " However, consider environment-specific adaptation; G\u00d7E may warrant targeted deployment strategies.")
    }
    impl
  } else if (h2_class == "moderate") {
    paste0(
      "Moderate genetic basis across environments. Selection possible but should be combined with multi-environment testing. ",
      "Environmental standardization and targeted management may enhance response."
    )
  } else {
    paste0(
      "Weak genetic basis across environments. Environmental and/or G\u00d7E effects dominate. ",
      "Focus on environmental improvement and management optimization before intensifying direct selection."
    )
  }
  text_parts$breeding <- paste0("\nBreeding Implication:\n", breed_body, "\n")

  # === Warnings ===
  if (length(warnings_vc$warnings) > 0) {
    text_parts$warnings <- "\n\u26a0 Cautions and Warnings:\n"
    for (name in names(warnings_vc$warnings)) {
      w <- warnings_vc$warnings[[name]]
      msg <- if (is.list(w)) w$message else w
      text_parts$warnings <- paste0(text_parts$warnings, "  \u2022 ", msg, "\n")
    }
  }

  paste(unlist(text_parts), collapse = "")
}
