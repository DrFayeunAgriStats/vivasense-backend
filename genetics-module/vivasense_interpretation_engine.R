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
  if (gam_percent < 10) return("low")
  if (gam_percent < 20) return("moderate")
  return("high")
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
  text_parts$numerical <- sprintf(
    "\nEstimated Variance Components:\n" %+%
    "  Genotypic variance (σ²G):      %.4f\n" %+%
    "  Phenotypic variance (σ²P):     %.4f\n" %+%
    "  Environmental variance (σ²E):  %.4f\n" %+%
    "  Broad-sense heritability (h²): %.4f\n",
    vc$sigma2_genotype,
    vc$sigma2_phenotypic,
    vc$sigma2_error,
    ifelse(is.na(h2), "not computed", sprintf("%.4f", h2))
  )
  
  if (!is.na(gp$GCV)) {
    text_parts$numerical <- paste0(text_parts$numerical, sprintf(
      "\nGenetic Parameters:\n" %+%
      "  Genotypic coefficient of variation (GCV): %.2f%%\n" %+%
      "  Phenotypic coefficient of variation (PCV): %.2f%%\n" %+%
      "  Genetic advance (absolute): %.3f\n" %+%
      "  Genetic advance as percent of mean (GAM%%): %.2f%%\n",
      gp$GCV, gp$PCV, gp$GAM, gp$GAM_percent
    ))
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
      interp <- sprintf(
        "The estimated broad-sense heritability (h² = %.3f) indicates HIGH genetic control of the trait within this environment. " %+%
        "The corresponding genetic advance as percent of mean (GAM = %.2f%%) is HIGH, suggesting substantial expected response to direct selection. " %+%
        "The joint pattern of high h² and high GAM indicates that additive gene effects are likely important; direct phenotypic selection should be effective.",
        h2, gp$GAM_percent
      )
    }
    # High h2 + Moderate GAM
    else if (h2_class == "high" && gam_class == "moderate") {
      interp <- sprintf(
        "The estimated broad-sense heritability (h² = %.3f) indicates HIGH genetic control within this environment. " %+%
        "However, the genetic advance as percent of mean (GAM = %.2f%%) is only MODERATE, suggesting that while the trait is heritable, " %+%
        "expected gains from selection are more modest. This pattern may reflect non-additive gene action, restricted allele frequency, or both.",
        h2, gp$GAM_percent
      )
    }
    # High h2 + Low GAM
    else if (h2_class == "high" && gam_class == "low") {
      interp <- sprintf(
        "The estimated broad-sense heritability (h² = %.3f) indicates HIGH genetic control, yet the genetic advance as percent of mean (GAM = %.2f%%) is LOW. " %+%
        "This dissociation suggests that while phenotypic variation is substantially genetic, the expected response to selection is limited. " %+%
        "Non-additive gene effects, low effective population size, or strong inbreeding depression may be responsible.",
        h2, gp$GAM_percent
      )
    }
    # Moderate h2 + High GAM
    else if (h2_class == "moderate" && gam_class == "high") {
      interp <- sprintf(
        "The estimated broad-sense heritability (h² = %.3f) indicates MODERATE genetic control, with the genetic advance as percent of mean (GAM = %.2f%%) being HIGH. " %+%
        "This combination suggests that while environmental influence is substantial, useful progress from direct selection is achievable. " %+%
        "Both genetic and environmental management should be considered.",
        h2, gp$GAM_percent
      )
    }
    # Moderate h2 + Moderate GAM
    else if (h2_class == "moderate" && gam_class == "moderate") {
      interp <- sprintf(
        "The estimated broad-sense heritability (h² = %.3f) and genetic advance as percent of mean (GAM = %.2f%%) both indicate MODERATE genetic control. " %+%
        "Selection may be useful, though environmental factors remain important. Progress should be expected to be steady but not rapid.",
        h2, gp$GAM_percent
      )
    }
    # Moderate h2 + Low GAM
    else if (h2_class == "moderate" && gam_class == "low") {
      interp <- sprintf(
        "The estimated broad-sense heritability (h² = %.3f) suggests MODERATE genetic control, but the genetic advance as percent of mean (GAM = %.2f%%) is LOW. " %+%
        "Direct phenotypic selection may be slow. Consider investigating additive effects more carefully or combining selection with environmental optimization.",
        h2, gp$GAM_percent
      )
    }
    # Low h2
    else {
      interp <- sprintf(
        "The estimated broad-sense heritability (h² = %.3f) indicates LOW genetic control under the present environment. " %+%
        "Phenotypic variation is dominated by environmental factors and/or measurement variation. " %+%
        "Direct phenotypic selection is unlikely to be reliable; focus on improving growing conditions and management practices.",
        h2
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
      cv_interp <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) is only slightly lower than the phenotypic coefficient of variation (PCV = %.2f%%). " %+%
        "Environmental influence on trait expression is limited, and trait variation reflects primarily genetic differences.",
        gp$GCV, gp$PCV
      )
    } else if (pcv_gcv_diff <= 7) {
      cv_interp <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) is moderately lower than the phenotypic coefficient of variation (PCV = %.2f%%). " %+%
        "Environmental factors exert appreciable influence on trait expression, though genetic variation remains substantial.",
        gp$GCV, gp$PCV
      )
    } else {
      cv_interp <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) is substantially lower than the phenotypic coefficient of variation (PCV = %.2f%%). " %+%
        "Environmental factors strongly affect trait expression, and environmental standardization or management may improve selection response.",
        gp$GCV, gp$PCV
      )
    }
    
    text_parts$cv_interp <- paste0("\n", cv_interp, "\n")
  }
  
  # === STEP 6: Breeding implication ===
  text_parts$breeding <- "\n"
  if (!isTRUE(result$heritability$h2_is_valid) || is.na(h2)) {
    text_parts$breeding <- paste0(
      text_parts$breeding,
      "Breeding Implication:\n" %+%
      "Heritability could not be reliably estimated from the present data. Redesign or expand the experiment to improve precision.\n"
    )
  } else if (h2_class == "high") {
    text_parts$breeding <- paste0(
      text_parts$breeding,
      "Breeding Implication:\n" %+%
      "Strong genetic basis for the trait. Direct phenotypic selection for this trait should be effective in this environment. " %+%
      "Prioritize identification and advancement of high-value individuals for next-generation breeding.\n"
    )
  } else if (h2_class == "moderate") {
    text_parts$breeding <- paste0(
      text_parts$breeding,
      "Breeding Implication:\n" %+%
      "Moderate genetic basis. Direct selection is possible but should be combined with attention to environmental standardization. " %+%
      "Consider multi-environment evaluation to assess stability of selected genotypes.\n"
    )
  } else {
    text_parts$breeding <- paste0(
      text_parts$breeding,
      "Breeding Implication:\n" %+%
      "Weak genetic basis under present conditions. Direct selection will be unreliable. " %+%
      "Prioritize improvement of growing conditions, management practices, and measurement precision before intensifying selection.\n"
    )
  }
  
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
  h2_class <- classify_heritability(h2)
  gam_class <- classify_gam(gp$GAM_percent)
  gcv_class <- classify_cv(gp$GCV)
  pcv_class <- classify_cv(gp$PCV)
  gxe_class <- classify_gxe_importance(vc$sigma2_genotype, vc$sigma2_ge)
  
  text_parts <- list()
  
  # Header
  text_parts$header <- sprintf(
    "Multi-Environment Genetic Analysis\n%s\n",
    paste(rep("=", 60), collapse = "")
  )
  
  # Metadata
  text_parts$metadata <- sprintf(
    "\nTrait: %s\nGenotypes: %d | Environments: %d | Replicates per G×E: %d\n" %+%
    "Grand mean: %.2f\nEnvironment model: %s\n" %+%
    "Analysis basis: Entry-mean broad-sense heritability across environments\n",
    "(not specified)",
    result$n_genotypes,
    result$n_environments,
    result$n_reps,
    result$grand_mean,
    ifelse(vc$heritability_basis == "fixed_environment_model", 
           "Fixed (standard)", 
           "Random (advanced)")
  )
  
  # === STEP 1: Report numerical values ===
  text_parts$numerical <- sprintf(
    "\nEstimated Variance Components:\n" %+%
    "  Genotypic variance (σ²G):           %.4f\n" %+%
    "  Genotype-by-environment variance (σ²G×E): %.4f\n" %+%
    "  Phenotypic variance on entry-mean basis (σ²P): %.4f\n" %+%
    "  Broad-sense heritability (h²):      %.4f\n" %+%
    "  Heritability formula: %s\n",
    vc$sigma2_genotype,
    vc$sigma2_ge,
    vc$sigma2_phenotypic,
    ifelse(is.na(h2), "not computed", sprintf("%.4f", h2)),
    result$heritability$formula
  )
  
  if (!is.na(gp$GCV)) {
    text_parts$numerical <- paste0(text_parts$numerical, sprintf(
      "\nGenetic Parameters (across-environment entry-means):\n" %+%
      "  Genotypic coefficient of variation (GCV): %.2f%%\n" %+%
      "  Phenotypic coefficient of variation (PCV): %.2f%%\n" %+%
      "  Genetic advance (absolute): %.3f\n" %+%
      "  Genetic advance as percent of mean (GAM%%): %.2f%%\n",
      gp$GCV, gp$PCV, gp$GAM, gp$GAM_percent
    ))
  }
  
  # === STEP 2-3: Interpret h2 and GAM jointly ===
  text_parts$interpretation <- "\nInterpretation:\n"
  
  if (!isTRUE(result$heritability$h2_is_valid) || is.na(h2)) {
    text_parts$interpretation <- paste0(
      text_parts$interpretation,
      "Heritability could not be computed. Phenotypic variance is zero or invalid.\n"
    )
  } else {
    interp <- sprintf(
      "The estimated broad-sense heritability across environments (h² = %.3f) is classified as %s. " %+%
      "The genetic advance as percent of mean (GAM = %.2f%%) is classified as %s. ",
      h2, h2_class, gp$GAM_percent, gam_class
    )
    
    # Joint interpretation pattern
    if (h2_class == "high" && gam_class == "high") {
      interp <- paste0(interp,
        "The combination of high h² and high GAM indicates strong genetic control with substantial expected response to selection. " %+%
        "Additive gene effects appear important; direct selection across environments should be effective."
      )
    } else if (h2_class == "high" && gam_class == "moderate") {
      interp <- paste0(interp,
        "High h² with moderate GAM suggests heritable control but limited immediate gains. Consider non-additive effects or linkage phase concerns."
      )
    } else if (h2_class == "moderate" && gam_class == "high") {
      interp <- paste0(interp,
        "Moderate h² with high GAM indicates that useful selection response is achievable despite environmental influence."
      )
    } else if (h2_class == "moderate" && gam_class == "moderate") {
      interp <- paste0(interp,
        "Moderate h² and moderate GAM indicate steady but not rapid genetic progress is achievable under selection."
      )
    } else if (h2_class == "low") {
      interp <- paste0(interp,
        "Low h² indicates that genetic control is weak under the present environmental conditions. " %+%
        "Environmental and/or G×E effects dominate trait expression; direct selection is less reliable."
      )
    }
    
    text_parts$interpretation <- paste0(text_parts$interpretation, interp, "\n")
  }
  
  # === STEP 4: Interpret GCV vs PCV ===
  if (!is.na(gp$GCV) && !is.na(gp$PCV)) {
    pcv_gcv_diff <- gp$PCV - gp$GCV
    
    cv_interp <- sprintf(
      "The genotypic coefficient of variation (GCV = %.2f%%) is ",
      gp$GCV
    )
    
    if (pcv_gcv_diff <= 2) {
      cv_interp <- paste0(cv_interp,
        sprintf("only slightly lower than the phenotypic coefficient of variation (PCV = %.2f%%). " %+%
        "Environmental influence across environments is limited.",
        gp$PCV)
      )
    } else if (pcv_gcv_diff <= 7) {
      cv_interp <- paste0(cv_interp,
        sprintf("moderately lower than the phenotypic coefficient of variation (PCV = %.2f%%). " %+%
        "Environmental variation across environments is appreciable.",
        gp$PCV)
      )
    } else {
      cv_interp <- paste0(cv_interp,
        sprintf("substantially lower than the phenotypic coefficient of variation (PCV = %.2f%%). " %+%
        "Environment and/or G×E effects substantially influence trait expression.",
        gp$PCV)
      )
    }
    
    text_parts$cv_interp <- paste0("\n", cv_interp, "\n")
  }
  
  # === STEP 5: Interpret G×E Variance ===
  text_parts$gxe_interp <- "\nGenotype-by-Environment Interaction:\n"
  
  if (gxe_class == "small") {
    text_parts$gxe_interp <- paste0(
      text_parts$gxe_interp,
      sprintf(
        "The G×E variance (σ²G×E = %.4f) is SMALL relative to genotypic variance (σ²G = %.4f). " %+%
        "Genotype performance is relatively stable across environments. " %+%
        "Selection across environments is more reliable; a single good genotype may perform well in multiple locations.",
        vc$sigma2_ge, vc$sigma2_genotype
      )
    )
  } else if (gxe_class == "moderate") {
    text_parts$gxe_interp <- paste0(
      text_parts$gxe_interp,
      sprintf(
        "The G×E variance (σ²G×E = %.4f) is MODERATE relative to genotypic variance (σ²G = %.4f). " %+%
        "Genotype performance shows meaningful environmental sensitivity. " %+%
        "Across-environment selection is possible, but attention to consistency across environments is warranted.",
        vc$sigma2_ge, vc$sigma2_genotype
      )
    )
  } else if (gxe_class == "large") {
    text_parts$gxe_interp <- paste0(
      text_parts$gxe_interp,
      sprintf(
        "The G×E variance (σ²G×E = %.4f) is LARGE relative to genotypic variance (σ²G = %.4f). " %+%
        "Genotype performance is strongly influenced by environment; genotypes rank differently across locations or seasons. " %+%
        "Further stability analysis is recommended; environment-specific breeding or targeted deployment may be more effective than single-population selection.",
        vc$sigma2_ge, vc$sigma2_genotype
      )
    )
  } else if (gxe_class == "large_dominates") {
    text_parts$gxe_interp <- paste0(
      text_parts$gxe_interp,
      "G×E variance dominates genotypic variance. Genotype ranks change substantially across environments. " %+%
      "This pattern suggests strong environment-specific adaptation; recommend environment-stratified breeding strategies and stability analysis."
    )
  }
  
  text_parts$gxe_interp <- paste0(text_parts$gxe_interp, "\n")
  
  # === STEP 6: Breeding implication ===
  text_parts$breeding <- "\n"
  
  if (!isTRUE(result$heritability$h2_is_valid) || is.na(h2)) {
    text_parts$breeding <- paste0(
      text_parts$breeding,
      "Breeding Implication:\n" %+%
      "Heritability could not be reliably estimated. Reconsider experimental design to improve precision.\n"
    )
  } else if (h2_class == "high") {
    breed_impl <- "Strong, stable genetic basis across environments. Direct across-environment selection should be effective."
    
    if (gxe_class == "small") {
      breed_impl <- paste0(breed_impl, 
        " Single high-performing genotype likely suitable for broad deployment."
      )
    } else if (gxe_class %in% c("moderate", "large")) {
      breed_impl <- paste0(breed_impl, 
        " However, consider environment-specific adaptation; G×E may warrant targeted deployment strategies."
      )
    }
    
    text_parts$breeding <- paste0(
      text_parts$breeding,
      "Breeding Implication:\n", breed_impl, "\n"
    )
  } else if (h2_class == "moderate") {
    text_parts$breeding <- paste0(
      text_parts$breeding,
      "Breeding Implication:\n" %+%
      "Moderate genetic basis across environments. Selection possible but should be combined with multi-environment testing. " %+%
      "Environmental standardization and targeted management may enhance response.\n"
    )
  } else {
    text_parts$breeding <- paste0(
      text_parts$breeding,
      "Breeding Implication:\n" %+%
      "Weak genetic basis across environments. Environmental and/or G×E effects dominate. " %+%
      "Focus on environmental improvement and management optimization before intensifying direct selection.\n"
    )
  }
  
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
