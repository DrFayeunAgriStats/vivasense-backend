# ============================================================================
# VIVASENSE FINAL INTERPRETATION ENGINE
# Strict Plant Breeding Standards (Lawrence Specification)
# Publication-Ready Output for Thesis & Journal Manuscripts
# ============================================================================

# String concatenation operator (not built into base R)
`%+%` <- paste0

library(jsonlite)

# ============================================================================
# SECTION 1 & 3: CLASSIFICATION ENGINE (STRICT, NO FUZZY BOUNDARIES)
# ============================================================================

classify_heritability <- function(h2) {
  if (is.na(h2)) return(NA_character_)
  if (h2 < 0.30) return("low")
  if (h2 >= 0.30 && h2 < 0.60) return("moderate")
  if (h2 >= 0.60) return("high")
}

classify_gam <- function(gam_percent) {
  if (is.na(gam_percent)) return(NA_character_)
  if (gam_percent < 5)  return("Low")      # < 5%  : low genetic advance
  if (gam_percent <= 10) return("Medium") # 5–10% : moderate genetic advance
  return("High")                           # > 10% : high genetic advance
}

classify_cv <- function(cv) {
  if (is.na(cv)) return(NA_character_)
  if (cv < 10) return("low")
  if (cv >= 10 && cv < 20) return("moderate")
  if (cv >= 20) return("high")
}

classify_gxe_importance <- function(sigma2_g, sigma2_ge) {
  if (is.na(sigma2_g) || is.na(sigma2_ge)) return(NA_character_)
  if (sigma2_g == 0) return("dominates")
  
  ratio <- sigma2_ge / sigma2_g
  if (ratio < 0.5) return("small")
  if (ratio >= 0.5 && ratio <= 1.0) return("moderate")
  if (ratio > 1.0) return("large")
}

# ============================================================================
# SECTION 4: INTERPRETATION ENGINE — 5-PARAGRAPH STRUCTURE (EXACT)
# ============================================================================

#' Generate publication-ready interpretation (single-environment)
#' 
#' Follows 5-paragraph structure:
#' 1. Numerical + H² classification
#' 2. H² + GAM joint interpretation (biological meaning)
#' 3. GCV vs PCV (environmental meaning)
#' 4. (N/A for single-env)
#' 5. Breeding implication
#'
generate_interpretation_single <- function(result) {
  
  vc <- result$variance_components
  gp <- result$genetic_parameters
  h2 <- result$heritability$h2_broad_sense
  n_g <- result$n_genotypes
  n_r <- result$n_reps
  mean <- result$grand_mean
  
  # Classifications
  h2_class <- classify_heritability(h2)
  gam_class <- classify_gam(gp$GAM_percent)
  gcv_class <- classify_cv(gp$GCV)
  pcv_class <- classify_cv(gp$PCV)
  
  paragraphs <- list()
  
  # === PARAGRAPH 1: Numerical + Heritability ===
  para1 <- sprintf(
    "Genetic analysis of %d genotypes across %d replicates yielded the following variance component estimates: " %+%
    "genotypic variance (σ²G) = %.4f, phenotypic variance (σ²P) = %.4f, and error variance (σ²E) = %.4f. " %+%
    "The estimated broad-sense heritability was H² = %.4f, which is classified as %s.",
    n_g, n_r,
    vc$sigma2_genotype,
    vc$sigma2_phenotypic,
    vc$sigma2_error,
    ifelse(is.na(h2), NA_character_, sprintf("%.3f", h2)),
    h2_class
  )
  paragraphs$para1 <- para1
  
  # === PARAGRAPH 2: H² + GAM Joint Interpretation ===
  para2 <- ""
  
  if (is.na(h2)) {
    para2 <- "Heritability could not be estimated due to data limitations."
  } else if (h2_class == "high" && gam_class == "High") {
    para2 <- sprintf(
      "The joint occurrence of high heritability (H² = %.3f) and high genetic advance as percent of mean (GAM = %.2f%%) " %+%
      "indicates strong genetic control with substantial expected response to direct selection. " %+%
      "This pattern suggests that additive gene effects are substantial; phenotypic selection should be effective.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "high" && gam_class == "Medium") {
    para2 <- sprintf(
      "Although broad-sense heritability was high (H² = %.3f), the genetic advance as percent of mean was moderate (GAM = %.2f%%). " %+%
      "This dissociation suggests that while genetic effects explain substantial phenotypic variation, " %+%
      "expected gains from single-generation selection are limited. Non-additive gene effects or restriction in allele frequency may account for the reduced additive response.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "high" && gam_class == "Low") {
    para2 <- sprintf(
      "Despite high heritability (H² = %.3f), genetic advance as percent of mean was low (GAM = %.2f%%). " %+%
      "This pattern suggests that genetic effects are predominantly non-additive; direct phenotypic selection is unlikely to generate rapid genetic improvement.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "moderate" && gam_class == "High") {
    para2 <- sprintf(
      "The moderate heritability (H² = %.3f) combined with high genetic advance as percent of mean (GAM = %.2f%%) indicates " %+%
      "that useful progress through direct selection is achievable, despite substantial environmental influence. " %+%
      "Environmental standardization may further improve selection response.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "moderate" && gam_class == "Medium") {
    para2 <- sprintf(
      "The estimated heritability (H² = %.3f) and genetic advance as percent of mean (GAM = %.2f%%) indicate moderate genetic control. " %+%
      "Genetic progress under direct selection should be steady, although environmental factors remain relevant.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "moderate" && gam_class == "Low") {
    para2 <- sprintf(
      "Although heritability was moderate (H² = %.3f), genetic advance as percent of mean was low (GAM = %.2f%%). " %+%
      "Direct selection response will be limited; consider combining selection with environmental optimization.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "low") {
    para2 <- sprintf(
      "The low heritability (H² = %.3f) indicates weak transmissible genetic control under the present conditions. " %+%
      "Phenotypic variation is dominated by environmental factors. Direct phenotypic selection is not recommended; " %+%
      "focus should be placed on optimizing environmental conditions and management practices.",
      h2
    )
  }
  
  paragraphs$para2 <- para2
  
  # === PARAGRAPH 3: GCV vs PCV (Environmental Meaning) ===
  para3 <- ""
  
  if (!is.na(gp$GCV) && !is.na(gp$PCV)) {
    pcv_gcv_diff <- gp$PCV - gp$GCV
    
    if (pcv_gcv_diff <= 2) {
      para3 <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) was very similar to the phenotypic coefficient of variation (PCV = %.2f%%), " %+%
        "indicating that environmental effects on trait expression were minimal. Phenotypic differences among genotypes reflect primarily genetic differences.",
        gp$GCV, gp$PCV
      )
    } else if (pcv_gcv_diff <= 7) {
      para3 <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) was moderately lower than the phenotypic coefficient of variation (PCV = %.2f%%), " %+%
        "indicating appreciable environmental influence on trait expression. However, genetic variation remains substantial.",
        gp$GCV, gp$PCV
      )
    } else {
      para3 <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) was substantially lower than the phenotypic coefficient of variation (PCV = %.2f%%), " %+%
        "indicating that environmental effects substantially modulate trait expression. Environmental standardization may reduce phenotypic variation and improve selection precision.",
        gp$GCV, gp$PCV
      )
    }
  } else {
    para3 <- "Genetic and phenotypic coefficients of variation could not be computed due to missing or invalid mean values."
  }
  
  paragraphs$para3 <- para3
  
  # === PARAGRAPH 4: Not applicable for single-environment ===
  paragraphs$para4 <- NA_character_
  
  # === PARAGRAPH 5: Breeding Implication (Mandatory) ===
  para5 <- ""
  
  if (is.na(h2)) {
    para5 <- "Breeding Implication: Heritability could not be reliably estimated. Expand or redesign the experiment."
  } else if (h2_class == "high") {
    para5 <- sprintf(
      "Breeding Implication: Strong genetic basis for the trait warrants direct phenotypic selection. " %+%
      "Prioritize identification and advancement of high-value individuals in the next generation."
    )
  } else if (h2_class == "moderate") {
    para5 <- sprintf(
      "Breeding Implication: Moderate genetic control allows for direct selection, but environmental standardization is advisable. " %+%
      "Consider multi-environment evaluation to assess genotype consistency."
    )
  } else {
    para5 <- sprintf(
      "Breeding Implication: Weak genetic control under present conditions. Focus on environmental improvement and management optimization before intensifying selection."
    )
  }
  
  paragraphs$para5 <- para5
  
  # Combine
  full_text <- paste(
    paragraphs$para1,
    paragraphs$para2,
    paragraphs$para3,
    paragraphs$para5,
    sep = "\n\n"
  )
  
  list(
    para1 = paragraphs$para1,
    para2 = paragraphs$para2,
    para3 = paragraphs$para3,
    para5 = paragraphs$para5,
    full_interpretation = full_text
  )
}


#' Generate publication-ready interpretation (multi-environment)
#' 
#' Follows 5-paragraph structure:
#' 1. Numerical + H² classification
#' 2. H² + GAM joint interpretation (biological meaning)
#' 3. GCV vs PCV (environmental meaning)
#' 4. G×E variance (stability meaning)
#' 5. Breeding implication
#'
generate_interpretation_multi <- function(result) {
  
  vc <- result$variance_components
  gp <- result$genetic_parameters
  h2 <- result$heritability$h2_broad_sense
  n_g <- result$n_genotypes
  n_e <- result$n_environments
  n_r <- result$n_reps
  mean <- result$grand_mean
  
  # Classifications
  h2_class <- classify_heritability(h2)
  gam_class <- classify_gam(gp$GAM_percent)
  gcv_class <- classify_cv(gp$GCV)
  pcv_class <- classify_cv(gp$PCV)
  gxe_class <- classify_gxe_importance(vc$sigma2_genotype, vc$sigma2_ge)
  
  paragraphs <- list()
  
  # === PARAGRAPH 1: Numerical + Heritability ===
  para1 <- sprintf(
    "Genetic analysis of %d genotypes evaluated across %d environments with %d replicates per genotype-environment combination " %+%
    "yielded variance component estimates on an entry-mean basis across environments. " %+%
    "Genotypic variance (σ²G) = %.4f, genotype-by-environment variance (σ²G×E) = %.4f, " %+%
    "phenotypic variance (σ²P) = %.4f. " %+%
    "The estimated broad-sense heritability was H² = %.4f, which is classified as %s.",
    n_g, n_e, n_r,
    vc$sigma2_genotype,
    vc$sigma2_ge,
    vc$sigma2_phenotypic,
    ifelse(is.na(h2), NA_character_, sprintf("%.3f", h2)),
    h2_class
  )
  paragraphs$para1 <- para1
  
  # === PARAGRAPH 2: H² + GAM Joint Interpretation ===
  para2 <- ""
  
  if (is.na(h2)) {
    para2 <- "Heritability could not be estimated due to data limitations."
  } else if (h2_class == "high" && gam_class == "High") {
    para2 <- sprintf(
      "The joint occurrence of high heritability (H² = %.3f) and high genetic advance as percent of mean (GAM = %.2f%%) " %+%
      "indicates strong genetic control with substantial expected response to across-environment selection. " %+%
      "This pattern suggests that additive gene effects are substantial and relatively stable across environments.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "high" && gam_class == "Medium") {
    para2 <- sprintf(
      "Although broad-sense heritability was high (H² = %.3f), genetic advance as percent of mean was moderate (GAM = %.2f%%). " %+%
      "This indicates substantial genetic effects with limited additive response; non-additive effects or genotype-by-environment " %+%
      "interactions may restrict the additive gains.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "moderate" && gam_class == "High") {
    para2 <- sprintf(
      "The moderate heritability (H² = %.3f) combined with high genetic advance as percent of mean (GAM = %.2f%%) indicates " %+%
      "that meaningful genetic progress across environments is achievable, despite appreciable environmental influence.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "moderate" && gam_class == "Medium") {
    para2 <- sprintf(
      "The heritability (H² = %.3f) and genetic advance as percent of mean (GAM = %.2f%%) indicate moderate genetic control across environments. " %+%
      "Steady genetic progress under selection is expected, though environmental factors remain relevant.",
      h2, gp$GAM_percent
    )
  } else if (h2_class == "low") {
    para2 <- sprintf(
      "The low heritability (H² = %.3f) indicates weak transmissible genetic control across environments. " %+%
      "Phenotypic variation is dominated by environmental and/or genotype-by-environment effects. " %+%
      "Direct across-environment selection is not recommended.",
      h2
    )
  } else {
    para2 <- "Heritability interpretation could not be generated."
  }
  
  paragraphs$para2 <- para2
  
  # === PARAGRAPH 3: GCV vs PCV (Environmental Meaning) ===
  para3 <- ""
  
  if (!is.na(gp$GCV) && !is.na(gp$PCV)) {
    pcv_gcv_diff <- gp$PCV - gp$GCV
    
    if (pcv_gcv_diff <= 2) {
      para3 <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) was very similar to the phenotypic coefficient of variation (PCV = %.2f%%), " %+%
        "indicating minimal environmental influence on trait expression across the environments tested.",
        gp$GCV, gp$PCV
      )
    } else if (pcv_gcv_diff <= 7) {
      para3 <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) was moderately lower than the phenotypic coefficient of variation (PCV = %.2f%%), " %+%
        "indicating that environmental effects across the test environments exert appreciable but not dominant influence on trait expression.",
        gp$GCV, gp$PCV
      )
    } else {
      para3 <- sprintf(
        "The genotypic coefficient of variation (GCV = %.2f%%) was substantially lower than the phenotypic coefficient of variation (PCV = %.2f%%), " %+%
        "indicating that environmental effects across locations and/or seasons substantially modulate trait expression. " %+%
        "Standardization of environmental conditions may improve selection precision.",
        gp$GCV, gp$PCV
      )
    }
  } else {
    para3 <- "Genetic and phenotypic coefficients of variation could not be computed due to missing or invalid mean values."
  }
  
  paragraphs$para3 <- para3
  
  # === PARAGRAPH 4: G×E Variance (Stability Meaning) ===
  para4 <- ""
  
  if (gxe_class == "small") {
    para4 <- sprintf(
      "The magnitude of genotype-by-environment variance (σ²G×E = %.4f) was small relative to genotypic variance (σ²G = %.4f). " %+%
      "This indicates that genotype performance is relatively stable across the environments evaluated. " %+%
      "A well-performing genotype selected in one environment is likely to perform well in others.",
      vc$sigma2_ge, vc$sigma2_genotype
    )
  } else if (gxe_class == "moderate") {
    para4 <- sprintf(
      "The magnitude of genotype-by-environment variance (σ²G×E = %.4f) was moderate relative to genotypic variance (σ²G = %.4f). " %+%
      "Genotype performance exhibits meaningful environmental sensitivity; genotypes rank somewhat differently across environments. " %+%
      "Attention to consistency across test environments is warranted in selection decisions.",
      vc$sigma2_ge, vc$sigma2_genotype
    )
  } else if (gxe_class == "large") {
    para4 <- sprintf(
      "The magnitude of genotype-by-environment variance (σ²G×E = %.4f) was large relative to genotypic variance (σ²G = %.4f). " %+%
      "Genotype performance is strongly influenced by environment; genotypes rank differently across environments. " %+%
      "Further stability analysis and consideration of environment-specific breeding strategies are recommended.",
      vc$sigma2_ge, vc$sigma2_genotype
    )
  } else {
    para4 <- "Genotype-by-environment variance could not be classified."
  }
  
  paragraphs$para4 <- para4
  
  # === PARAGRAPH 5: Breeding Implication (Mandatory) ===
  para5 <- ""
  
  if (is.na(h2)) {
    para5 <- "Breeding Implication: Heritability could not be reliably estimated. Redesign or expand the multi-environment trial."
  } else if (h2_class == "high") {
    if (gxe_class %in% c("small", NA_character_)) {
      para5 <- sprintf(
        "Breeding Implication: Strong genetic basis across environments with stable genotype performance. " %+%
        "Direct across-environment selection is effective; selected genotypes are suitable for broad deployment."
      )
    } else if (gxe_class == "moderate") {
      para5 <- sprintf(
        "Breeding Implication: Strong genetic basis, but moderate G×E indicates some environment-specific adaptation. " %+%
        "Across-environment selection is viable, but consider zone-specific variety recommendations."
      )
    } else {
      para5 <- sprintf(
        "Breeding Implication: Strong genetic basis, but large G×E indicates substantial environmental specificity. " %+%
        "Environment-stratified breeding and targeted variety deployment are recommended over single-population selection."
      )
    }
  } else if (h2_class == "moderate") {
    para5 <- sprintf(
      "Breeding Implication: Moderate genetic control across environments. Direct selection is possible, " %+%
      "but selection intensity should be balanced with multi-environment consistency. Environmental management improvements may enhance gains."
    )
  } else {
    para5 <- sprintf(
      "Breeding Implication: Weak genetic control across environments; environmental and/or G×E effects dominate. " %+%
      "Focus on environmental improvement and management optimization rather than direct genotype selection."
    )
  }
  
  paragraphs$para5 <- para5
  
  # MANDATORY PHRASE FOR MULTI-ENVIRONMENT
  mandatory_phrase <- "Estimates were obtained on an entry-mean basis across environments."
  
  # Combine
  full_text <- paste(
    paragraphs$para1,
    mandatory_phrase,
    paragraphs$para2,
    paragraphs$para3,
    paragraphs$para4,
    paragraphs$para5,
    sep = "\n\n"
  )
  
  list(
    para1 = paragraphs$para1,
    para2 = paragraphs$para2,
    para3 = paragraphs$para3,
    para4 = paragraphs$para4,
    para5 = paragraphs$para5,
    mandatory_phrase = mandatory_phrase,
    full_interpretation = full_text
  )
}


# ============================================================================
# SECTION 9: QUALITY CHECK BEFORE OUTPUT
# ============================================================================

quality_check <- function(interpretation, computation_mode) {
  
  checks <- list(
    has_numerical_values = grepl("σ²|H²|GAM|GCV|PCV", interpretation$full_interpretation),
    has_classification = grepl("low|moderate|high", interpretation$full_interpretation),
    has_gam_meaning = grepl("genetic advance|gain|response|selection", interpretation$full_interpretation),
    has_additive_implication = grepl("additive|non-additive|gene effect", interpretation$full_interpretation, 
                                      ignore.case = TRUE),
    has_environment_meaning = grepl("environmental|GCV|PCV", interpretation$full_interpretation),
    has_breeding_implication = grepl("Breeding Implication", interpretation$full_interpretation)
  )
  
  if (computation_mode == "multi_environment") {
    checks$has_gxe_meaning <- grepl("G×E|genotype.by.environment|environment.specific|stability", 
                                     interpretation$full_interpretation, ignore.case = TRUE)
    checks$has_mandatory_phrase <- grepl("entry-mean basis across environments", interpretation$full_interpretation)
  }
  
  list(
    all_checks_pass = all(unlist(checks)),
    checks = checks
  )
}


# ============================================================================
# SECTION 8: FINAL OUTPUT FUNCTION
# ============================================================================

#' Generate final VivaSense output JSON
#'
generate_vivasense_output <- function(result, computation_mode) {
  
  # Generate interpretation
  if (computation_mode == "single_environment") {
    interp <- generate_interpretation_single(result)
  } else {
    interp <- generate_interpretation_multi(result)
  }
  
  # Quality check
  qc <- quality_check(interp, computation_mode)
  
  if (!qc$all_checks_pass) {
    warning("Quality check failed on some criteria: ", 
            paste(names(qc$checks)[!unlist(qc$checks)], collapse = ", "))
  }
  
  # Classification summary
  h2_class <- classify_heritability(result$heritability$h2_broad_sense)
  gam_class <- classify_gam(result$genetic_parameters$GAM_percent)
  gcv_class <- classify_cv(result$genetic_parameters$GCV)
  pcv_class <- classify_cv(result$genetic_parameters$PCV)
  
  classification_summary <- list(
    heritability_level = h2_class,
    heritability_value = result$heritability$h2_broad_sense,
    gam_level = gam_class,
    gam_value = result$genetic_parameters$GAM_percent,
    gcv_level = gcv_class,
    pcv_level = pcv_class
  )
  
  if (computation_mode == "multi_environment") {
    classification_summary$gxe_importance <- classify_gxe_importance(
      result$variance_components$sigma2_genotype,
      result$variance_components$sigma2_ge
    )
  }
  
  # Extract breeding implication
  breeding_impl <- if (computation_mode == "single_environment") {
    interp$para5
  } else {
    interp$para5
  }
  
  # Caution note (if warnings exist)
  caution_note <- NA_character_
  if (length(result$variance_warnings) > 0) {
    caution_note <- paste(
      "CAUTIONS:",
      paste(names(result$variance_warnings), collapse = "; "),
      sep = " "
    )
  }
  
  # Final output structure
  output <- list(
    computation_mode = computation_mode,
    estimation_basis = result$heritability$interpretation_basis,
    variance_components = list(
      sigma2_genotype = result$variance_components$sigma2_genotype,
      sigma2_ge = result$variance_components$sigma2_ge,
      sigma2_error = result$variance_components$sigma2_error,
      sigma2_phenotypic = result$variance_components$sigma2_phenotypic
    ),
    genetic_parameters = list(
      GCV_percent = result$genetic_parameters$GCV,
      PCV_percent = result$genetic_parameters$PCV,
      GAM_absolute = result$genetic_parameters$GAM,
      GAM_percent = result$genetic_parameters$GAM_percent
    ),
    heritability = list(
      h2_broad_sense = result$heritability$h2_broad_sense,
      h2_is_valid = result$heritability$h2_is_valid,
      formula = result$heritability$formula
    ),
    classification_summary = classification_summary,
    interpretation_paragraph = interp$full_interpretation,
    breeding_implication = breeding_impl,
    caution_note = caution_note,
    quality_check = list(
      all_criteria_met = qc$all_checks_pass,
      details = qc$checks
    )
  )
  
  output
}
