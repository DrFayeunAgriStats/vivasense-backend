# vivasense_blup.R
#
# BLUP (Best Linear Unbiased Prediction) using lme4
#
# Reference:
#   Henderson, C.R. (1975). Best linear unbiased estimation and prediction
#   under a selection model. Biometrics, 31(2), 423-447.
#
# This script is provided as an R-based reference implementation using lme4.
# The primary computation is performed by analysis_blup_routes.py using
# statsmodels.MixedLM for reliability and portability.
#
# Function: compute_blup_analysis(observations, trait_name, model_type)
#   observations : data.frame with columns genotype, [environment], [rep], trait_value
#   trait_name   : character string
#   model_type   : "single-environment" or "multi-environment"
#
# Returns a list suitable for JSON serialisation.

suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
  library(dplyr)
})

compute_blup_analysis <- function(observations, trait_name = "trait",
                                   model_type = "single-environment") {

  df <- as.data.frame(observations)
  df$trait_value <- suppressWarnings(as.numeric(df$trait_value))
  df$genotype    <- as.character(df$genotype)
  df <- df[!is.na(df$trait_value), ]

  n_genos <- length(unique(df$genotype))
  if (n_genos < 2) stop("BLUP requires at least 2 genotypes.")

  # Fit mixed model: genotype as random, environment as fixed (if multi-env)
  if (model_type == "multi-environment" && "environment" %in% colnames(df)) {
    df$environment <- as.character(df$environment)
    formula_str <- paste0(trait_name, " ~ environment + (1 | genotype)")
    # Rename trait column for lmer
    df$__trait__ <- df[[trait_name]]
    formula_str  <- "__trait__ ~ environment + (1 | genotype)"
  } else {
    df$__trait__ <- df$trait_value
    formula_str  <- "__trait__ ~ 1 + (1 | genotype)"
    model_type   <- "single-environment"
  }

  fit <- tryCatch(
    lmer(as.formula(formula_str), data = df, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa")),
    error = function(e) stop(paste("lmer failed:", conditionMessage(e)))
  )

  # Extract BLUPs (random effects)
  re      <- ranef(fit)$genotype
  blups   <- setNames(re[[1]], rownames(re))
  se_re   <- se.ranef(fit)$genotype[, 1]
  names(se_re) <- rownames(re)

  # Variance components
  vc          <- as.data.frame(VarCorr(fit))
  sigma2_g    <- vc$vcov[vc$grp == "genotype"]
  sigma2_e    <- sigma(fit)^2
  sigma2_g    <- max(sigma2_g, 1e-10)

  genotypes <- names(blups)
  blup_rows <- lapply(genotypes, function(g) {
    blup_val    <- blups[g]
    se_val      <- se_re[g]
    pev         <- se_val^2
    reliability <- max(0, min(1, 1 - pev / sigma2_g))
    list(
      genotype    = g,
      blup        = blup_val,
      se          = se_val,
      reliability = reliability
    )
  })

  blup_df <- do.call(rbind, lapply(blup_rows, as.data.frame))
  blup_df <- blup_df[order(-blup_df$blup), ]
  blup_df$rank <- seq_len(nrow(blup_df))

  n_top <- max(1, ceiling(nrow(blup_df) * 0.10))
  best_genotypes <- blup_df$genotype[1:n_top]

  list(
    status             = "success",
    trait              = trait_name,
    model_type         = model_type,
    genotype_blups     = blup_df,
    best_genotypes     = as.list(best_genotypes),
    variance_components = list(
      sigma2_genotype  = sigma2_g,
      sigma2_residual  = sigma2_e,
      sigma2_phenotypic = sigma2_g + sigma2_e
    ),
    interpretation = paste0(
      "BLUP analysis for ", trait_name, " (", model_type, " model). ",
      "Top genotypes: ", paste(best_genotypes, collapse = ", "), "."
    )
  )
}

# ── Standalone execution ──────────────────────────────────────────────────────
if (!interactive()) {
  input_json <- readLines("stdin") |> paste(collapse = "\n")
  input_data <- jsonlite::fromJSON(input_json)
  result <- compute_blup_analysis(
    observations = input_data$observations,
    trait_name   = input_data$trait_name,
    model_type   = input_data$model_type
  )
  cat(jsonlite::toJSON(result, auto_unbox = TRUE, na = "null", digits = 6))
}
