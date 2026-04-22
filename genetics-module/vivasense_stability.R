# vivasense_stability.R
#
# Eberhart-Russell (1966) Stability Analysis
#
# Reference:
#   Eberhart, S.A. and Russell, W.A. (1966). Stability parameters for
#   comparing varieties. Crop Science, 6(1), 36-40.
#
# This script is provided as an R-based reference implementation.
# The primary computation is performed by analysis_stability_routes.py
# using Python (numpy/scipy) for reliability and portability.
#
# Usage (standalone):
#   Rscript vivasense_stability.R  # (supply data via stdin as JSON)
#
# Function: compute_stability_analysis(observations, trait_name)
#   observations : data.frame with columns genotype, environment, trait_value
#   trait_name   : character string (used in output labels)
#
# Returns a JSON string with:
#   status, trait, n_genotypes, n_environments, genotype_stability,
#   environment_means, grand_mean, best_stable_genotypes, interpretation

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
})

compute_stability_analysis <- function(observations, trait_name = "trait") {

  df <- as.data.frame(observations)
  df$trait_value  <- suppressWarnings(as.numeric(df$trait_value))
  df$genotype     <- as.character(df$genotype)
  df$environment  <- as.character(df$environment)
  df <- df[!is.na(df$trait_value), ]

  n_envs  <- length(unique(df$environment))
  n_genos <- length(unique(df$genotype))

  if (n_envs < 2) {
    stop("Stability analysis requires data from at least 2 environments.")
  }
  if (n_genos < 3) {
    stop("Stability analysis requires at least 3 genotypes.")
  }

  # Cell means (average across reps within genotype × environment)
  cell_means <- df %>%
    group_by(genotype, environment) %>%
    summarise(cell_mean = mean(trait_value, na.rm = TRUE), .groups = "drop")

  # Environmental means and index
  env_means <- cell_means %>%
    group_by(environment) %>%
    summarise(env_mean = mean(cell_mean, na.rm = TRUE), .groups = "drop")
  grand_mean <- mean(env_means$env_mean)
  env_means$env_index <- env_means$env_mean - grand_mean

  # Eberhart-Russell regression per genotype
  genotypes <- sort(unique(cell_means$genotype))
  results_list <- lapply(genotypes, function(g) {
    gd <- cell_means[cell_means$genotype == g, ]
    gd <- merge(gd, env_means, by = "environment")
    x  <- gd$env_index
    y  <- gd$cell_mean
    n_e <- nrow(gd)
    if (n_e < 2) {
      return(list(genotype = g, mean = mean(y), bi = 1.0, s2di = NA_real_))
    }
    fit <- lm(y ~ x)
    bi  <- coef(fit)[["x"]]
    ss_res <- sum(residuals(fit)^2)
    s2di   <- if (n_e > 2) ss_res / (n_e - 2) else 0.0
    list(
      genotype = g,
      mean     = mean(y, na.rm = TRUE),
      bi       = bi,
      s2di     = s2di
    )
  })

  results_df <- do.call(rbind, lapply(results_list, as.data.frame))
  results_df <- results_df[order(-results_df$mean), ]
  results_df$rank <- seq_len(nrow(results_df))

  # Classify
  s2di_threshold <- quantile(results_df$s2di, 0.75, na.rm = TRUE)
  results_df$stability_class <- with(results_df, ifelse(
    bi >= 0.9 & bi <= 1.1 & s2di <= s2di_threshold, "stable",
    ifelse(bi > 1.1 & s2di <= s2di_threshold, "responsive_favorable",
    ifelse(bi < 0.9 & s2di <= s2di_threshold, "responsive_poor",
    "unpredictable"))
  ))

  # Best stable genotypes
  best_stable <- results_df$genotype[
    results_df$stability_class == "stable" &
    results_df$mean >= grand_mean
  ]
  if (length(best_stable) == 0) {
    best_stable <- results_df$genotype[1:min(3, nrow(results_df))]
  }

  env_means_named <- setNames(
    as.list(env_means$env_mean),
    env_means$environment
  )

  interpretation <- paste0(
    "Eberhart-Russell stability analysis for ", trait_name, " across ",
    n_envs, " environments. Grand mean = ", round(grand_mean, 3),
    ". Recommended stable genotypes: ", paste(best_stable, collapse = ", "), "."
  )

  list(
    status               = "success",
    trait                = trait_name,
    n_genotypes          = n_genos,
    n_environments       = n_envs,
    grand_mean           = grand_mean,
    genotype_stability   = results_df,
    environment_means    = env_means_named,
    best_stable_genotypes = as.list(best_stable),
    interpretation       = interpretation
  )
}

# ── Standalone execution (when called as a script) ────────────────────────────
if (!interactive()) {
  input_json <- readLines("stdin") |> paste(collapse = "\n")
  input_data <- jsonlite::fromJSON(input_json)
  result <- compute_stability_analysis(
    observations = input_data$observations,
    trait_name   = input_data$trait_name
  )
  cat(jsonlite::toJSON(result, auto_unbox = TRUE, na = "null", digits = 6))
}
