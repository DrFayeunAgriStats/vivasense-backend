# vivasense_stability.R
#
# Stability Analysis Functions
#
# References:
#   Eberhart, S.A. and Russell, W.A. (1966). Stability parameters for
#   comparing varieties. Crop Science, 6(1), 36-40.
#   Purchase, J.L. et al. (2000). Genotype x environment interaction of
#   winter wheat in South Africa. Euphytica, 111, 35-42. (ASV metric)
#   Yan, W. and Kang, M.S. (2003). GGE Biplot Analysis. CRC Press.
#
# This script provides R-based reference implementations.
# The primary computation is performed by analysis_stability_routes.py
# using Python (numpy/scipy) for reliability and portability.

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

# ============================================================================
# AMMI ANALYSIS (Additive Main effects and Multiplicative Interaction)
# ============================================================================

#' Compute AMMI analysis using agricolae
#'
#' @param data data.frame with columns: genotype, environment, rep, trait_value
#' @param trait_name character
#' @param n_components integer, number of IPCA axes (default 2)
#' @return list with AMMI results
compute_ammi_analysis <- function(data, trait_name, n_components = 2) {

  df <- as.data.frame(data)
  df$trait_value  <- suppressWarnings(as.numeric(df$trait_value))
  df$genotype     <- as.character(df$genotype)
  df$environment  <- as.character(df$environment)
  df <- df[!is.na(df$trait_value), ]

  n_env <- length(unique(df$environment))
  if (n_env < 2) {
    return(list(
      status  = "error",
      message = "AMMI requires at least 2 environments"
    ))
  }

  # Compute genotype x environment means
  ge_means <- aggregate(
    trait_value ~ genotype + environment,
    data = df,
    FUN  = mean
  )

  # Run AMMI using agricolae::AMMI()
  ammi_result <- tryCatch({
    agricolae::AMMI(
      ENV     = ge_means$environment,
      GEN     = ge_means$genotype,
      REP     = 1,
      Y       = ge_means$trait_value,
      console = FALSE
    )
  }, error = function(e) {
    list(status = "error", message = paste("AMMI failed:", e$message))
  })

  if (!is.null(ammi_result$status) && ammi_result$status == "error") {
    return(ammi_result)
  }

  # Extract IPCA scores
  biplot      <- ammi_result$biplot
  geno_scores <- biplot[biplot$type == "GEN", ]
  env_scores  <- biplot[biplot$type == "ENV", ]

  # Variance explained
  pct_var <- ammi_result$analysis$percent

  # AMMI Stability Value (ASV) - Purchase et al. 2000
  if (length(pct_var) >= 2 && pct_var[2] > 0) {
    ss_ratio         <- sqrt(pct_var[1] / pct_var[2])
    geno_scores$asv  <- sqrt(
      (ss_ratio * geno_scores$PC1)^2 + geno_scores$PC2^2
    )
  } else {
    geno_scores$asv  <- abs(geno_scores$PC1)
  }

  geno_scores$asv_rank <- rank(geno_scores$asv)

  # Stability classification
  quantiles <- quantile(geno_scores$asv, probs = c(0.25, 0.50, 0.75))
  geno_scores$stability_class <- sapply(geno_scores$asv, function(asv) {
    if (asv <= quantiles[1]) return("highly stable")
    if (asv <= quantiles[2]) return("stable")
    if (asv <= quantiles[3]) return("moderately stable")
    return("unstable")
  })

  list(
    status             = "success",
    trait              = trait_name,
    variance_explained = as.list(pct_var),
    cumulative_variance = as.list(cumsum(pct_var)),
    genotype_scores    = geno_scores,
    environment_scores = env_scores,
    biplot_data        = list(
      genotypes    = geno_scores[, c("Code", "Mean", "PC1", "PC2")],
      environments = env_scores[, c("Code", "Mean", "PC1", "PC2")]
    )
  )
}

# ============================================================================
# GGE BIPLOT ANALYSIS (Genotype + Genotype x Environment)
# ============================================================================

#' Compute GGE biplot using metan package (if available) or base SVD
#'
#' @param data data.frame with columns: genotype, environment, rep, trait_value
#' @param trait_name character
#' @param biplot_type character: "which-won-where", "mean-stability", "discriminativeness"
#' @return list with GGE biplot results
compute_gge_biplot <- function(data, trait_name, biplot_type = "which-won-where") {

  df <- as.data.frame(data)
  df$trait_value  <- suppressWarnings(as.numeric(df$trait_value))
  df$genotype     <- as.character(df$genotype)
  df$environment  <- as.character(df$environment)
  df <- df[!is.na(df$trait_value), ]

  n_env <- length(unique(df$environment))
  if (n_env < 2) {
    return(list(
      status  = "error",
      message = "GGE Biplot requires at least 2 environments"
    ))
  }

  # Compute genotype x environment means
  ge_means <- aggregate(
    trait_value ~ genotype + environment,
    data = df,
    FUN  = mean
  )

  genotypes    <- sort(unique(ge_means$genotype))
  environments <- sort(unique(ge_means$environment))
  n_g <- length(genotypes)
  n_e <- length(environments)

  # Pivot to GE matrix
  ge_wide <- reshape(
    ge_means,
    idvar     = "genotype",
    timevar   = "environment",
    direction = "wide"
  )
  rownames(ge_wide) <- ge_wide$genotype
  ge_matrix <- as.matrix(ge_wide[genotypes, paste0("trait_value.", environments)])

  # Impute missing cells with column mean
  for (j in seq_len(ncol(ge_matrix))) {
    col_na         <- is.na(ge_matrix[, j])
    ge_matrix[col_na, j] <- mean(ge_matrix[, j], na.rm = TRUE)
  }

  grand_mean   <- mean(ge_matrix)
  geno_means   <- rowMeans(ge_matrix)
  env_means    <- colMeans(ge_matrix)

  # Environment-centred matrix (GGE = G + GxE)
  gge_matrix <- sweep(ge_matrix, 2, env_means, "-")

  # SVD
  svd_result  <- svd(gge_matrix)
  U <- svd_result$u
  S <- svd_result$d
  Vt <- t(svd_result$v)

  ss_total    <- sum(S^2)
  pct_var     <- if (ss_total > 0) (S^2 / ss_total * 100) else rep(0, length(S))

  n_pc <- min(2, length(S))
  # Symmetric partitioning
  geno_scores_mat <- U[, seq_len(n_pc)] * rep(sqrt(S[seq_len(n_pc)]), each = n_g)
  env_scores_mat  <- t(Vt[seq_len(n_pc), ]) * rep(sqrt(S[seq_len(n_pc)]), each = n_e)

  geno_df <- data.frame(
    genotype = genotypes,
    mean     = geno_means,
    PC1      = geno_scores_mat[, 1],
    PC2      = if (n_pc > 1) geno_scores_mat[, 2] else rep(0, n_g),
    row.names = NULL
  )

  env_df <- data.frame(
    environment = environments,
    mean        = env_means,
    PC1         = env_scores_mat[, 1],
    PC2         = if (n_pc > 1) env_scores_mat[, 2] else rep(0, n_e),
    row.names   = NULL
  )

  # Which-Won-Where
  which_won_where <- NULL
  if (biplot_type == "which-won-where") {
    winning_genos <- sapply(environments, function(env) {
      env_data <- ge_means[ge_means$environment == env, ]
      env_data$genotype[which.max(env_data$trait_value)]
    })
    names(winning_genos) <- environments

    mega_map <- tapply(names(winning_genos), winning_genos, c)
    mega_envs <- lapply(seq_along(mega_map), function(i) {
      winner <- names(mega_map)[i]
      envs   <- mega_map[[i]]
      yields <- sapply(envs, function(e) {
        d <- ge_means[ge_means$genotype == winner & ge_means$environment == e, ]
        if (nrow(d) > 0) mean(d$trait_value) else NA_real_
      })
      list(
        id          = i,
        environments = as.list(envs),
        best_genotype = winner,
        mean_yield  = round(mean(yields, na.rm = TRUE), 4)
      )
    })

    which_won_where <- list(
      mega_environments = mega_envs,
      winning_genotypes = as.list(winning_genos)
    )
  }

  # Mean vs Stability
  mean_vs_stability <- NULL
  if (biplot_type == "mean-stability") {
    distances <- sqrt(geno_df$PC1^2 + geno_df$PC2^2)
    norm_means <- (geno_means - min(geno_means)) / (max(geno_means) - min(geno_means) + 1e-9)
    scores <- norm_means / (distances + 1e-9)
    ideal_idx <- which.max(scores)

    dist_ranks <- rank(distances)
    geno_dist_df <- data.frame(
      genotype          = genotypes,
      distance_from_ideal = round(distances, 6),
      rank              = dist_ranks
    )
    geno_dist_df <- geno_dist_df[order(geno_dist_df$rank), ]

    mean_vs_stability <- list(
      ideal_genotype    = genotypes[ideal_idx],
      ideal_coordinates = list(PC1 = geno_df$PC1[ideal_idx], PC2 = geno_df$PC2[ideal_idx]),
      genotype_distances = geno_dist_df
    )
  }

  list(
    status             = "success",
    trait              = trait_name,
    variance_explained = as.list(round(pct_var[seq_len(n_pc)], 4)),
    cumulative_variance = round(sum(pct_var[seq_len(n_pc)]), 4),
    genotype_scores    = geno_df,
    environment_scores = env_df,
    which_won_where    = which_won_where,
    mean_vs_stability  = mean_vs_stability,
    biplot_data        = list(
      genotypes    = geno_df,
      environments = env_df
    )
  )
}