# vivasense_pca.R
#
# Principal Component Analysis (PCA) — R reference implementation
#
# Reference:
#   Pearson, K. (1901). On lines and planes of closest fit to systems of
#   points in space. Philosophical Magazine, 2(11), 559-572.
#
# This script is provided as an R-based reference implementation.
# The primary computation is performed by analysis_pca_routes.py using
# scikit-learn for reliability and portability.
#
# Function: compute_pca_analysis(observations, trait_cols, scale, n_components)
#   observations : data.frame with columns genotype, trait1, trait2, ...
#   trait_cols   : character vector of trait column names
#   scale        : logical (standardise before PCA; default TRUE)
#   n_components : integer or NULL (default: all)
#
# Returns a list with variance_explained, loadings, scores, interpretation.

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
})

compute_pca_analysis <- function(observations, trait_cols,
                                  scale = TRUE, n_components = NULL) {

  df <- as.data.frame(observations)
  df$genotype <- as.character(df$genotype)

  # Coerce trait columns to numeric
  for (tc in trait_cols) {
    df[[tc]] <- suppressWarnings(as.numeric(df[[tc]]))
  }

  # Aggregate to genotype means
  geno_means <- df %>%
    group_by(genotype) %>%
    summarise(across(all_of(trait_cols), \(x) mean(x, na.rm = TRUE)),
              .groups = "drop")

  # Drop traits/genotypes with any NA
  mat <- as.matrix(geno_means[, trait_cols])
  complete_rows <- complete.cases(mat)
  mat <- mat[complete_rows, ]
  geno_labels <- geno_means$genotype[complete_rows]

  n_genotypes <- nrow(mat)
  n_traits    <- ncol(mat)

  if (n_genotypes < 3) stop("PCA requires at least 3 genotypes with complete data.")
  if (n_traits    < 2) stop("PCA requires at least 2 traits with non-missing data.")

  max_components <- min(n_genotypes, n_traits)
  if (is.null(n_components)) {
    n_components <- max_components
  } else {
    n_components <- min(as.integer(n_components), max_components)
  }

  # Run PCA
  pca_result <- prcomp(mat, center = TRUE, scale. = scale)

  # Variance explained
  var_exp <- (pca_result$sdev^2 / sum(pca_result$sdev^2)) * 100
  var_exp <- var_exp[1:n_components]
  cum_var <- cumsum(var_exp)

  # Loadings (rotation matrix): rows = traits, cols = PCs
  loadings_mat <- pca_result$rotation[, 1:n_components, drop = FALSE]
  loadings_list <- lapply(rownames(loadings_mat), function(trait) {
    as.list(unname(loadings_mat[trait, ]))
  })
  names(loadings_list) <- rownames(loadings_mat)

  # Scores: rows = genotypes, cols = PCs
  scores_mat <- pca_result$x[, 1:n_components, drop = FALSE]
  scores_list <- lapply(seq_len(nrow(scores_mat)), function(i) {
    list(
      genotype = geno_labels[i],
      scores   = as.list(unname(scores_mat[i, ]))
    )
  })

  list(
    status             = "success",
    n_traits           = n_traits,
    n_genotypes        = n_genotypes,
    n_components       = n_components,
    variance_explained = as.list(round(var_exp, 4)),
    cumulative_variance = as.list(round(cum_var, 4)),
    loadings           = loadings_list,
    scores             = scores_list,
    interpretation     = paste0(
      "PCA computed on ", n_traits, " traits across ", n_genotypes, " genotypes. ",
      "PC1 explains ", round(var_exp[1], 1), "% of variance."
    )
  )
}

# ── Standalone execution ──────────────────────────────────────────────────────
if (!interactive()) {
  input_json <- readLines("stdin") |> paste(collapse = "\n")
  input_data <- jsonlite::fromJSON(input_json)
  result <- compute_pca_analysis(
    observations = input_data$observations,
    trait_cols   = input_data$trait_cols,
    scale        = isTRUE(input_data$scale),
    n_components = input_data$n_components
  )
  cat(jsonlite::toJSON(result, auto_unbox = TRUE, na = "null", digits = 6))
}
