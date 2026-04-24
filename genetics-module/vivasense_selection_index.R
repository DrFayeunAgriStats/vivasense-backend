# vivasense_selection_index.R
# VivaSense — Smith-Hazel Selection Index (R reference / validation scripts)
#
# Use these for cross-checking Python results.
# Not invoked at runtime — Python-native implementation is used instead.

selection_index_vivasense <- function(geno_means, traits, econ_weights, h2_values) {
  # geno_means: data.frame with genotype rows and trait columns (already averaged)
  # econ_weights: named numeric vector of economic weights
  # h2_values: named numeric vector of heritabilities

  X <- as.matrix(geno_means[, traits])
  a <- econ_weights[traits]

  # Phenotypic covariance matrix (P)
  P <- cov(X)

  # Genetic covariance matrix (G): simplified from phenotypic correlations × h2
  pheno_corr <- cor(X)
  h2 <- h2_values[traits]
  gen_var <- h2 * diag(P)
  gen_sd  <- sqrt(gen_var)
  G <- pheno_corr * outer(gen_sd, gen_sd)

  # Index weights: b = P^{-1} G a
  b <- solve(P, G %*% a)

  # Index scores
  scores <- X %*% b
  ranked <- order(scores, decreasing = TRUE)

  cat("=== Smith-Hazel Selection Index Weights ===\n")
  names(b) <- traits
  print(b)

  cat("\n=== Genotype Rankings ===\n")
  print(data.frame(
    genotype = rownames(geno_means)[ranked],
    index    = scores[ranked]
  ))

  invisible(list(weights = b, scores = scores))
}

# For GBLUP-based selection index (requires sommer):
# library(sommer)
# # Fit mixed model to estimate G matrix, then compute Smith-Hazel index
