# vivasense_manova.R
# VivaSense — MANOVA (R reference / validation scripts)
#
# Use these for cross-checking Python results.
# Not invoked at runtime — Python-native implementation is used instead.

manova_vivasense <- function(df, traits, factor_col, test = "Wilks") {
  # Build multivariate response
  Y <- as.matrix(df[, traits])
  f <- as.factor(df[[factor_col]])

  fit <- manova(Y ~ f)

  cat("=== MANOVA Summary (", test, "test) ===\n")
  print(summary(fit, test = test))

  cat("\n=== Univariate ANOVAs ===\n")
  print(summary.aov(fit))

  invisible(fit)
}

# Example usage (requires car package for additional tests):
# library(car)
# fit <- manova(cbind(yield, biomass, height) ~ genotype, data = df)
# summary(fit, test = "Wilks")
# summary.aov(fit)
