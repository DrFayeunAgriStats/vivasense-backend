# vivasense_nonparametric.R
# VivaSense — Non-parametric Tests (R reference / validation scripts)
#
# Use these for cross-checking Python results.
# Not invoked at runtime — Python-native implementation is used instead.

kruskal_test_vivasense <- function(df, trait, group) {
  formula <- as.formula(paste(trait, "~", group))
  result  <- kruskal.test(formula, data = df)
  cat("Kruskal-Wallis H =", result$statistic, "\n")
  cat("df =", result$parameter, "\n")
  cat("p-value =", result$p.value, "\n")
  invisible(result)
}

friedman_test_vivasense <- function(df, trait, group, block) {
  formula <- as.formula(paste(trait, "~", group, "|", block))
  result  <- friedman.test(formula, data = df)
  cat("Friedman Chi-sq =", result$statistic, "\n")
  cat("df =", result$parameter, "\n")
  cat("p-value =", result$p.value, "\n")
  invisible(result)
}

# Example usage:
# kruskal_test_vivasense(df, "disease_score", "genotype")
# friedman_test_vivasense(df, "score", "treatment", "block")
