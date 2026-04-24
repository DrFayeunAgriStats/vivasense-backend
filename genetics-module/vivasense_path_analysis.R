# vivasense_path_analysis.R
# VivaSense — Path Analysis (R reference / validation scripts)
#
# Use these for cross-checking Python results.
# Not invoked at runtime — Python-native implementation is used instead.

path_analysis_vivasense <- function(df, outcome, predictors, standardize = TRUE) {
  formula <- as.formula(paste(outcome, "~", paste(predictors, collapse = " + ")))

  if (standardize) {
    df_std <- as.data.frame(scale(df[, c(outcome, predictors)]))
    fit <- lm(formula, data = df_std)
  } else {
    fit <- lm(formula, data = df)
  }

  cat("=== Path Analysis (Direct Effects) ===\n")
  print(summary(fit))

  cat("\n=== Correlation Matrix ===\n")
  print(cor(df[, c(outcome, predictors)]))

  invisible(fit)
}

# For SEM-based path analysis (requires lavaan):
# library(lavaan)
# model <- 'yield ~ biomass + height + days_to_flower'
# fit <- sem(model, data = df)
# summary(fit, standardized = TRUE)
