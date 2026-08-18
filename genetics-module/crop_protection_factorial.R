suppressPackageStartupMessages({
  library(agricolae)
})

# Crop-protection factorial-CRD adapter.
#
# This deliberately does not alter compute_single_environment().  Replicate is
# retained in `data` for experimental-unit provenance, but it is absent from the
# fitted model because laboratory CRD replicates are independent units rather
# than blocks.
compute_crop_protection_factorial_crd <- function(data) {
  required <- c("treatment", "dose", "replicate", "inference_value")
  missing <- setdiff(required, colnames(data))
  if (length(missing) > 0) {
    stop(sprintf("Missing adapter columns: %s", paste(missing, collapse = ", ")))
  }

  data$treatment <- factor(data$treatment)
  data$dose <- factor(data$dose, levels = unique(data$dose))
  data$replicate <- factor(data$replicate)

  # Replicate is intentionally not in this formula.
  model <- aov(inference_value ~ treatment * dose, data = data)
  at <- anova(model)
  residual_row <- which(rownames(at) == "Residuals")
  mse <- as.numeric(at[residual_row, "Mean Sq"])
  error_df <- as.integer(at[residual_row, "Df"])

  cell_key <- interaction(data$treatment, data$dose, sep = "\037", drop = TRUE)
  cell_n <- table(cell_key)
  balanced <- length(unique(as.integer(cell_n))) == 1L

  hsd <- HSD.test(
    model,
    c("treatment", "dose"),
    DFerror = error_df,
    MSerror = mse,
    group = TRUE,
    console = FALSE,
    unbalanced = !balanced
  )

  group_rows <- rownames(hsd$groups)
  group_parts <- strsplit(group_rows, ":", fixed = TRUE)
  group_lookup <- setNames(
    as.character(hsd$groups$groups),
    vapply(group_parts, function(x) paste(x, collapse = "\037"), character(1))
  )

  split_rows <- split(seq_len(nrow(data)), cell_key)
  interaction_means <- lapply(split_rows, function(idx) {
    inference <- data$inference_value[idx]
    display <- if ("display_value" %in% colnames(data)) data$display_value[idx] else inference
    n_cell <- length(idx)
    treatment <- as.character(data$treatment[idx[1]])
    dose <- as.character(data$dose[idx[1]])
    key <- paste(treatment, dose, sep = "\037")

    list(
      treatment = treatment,
      dose = dose,
      n = n_cell,
      mean_inference_scale = mean(inference),
      mean_display_scale = mean(display),
      se_inference_scale = sqrt(mse / n_cell),
      se_display_scale = if (n_cell > 1L) stats::sd(display) / sqrt(n_cell) else NA_real_,
      letter = unname(group_lookup[[key]])
    )
  })

  marginal <- function(group_name) {
    groups <- split(seq_len(nrow(data)), data[[group_name]])
    lapply(names(groups), function(level) {
      idx <- groups[[level]]
      display <- if ("display_value" %in% colnames(data)) data$display_value[idx] else data$inference_value[idx]
      list(
        level = level,
        n = length(idx),
        mean_inference_scale = mean(data$inference_value[idx]),
        mean_display_scale = mean(display),
        se_display_scale = if (length(idx) > 1L) stats::sd(display) / sqrt(length(idx)) else NA_real_
      )
    })
  }

  source_names <- rownames(at)
  source_names[source_names == "treatment"] <- "Treatment"
  source_names[source_names == "dose"] <- "Dose"
  source_names[source_names == "treatment:dose"] <- "Treatment:Dose"
  source_names[source_names == "Residuals"] <- "Error"

  anova_rows <- lapply(seq_len(nrow(at)), function(i) {
    list(
      source = source_names[i],
      df = as.integer(at[i, "Df"]),
      ss = as.numeric(at[i, "Sum Sq"]),
      ms = as.numeric(at[i, "Mean Sq"]),
      f_value = if (is.na(at[i, "F value"])) NA_real_ else as.numeric(at[i, "F value"]),
      p_value = if (is.na(at[i, "Pr(>F)"])) NA_real_ else as.numeric(at[i, "Pr(>F)"])
    )
  })

  interaction_p <- as.numeric(at[which(rownames(at) == "treatment:dose"), "Pr(>F)"])

  list(
    model_formula = "response ~ treatment * dose",
    anova = anova_rows,
    residual_mean_square = mse,
    error_df = error_df,
    balanced = balanced,
    common_cell_n = if (balanced) as.integer(cell_n[[1]]) else NA_integer_,
    common_interaction_se = if (balanced) sqrt(mse / as.integer(cell_n[[1]])) else NA_real_,
    interaction = list(
      significant = is.finite(interaction_p) && interaction_p < 0.05,
      means = unname(interaction_means)
    ),
    marginal_means = list(
      treatment = marginal("treatment"),
      dose = marginal("dose")
    )
  )
}
