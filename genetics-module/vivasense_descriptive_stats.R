# VivaSense Descriptive Statistics Engine
# Dependencies: jsonlite, car, moments, dplyr, tidyr

suppressPackageStartupMessages({
  library(jsonlite)
  library(car)
  library(moments)
  library(dplyr)
  library(tidyr)
})

compute_descriptive_stats <- function(data, trait, group_var = NULL) {
  x <- data[[trait]]
  x_clean <- na.omit(x)
  
  if (length(x_clean) == 0) return(NULL)
  
  overall <- list(
    n = length(x_clean),
    mean = mean(x_clean),
    sd = sd(x_clean),
    min = min(x_clean),
    q1 = unname(quantile(x_clean, 0.25)),
    median = median(x_clean),
    q3 = unname(quantile(x_clean, 0.75)),
    max = max(x_clean),
    cv_percent = ifelse(mean(x_clean) != 0, (sd(x_clean) / mean(x_clean)) * 100, NA),
    skewness = skewness(x_clean),
    kurtosis = kurtosis(x_clean),
    n_missing = sum(is.na(x))
  )
  
  by_group <- NULL
  if (!is.null(group_var) && group_var %in% names(data)) {
    by_group <- data %>%
      group_by(!!sym(group_var)) %>%
      summarise(
        n = sum(!is.na(!!sym(trait))),
        mean = mean(!!sym(trait), na.rm = TRUE),
        sd = sd(!!sym(trait), na.rm = TRUE),
        min = min(!!sym(trait), na.rm = TRUE),
        max = max(!!sym(trait), na.rm = TRUE),
        cv_percent = ifelse(mean(!!sym(trait), na.rm=TRUE) != 0, 
                            (sd(!!sym(trait), na.rm=TRUE) / mean(!!sym(trait), na.rm=TRUE)) * 100, NA),
        .groups = 'drop'
      ) %>%
      as.data.frame()
  }
  
  return(list(overall = overall, by_group = by_group))
}

test_normality_shapiro <- function(data, trait, group_var = NULL) {
  x <- na.omit(data[[trait]])
  res <- list(W = NA, p_value = NA, normal_05 = NA, interpretation = "Insufficient data")
  
  if (length(x) >= 3 && length(x) <= 5000 && var(x) > 0) {
    sw <- tryCatch(shapiro.test(x), error = function(e) NULL)
    if (!is.null(sw)) {
      res$W <- sw$statistic
      res$p_value <- sw$p.value
      res$normal_05 <- sw$p.value >= 0.05
      res$interpretation <- ifelse(sw$p.value >= 0.05, "Normal", "Non-normal")
    }
  }
  return(res)
}

test_homogeneity_levene <- function(data, trait, group_var) {
  res <- list(statistic = NA, p_value = NA, homogeneous_05 = NA, interpretation = "Insufficient data")
  
  clean_data <- data[!is.na(data[[trait]]) & !is.na(data[[group_var]]), ]
  if (nrow(clean_data) > 0 && length(unique(clean_data[[group_var]])) > 1) {
    clean_data[[group_var]] <- as.factor(clean_data[[group_var]])
    lt <- tryCatch(car::leveneTest(clean_data[[trait]] ~ clean_data[[group_var]], center=median), error = function(e) NULL)
    
    if (!is.null(lt)) {
      res$statistic <- lt$`F value`[1]
      res$p_value <- lt$`Pr(>F)`[1]
      res$homogeneous_05 <- lt$`Pr(>F)`[1] >= 0.05
      res$interpretation <- ifelse(lt$`Pr(>F)`[1] >= 0.05, "Homogeneous", "Heterogeneous")
    }
  }
  return(res)
}

detect_outliers_iqr <- function(data, trait) {
  x <- data[[trait]]
  clean_indices <- which(!is.na(x))
  x_clean <- x[clean_indices]
  
  if (length(x_clean) < 4) return(data.frame())
  
  q1 <- unname(quantile(x_clean, 0.25))
  q3 <- unname(quantile(x_clean, 0.75))
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  outliers_idx <- which(x_clean < lower_bound | x_clean > upper_bound)
  
  if (length(outliers_idx) == 0) {
    return(data.frame(row_index=integer(), value=numeric(), lower_bound=numeric(), upper_bound=numeric(), outlier_type=character()))
  }
  
  vals <- x_clean[outliers_idx]
  types <- ifelse(vals < lower_bound, "Low", "High")
  
  outliers_df <- data.frame(
    row_index = clean_indices[outliers_idx],
    value = vals,
    lower_bound = lower_bound,
    upper_bound = upper_bound,
    outlier_type = types,
    stringsAsFactors = FALSE
  )
  return(outliers_df)
}

assess_missing_data <- function(data) {
  n_total <- nrow(data)
  cols <- names(data)
  
  missing_summary <- lapply(cols, function(col) {
    n_miss <- sum(is.na(data[[col]]))
    pct <- (n_miss / n_total) * 100
    pattern <- ifelse(n_miss == 0, "Complete", 
                      ifelse(pct < 5, "Minor (<5%)", 
                             ifelse(pct < 20, "Moderate (5-20%)", "Severe (>20%)")))
    
    list(
      column = col,
      n = n_total,
      n_missing = n_miss,
      pct_missing = pct,
      pattern = pattern
    )
  })
  
  return(bind_rows(missing_summary))
}

# Optional command line JSON interface can be added below for subprocess execution
if (!interactive() && exists("sys.calls") && length(sys.calls()) == 0) {
  # Suppress direct execution if sourced from Python wrapper
}