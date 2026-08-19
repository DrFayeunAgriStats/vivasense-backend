suppressPackageStartupMessages(library(agricolae))

compute_crop_protection_factorial_crd <- function(data, factor_names) {
  required <- c(factor_names, "replicate", "inference_value")
  missing <- setdiff(required, colnames(data))
  if (length(missing)) stop(sprintf("Missing adapter columns: %s", paste(missing, collapse=", ")))
  for (name in factor_names) data[[name]] <- factor(data[[name]], levels=unique(data[[name]]))
  data$replicate <- factor(data$replicate)
  formula <- as.formula(sprintf("inference_value ~ %s", paste(factor_names, collapse=" * ")))
  model <- aov(formula, data=data)
  at <- anova(model)
  residual_row <- which(rownames(at) == "Residuals")
  mse <- as.numeric(at[residual_row, "Mean Sq"]); error_df <- as.integer(at[residual_row, "Df"])
  cell_key <- interaction(data[factor_names], sep="\037", drop=TRUE)
  cell_n <- table(cell_key); balanced <- length(unique(as.integer(cell_n))) == 1L
  hsd <- HSD.test(model, factor_names, DFerror=error_df, MSerror=mse,
                  group=TRUE, console=FALSE, unbalanced=!balanced)
  lookup <- setNames(as.character(hsd$groups$groups),
                     vapply(strsplit(rownames(hsd$groups), ":", fixed=TRUE),
                            function(x) paste(x, collapse="\037"), character(1)))
  split_rows <- split(seq_len(nrow(data)), cell_key)
  cell_means <- lapply(split_rows, function(idx) {
    inference <- data$inference_value[idx]
    display <- if ("display_value" %in% colnames(data)) data$display_value[idx] else inference
    levels <- lapply(factor_names, function(name) as.character(data[[name]][idx[1]])); names(levels) <- factor_names
    key <- paste(unlist(levels), collapse="\037"); n <- length(idx)
    c(levels, list(n=n, mean_inference_scale=mean(inference), mean_display_scale=mean(display),
      se_inference_scale=sqrt(mse/n),
      se_display_scale=if(n>1) sd(display)/sqrt(n) else NA_real_, letter=unname(lookup[[key]])))
  })
  marginal_for <- function(names) {
    key <- interaction(data[names], sep="\037", drop=TRUE)
    lapply(split(seq_len(nrow(data)), key), function(idx) {
      levels <- lapply(names, function(name) as.character(data[[name]][idx[1]])); names(levels) <- names
      display <- if("display_value" %in% colnames(data)) data$display_value[idx] else data$inference_value[idx]
      c(levels, list(n=length(idx), mean_inference_scale=mean(data$inference_value[idx]),
                     mean_display_scale=mean(display), se_display_scale=sd(display)/sqrt(length(idx))))
    })
  }
  marginal <- list()
  for (size in seq_len(min(2L, length(factor_names)))) {
    for (combo in combn(factor_names, size, simplify=FALSE)) marginal[[paste(combo,collapse=":")]] <- marginal_for(combo)
  }
  source <- rownames(at); source[source=="Residuals"] <- "Error"
  anova_rows <- lapply(seq_len(nrow(at)), function(i) list(source=source[i], df=as.integer(at[i,"Df"]),
    ss=as.numeric(at[i,"Sum Sq"]), ms=as.numeric(at[i,"Mean Sq"]),
    f_value=if(is.na(at[i,"F value"])) NA_real_ else as.numeric(at[i,"F value"]),
    p_value=if(is.na(at[i,"Pr(>F)"])) NA_real_ else as.numeric(at[i,"Pr(>F)"])))
  interaction_rows <- which(grepl(":", rownames(at)))
  interactions <- lapply(interaction_rows, function(i) list(source=rownames(at)[i], p_value=as.numeric(at[i,"Pr(>F)"]),
                                                             order=length(strsplit(rownames(at)[i],":",fixed=TRUE)[[1]])))
  list(model_formula=paste("response ~",paste(factor_names,collapse=" * ")), anova=anova_rows,
       residual_mean_square=mse,error_df=error_df,balanced=balanced,
       common_cell_n=if(balanced)as.integer(cell_n[[1]])else NA_integer_,
       common_interaction_se=if(balanced)sqrt(mse/as.integer(cell_n[[1]]))else NA_real_,
       cell_means=unname(cell_means),interaction=list(means=unname(cell_means),tests=interactions),
       marginal_means=marginal)
}
