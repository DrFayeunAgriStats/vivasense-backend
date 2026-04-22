# vivasense_cluster.R
#
# Hierarchical Cluster Analysis — R reference implementation
#
# References:
#   Ward, J.H. (1963). Hierarchical grouping to optimise an objective function.
#   Journal of the American Statistical Association, 58(301), 236-244.
#
#   Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation
#   and validation of cluster analysis. Journal of Computational and Applied
#   Mathematics, 20, 53-65.
#
# This script is provided as an R-based reference implementation.
# The primary computation is performed by analysis_cluster_routes.py using
# scipy.cluster.hierarchy + sklearn.metrics for reliability and portability.
#
# Requires: cluster package (for silhouette)
#
# Function: compute_cluster_analysis(observations, trait_cols, method, k, scale)
#   observations : data.frame with columns genotype, trait1, trait2, ...
#   trait_cols   : character vector
#   method       : linkage method ("ward.D2", "complete", "average", "single")
#   k            : integer or NULL (auto-detect via silhouette)
#   scale        : logical

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  if (requireNamespace("cluster", quietly = TRUE)) library(cluster)
})

compute_cluster_analysis <- function(observations, trait_cols,
                                      method = "ward.D2", k = NULL,
                                      scale = TRUE) {

  df <- as.data.frame(observations)
  df$genotype <- as.character(df$genotype)
  for (tc in trait_cols) df[[tc]] <- suppressWarnings(as.numeric(df[[tc]]))

  # Aggregate to genotype means
  geno_means <- df %>%
    group_by(genotype) %>%
    summarise(across(all_of(trait_cols), \(x) mean(x, na.rm = TRUE)),
              .groups = "drop")

  mat <- as.matrix(geno_means[, trait_cols])
  complete_rows <- complete.cases(mat)
  mat <- mat[complete_rows, ]
  geno_labels <- geno_means$genotype[complete_rows]

  n_genos <- nrow(mat)
  n_traits <- ncol(mat)
  if (n_genos < 4) stop("Cluster analysis requires at least 4 genotypes.")
  if (n_traits < 2) stop("Cluster analysis requires at least 2 traits.")

  # Standardise
  if (scale) mat_scaled <- scale(mat) else mat_scaled <- mat

  # Compute distance matrix and hierarchical clustering
  dist_mat <- dist(mat_scaled, method = "euclidean")
  hc       <- hclust(dist_mat, method = method)

  # Determine optimal k via silhouette (if not provided)
  max_k <- min(10L, n_genos - 1L)
  min_k <- 2L

  if (!is.null(k)) {
    optimal_k <- max(min_k, min(as.integer(k), max_k))
  } else {
    best_k   <- min_k
    best_sil <- -Inf
    for (ck in min_k:max_k) {
      labels_ck <- cutree(hc, k = ck)
      if (length(unique(labels_ck)) < 2) next
      if (requireNamespace("cluster", quietly = TRUE)) {
        sil <- cluster::silhouette(labels_ck, dist_mat)
        avg_sil <- mean(sil[, "sil_width"])
      } else {
        avg_sil <- 0
      }
      if (avg_sil > best_sil) {
        best_sil <- avg_sil
        best_k   <- ck
      }
    }
    optimal_k <- best_k
  }

  labels <- cutree(hc, k = optimal_k)

  # Per-genotype silhouette scores
  if (requireNamespace("cluster", quietly = TRUE) && length(unique(labels)) > 1) {
    sil_obj <- cluster::silhouette(labels, dist_mat)
    sil_scores <- sil_obj[, "sil_width"]
  } else {
    sil_scores <- rep(0, n_genos)
  }

  # Assignments
  assignments <- data.frame(
    genotype         = geno_labels,
    cluster_id       = as.integer(labels),
    silhouette_score = round(sil_scores, 4),
    stringsAsFactors = FALSE
  )
  assignments <- assignments[order(assignments$cluster_id, assignments$genotype), ]

  # Cluster summary (un-scaled means for interpretability)
  cluster_ids <- sort(unique(as.integer(labels)))
  cluster_summary <- lapply(cluster_ids, function(cid) {
    idx <- which(labels == cid)
    means_vec <- colMeans(mat[idx, , drop = FALSE])
    list(
      cluster_id    = cid,
      size          = length(idx),
      mean_per_trait = as.list(round(means_vec, 4))
    )
  })

  # Linkage matrix for dendrogram
  Z <- cbind(hc$merge, hc$height,
             table(labels)[order(as.integer(names(table(labels))))])

  list(
    status             = "success",
    n_genotypes        = n_genos,
    n_traits           = n_traits,
    method             = method,
    optimal_k          = optimal_k,
    cluster_assignments = assignments,
    cluster_summary    = cluster_summary,
    silhouette_scores  = as.list(round(sil_scores, 4)),
    dendrogram_data    = list(
      linkage_matrix   = hc$merge,
      heights          = hc$height,
      labels           = geno_labels,
      method           = method
    ),
    interpretation = paste0(
      "Hierarchical cluster analysis (", method, " linkage) identified ",
      optimal_k, " clusters among ", n_genos, " genotypes. "
    )
  )
}

# ── Standalone execution ──────────────────────────────────────────────────────
if (!interactive()) {
  input_json <- readLines("stdin") |> paste(collapse = "\n")
  input_data <- jsonlite::fromJSON(input_json)
  result <- compute_cluster_analysis(
    observations = input_data$observations,
    trait_cols   = input_data$trait_cols,
    method       = input_data$method %||% "ward.D2",
    k            = input_data$k,
    scale        = isTRUE(input_data$scale)
  )
  cat(jsonlite::toJSON(result, auto_unbox = TRUE, na = "null", digits = 6))
}
