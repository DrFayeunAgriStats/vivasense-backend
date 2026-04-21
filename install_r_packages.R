# install_r_packages.R

CRAN_MIRROR <- "https://cloud.r-project.org/"

# Step 1: Install cpp11 and Rcpp dependencies first
cpp_packages <- c("cpp11", "Rcpp", "RcppArmadillo", "RcppProgress", "RcppEigen")
message("Installing C++ interface packages first: ", paste(cpp_packages, collapse = ", "))
install.packages(
  cpp_packages,
  repos = CRAN_MIRROR,
  dependencies = TRUE,
  Ncpus = 1L
)

required_packages <- c(
  "car", "lme4", "emmeans", "multcomp", "lmerTest", "pbkrtest",
  "agricolae", "sommer", "dplyr", "tidyr", "ggplot2", "jsonlite",
  "readr", "stringr", "purrr", "broom", "rlang", "tibble"
)

missing_pkgs <- required_packages[
  !sapply(required_packages, requireNamespace, quietly = TRUE)
]

if (length(missing_pkgs) > 0) {
  message("Installing missing packages (", length(missing_pkgs), "): ",
          paste(missing_pkgs, collapse = ", "))

  install.packages(
    missing_pkgs,
    repos = CRAN_MIRROR,
    dependencies = c("Depends", "Imports"),
    Ncpus = 1L
  )
} else {
  message("All required R packages already present - skipping installation.")
}

failed <- character(0)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    failed <- c(failed, pkg)
    message("FAIL: package '", pkg, "' could not be loaded.")
  } else {
    message("OK:   ", pkg)
  }
}

if (length(failed) > 0) {
  stop(
    "FATAL: The following R packages are unavailable after installation: ",
    paste(failed, collapse = ", ")
  )
}

message("\nAll ", length(required_packages),
        " R packages installed and verified successfully.")