# install_r_packages.R

CRAN_MIRROR <- "https://cloud.r-project.org/"
INSTALL_NCPUS <- 2L

message("=== Phase 1: Installing C++ interface packages ===")
base_cpp_packages <- c("Rcpp", "cpp11")
install.packages(
  base_cpp_packages,
  repos = CRAN_MIRROR,
  dependencies = TRUE,
  Ncpus = INSTALL_NCPUS
)

message("\n=== Phase 2: Installing Rcpp ecosystem packages ===")
rcpp_packages <- c("RcppArmadillo", "RcppEigen", "RcppProgress")
install.packages(
  rcpp_packages,
  repos = CRAN_MIRROR,
  dependencies = TRUE,
  Ncpus = INSTALL_NCPUS
)

message("\n=== Phase 3: Installing packages that depend on cpp11 ===")
cpp11_dependent <- c("tzdb", "isoband")
install.packages(
  cpp11_dependent,
  repos = CRAN_MIRROR,
  dependencies = TRUE,
  Ncpus = INSTALL_NCPUS
)

message("\n=== Phase 4: Installing application packages ===")
required_packages <- c(
  "car", "lme4", "emmeans", "multcomp", "lmerTest", "pbkrtest",
  "agricolae", "sommer", "dplyr", "tidyr", "ggplot2", "jsonlite",
  "readr", "stringr", "purrr", "broom", "rlang", "tibble"
)

install.packages(
  required_packages,
  repos = CRAN_MIRROR,
  dependencies = c("Depends", "Imports"),
  Ncpus = INSTALL_NCPUS
)

message("\n=== Verifying all packages ===")
all_packages <- c(base_cpp_packages, rcpp_packages, cpp11_dependent, required_packages)
failed <- character(0)

for (pkg in all_packages) {
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

message("\nAll ", length(all_packages),
        " R packages installed and verified successfully.")