# install_packages.R — CRAN fallback installer
# Run during Render build: Rscript install_packages.R
# Installs any packages that apt-get didn't provide.

required <- c("jsonlite", "agricolae", "dplyr", "tidyr")

missing_pkgs <- required[!sapply(required, requireNamespace, quietly = TRUE)]

if (length(missing_pkgs) > 0) {
  message("Installing missing R packages: ", paste(missing_pkgs, collapse = ", "))
  install.packages(
    missing_pkgs,
    repos  = "https://cloud.r-project.org",
    quiet  = FALSE,
    Ncpus  = 2L
  )
} else {
  message("All required R packages are already installed.")
}
