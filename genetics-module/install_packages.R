# install_packages.R — startup-time CRAN fallback installer
# Called from app_genetics.py before the R engine initialises.
# Installs any packages not already present (e.g. after a cold deploy).
#
# NOTE: asreml requires a paid commercial licence and cannot be installed
# from CRAN. Use 'sommer' as a free alternative for mixed-model analysis.

required <- c("jsonlite", "agricolae", "dplyr", "tidyr", "ggplot2", "sommer")

missing_pkgs <- required[!sapply(required, requireNamespace, quietly = TRUE)]

if (length(missing_pkgs) > 0) {
  message("Installing missing R packages: ", paste(missing_pkgs, collapse = ", "))
  install.packages(
    missing_pkgs,
    repos        = "https://cloud.r-project.org",
    dependencies = TRUE,
    Ncpus        = 2L
  )
  message("Done.")
} else {
  message("All required R packages already installed.")
}
