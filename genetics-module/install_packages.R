# VivaSense Genetics - R Environment Setup

# Use a reliable CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Required packages for ANOVA and Genetic Parameters modules
required_packages <- c(
  "jsonlite", 
  "agricolae", 
  "dplyr", 
  "tidyr", 
  "readr", 
  "ggplot2", 
  "sommer", 
  "lme4",
  "pbkrtest",
  "car"
)

# Install any missing packages with dependencies
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, dependencies = TRUE)
  }
}

# Verification check (Outputs directly to Render logs via Python subprocess stdout)
for (pkg in required_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("[SUCCESS] %s is installed and ready.\n", pkg))
  } else {
    cat(sprintf("[ERROR] %s failed to install!\n", pkg))
  }
}