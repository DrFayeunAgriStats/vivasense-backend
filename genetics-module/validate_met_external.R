# ============================================================================
# External validation — MET combined-ANOVA F-test denominators, real trial data
#
# Companion to test_met_environment_ftest.R. That test runs against a committed
# SYNTHETIC fixture and is the gate for CI. This script runs the same checks
# against a REAL multi-environment trial held OUTSIDE the repository.
#
# The separation is deliberate. Real trial data does not belong in version
# control; a synthetic fixture with an engineered variance structure is the
# better regression gate anyway, because it can be built so a wrong denominator
# is catastrophically wrong rather than marginally wrong. But real data exercises
# the messiness synthetic data cannot — genuine imbalance, real replicate
# behaviour — so it is still worth running by hand before a release.
#
# Usage (dataset path is required; nothing is hardcoded to a local machine):
#
#     Rscript validate_met_external.R /path/to/trial.csv [dataset-key]
#
# The CSV must have columns: Location, Year, Replication, Genotype, <trait>.
# Environment is formed as Location × Year, matching the combined-ANOVA
# structure this validation covers (RCBD, replication nested in environment).
#
# Reference values below were derived INDEPENDENTLY of the engine — mean squares
# from anova(lm(...)), each F formed by hand as MS(effect)/MS(named denominator),
# each p from pf() with that denominator's df — and are frozen full-precision.
# They are numbers ABOUT a dataset, not the dataset: recording them here leaks
# nothing and keeps the validation reproducible by anyone holding the data.
#
# To validate a different trial, add an entry to REFERENCES and pass its key.
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript validate_met_external.R <path-to-csv> [dataset-key]\n",
       "  Dataset is NOT stored in this repository — supply it from wherever ",
       "you hold it privately.")
}
csv_path    <- args[[1]]
dataset_key <- if (length(args) >= 2) args[[2]] else "sayo_met"

if (!file.exists(csv_path)) {
  stop(sprintf("Dataset not found: %s", csv_path))
}

source("vivasense_genetics.R")

# ── Per-dataset expectations ────────────────────────────────────────────────
#
#   Source            Correct denominator
#   ----------------  -------------------
#   Environment       Rep(Environment)
#   Genotype          Residual
#   Environment x G   Residual
#   Rep(Environment)  Residual
#
REFERENCES <- list(
  sayo_met = list(
    description = "Sayo multi-environment maize trial (3 locations x 3 years, 20 genotypes, 3 reps)",
    trait = "Days_to_Silking",
    terms = list(
      environment            = c(df = 8,   f = 27.4198580891912, p = 1.47636476986597e-08),
      genotype               = c(df = 19,  f = 10.8226358068169, p = 2.90204883410457e-25),
      `environment:rep`      = c(df = 18,  f = 2.14496064203315, p = 4.61104428373205e-03),
      `environment:genotype` = c(df = 152, f = 1.01074060003622, p = 4.62098647248087e-01)
    ),
    residual_df = 342,
    # What the defect produced: MS(Environment) / MS(Residual) instead of
    # MS(Environment) / MS(Rep(Environment)). Kept so the guard is explicit.
    wrong_environment_f = 58.8145164114536
  )
)

if (!(dataset_key %in% names(REFERENCES))) {
  stop(sprintf("Unknown dataset key '%s'. Known: %s",
               dataset_key, paste(names(REFERENCES), collapse = ", ")))
}
ref <- REFERENCES[[dataset_key]]

# ── Load and shape ──────────────────────────────────────────────────────────
raw <- read.csv(csv_path, fileEncoding = "UTF-8-BOM", check.names = FALSE,
                stringsAsFactors = FALSE)

required <- c("Location", "Year", "Replication", "Genotype", ref$trait)
missing  <- setdiff(required, colnames(raw))
if (length(missing) > 0) {
  stop(sprintf("Dataset is missing required column(s): %s",
               paste(missing, collapse = ", ")))
}

met <- data.frame(
  genotype    = factor(raw$Genotype),
  environment = factor(paste(raw$Location, raw$Year, sep = " × ")),
  rep         = factor(raw$Replication),
  trait_value = as.numeric(raw[[ref$trait]]),
  stringsAsFactors = FALSE
)

cat(sprintf("\n=== External MET validation: %s ===\n", dataset_key))
cat(sprintf("%s\n", ref$description))
cat(sprintf("Trait: %s | %d rows, %d genotypes, %d environments, %d reps\n\n",
            ref$trait, nrow(met), nlevels(met$genotype),
            nlevels(met$environment), nlevels(met$rep)))

res <- compute_multi_environment(met, trait_name = ref$trait)
at  <- res$anova_table

TOL_F <- 1e-9      # absolute
TOL_P <- 1e-9      # relative
failures <- 0L

for (term in names(ref$terms)) {
  want <- ref$terms[[term]]
  if (!(term %in% rownames(at))) {
    failures <- failures + 1L
    cat(sprintf("  FAIL  %-22s absent from ANOVA table (rows: %s)\n",
                term, paste(rownames(at), collapse = ", ")))
    next
  }
  got_df <- as.integer(at[term, "Df"])
  got_f  <- at[term, "F value"]
  got_p  <- at[term, "Pr(>F)"]

  ok_df <- !is.na(got_df) && got_df == as.integer(want[["df"]])
  ok_f  <- !is.na(got_f)  && abs(got_f - want[["f"]]) <= TOL_F
  ok_p  <- !is.na(got_p)  && abs(got_p - want[["p"]]) <= TOL_P * abs(want[["p"]])

  if (ok_df && ok_f && ok_p) {
    cat(sprintf("  OK    %-22s df=%3d  F=%.12f  p=%.12e\n", term, got_df, got_f, got_p))
  } else {
    failures <- failures + 1L
    cat(sprintf("  FAIL  %-22s df=%s (want %s)  F=%.12f (want %.12f)  p=%.12e (want %.12e)\n",
                term, format(got_df), format(want[["df"]]),
                got_f, want[["f"]], got_p, want[["p"]]))
  }
}

# Residual carries no test of its own.
if ("Residuals" %in% rownames(at)) {
  rdf <- as.integer(at["Residuals", "Df"])
  if (!is.na(rdf) && rdf == as.integer(ref$residual_df) &&
      is.na(at["Residuals", "F value"]) && is.na(at["Residuals", "Pr(>F)"])) {
    cat(sprintf("  OK    %-22s df=%3d  (no F/p, as expected)\n", "Residuals", rdf))
  } else {
    failures <- failures + 1L
    cat(sprintf("  FAIL  %-22s df=%s (want %d) or carries an unexpected F/p\n",
                "Residuals", format(rdf), as.integer(ref$residual_df)))
  }
}

# Guard: the original defect must not reappear.
f_env <- at["environment", "F value"]
if (!is.na(f_env) && abs(f_env - ref$wrong_environment_f) < 1e-6) {
  failures <- failures + 1L
  cat(sprintf("\n  FAIL  Environment is tested against the pooled residual (F = %.10g)\n", f_env))
} else {
  cat("\n  OK    Environment is not tested against the pooled residual\n")
}

cat("\n================================\n")
if (failures == 0L) {
  cat("EXTERNAL VALIDATION PASSED\n")
  quit(status = 0)
} else {
  cat(sprintf("%d CHECK(S) FAILED\n", failures))
  quit(status = 1)
}
