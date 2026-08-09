# ============================================================================
# Generator for the synthetic MET-RCBD regression fixture.
#
# Committed so the fixture is reproducible and auditable rather than an opaque
# blob of numbers. Run from genetics-module/testdata/:
#
#     Rscript make_met_fixture.R
#
# Structure mirrors the real multi-environment trials VivaSense analyses:
#
#   Environment = Location × Year  -> 3 locations × 3 years = 9 environments
#   20 genotypes, 3 replications per environment              = 540 rows
#   Replication is nested within environment (labels 1..3 reused per env)
#
# The variance structure is chosen so the regression test is meaningful:
#
#   * a strong environment main effect          -> Environment clearly significant
#   * genuine rep-within-environment variation  -> MS(Rep(Env)) well above the
#     residual, so testing Environment against the pooled residual gives a very
#     different answer from testing it against Rep(Environment). Without this,
#     the two denominators nearly coincide and the test proves nothing.
#   * a clear genotype effect, weak G×E, modest plot noise
#
# NOTE: this fixture exists to pin the F-test denominators, not to be realistic
# agronomy. The real Sayo trial data remains the external validation dataset and
# is deliberately NOT committed here.
# ============================================================================

set.seed(20260809)

locations <- c("Loc1", "Loc2", "Loc3")
years     <- c(2023, 2024, 2025)
genotypes <- sprintf("G%02d", 1:20)
reps      <- 1:3

grid <- expand.grid(
  Genotype    = genotypes,
  Replication = reps,
  Year        = years,
  Location    = locations,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

env_key <- paste(grid$Location, grid$Year, sep = "|")
env_levels <- unique(env_key)

# Environment main effects — large, spread across the 9 environments.
env_effect <- setNames(seq(-6, 6, length.out = length(env_levels)), env_levels)

# Rep-within-environment effects: a separate draw for every environment × rep
# combination. This is the whole-environment stratum the fix is about.
rep_key <- paste(env_key, grid$Replication, sep = "|")
rep_levels <- unique(rep_key)
rep_effect <- setNames(rnorm(length(rep_levels), 0, 1.2), rep_levels)

# Genotype main effects.
geno_effect <- setNames(seq(-3, 3, length.out = length(genotypes)), genotypes)

# Weak G×E: present but small relative to the genotype main effect.
ge_key <- paste(env_key, grid$Genotype, sep = "|")
ge_levels <- unique(ge_key)
ge_effect <- setNames(rnorm(length(ge_levels), 0, 0.35), ge_levels)

grid$Days_to_Silking <- round(
  58 +
    env_effect[env_key] +
    rep_effect[rep_key] +
    geno_effect[grid$Genotype] +
    ge_effect[ge_key] +
    rnorm(nrow(grid), 0, 1.5),
  2
)

out <- grid[order(grid$Location, grid$Year, grid$Replication, grid$Genotype),
            c("Location", "Year", "Replication", "Genotype", "Days_to_Silking")]

write.csv(out, "met_rcbd_synthetic.csv", row.names = FALSE, quote = FALSE)
cat(sprintf("Wrote met_rcbd_synthetic.csv: %d rows, %d environments, %d genotypes, %d reps\n",
            nrow(out), length(env_levels), length(genotypes), length(reps)))
