# ============================================================================
# Generator for the synthetic three-factor factorial RCBD regression fixtures.
#
# Committed so the fixtures are reproducible and auditable. Run from
# genetics-module/testdata/:
#
#     Rscript make_factorial3_fixture.R
#
# Structure (all three fixtures share it):
#
#   Irrigation (A, 2 levels) x Variety (B, 3) x Spacing (C, 2) = 12 cells
#   3 replicate blocks                                          = 36 rows
#   Complete and equally replicated, so the full A*B*C model is estimable.
#
#   df: rep 2 | A 1 | B 2 | C 1 | AB 2 | AC 1 | BC 2 | ABC 2 | residual 22
#
# Three variants drive the FAC-05 mean-separation decision tree, which selects
# what to present based on the highest significant order of interaction:
#
#   f3_threeway.csv  — a real three-way interaction  -> simple effects
#   f3_twoway.csv    — no three-way, a real A:B      -> AB interaction means
#   f3_none.csv      — additive effects only         -> marginal means
#
# Effects are large relative to the noise so the branch taken is unambiguous
# and the test cannot flip on a seed change.
# ============================================================================

set.seed(20260811)

A <- c("Full", "Deficit")          # Irrigation
B <- c("V1", "V2", "V3")           # Variety
C <- c("Narrow", "Wide")           # Spacing
REPS <- c("R1", "R2", "R3")

grid <- expand.grid(
  Spacing = C, Variety = B, Irrigation = A, Rep = REPS,
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)

base    <- 10
a_eff   <- c(Full = 1.5, Deficit = -1.5)
b_eff   <- c(V1 = -1.0, V2 = 0.0, V3 = 1.0)
c_eff   <- c(Narrow = -0.8, Wide = 0.8)
rep_eff <- c(R1 = -0.3, R2 = 0.0, R3 = 0.3)

additive <- base +
  a_eff[grid$Irrigation] +
  b_eff[grid$Variety] +
  c_eff[grid$Spacing] +
  rep_eff[grid$Rep]

# A:B interaction — variety response depends on irrigation.
ab <- ifelse(grid$Irrigation == "Full",
             c(V1 = -1.6, V2 = 0.0, V3 = 1.6)[grid$Variety],
             c(V1 =  1.6, V2 = 0.0, V3 = -1.6)[grid$Variety])

# A:B:C — the A:B pattern itself reverses between spacings.
abc <- ab * ifelse(grid$Spacing == "Wide", -1, 1) * 1.4

noise <- function(n) rnorm(n, 0, 0.25)

write_one <- function(values, path) {
  out <- data.frame(
    Rep        = grid$Rep,
    Irrigation = grid$Irrigation,
    Variety    = grid$Variety,
    Spacing    = grid$Spacing,
    Yield      = round(values, 3),
    stringsAsFactors = FALSE
  )
  out <- out[order(out$Rep, out$Irrigation, out$Variety, out$Spacing), ]
  write.csv(out, path, row.names = FALSE, quote = FALSE)
  cat(sprintf("Wrote %s: %d rows\n", path, nrow(out)))
}

write_one(additive + ab + abc + noise(nrow(grid)), "f3_threeway.csv")
write_one(additive + ab        + noise(nrow(grid)), "f3_twoway.csv")
write_one(additive             + noise(nrow(grid)), "f3_none.csv")
