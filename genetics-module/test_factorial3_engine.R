# ============================================================================
# Regression test — three-factor factorial (FAC-01..FAC-07)
#
# Structure: Irrigation (A,2) x Variety (B,3) x Spacing (C,2) in 3 blocks = 36
# rows, complete and equally replicated.
#
#   trait_value ~ rep + genotype * factor * factor_c
#
# `a * b * c` expands to main effects plus EVERY interaction among them, so the
# three-way term is fitted by default — no automatic reduction (FAC-01).
#
# rep stays an additive blocking main effect, unchanged by the arity of the
# treatment structure: no nesting, no error stratum of its own. All treatment
# terms — including the added AC, BC and ABC — are tested against the pooled
# residual, exactly as AB already was (FAC-03).
#
# Expected values are derived INDEPENDENTLY of the engine: mean squares, F and
# p read from a direct anova(aov(...)) on the same fixture, frozen here as
# full-precision literals. Not regenerated from the code under test.
#
# Fixtures: testdata/f3_{threeway,twoway,none}.csv, built by
# testdata/make_factorial3_fixture.R (committed, seeded, reproducible).
#
# Run from inside genetics-module/:
#     Rscript test_factorial3_engine.R
# ============================================================================

source("vivasense_genetics.R")

TOL_ABS   <- 1e-9
TOL_REL_P <- 1e-9
failures  <- 0L

expect <- function(label, ok, detail = "") {
  if (isTRUE(ok)) {
    cat(sprintf("  PASS  %s\n", label))
  } else {
    failures <<- failures + 1L
    cat(sprintf("  FAIL  %s %s\n", label, detail))
  }
}

expect_close <- function(label, actual, expected, tol = TOL_ABS, relative = FALSE) {
  ok <- !is.null(actual) && length(actual) == 1L && !is.na(actual) &&
    if (relative) abs(actual - expected) <= tol * abs(expected)
    else abs(actual - expected) <= tol
  expect(label, ok,
         if (isTRUE(ok)) "" else sprintf("got %s, expected %.15g",
                                         format(actual), expected))
}

load_fixture <- function(name) {
  raw <- read.csv(file.path("testdata", paste0(name, ".csv")), stringsAsFactors = FALSE)
  data.frame(
    rep         = factor(raw$Rep),
    genotype    = factor(raw$Irrigation),
    factor      = factor(raw$Variety),
    factor_c    = factor(raw$Spacing),
    trait_value = as.numeric(raw$Yield),
    stringsAsFactors = FALSE
  )
}

# ── 1. Full A*B*C model against independently derived references ────────────
cat("\n=== Full three-factor model (f3_threeway) ===\n")
res3 <- compute_single_environment(load_fixture("f3_threeway"), trait_name = "Yield")
at3  <- res3$anova_table

reference <- list(
  rep                        = list(df = 2L,  ms = 1.06656144444445,    f = 24.9994218313359,   p = 2.16804729807599e-06),
  genotype                   = list(df = 1L,  ms = 84.0461121111111,    f = 1969.97952709968,   p = 5.04104844800083e-23),
  `factor`                   = list(df = 2L,  ms = 14.7908337777778,    f = 346.686348708622,   p = 2.32698464495904e-17),
  factor_c                   = list(df = 1L,  ms = 23.9773444444444,    f = 562.011453990007,   p = 3.710582366167e-17),
  `genotype:factor`          = list(df = 2L,  ms = 30.2008084444444,    f = 707.884907974811,   p = 1.07663007574054e-20),
  `genotype:factor_c`        = list(df = 1L,  ms = 0.109781777777779,   f = 2.5732047472335,    p = 0.122948233859415),
  `factor:factor_c`          = list(df = 2L,  ms = 0.0351697777777789,  f = 0.824353922561874,  p = 0.451614536342747),
  `genotype:factor:factor_c` = list(df = 2L,  ms = 59.7718951111111,    f = 1401.00959707893,   p = 6.41365216703904e-24)
)

for (term in names(reference)) {
  ref <- reference[[term]]
  if (!(term %in% rownames(at3))) {
    expect(sprintf("%s present", term), FALSE,
           sprintf("(rows: %s)", paste(rownames(at3), collapse = ", ")))
    next
  }
  expect(sprintf("%-26s df", term), as.integer(at3[term, "Df"]) == ref$df)
  expect_close(sprintf("%-26s MS", term), at3[term, "Mean Sq"], ref$ms)
  expect_close(sprintf("%-26s F",  term), at3[term, "F value"], ref$f)
  expect_close(sprintf("%-26s p",  term), at3[term, "Pr(>F)"], ref$p,
               tol = TOL_REL_P, relative = TRUE)
}
expect("Residuals df = 22", as.integer(at3["Residuals", "Df"]) == 22L)
expect_close("Residuals MS", at3["Residuals", "Mean Sq"], 0.042663444444445)

# FAC-03: every added term is tested against the pooled residual, as AB was.
cat("\n--- FAC-03: added terms use the pooled residual ---\n")
ms_res <- at3["Residuals", "Mean Sq"]
for (term in c("genotype:factor", "genotype:factor_c",
               "factor:factor_c", "genotype:factor:factor_c")) {
  expect_close(sprintf("%-26s F = MS/MS_resid", term),
               at3[term, "F value"],
               at3[term, "Mean Sq"] / ms_res, tol = 1e-9)
}

expect("n_treatment_factors = 3", identical(as.integer(res3$n_treatment_factors), 3L))

# ── 2. FAC-05 decision tree ─────────────────────────────────────────────────
cat("\n=== FAC-05 mean-separation decision tree ===\n")

expect("three-way significant -> 'three_way'",
       identical(res3$mean_separation_basis$selected, "three_way"))
expect("three-way -> cells conditioned on all three factors",
       !is.null(res3$interaction_separation$genotype) &&
       !is.null(res3$interaction_separation$factor) &&
       !is.null(res3$interaction_separation$factor_c))
expect("three-way -> 12 cells", length(res3$interaction_separation$mean) == 12L)

res2 <- compute_single_environment(load_fixture("f3_twoway"), trait_name = "Yield")
expect("three-way NS + AB significant -> 'two_way'",
       identical(res2$mean_separation_basis$selected, "two_way"))
expect("two-way -> AB reported as significant term",
       grepl("genotype:factor", res2$mean_separation_basis$significant_terms, fixed = TRUE))
expect("two-way -> AB interaction means present",
       !is.null(res2$two_way_interaction_means[["genotype:factor"]]))
expect("two-way -> primary result has no factor_c column",
       is.null(res2$interaction_separation$factor_c))

res0 <- compute_single_environment(load_fixture("f3_none"), trait_name = "Yield")
expect("no interaction significant -> 'marginal'",
       identical(res0$mean_separation_basis$selected, "marginal"))
expect("marginal -> no interaction means presented",
       is.null(res0$interaction_separation))
expect("marginal -> marginal means still available for A",
       !is.null(res0$mean_separation))
expect("marginal -> marginal means available for B and C",
       !is.null(res0$mean_separation_b) && !is.null(res0$mean_separation_c))

# ── 3. Interaction-label parsing on three-piece rownames ────────────────────
# The previous logic used sub(":.*","") / sub("^[^:]*:","") — on "G1:F1:C1"
# that returned "G1" and "F1:C1", silently attributing the third factor's level
# to the second. Every parsed level must be a real level of its own factor.
cat("\n=== Three-piece label parsing ===\n")
cells <- res3$interaction_separation
lv_a <- c("Full", "Deficit"); lv_b <- c("V1", "V2", "V3"); lv_c <- c("Narrow", "Wide")
expect("factor A levels parsed cleanly",  all(cells$genotype %in% lv_a))
expect("factor B levels parsed cleanly",  all(cells$factor   %in% lv_b))
expect("factor C levels parsed cleanly",  all(cells$factor_c %in% lv_c))
expect("no residual ':' left in any parsed level",
       !any(grepl(":", c(cells$genotype, cells$factor, cells$factor_c), fixed = TRUE)))
expect("all 12 combinations distinct",
       length(unique(paste(cells$genotype, cells$factor, cells$factor_c))) == 12L)

# ── 4. Two-factor factorial must be unchanged ───────────────────────────────
cat("\n=== Two-factor factorial unchanged ===\n")
two <- load_fixture("f3_threeway")
two$factor_c <- NULL
res_two <- compute_single_environment(two, trait_name = "Yield")
at2 <- res_two$anova_table

expect("2-factor rows are rep/genotype/factor/interaction/Residuals",
       identical(rownames(at2),
                 c("rep", "genotype", "factor", "genotype:factor", "Residuals")))
expect("no factor_c term appears", !any(grepl("factor_c", rownames(at2), fixed = TRUE)))
expect("n_treatment_factors = 2", identical(as.integer(res_two$n_treatment_factors), 2L))
expect("2-factor keeps unconditional marginal means (A)", !is.null(res_two$mean_separation))
expect("2-factor keeps unconditional marginal means (B)", !is.null(res_two$mean_separation_b))
expect("2-factor keeps interaction means", !is.null(res_two$interaction_separation))
expect("2-factor interaction keys are genotype/factor only",
       !is.null(res_two$interaction_separation$genotype) &&
       !is.null(res_two$interaction_separation$factor) &&
       is.null(res_two$interaction_separation$factor_c))
expect("2-factor is NOT put through the decision tree",
       is.null(res_two$mean_separation_basis))
expect("2-factor design label unchanged", identical(res_two$design, "factorial_rcbd"))

# Two-factor CRD likewise.
two_crd <- two
two_crd$rep <- NULL
two_crd$rep <- factor(rep("R1", nrow(two_crd)))
res_crd <- compute_single_environment(two_crd, trait_name = "Yield", crd_mode = TRUE)
expect("2-factor CRD still fits", !is.null(res_crd$anova_table))
expect("2-factor CRD design label unchanged", identical(res_crd$design, "factorial_crd"))

# ── 5. Guards ───────────────────────────────────────────────────────────────
cat("\n=== Guards ===\n")
orphan <- load_fixture("f3_threeway")
orphan$factor <- NULL
expect("third factor without the second is rejected",
       inherits(try(compute_single_environment(orphan, trait_name = "Yield"),
                    silent = TRUE), "try-error"))

single_level <- load_fixture("f3_threeway")
single_level$factor_c <- factor(rep("OnlyOne", nrow(single_level)))
expect("single-level third factor is rejected",
       inherits(try(compute_single_environment(single_level, trait_name = "Yield"),
                    silent = TRUE), "try-error"))

cat("\n================================\n")
if (failures == 0L) {
  cat("ALL CHECKS PASSED\n"); quit(status = 0)
} else {
  cat(sprintf("%d CHECK(S) FAILED\n", failures)); quit(status = 1)
}
