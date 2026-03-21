# ============================================================================
# VivaSense Genetics Module - Test Cases
# Three scenarios: valid single-env, valid multi-env, negative variance edge case
# ============================================================================

# Source the genetics engine
source("vivasense_genetics.R")

cat("\n")
cat("================================================================================\n")
cat("TEST CASE 1: Single-Environment Analysis (Valid Case)\n")
cat("================================================================================\n")
cat("\nScenario: Plant height measured in single location across 10 genotypes, 3 replicates\n")
cat("Expected: Clear genetic signal, moderate-to-high heritability\n\n")

# Create test data: RCBD with clear genetic effect
set.seed(42)
test1_data <- expand.grid(
  genotype = paste0("G", sprintf("%02d", 1:10)),
  rep = c("R1", "R2", "R3")
)

# Add genetic main effect + environmental error
# True genetic effect: 10 to 40 cm difference between extremes
genetic_effect <- data.frame(
  genotype = unique(test1_data$genotype),
  genetic_value = seq(from = 30, to = 55, length.out = 10)
)

test1_data <- test1_data %>%
  left_join(genetic_effect, by = "genotype") %>%
  mutate(
    error = rnorm(nrow(.), mean = 0, sd = 2),  # Environmental error SD = 2
    trait_value = genetic_value + error
  ) %>%
  select(genotype, rep, trait_value)

cat("Data structure (first 6 rows):\n")
print(head(test1_data, 6))
cat("\nData summary:\n")
cat("  Genotypes: ", nlevels(factor(test1_data$genotype)), "\n")
cat("  Replicates per genotype: ", nrow(test1_data) / nlevels(factor(test1_data$genotype)), "\n")
cat("  Trait range: ", round(min(test1_data$trait_value), 2), " to ", 
    round(max(test1_data$trait_value), 2), "\n")
cat("  Trait SD: ", round(sd(test1_data$trait_value), 2), "\n\n")

# Run analysis
result1 <- genetics_analysis(
  data = test1_data,
  mode = "single",
  trait_name = "Plant Height (cm)"
)

cat("Analysis Status: ", result1$status, "\n\n")
cat(result1$interpretation)

# Export to JSON
json1 <- export_to_json(result1)
cat("\n[JSON Output - First 500 chars]:\n")
cat(substr(json1, 1, 500), "...\n\n")

# Save full JSON
writeLines(json1, "test_case_1_output.json")
cat("Full JSON saved to: test_case_1_output.json\n\n")


cat("\n")
cat("================================================================================\n")
cat("TEST CASE 2: Multi-Environment Analysis (Valid Case with G×E)\n")
cat("================================================================================\n")
cat("\nScenario: Plant height across 15 genotypes, 2 environments (E1, E2), 3 replicates\n")
cat("Design: 15 G × 2 E × 3 reps = 90 observations\n")
cat("Expected: Genetic signal present, moderate G×E interaction\n\n")

# Create multi-environment data
set.seed(123)
test2_data <- expand.grid(
  genotype = paste0("G", sprintf("%02d", 1:15)),
  environment = c("E1", "E2"),
  rep = c("R1", "R2", "R3")
)

# Genetic effect (consistent across environments)
genetic_effect_2 <- data.frame(
  genotype = unique(test2_data$genotype),
  genetic_value = seq(from = 35, to = 65, length.out = 15)
)

# Environment main effect
env_effect <- data.frame(
  environment = c("E1", "E2"),
  env_value = c(0, 5)  # E2 is slightly higher
)

# G×E interaction (some genotypes perform better in E2)
ge_effect <- expand.grid(
  genotype = unique(test2_data$genotype),
  environment = c("E1", "E2")
)
ge_effect$ge_value <- rnorm(nrow(ge_effect), mean = 0, sd = 1.5)

# Assemble
test2_data <- test2_data %>%
  left_join(genetic_effect_2, by = "genotype") %>%
  left_join(env_effect, by = "environment") %>%
  left_join(ge_effect, by = c("genotype", "environment")) %>%
  mutate(
    error = rnorm(nrow(.), mean = 0, sd = 2.5),
    trait_value = genetic_value + env_value + ge_value + error
  ) %>%
  select(genotype, environment, rep, trait_value)

cat("Data structure (first 9 rows):\n")
print(head(test2_data, 9))
cat("\nData summary:\n")
cat("  Genotypes: ", nlevels(factor(test2_data$genotype)), "\n")
cat("  Environments: ", nlevels(factor(test2_data$environment)), "\n")
cat("  Replicates per G×E: ", 3, "\n")
cat("  Total observations: ", nrow(test2_data), "\n")
cat("  Trait range: ", round(min(test2_data$trait_value), 2), " to ", 
    round(max(test2_data$trait_value), 2), "\n")
cat("  Trait SD: ", round(sd(test2_data$trait_value), 2), "\n\n")

# Run analysis with FIXED environment model (default)
result2 <- genetics_analysis(
  data = test2_data,
  mode = "multi",
  trait_name = "Plant Height (cm)",
  random_environment = FALSE
)

cat("Analysis Status: ", result2$status, "\n")
cat("Environment Model: Fixed (standard)\n\n")
cat(result2$interpretation)

# Export to JSON
json2 <- export_to_json(result2)
writeLines(json2, "test_case_2_output.json")
cat("\nFull JSON saved to: test_case_2_output.json\n\n")


cat("\n")
cat("================================================================================\n")
cat("TEST CASE 3: Edge Case - Negative Variance Component\n")
cat("================================================================================\n")
cat("\nScenario: Very small sample (5 genotypes, 2 reps), high environmental noise\n")
cat("Expected: Weak genetic signal, negative or near-zero genetic variance\n")
cat("Expected Warnings: Negative σ²G, low h², weak genetic signal\n\n")

# Create data with very weak genetic effect and high noise
set.seed(999)
test3_data <- expand.grid(
  genotype = paste0("G", 1:5),
  rep = c("R1", "R2")
)

# VERY small genetic effect (only 2 cm difference between extremes)
genetic_effect_3 <- data.frame(
  genotype = unique(test3_data$genotype),
  genetic_value = seq(from = 48, to = 50, length.out = 5)  # Tiny range
)

test3_data <- test3_data %>%
  left_join(genetic_effect_3, by = "genotype") %>%
  mutate(
    # High environmental error (SD = 5, much larger than genetic effect)
    error = rnorm(nrow(.), mean = 0, sd = 5),
    trait_value = genetic_value + error
  ) %>%
  select(genotype, rep, trait_value)

cat("Data structure:\n")
print(test3_data)
cat("\nData summary:\n")
cat("  Genotypes: ", nlevels(factor(test3_data$genotype)), "\n")
cat("  Replicates per genotype: ", 2, "\n")
cat("  Total observations: ", nrow(test3_data), "\n")
cat("  Trait range: ", round(min(test3_data$trait_value), 2), " to ", 
    round(max(test3_data$trait_value), 2), "\n")
cat("  Trait SD: ", round(sd(test3_data$trait_value), 2), "\n")
cat("  [NOTE: Trait SD >> genetic effect range, expect negative or zero genetic variance]\n\n")

# Run analysis
result3 <- genetics_analysis(
  data = test3_data,
  mode = "single",
  trait_name = "Plant Height (cm)"
)

cat("Analysis Status: ", result3$status, "\n\n")
cat(result3$interpretation)

# Export to JSON
json3 <- export_to_json(result3)
writeLines(json3, "test_case_3_output.json")
cat("\nFull JSON saved to: test_case_3_output.json\n\n")

# Print variance components for detailed inspection
cat("Variance Components (Detailed):\n")
vc3 <- result3$result$variance_components
cat("  σ²G = ", round(vc3$sigma2_genotype, 6), " [Expected: ≤ 0 or very small]\n")
cat("  σ²E = ", round(vc3$sigma2_error, 6), " [Expected: large]\n")
cat("  σ²P = ", round(vc3$sigma2_phenotypic, 6), "\n")
cat("  h² = ", round(result3$result$heritability$h2_broad_sense, 4), " [Expected: very low]\n\n")

cat("Warnings Raised:\n")
if (length(result3$variance_warnings$warnings) > 0) {
  for (w in names(result3$variance_warnings$warnings)) {
    cat("  ✗", w, "\n")
  }
} else {
  cat("  (No warnings)\n")
}

cat("\n")
cat("================================================================================\n")
cat("TEST SUMMARY\n")
cat("================================================================================\n")
cat("\nTest 1 (Single-Env Valid):\n")
cat("  Status: ", result1$status, "\n")
cat("  h²: ", round(result1$result$heritability$h2_broad_sense, 3), "\n")
cat("  Warnings: ", length(result1$variance_warnings$warnings), "\n")

cat("\nTest 2 (Multi-Env Valid):\n")
cat("  Status: ", result2$status, "\n")
cat("  h²: ", round(result2$result$heritability$h2_broad_sense, 3), "\n")
cat("  σ²G×E: ", round(result2$result$variance_components$sigma2_ge, 3), "\n")
cat("  Warnings: ", length(result2$variance_warnings$warnings), "\n")

cat("\nTest 3 (Negative Variance Edge Case):\n")
cat("  Status: ", result3$status, "\n")
cat("  h²: ", round(result3$result$heritability$h2_broad_sense, 3), "\n")
cat("  σ²G (clamped): ", round(result3$result$variance_components$sigma2_genotype, 3), "\n")
cat("  Warnings: ", length(result3$variance_warnings$warnings), " raised\n")

cat("\n✓ All test cases completed successfully.\n")
cat("  JSON outputs: test_case_[1-3]_output.json\n\n")
