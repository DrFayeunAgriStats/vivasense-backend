export const WEEK_TITLES: Record<number, string> = {
  0: "Software Setup & Tidy Data",
  1: "Experimental Design Foundations",
  2: "ANOVA Conceptual Foundations",
  3: "ANOVA in R & Interpretation",
  4: "Correlation & Regression",
  5: "PCA & Multivariate Analysis",
  6: "Thesis Writing & Integration",
};

export const WEEK_TOPICS: Record<number, string[]> = {
  0: [
    "Software Setup & Tidy Data",
    "R/RStudio Installation",
    "VivaSense Orientation",
    "Reproducibility Philosophy",
    "Swirl Introduction",
    "Analytical Mindset",
  ],
  1: [
    "Experimental Design Foundations",
    "CRD, RCBD, Factorial",
    "Fixed vs Random Factors",
    "Experimental vs Sampling Units",
    "Descriptive Statistics & Plots",
  ],
  2: [
    "What ANOVA Actually Tests",
    "Partitioning Variance",
    "F-ratio Logic",
    "Reading ANOVA Tables",
    "Connecting Design to Model",
  ],
  3: [
    "Running ANOVA in R",
    "ANOVA Assumptions & Diagnostics",
    "Mean Separation",
    "Writing ANOVA Interpretations",
    "Biological Meaning of Output",
  ],
  4: [
    "Correlation vs Causation",
    "Simple Linear Regression",
    "Interpreting Coefficients",
    "Visual Diagnostics",
    "Agricultural Applications",
  ],
  5: [
    "Why Multivariate Analysis",
    "PCA Conceptual Logic",
    "Eigenvalues & Variance Explained",
    "Loading Plots & Biplots",
    "Trait Clustering",
    "Crop Improvement Application",
  ],
  6: [
    "Structuring Chapter 4 (Results)",
    "Writing Statistical Interpretations",
    "Avoiding Thesis Errors",
    "VivaSense + AI Workflow",
    "Final Integration",
    "Competency Assessment",
  ],
};

export const QUICK_ACTIONS: Record<number, string[]> = {
  0: [
    "Begin Week 0",
    "What is Tidy Data?",
    "Show folder structure",
    "What is reproducibility?",
    "Introduce me to swirl",
  ],
  1: [
    "Begin Week 1",
    "Explain CRD vs RCBD",
    "Fixed vs random factors",
    "What is an experimental unit?",
    "Show descriptive plots in R",
  ],
  2: [
    "Begin Week 2",
    "What does ANOVA actually test?",
    "Explain F-ratio logic",
    "How to read an ANOVA table",
    "Partition the variance",
  ],
  3: [
    "Begin Week 3",
    "Run ANOVA in R",
    "Check ANOVA assumptions",
    "Explain mean separation",
    "Write an ANOVA interpretation",
  ],
  4: [
    "Begin Week 4",
    "Correlation vs causation",
    "Run simple regression in R",
    "Interpret regression coefficients",
    "Show visual diagnostics",
  ],
  5: [
    "Begin Week 5",
    "Explain PCA step by step",
    "What are eigenvalues?",
    "Interpret a biplot",
    "Show PCA in R",
  ],
  6: [
    "Begin Week 6",
    "Structure Chapter 4 Results",
    "Common thesis errors in stats",
    "Show VivaSense workflow",
    "Submit competency assessment",
  ],
};

export const R_CODE: Record<number, string> = {
  0: `# Week 0: R Installation Check & Setup
# Check R version
R.version.string

# Install swirl for interactive learning
install.packages("swirl")
library(swirl)
swirl()

# FIA Folder Structure Setup
dir.create("FIA-ADAP", showWarnings = FALSE)
setwd("FIA-ADAP")
dir.create("data-raw", showWarnings = FALSE)
dir.create("data-clean", showWarnings = FALSE)
dir.create("scripts", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

# Verify installation
cat("R is ready! Version:", R.version.string, "\\n")
cat("Working directory:", getwd(), "\\n")`,

  1: `# Week 1: Descriptive Statistics & Plots
# Load and inspect your data
dat <- read.csv("data-raw/experiment.csv")
head(dat)
str(dat)
summary(dat)

# Ensure treatment is a factor
dat$Treatment <- as.factor(dat$Treatment)

# Descriptive statistics by group
tapply(dat$Yield, dat$Treatment, mean)
tapply(dat$Yield, dat$Treatment, sd)

# Boxplot — always visualise before analysis
boxplot(Yield ~ Treatment, data = dat,
        col = "lightgreen",
        main = "Yield by Treatment",
        ylab = "Yield (kg/ha)",
        xlab = "Treatment")

# Histogram of response variable
hist(dat$Yield, breaks = 15, col = "steelblue",
     main = "Distribution of Yield",
     xlab = "Yield (kg/ha)")`,

  2: `# Week 2: ANOVA Table & Interpretation
# Fit the ANOVA model
model <- aov(Yield ~ Treatment, data = dat)

# View the ANOVA table
summary(model)        # Quick summary
anova(model)          # Detailed ANOVA table
summary.aov(model)    # Alternative view

# Understanding the output:
# Df     = Degrees of freedom
# Sum Sq = Sum of Squares (variance explained)
# Mean Sq = Mean Square (Sum Sq / Df)
# F value = F-ratio (Treatment MS / Residual MS)
# Pr(>F) = p-value

# Total variance = Treatment SS + Residual SS
# If p < 0.05, treatments differ significantly`,

  3: `# Week 3: Full ANOVA Pipeline
# Step 1: Fit the model
model <- aov(Yield ~ Treatment, data = dat)
summary(model)

# Step 2: Check assumptions
# Normality of residuals
shapiro.test(residuals(model))
qqnorm(residuals(model))
qqline(residuals(model), col = "red")

# Homogeneity of variance
library(car)
leveneTest(Yield ~ Treatment, data = dat)

# Step 3: Diagnostic plots
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))

# Step 4: Mean separation (if ANOVA is significant)
tukey <- TukeyHSD(model)
print(tukey)
plot(tukey)

# Step 5: Compact letter display
library(multcomp)
cld(glht(model, linfct = mcp(Treatment = "Tukey")))`,

  4: `# Week 4: Correlation & Regression
# Correlation analysis
cor(dat$PlantHeight, dat$Yield)
cor.test(dat$PlantHeight, dat$Yield)

# Correlation matrix for multiple traits
num_vars <- dat[, sapply(dat, is.numeric)]
cor_matrix <- cor(num_vars, use = "complete.obs")
print(round(cor_matrix, 3))

# Simple linear regression
reg_model <- lm(Yield ~ PlantHeight, data = dat)
summary(reg_model)

# Interpretation:
# Coefficients: intercept and slope
# R-squared: proportion of variance explained
# p-value: significance of relationship

# Visualise regression
plot(dat$PlantHeight, dat$Yield,
     xlab = "Plant Height (cm)", ylab = "Yield (kg/ha)",
     main = "Yield vs Plant Height", pch = 19, col = "darkgreen")
abline(reg_model, col = "red", lwd = 2)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(reg_model)
par(mfrow = c(1, 1))`,

  5: `# Week 5: PCA & Multivariate Analysis
# Prepare numeric data (exclude grouping columns)
num_data <- dat[, sapply(dat, is.numeric)]

# Run PCA with scaling
pca_result <- prcomp(num_data, scale. = TRUE, center = TRUE)

# Summary — eigenvalues and variance explained
summary(pca_result)

# Scree plot — how many PCs to retain?
screeplot(pca_result, type = "lines", main = "Scree Plot")

# Loadings — which traits contribute most?
print(pca_result$rotation[, 1:3])

# Basic biplot
biplot(pca_result, main = "PCA Biplot")

# Enhanced biplot with factoextra
library(factoextra)
fviz_pca_biplot(pca_result,
                label = "var",
                col.var = "darkgreen",
                col.ind = "steelblue",
                title = "PCA Biplot — Trait Relationships")

# Variance explained per PC
fviz_eig(pca_result, addlabels = TRUE)`,

  6: `# Week 6: Full Integrated Pipeline
# ============================================
# FIA-ADAP Complete Analysis Template
# ============================================

# 1. Load and inspect
dat <- read.csv("data-raw/experiment.csv")
str(dat)
summary(dat)

# 2. Descriptive statistics (Week 1)
tapply(dat$Yield, dat$Treatment, mean)
boxplot(Yield ~ Treatment, data = dat, col = "lightgreen")

# 3. ANOVA (Weeks 2-3)
model <- aov(Yield ~ Treatment, data = dat)
summary(model)
shapiro.test(residuals(model))
TukeyHSD(model)

# 4. Correlation & Regression (Week 4)
cor.test(dat$PlantHeight, dat$Yield)
reg <- lm(Yield ~ PlantHeight, data = dat)
summary(reg)

# 5. PCA (Week 5)
pca <- prcomp(dat[, sapply(dat, is.numeric)], scale. = TRUE)
summary(pca)
biplot(pca)

# 6. Interpretation Template:
# "Treatment had a significant effect on yield
#  (F = X.XX, p < 0.05). Tukey HSD revealed that
#  Treatment A (mean = XX) was significantly higher
#  than Treatment B (mean = XX). Plant height showed
#  a strong positive correlation with yield (r = 0.XX,
#  p < 0.01). PCA explained XX% of total variance
#  in the first two components."`,
};
