export interface Question {
  id: number;
  question: string;
  options: string[];
  correct: number;
  explanation: string;
}

export interface Quiz {
  title: string;
  description: string;
  passingScore: number;
  questions: Question[];
}

export type WeekKey = "week0" | "week1" | "week2" | "week3" | "week4" | "week5" | "week6";

export const WEEK_ORDER: WeekKey[] = ["week0", "week1", "week2", "week3", "week4", "week5", "week6"];

export const quizzes: Record<WeekKey, Quiz> = {
  week0: {
    title: "Week 0 Quiz: Research Data Fundamentals",
    description: "Foundation concepts for agricultural data analysis",
    passingScore: 7,
    questions: [
      { id: 1, question: "In agricultural research, what is a variable?", options: ["A characteristic or measurement that can change across observations", "A factor that the researcher deliberately changes", "The average value in a dataset", "A number used only in calculations"], correct: 0, explanation: "A variable is any characteristic that varies across observations (yield, rainfall, pest count, etc.)." },
      { id: 2, question: "What is the difference between quantitative and qualitative data?", options: ["Quantitative is numerical; qualitative is descriptive", "Quantitative is always more accurate", "Qualitative data cannot be analyzed statistically", "There is no practical difference"], correct: 0, explanation: "Quantitative data = numbers (yield in kg). Qualitative data = descriptions (crop variety name)." },
      { id: 3, question: "What is a population in statistical terms?", options: ["All individuals or units of interest in a study", "The number of people in a country", "The sample used in an experiment", "A group of organisms of the same species"], correct: 0, explanation: "Population = all possible units. Sample = subset studied to infer about the population." },
      { id: 4, question: "Which best describes a representative sample?", options: ["A sample that mirrors the population's key characteristics", "A sample of any size from the population", "A sample that is easy to collect", "The largest possible sample"], correct: 0, explanation: "Representative samples have similar proportions as the population. Random sampling helps achieve this." },
      { id: 5, question: "What is the unit of analysis in research?", options: ["The individual subject or object being measured", "The total sample size", "The research hypothesis", "The statistical test being used"], correct: 0, explanation: "Unit of analysis is what you measure (a farm, a plant, a field plot)." },
      { id: 6, question: "In a study of rice yield across 50 farmers, what is the population?", options: ["All rice farmers in Nigeria (or the defined region)", "The 50 farmers studied", "The total rice production in Nigeria", "The average yield"], correct: 0, explanation: "Population = ALL rice farmers in the region. The 50 farmers = sample." },
      { id: 7, question: "What does 'validity' mean in research?", options: ["The study measures what it claims to measure", "The results are always statistically significant", "The sample size is large enough", "The data has no errors"], correct: 0, explanation: "Validity = does the study measure the intended construct?" },
      { id: 8, question: "What is reliability in research?", options: ["Consistency of measurement across repetitions", "Whether results are statistically significant", "How many people participated", "The accuracy of the final conclusion"], correct: 0, explanation: "Reliability = reproducibility. Reliable instruments give the same result repeatedly." },
      { id: 9, question: "Which is an example of categorical data?", options: ["Crop variety (Hybrid, Improved, Local)", "Plant height in centimeters", "Soil moisture percentage", "Yield per hectare"], correct: 0, explanation: "Categorical = categories without numerical ranking (crop type, farmer gender, region)." },
      { id: 10, question: "Why is random sampling important in research?", options: ["It reduces bias and increases representativeness", "It guarantees the largest sample size", "It eliminates the need for statistics", "It makes data collection faster"], correct: 0, explanation: "Random sampling ensures each unit has equal chance of selection, reducing selection bias." },
    ],
  },
  week1: {
    title: "Week 1 Quiz: Descriptive Statistics & Summarizing Data",
    description: "Learn to summarize and describe agricultural datasets",
    passingScore: 7,
    questions: [
      { id: 1, question: "What does the mean measure?", options: ["The average value of a dataset", "The most frequently occurring value", "The middle value when data is ordered", "The range of values"], correct: 0, explanation: "Mean = sum of all values / number of observations." },
      { id: 2, question: "When is the median more useful than the mean?", options: ["When data has outliers or is skewed", "When sample size is large", "When data is normally distributed", "When measuring categorical variables"], correct: 0, explanation: "Median is robust to outliers. If 4 farms yield 5 tonnes and 1 farm yields 100 tonnes, median = 5 (realistic)." },
      { id: 3, question: "What does standard deviation measure?", options: ["How spread out data is from the mean", "The average value of the dataset", "The difference between max and min", "The median of the dataset"], correct: 0, explanation: "Standard deviation quantifies variability. High SD = data spread wide." },
      { id: 4, question: "If rainfall has mean 800 mm and SD 150 mm, what does this tell you?", options: ["Most years have rainfall between 650-950 mm (±1 SD)", "Rainfall never exceeds 950 mm", "The maximum rainfall is 950 mm", "Rainfall is always exactly 800 mm"], correct: 0, explanation: "In normally distributed data, ~68% of observations fall within ±1 SD." },
      { id: 5, question: "What is a quartile?", options: ["A value that divides data into 4 equal parts (Q1, Q2, Q3)", "One-quarter of the data", "The average of four values", "The 25th observation in a dataset"], correct: 0, explanation: "Q1 = 25th percentile, Q2 = median, Q3 = 75th percentile." },
      { id: 6, question: "What does the range measure?", options: ["The difference between maximum and minimum values", "The average spread from the mean", "The middle 50% of data", "How many values fall within 1 SD"], correct: 0, explanation: "Range = max - min. Simple but affected by outliers." },
      { id: 7, question: "What is skewness?", options: ["Asymmetry in the distribution of data", "The standard deviation of the data", "How many outliers are present", "The strength of correlation"], correct: 0, explanation: "Right-skewed: tail extends right. Left-skewed: tail extends left." },
      { id: 8, question: "In a frequency distribution, what does 'frequency' mean?", options: ["How many times a value or range occurs", "How often data is collected", "The average value in a group", "The probability of an event"], correct: 0, explanation: "Frequency table counts occurrences in each class/bin." },
      { id: 9, question: "What is a histogram?", options: ["A bar chart showing frequency distribution of continuous data", "A line graph over time", "A comparison between two groups", "A pie chart showing proportions"], correct: 0, explanation: "Histograms visualize numerical data distribution. Bars touch (continuous)." },
      { id: 10, question: "What is the coefficient of variation (CV)?", options: ["The ratio of standard deviation to mean (SD/mean × 100%)", "The difference between SD and mean", "Another name for variance", "The range divided by sample size"], correct: 0, explanation: "CV allows comparing variability across datasets with different scales." },
    ],
  },
  week2: {
    title: "Week 2 Quiz: Data Visualization & Exploratory Analysis",
    description: "Visualizing agricultural data to identify patterns",
    passingScore: 7,
    questions: [
      { id: 1, question: "When should you use a box plot?", options: ["To display distribution, median, quartiles, and outliers", "To show changes over time", "To compare two continuous variables", "To display categorical proportions"], correct: 0, explanation: "Box plots efficiently summarize data: median, quartiles, whiskers, and outliers." },
      { id: 2, question: "What does a scatter plot reveal?", options: ["The relationship between two continuous variables", "The trend over time", "The frequency of categories", "The average value of groups"], correct: 0, explanation: "Scatter plots show bivariate relationships. Pattern = linear, curved, etc." },
      { id: 3, question: "Which visualization is best for showing trends over time?", options: ["Line graph", "Pie chart", "Box plot", "Bar chart"], correct: 0, explanation: "Line graphs connect points over time, making trends visible." },
      { id: 4, question: "What is the advantage of a violin plot over a box plot?", options: ["It shows the full distribution shape, not just quartiles", "It works only for categorical data", "It always requires more data", "It eliminates outlier detection"], correct: 0, explanation: "Violin plots show density: wider = more data. Reveals bimodal distributions." },
      { id: 5, question: "When is a pie chart appropriate?", options: ["To show proportions of a whole (categorical data)", "To compare continuous variables", "To display time series data", "To show relationships between two variables"], correct: 0, explanation: "Pie charts work for parts-of-whole. Avoid for many categories." },
      { id: 6, question: "What can a heatmap show in agricultural research?", options: ["Intensity/magnitude across two dimensions (e.g., yield by month and field)", "Trends over time", "Proportions of categories", "Outliers in univariate data"], correct: 0, explanation: "Heatmaps use color intensity for spatial patterns, time×location interactions." },
      { id: 7, question: "What is the purpose of a Q-Q plot?", options: ["To assess if data follows a normal distribution", "To compare means of two groups", "To show proportions", "To identify correlation strength"], correct: 0, explanation: "Q-Q plots compare quantiles to a normal distribution. Points on diagonal = normal." },
      { id: 8, question: "When comparing yield across three crop varieties, which visualization is best?", options: ["Grouped box plot or violin plot", "Single line graph", "Pie chart", "Scatter plot"], correct: 0, explanation: "Multiple box plots side-by-side allow comparing distributions." },
      { id: 9, question: "What does an outlier look like in a scatter plot?", options: ["A point far from the main cluster", "A point on the regression line", "The leftmost point", "Any point above the mean"], correct: 0, explanation: "Outliers are extreme values that don't fit the pattern." },
      { id: 10, question: "When should categorical and continuous data be visualized together?", options: ["Using grouped/faceted plots (box plots by category, scatter by group)", "Always in separate plots", "Never together", "Only with categorical data converted to numbers"], correct: 0, explanation: "Faceted/grouped plots show both: yield (continuous) by region (categorical)." },
    ],
  },
  week3: {
    title: "Week 3 Quiz: Probability & Distributions",
    description: "Understanding randomness and probability in agricultural systems",
    passingScore: 7,
    questions: [
      { id: 1, question: "What is probability?", options: ["The likelihood of an event occurring, ranging 0 to 1", "The number of times an event happened", "The average outcome of an experiment", "A synonym for statistics"], correct: 0, explanation: "Probability ranges 0-1. P = 0: impossible. P = 1: certain." },
      { id: 2, question: "What is the normal distribution?", options: ["A symmetric, bell-shaped distribution where most data cluster near the mean", "Any distribution used for continuous data", "A distribution skewed to the right", "The distribution of categorical data"], correct: 0, explanation: "Normal (Gaussian) distribution: symmetric, mean=median=mode." },
      { id: 3, question: "What does the 68-95-99.7 rule state?", options: ["In normal distribution: 68% within ±1 SD, 95% within ±2 SD, 99.7% within ±3 SD", "The percentage of outliers in any dataset", "The probability of any specific value", "The relationship between mean and median"], correct: 0, explanation: "This rule quantifies normality. 95% of farms expect yield within ±2 SD of mean." },
      { id: 4, question: "What is a z-score?", options: ["The number of standard deviations a value is from the mean", "A measure of central tendency", "The variance of data", "A type of probability distribution"], correct: 0, explanation: "z = (value - mean) / SD. Standardizes data for comparison." },
      { id: 5, question: "What is a binomial distribution?", options: ["Distribution of outcomes with two possible results in n independent trials", "A normal distribution with two peaks", "The distribution of continuous data", "A distribution for ranking categorical data"], correct: 0, explanation: "Binomial: repeated trials, each with p(success). E.g., seed germination rates." },
      { id: 6, question: "When is the Poisson distribution useful in agriculture?", options: ["For counting events (pest count, diseases) in a fixed area/time", "For measuring continuous yields", "For ranking categorical variables", "For comparing two means"], correct: 0, explanation: "Poisson models counts of rare events. E.g., number of aphids per leaf." },
      { id: 7, question: "What does a probability density function (PDF) show?", options: ["The likelihood of values within a range (area under curve = probability)", "The exact probability of a specific value", "The cumulative count of observations", "The relationship between two variables"], correct: 0, explanation: "For continuous data, probability is the area under the curve." },
      { id: 8, question: "What is the central limit theorem?", options: ["Distribution of sample means approaches normal, even if population isn't normal", "Mean and median are always equal", "Large samples are always normally distributed", "Statistics must be normally distributed"], correct: 0, explanation: "CLT: sample means follow normal distribution (for large n). Fundamental to hypothesis testing." },
      { id: 9, question: "What is skewness and when is it a problem?", options: ["Asymmetry in distribution; problematic for tests assuming normality", "Outliers in the dataset", "The variance of the data", "A measure of correlation"], correct: 0, explanation: "Skewness affects t-tests, ANOVA which assume normality." },
      { id: 10, question: "What is kurtosis?", options: ["The heaviness of tails in a distribution (peaked vs. flat)", "The center of a distribution", "The spread of data", "The number of standard deviations"], correct: 0, explanation: "High kurtosis: heavy tails, peaked. Low kurtosis: light tails, flat." },
    ],
  },
  week4: {
    title: "Week 4 Quiz: Correlation, Causation & Simple Linear Regression",
    description: "Understanding relationships between variables",
    passingScore: 7,
    questions: [
      { id: 1, question: "A researcher observes that farms with higher pesticide use have lower pest damage. Can she conclude pesticides cause the reduction?", options: ["Yes, correlation implies causation", "No, reverse causality may be at play", "Yes, if the correlation is statistically significant", "Only if the sample size is very large"], correct: 1, explanation: "High pest pressure may drive pesticide use, not the other way around. Correlation ≠ causation." },
      { id: 2, question: "In ŷ = 2.5 + 0.8x (y = yield, x = nitrogen), what does the slope 0.8 mean?", options: ["For every 1 kg/ha nitrogen, yield increases by 0.8 tonnes/ha", "For every 0.8 kg/ha nitrogen, yield increases by 1 tonne/ha", "The intercept is 0.8", "The correlation coefficient is 0.8"], correct: 0, explanation: "The slope represents the change in y for each 1-unit change in x." },
      { id: 3, question: "What does R² = 0.72 mean in regression?", options: ["72% of variation in y is explained by x", "The model is 72% correct", "The correlation coefficient is 0.72", "There is a 72% probability the relationship is causal"], correct: 0, explanation: "R² is the coefficient of determination. 0.72 = 72% of variance explained." },
      { id: 4, question: "Cassava farmers in wetter regions have higher yields. Which is a confounding variable?", options: ["Soil type (clay soils retain more water)", "The farmer's experience level", "Both A and B are confounders", "Neither — rainfall directly causes yield"], correct: 2, explanation: "Both soil type and farmer experience could confound the rainfall-yield relationship." },
      { id: 5, question: "In regression diagnostics, what does a Q-Q plot primarily test?", options: ["Whether residuals are normally distributed", "Whether the relationship is linear", "Whether variances are equal", "Whether observations are independent"], correct: 0, explanation: "Q-Q plot compares residuals to a normal distribution." },
      { id: 6, question: "In sorghum_yield = 1.2 + 0.15 × rainfall_mm, the intercept (1.2) represents:", options: ["Predicted yield when rainfall = 0 mm", "The slope of the line", "The correlation coefficient", "The R² value"], correct: 0, explanation: "The intercept is the predicted value of y when x = 0." },
      { id: 7, question: "Strong negative correlation (r = −0.88, p < 0.001) between pest and predator count. Can we claim predators cause pest decrease?", options: ["Yes, because p < 0.05 and r is strong", "No, without experimental manipulation we cannot infer causation", "Yes, if the sample size is >30", "Only if R² > 0.70"], correct: 1, explanation: "Only experimental designs with random assignment can establish causal relationships." },
      { id: 8, question: "What is the residual standard error in regression?", options: ["Average distance of observations from the regression line", "SD of the independent variable", "The slope of the regression line", "The R² value"], correct: 0, explanation: "Residual standard error measures prediction accuracy." },
      { id: 9, question: "Model 1: R² = 0.54 (rainfall). Model 2: R² = 0.68 (rainfall + soil pH + potassium). Conclusion?", options: ["Model 2 is definitely better", "Model 2 explains 14% more variation, but check if complexity is justified", "Model 1 is simpler and therefore superior", "P-values tell us which is better"], correct: 1, explanation: "Adding variables always increases R². Use adjusted R² or F-tests to justify complexity." },
      { id: 10, question: "To claim compost causes higher yields, what must happen?", options: ["Correlation must be >0.8", "Sample size must exceed 100", "Treatments must be randomly assigned (experimental design)", "Regression R² must be >0.7"], correct: 2, explanation: "Random assignment is essential for causal inference." },
    ],
  },
  week5: {
    title: "Week 5 Quiz: Multiple Regression & Model Selection",
    description: "Extending regression to multiple predictors",
    passingScore: 7,
    questions: [
      { id: 1, question: "What does adjusted R² account for that regular R² does not?", options: ["The number of predictor variables", "The sample size", "Measurement error", "Non-linear relationships"], correct: 0, explanation: "Adjusted R² penalizes adding variables that don't improve model fit." },
      { id: 2, question: "What is multicollinearity?", options: ["High correlation between predictor variables", "Correlation between predictor and outcome", "More than one outcome variable", "Non-linear relationships"], correct: 0, explanation: "Multicollinearity makes it hard to estimate individual effects. Check with VIF." },
      { id: 3, question: "What does the F-statistic test in multiple regression?", options: ["Whether the overall model is significant", "Whether a single predictor is significant", "The slope of one predictor", "The normality of residuals"], correct: 0, explanation: "F-test: H0 = all predictors have zero effect." },
      { id: 4, question: "What does VIF > 5 indicate?", options: ["Problematic multicollinearity", "Predictor is not significant", "Model is overfitting", "Sample size is too small"], correct: 0, explanation: "VIF > 5-10 suggests multicollinearity. Consider removing one correlated variable." },
      { id: 5, question: "Which criterion prevents overfitting when comparing models?", options: ["Adjusted R² or AIC (penalizes complexity)", "Regular R² (always increases)", "Sample size", "P-value of the last predictor"], correct: 0, explanation: "Adjusted R² and AIC both penalize adding predictors." },
      { id: 6, question: "Difference between backward and forward selection?", options: ["Backward: start full, remove weak. Forward: start empty, add strong", "Backward is always better", "They produce identical models", "Forward is only for small datasets"], correct: 0, explanation: "Backward: full model → remove weakest. Forward: empty → add strongest." },
      { id: 7, question: "A p-value > 0.05 for a predictor indicates:", options: ["Not statistically significant", "The model is invalid", "Sample size is too small", "Evidence the effect is real"], correct: 0, explanation: "p > 0.05 suggests no significant relationship after accounting for other predictors." },
      { id: 8, question: "What is interaction in regression?", options: ["Effect of one predictor depends on another's value", "Correlation between predictors", "The intercept of the model", "Non-linear relationships"], correct: 0, explanation: "Example: fertilizer effect differs by rainfall level." },
      { id: 9, question: "How do you detect heteroscedasticity?", options: ["Residuals vs. fitted plot: cone/funnel shape", "Q-Q plot", "Correlation matrix", "Histogram of predictors"], correct: 0, explanation: "Heteroscedasticity = non-constant variance shown as widening spread." },
      { id: 10, question: "When should you standardize predictors?", options: ["When predictors have different scales", "Always", "Never", "Only for categorical variables"], correct: 0, explanation: "Standardizing puts all predictors on the same scale for comparing effect sizes." },
    ],
  },
  week6: {
    title: "Week 6 Quiz: ANOVA & Experimental Design",
    description: "Comparing groups and designing experiments",
    passingScore: 7,
    questions: [
      { id: 1, question: "What is ANOVA testing?", options: ["Whether means of 3+ groups differ significantly", "Whether data is normally distributed", "Whether two variables are correlated", "Whether sample size is adequate"], correct: 0, explanation: "ANOVA tests H0: all group means are equal." },
      { id: 2, question: "What is the null hypothesis in one-way ANOVA?", options: ["All group means are equal (μ1 = μ2 = μ3)", "At least one group mean is different", "Variance within groups is zero", "Sample means equal population means"], correct: 0, explanation: "H0: μ1 = μ2 = μ3. Reject if p < 0.05." },
      { id: 3, question: "What does the F-statistic measure in ANOVA?", options: ["Ratio of between-group to within-group variance", "The mean of the data", "The standard deviation of groups", "The sample size"], correct: 0, explanation: "F = MS_between / MS_within. Large F: groups differ a lot." },
      { id: 4, question: "What is a post-hoc test?", options: ["Test after ANOVA to identify which groups differ", "Test done before ANOVA", "Alternative to ANOVA", "Test of model assumptions"], correct: 0, explanation: "Post-hoc (Tukey, LSD) compares pairs after significant ANOVA." },
      { id: 5, question: "What is the purpose of blocking in experimental design?", options: ["Reduce error by grouping similar units", "Increase sample size", "Randomize treatments", "Increase bias"], correct: 0, explanation: "Blocking controls nuisance variation, increases power to detect treatment effects." },
      { id: 6, question: "What does 'replication' mean in experimental design?", options: ["Each treatment applied to multiple units", "Repeating an experiment later", "Using the same measurement tool", "Checking statistical assumptions"], correct: 0, explanation: "Replication (≥3 reps) allows estimation of variation and confidence in estimates." },
      { id: 7, question: "What is randomization in experimental design?", options: ["Randomly assigning treatments to units to minimize bias", "Randomly selecting a sample", "Using random numbers in analysis", "Repeating an experiment"], correct: 0, explanation: "Randomization balances unknown factors, enables causal inference." },
      { id: 8, question: "What is a completely randomized design (CRD)?", options: ["All treatments randomly assigned to all units; no blocking", "Treatments assigned in fixed order", "Designs with spatial blocking", "Designs with multiple outcomes"], correct: 0, explanation: "CRD: simplest design. Appropriate when variation is uniform." },
      { id: 9, question: "What is a randomized complete block design (RCBD)?", options: ["Units grouped into blocks; all treatments in each block, randomized", "Treatments not randomized", "No blocking used", "More than 3 treatment levels"], correct: 0, explanation: "RCBD: field divided into blocks. Each block receives all treatments." },
      { id: 10, question: "What assumption must be met for valid ANOVA results?", options: ["Normality of residuals and homogeneity of variance", "Data must be perfectly normal", "Groups must have identical means", "Sample sizes must be huge"], correct: 0, explanation: "Key: normality (Q-Q plot) and homogeneity (Levene's test). Minor violations acceptable with large samples." },
    ],
  },
};
