import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  BookOpen,
  AlertTriangle,
  CheckCircle2,
} from "lucide-react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const curriculum = [
  {
    week: "Week 1",
    title: "Experimental Design Foundations",
    topics: [
      "Completely Randomized Design (CRD)",
      "Randomized Complete Block Design (RCBD)",
      "Factorial Experiments",
      "Split-plot and Split-split-plot Designs",
    ],
    whyItMatters: "The choice of experimental design determines the validity of your entire study. A wrong design choice can invalidate months of field work and make your data unanalysable.",
    commonMistakes: [
      "Using CRD when blocking is necessary",
      "Incorrect randomization procedures",
      "Confusing treatment factors with blocking factors",
      "Wrong error terms in factorial designs",
    ],
  },
  {
    week: "Week 2",
    title: "ANOVA & Mean Separation",
    topics: [
      "One-way and Two-way ANOVA",
      "ANOVA assumptions and diagnostics",
      "LSD, Duncan, Tukey, and SNK tests",
      "When to use which mean separation test",
    ],
    whyItMatters: "ANOVA is the backbone of agricultural experimentation. Understanding when and why to use it—and how to interpret results—is essential for valid conclusions.",
    commonMistakes: [
      "Ignoring ANOVA assumptions",
      "Using inappropriate mean separation tests",
      "Over-interpreting non-significant results",
      "Misreading interaction effects",
    ],
  },
  {
    week: "Week 3",
    title: "Regression & Correlation Analysis",
    topics: [
      "Simple linear regression",
      "Multiple regression",
      "Correlation analysis and interpretation",
      "Model diagnostics and validation",
    ],
    whyItMatters: "Understanding relationships between variables is crucial for predictive modeling and understanding cause-effect relationships in agricultural systems.",
    commonMistakes: [
      "Confusing correlation with causation",
      "Ignoring multicollinearity",
      "Overfitting regression models",
      "Misinterpreting R² values",
    ],
  },
  {
    week: "Week 4",
    title: "G×E Analysis: AMMI & GGE Biplot",
    topics: [
      "Understanding genotype × environment interaction",
      "AMMI (Additive Main effects and Multiplicative Interaction)",
      "GGE biplot analysis and interpretation",
      "Stability analysis and variety recommendations",
    ],
    tools: "R, GRAPES",
    whyItMatters: "Multi-environment trial analysis is essential for plant breeding and variety recommendation. These techniques separate main effects from interactions for better variety selection.",
    commonMistakes: [
      "Ignoring significant G×E when making recommendations",
      "Misinterpreting biplot distances and angles",
      "Using inappropriate number of principal components",
      "Failing to identify mega-environments",
    ],
  },
  {
    week: "Week 5",
    title: "Multivariate Analysis",
    topics: [
      "Principal Component Analysis (PCA)",
      "Cluster Analysis (Hierarchical and K-means)",
      "Discriminant Analysis basics",
      "Interpreting and presenting multivariate results",
    ],
    tools: "R, PAST",
    whyItMatters: "When you have many variables measured on your samples (e.g., morphological traits, yield components), multivariate techniques help you see patterns and group similar observations.",
    commonMistakes: [
      "Not standardizing data before PCA",
      "Choosing arbitrary number of clusters",
      "Misinterpreting principal component loadings",
      "Ignoring outliers in cluster analysis",
    ],
  },
  {
    week: "Week 6",
    title: "Writing Results & Defense Preparation",
    topics: [
      "Structuring your results section",
      "Creating publication-quality tables and figures",
      "Statistical notation and APA formatting",
      "Defending your methodology during supervision",
    ],
    whyItMatters: "Knowing how to analyse data is only half the battle. You must be able to present and defend your choices clearly to supervisors, examiners, and journal reviewers.",
    commonMistakes: [
      "Reporting results without interpretation",
      "Using inconsistent table formats",
      "Failing to explain methodology choices",
      "Presenting too much or too little statistical detail",
    ],
  },
];

export default function Curriculum() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Curriculum
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
              A structured 6-week journey from experimental design fundamentals 
              to confident thesis defense. Each week builds on the previous one.
            </p>
            <p className="text-primary-foreground/70 text-sm italic">
              FIA-ADAP™ Foundations is the entry-level programme within the FIA-ADAP™ 
              programme family at Field-to-Insight Academy.
            </p>
          </div>
        </div>
      </section>

      {/* Curriculum Overview */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
                6-Week Breakdown
              </h2>
              <p className="text-lg text-muted-foreground">
                Each week includes 2 live sessions, practical exercises, and 
                opportunities to apply techniques to your own data.
              </p>
            </div>

            <Accordion type="single" collapsible className="space-y-4">
              {curriculum.map((week, index) => (
                <AccordionItem
                  key={week.week}
                  value={week.week}
                  className="bg-card border border-border rounded-xl overflow-hidden"
                >
                  <AccordionTrigger className="px-6 py-5 hover:no-underline hover:bg-muted/50">
                    <div className="flex items-center gap-4 text-left">
                      <div className="w-12 h-12 rounded-lg bg-primary flex items-center justify-center text-primary-foreground font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <div className="text-sm text-primary font-semibold">
                          {week.week}
                        </div>
                        <div className="text-lg font-semibold text-foreground">
                          {week.title}
                        </div>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="px-6 pb-6">
                    <div className="pl-16 space-y-6">
                      {/* Topics */}
                      <div>
                        <h4 className="flex items-center gap-2 font-semibold text-foreground mb-3">
                          <BookOpen className="w-4 h-4 text-primary" />
                          What You'll Learn
                        </h4>
                        <ul className="grid md:grid-cols-2 gap-2">
                          {week.topics.map((topic) => (
                            <li key={topic} className="flex items-start gap-2">
                              <CheckCircle2 className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
                              <span className="text-muted-foreground text-sm">{topic}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Why It Matters */}
                      <div className="bg-muted rounded-lg p-4">
                        <h4 className="font-semibold text-foreground mb-2">
                          Why This Matters in Research
                        </h4>
                        <p className="text-muted-foreground text-sm">
                          {week.whyItMatters}
                        </p>
                      </div>

                      {/* Common Mistakes */}
                      <div>
                        <h4 className="flex items-center gap-2 font-semibold text-foreground mb-3">
                          <AlertTriangle className="w-4 h-4 text-accent" />
                          Common Mistakes We'll Help You Avoid
                        </h4>
                        <ul className="space-y-2">
                          {week.commonMistakes.map((mistake) => (
                            <li key={mistake} className="flex items-start gap-2">
                              <span className="text-destructive">✗</span>
                              <span className="text-muted-foreground text-sm">{mistake}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </div>
        </div>
      </section>

      {/* Software Tools */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-6">
              Software Tools Covered
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              You'll gain hands-on experience with industry-standard tools used 
              in agricultural research.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {["R / RStudio", "SAS Studio", "GRAPES", "PAST", "OPSTAT", "Excel"].map((tool) => (
                <div
                  key={tool}
                  className="bg-card rounded-xl p-6 border border-border"
                >
                  <span className="font-semibold text-foreground">{tool}</span>
                </div>
              ))}
            </div>
            <p className="text-muted-foreground mt-6 text-sm">
              Note: Tools are means to an end. We emphasize understanding the 
              methodology—any software can then be used confidently.
            </p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
            Ready to Master Agricultural Data Analysis?
          </h2>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join the next cohort and transform your approach to research data.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="gold" size="lg" asChild>
              <Link to="/apply">
                Apply Now
                <ArrowRight className="w-4 h-4" />
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link to="/pricing">View Investment</Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
}
