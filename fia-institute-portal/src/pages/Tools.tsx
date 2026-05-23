import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  Microscope,
  BarChart3,
  FileSpreadsheet,
  Code,
  Lightbulb,
  Target,
  CheckCircle2,
  Dna,
  Calculator,
  LineChart,
} from "lucide-react";

const frameworkSteps = [
  {
    step: "1",
    title: "Field Design",
    description: "Proper experimental design before data collection",
    details: "Choose the right design (CRD, RCBD, Factorial, Split-plot) based on your research question, resources, and field conditions.",
    icon: Microscope,
  },
  {
    step: "2",
    title: "Data Analysis",
    description: "Appropriate statistical methods for your data",
    details: "Apply the correct analytical techniques—ANOVA, regression, multivariate analysis—matched to your design and objectives.",
    icon: BarChart3,
  },
  {
    step: "3",
    title: "Insight",
    description: "Meaningful interpretation of results",
    details: "Transform statistical outputs into biological and agronomic understanding that answers your research questions.",
    icon: Lightbulb,
  },
  {
    step: "4",
    title: "Decision",
    description: "Defensible conclusions and recommendations",
    details: "Make evidence-based recommendations and confidently defend your methodology to supervisors, reviewers, and stakeholders.",
    icon: Target,
  },
];

const softwareTools = [
  {
    name: "R / RStudio",
    description: "Open-source statistical computing environment. Powerful, flexible, and the gold standard in academic research and reproducible analysis.",
    uses: ["ANOVA", "Regression", "PCA", "GGE Biplot", "AMMI", "Publication graphics"],
    bestFor: "Comprehensive statistical analysis, reproducible research, and publication-quality visualizations.",
    commonMisuse: "Using packages without understanding the underlying methods or ignoring diagnostic outputs.",
    icon: Code,
  },
  {
    name: "SAS Studio",
    description: "Industry-standard statistical software widely used by research institutions, government agencies, and journals.",
    uses: ["PROC GLM", "PROC MIXED", "AMMI Analysis", "Stability analysis", "Multi-location trials"],
    bestFor: "Large-scale agricultural trials, institutional reporting, and journal submissions requiring SAS outputs.",
    commonMisuse: "Copying code without understanding the model specification or correct error terms.",
    icon: BarChart3,
  },
  {
    name: "GRAPES",
    description: "Specialized software for plant breeding and genetics analysis. Designed specifically for agricultural genetic research.",
    uses: ["GGE Biplot analysis", "AMMI models", "Stability analysis", "G×E interaction", "Variety selection"],
    bestFor: "Genotype × environment interaction analysis and multi-environment trial evaluation in plant breeding.",
    commonMisuse: "Misinterpreting biplot distances and angles or selecting wrong model components.",
    icon: Dna,
  },
  {
    name: "PAST",
    description: "Paleontological Statistics software package. Excellent for multivariate analysis with intuitive interfaces.",
    uses: ["PCA", "Cluster Analysis", "Discriminant Analysis", "Diversity indices", "Ordination"],
    bestFor: "Multivariate analysis, morphometric studies, and exploratory data analysis with visual outputs.",
    commonMisuse: "Not standardizing data before PCA or choosing arbitrary cluster numbers.",
    icon: LineChart,
  },
  {
    name: "OPSTAT",
    description: "Agricultural statistics software developed for field experiment analysis. User-friendly for common agricultural designs.",
    uses: ["ANOVA", "Mean separation", "Experimental designs", "Correlation", "Path analysis"],
    bestFor: "Standard agricultural experimental designs (CRD, RCBD, Factorial) and routine field trial analysis.",
    commonMisuse: "Using default settings without verifying assumptions or design specifications.",
    icon: Calculator,
  },
  {
    name: "Microsoft Excel",
    description: "Essential for data preparation, organization, and preliminary analysis. The starting point for all data workflows.",
    uses: ["Data entry", "Data cleaning", "Pivot tables", "Basic calculations", "Preliminary charts"],
    bestFor: "Data structuring, exploratory analysis, and preparing datasets for specialized software.",
    commonMisuse: "Performing complex analyses in Excel when specialized software provides better diagnostics.",
    icon: FileSpreadsheet,
  },
];

export default function Tools() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Tools & Framework
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
              Our methodology and the software tools you'll master during 
              the FIA–ADAP™ program.
            </p>
            <p className="text-primary-foreground/70 text-sm italic">
              FIA-ADAP™ Foundations is the entry-level programme within the FIA-ADAP™ 
              programme family at Field-to-Insight Academy.
            </p>
          </div>
        </div>
      </section>

      {/* Framework Section */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="text-center mb-16">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              The Field-to-Insight Framework
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              A systematic approach to agricultural data analysis that takes you 
              from experimental design to defensible conclusions.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            {frameworkSteps.map((step, index) => (
              <div key={step.title} className="relative">
                <div className="card-elevated p-6 h-full">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold">
                      {step.step}
                    </div>
                    <step.icon className="w-6 h-6 text-accent" />
                  </div>
                  <h3 className="font-serif text-xl font-semibold text-foreground mb-2">
                    {step.title}
                  </h3>
                  <p className="text-primary font-medium text-sm mb-3">
                    {step.description}
                  </p>
                  <p className="text-muted-foreground text-sm">
                    {step.details}
                  </p>
                </div>
                {index < frameworkSteps.length - 1 && (
                  <div className="hidden lg:block absolute top-1/2 -right-3 transform -translate-y-1/2">
                    <ArrowRight className="w-6 h-6 text-muted-foreground/50" />
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="bg-muted rounded-2xl p-8 text-center">
            <h3 className="font-serif text-xl font-semibold text-foreground mb-4">
              The Framework in Practice
            </h3>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Every topic in the curriculum is taught through this framework lens. 
              You don't just learn techniques in isolation—you understand how they 
              fit into the complete journey from research question to defensible answer.
            </p>
          </div>
        </div>
      </section>

      {/* Software Tools */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              Software Tools
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Gain practical experience with tools used by agricultural 
              researchers worldwide. Each tool is taught with its purpose, strengths, 
              and common misuses corrected.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {softwareTools.map((tool) => (
              <div key={tool.name} className="bg-card rounded-xl p-6 border border-border">
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                    <tool.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="font-serif text-xl font-semibold text-foreground">
                    {tool.name}
                  </h3>
                </div>
                <p className="text-muted-foreground text-sm mb-4">
                  {tool.description}
                </p>
                
                <div className="mb-4">
                  <p className="text-xs font-medium text-foreground mb-2 uppercase tracking-wide">Best For:</p>
                  <p className="text-muted-foreground text-sm">{tool.bestFor}</p>
                </div>

                <div className="mb-4">
                  <p className="text-xs font-medium text-foreground mb-2 uppercase tracking-wide">Used for:</p>
                  <div className="flex flex-wrap gap-1">
                    {tool.uses.map((use) => (
                      <span
                        key={use}
                        className="px-2 py-1 bg-muted rounded text-xs text-muted-foreground"
                      >
                        {use}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="pt-4 border-t border-border">
                  <p className="text-xs font-medium text-destructive mb-1">Common Misuse FIA Corrects:</p>
                  <p className="text-muted-foreground text-xs">{tool.commonMisuse}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Philosophy Note */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <div className="bg-primary/5 rounded-2xl p-8 border border-primary/20">
              <h3 className="font-serif text-2xl font-bold text-foreground mb-4 text-center">
                Tools Are Secondary to Understanding
              </h3>
              <p className="text-muted-foreground text-center mb-6">
                At FIA, we believe that software is a means to an end, not the end itself. 
                Our emphasis is on:
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                {[
                  "Understanding the statistical concepts",
                  "Knowing when and why to use each method",
                  "Interpreting results correctly",
                ].map((item) => (
                  <div key={item} className="flex flex-col items-center text-center">
                    <CheckCircle2 className="w-5 h-5 text-primary mb-2" />
                    <span className="text-foreground text-sm">{item}</span>
                  </div>
                ))}
              </div>
              <p className="text-muted-foreground text-center mt-6 text-sm">
                Once you understand the methodology, you can confidently use any software—
                R, SAS, SPSS, GenStat, or others your institution provides.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-muted">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
            Ready to Master These Tools?
          </h2>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join FIA–ADAP™ Foundations and learn to apply the Field-to-Insight 
            Framework to your own research.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="gold" size="lg" asChild>
              <Link to="/apply">
                Apply Now
                <ArrowRight className="w-4 h-4" />
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link to="/curriculum">View Curriculum</Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
}
