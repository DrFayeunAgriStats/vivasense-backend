import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  Award,
  CheckCircle2,
  Target,
  FileCheck,
  Shield,
  TrendingUp,
} from "lucide-react";

const learningOutcomes = [
  {
    category: "Design",
    outcomes: [
      "Select the appropriate experimental design for any agricultural research scenario",
      "Implement proper randomization and blocking procedures",
      "Identify and avoid common design errors that invalidate experiments",
      "Plan data collection to support intended statistical analyses",
    ],
  },
  {
    category: "Analysis",
    outcomes: [
      "Conduct ANOVA and choose the correct mean separation tests",
      "Perform regression and correlation analysis with proper diagnostics",
      "Analyse genotype × environment interactions using AMMI and GGE biplot",
      "Apply multivariate techniques (PCA, Cluster Analysis) to complex datasets",
    ],
  },
  {
    category: "Interpretation",
    outcomes: [
      "Translate statistical outputs into meaningful biological insights",
      "Distinguish between statistically significant and practically important results",
      "Identify and communicate the limitations of your analyses",
      "Draw evidence-based conclusions from your data",
    ],
  },
  {
    category: "Defense",
    outcomes: [
      "Write clear, publication-ready results sections",
      "Create professional tables and figures for theses and journals",
      "Confidently explain and justify every methodological choice",
      "Respond effectively to supervisor and examiner questions",
    ],
  },
];

const certificationFeatures = [
  {
    icon: Target,
    title: "Assessment-Based",
    description: "Certification requires demonstrating competence through practical assessments, not just attendance.",
  },
  {
    icon: FileCheck,
    title: "Practical Evaluation",
    description: "Work with real datasets to show you can apply what you've learned to actual research scenarios.",
  },
  {
    icon: Shield,
    title: "Competence Focus",
    description: "The Certificate of Competence signifies verified ability, not just course completion.",
  },
  {
    icon: TrendingUp,
    title: "Career Value",
    description: "Demonstrate to supervisors, employers, and institutions that you have genuine analytical skills.",
  },
];

export default function Outcomes() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Learning Outcomes & Certification
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
              What you will achieve and how your competence is recognized 
              upon completing FIA–ADAP™ Foundations.
            </p>
            <p className="text-primary-foreground/70 text-sm italic">
              FIA-ADAP™ Foundations is the entry-level programme within the FIA-ADAP™ 
              programme family at Field-to-Insight Academy.
            </p>
          </div>
        </div>
      </section>

      {/* Learning Outcomes */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              What You Will Be Able To Do
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              FIA–ADAP™ Foundations develops four core competencies that transform 
              how you approach agricultural research.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {learningOutcomes.map((category) => (
              <div key={category.category} className="card-elevated p-8">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                    <CheckCircle2 className="w-5 h-5 text-primary-foreground" />
                  </div>
                  <h3 className="font-serif text-2xl font-bold text-foreground">
                    {category.category}
                  </h3>
                </div>
                <ul className="space-y-3">
                  {category.outcomes.map((outcome) => (
                    <li key={outcome} className="flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                      <span className="text-muted-foreground">{outcome}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Certification */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 text-accent text-sm font-medium mb-6">
                <Award className="w-4 h-4" />
                <span>Certificate of Competence</span>
              </div>
              <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
                Certification That Means Something
              </h2>
              <p className="text-lg text-muted-foreground mb-8">
                Unlike certificates of attendance or participation, the FIA Certificate 
                of Competence is assessment-based. It signifies that you have demonstrated 
                genuine ability in agricultural data analysis.
              </p>
              
              <div className="bg-card rounded-xl p-6 border border-border">
                <h4 className="font-semibold text-foreground mb-4">
                  To Earn Your Certificate:
                </h4>
                <ol className="space-y-3">
                  {[
                    "Complete all 12 live sessions (recordings available for makeup)",
                    "Submit weekly practical exercises",
                    "Pass the final competence assessment",
                    "Demonstrate ability to apply techniques to real datasets",
                  ].map((step, index) => (
                    <li key={step} className="flex items-start gap-3">
                      <span className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold flex-shrink-0">
                        {index + 1}
                      </span>
                      <span className="text-muted-foreground">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {certificationFeatures.map((feature) => (
                <div key={feature.title} className="bg-card rounded-xl p-6 border border-border">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <feature.icon className="w-5 h-5 text-primary" />
                  </div>
                  <h4 className="font-semibold text-foreground mb-2">
                    {feature.title}
                  </h4>
                  <p className="text-muted-foreground text-sm">
                    {feature.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Why This Matters */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-6">
              Why This Approach Matters
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              In an era of easy online certificates, FIA stands apart by prioritizing 
              verified competence over mere completion.
            </p>
            <div className="grid md:grid-cols-3 gap-6 text-left">
              <div className="bg-muted rounded-xl p-6">
                <h4 className="font-semibold text-foreground mb-2">For Students</h4>
                <p className="text-muted-foreground text-sm">
                  Confidence that you actually know what you're doing—not just 
                  that you watched videos.
                </p>
              </div>
              <div className="bg-muted rounded-xl p-6">
                <h4 className="font-semibold text-foreground mb-2">For Supervisors</h4>
                <p className="text-muted-foreground text-sm">
                  Assurance that FIA graduates have demonstrated practical ability 
                  in data analysis.
                </p>
              </div>
              <div className="bg-muted rounded-xl p-6">
                <h4 className="font-semibold text-foreground mb-2">For Institutions</h4>
                <p className="text-muted-foreground text-sm">
                  A meaningful credential that indicates research-ready competence 
                  in methodology.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold mb-6">
            Ready to Earn Your Certificate of Competence?
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            Join FIA–ADAP™ Foundations and develop verified expertise in 
            agricultural data analysis.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="hero" size="xl" asChild>
              <Link to="/apply">
                Apply Now
                <ArrowRight className="w-5 h-5" />
              </Link>
            </Button>
            <Button variant="hero-outline" size="xl" asChild>
              <Link to="/pricing">View Investment</Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
}
