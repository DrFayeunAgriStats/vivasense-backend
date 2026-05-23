import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  FlaskConical,
  BarChart3,
  Microscope,
  TrendingUp,
  Layers,
} from "lucide-react";

const services = [
  {
    icon: FlaskConical,
    title: "Experimental Design",
    description:
      "CRD, RCBD, Split-plot, factorial, and augmented designs tailored for agricultural field trials.",
  },
  {
    icon: BarChart3,
    title: "Statistical Analysis",
    description:
      "ANOVA, regression, correlation, mean separation tests, and multi-factor analysis for crop research data.",
  },
  {
    icon: TrendingUp,
    title: "G×E Analysis",
    description:
      "Genotype-by-environment interaction analysis using AMMI, GGE biplot, and stability models.",
  },
  {
    icon: Microscope,
    title: "Heritability Analysis",
    description:
      "Broad-sense and narrow-sense heritability estimation, genetic advance, and variance component analysis.",
  },
  {
    icon: Layers,
    title: "PCA & Cluster Analysis",
    description:
      "Principal component analysis, hierarchical clustering, and multivariate techniques for germplasm evaluation.",
  },
];

export default function DataAnalysisServices() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-4">
              Agricultural Data Analysis Services
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
              Professional data analysis for agricultural researchers, postgraduate students,
              and institutions requiring rigorous statistical interpretation.
            </p>
            <p className="text-primary-foreground/70 leading-relaxed">
              From experimental design to publication-ready results, we provide
              end-to-end support for your research data.
            </p>
          </div>
        </div>
      </section>

      {/* Services */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              What We Offer
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Comprehensive data analysis services covering the full spectrum of
              agricultural research methodology.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {services.map((s) => (
              <div
                key={s.title}
                className="card-elevated p-6 hover:border-primary transition-colors"
              >
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <s.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-serif text-xl font-semibold text-foreground mb-2">
                  {s.title}
                </h3>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {s.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold mb-6">
            Need Help With Your Data?
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            Get in touch to discuss your data analysis needs and let our team
            provide expert statistical support for your research.
          </p>
          <Button variant="hero" size="xl" asChild>
            <Link to="/contact">
              Request Data Analysis
              <ArrowRight className="w-5 h-5" />
            </Link>
          </Button>
        </div>
      </section>
    </Layout>
  );
}
