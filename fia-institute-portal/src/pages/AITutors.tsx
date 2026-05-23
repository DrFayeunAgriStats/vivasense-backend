import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  GraduationCap,
  Leaf,
  FlaskConical,
  PenTool,
  Sprout,
  Sparkles,
} from "lucide-react";

const tutors = [
  {
    icon: GraduationCap,
    title: "FIA-ADAP Foundation Tutor",
    description:
      "AI-powered learning assistant for the FIA-ADAP Foundations programme. Covers experimental design, data analysis, and agricultural statistics across a structured 7-week curriculum.",
    href: "/adap-tutor",
    status: "Live",
  },
  {
    icon: Leaf,
    title: "Plant Genetics Tutor",
    description:
      "Interactive AI tutor for plant genetics concepts including inheritance patterns, gene action, and breeding methodology.",
    href: "/plant-genetics-mastery-tutor",
    status: "Live",
  },
  {
    icon: FlaskConical,
    title: "Biometrical Genetics Tutor",
    description:
      "Postgraduate-level AI tutor covering quantitative genetics, heritability estimation, combining ability analysis, and genetic parameter computation.",
    href: "/biometrical-genetics-mastery-tutor",
    status: "Live",
  },
  {
    icon: Sprout,
    title: "Plant Improvement AI Tutor",
    description:
      "Interactive AI-assisted learning for undergraduate crop improvement students — covering centres of origin, domestication, plant introduction, and Nigerian crop case studies.",
    href: "/ai-tutors/plant-improvement",
    status: "Live",
  },
  {
    icon: PenTool,
    title: "Thesis Mentor",
    description:
      "AI-powered research and thesis guidance for undergraduate, MSc, and PhD students — covering topic development, literature review, methodology, data analysis, results interpretation, and thesis writing.",
    href: "/thesis-mentor",
    status: "Live",
  },
];

export default function AITutors() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              AI Tutors
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Intelligent, domain-specific AI tutors designed to guide agricultural
              researchers through complex concepts with rigour and clarity.
            </p>
          </div>
        </div>
      </section>

      {/* Tutor Cards */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {tutors.map((tutor) => (
              <div key={tutor.title} className="card-elevated p-8 flex flex-col">
                <div className="flex items-center justify-between mb-6">
                  <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center">
                    <tutor.icon className="w-7 h-7 text-primary" />
                  </div>
                  <span className="px-3 py-1 rounded-full text-xs font-semibold text-primary-foreground bg-primary">
                    {tutor.status}
                  </span>
                </div>
                <h3 className="font-serif text-2xl font-bold text-foreground mb-3">
                  {tutor.title}
                </h3>
                <p className="text-muted-foreground leading-relaxed mb-6 flex-1">
                  {tutor.description}
                </p>
                <Button variant="default" asChild className="w-full">
                  <Link to={tutor.href}>
                    Open Tutor
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                </Button>
              </div>
            ))}
          </div>

          {/* Coming Soon */}
          <div className="max-w-5xl mx-auto mt-8">
            <div className="card-elevated p-8 text-center border-dashed">
              <div className="w-14 h-14 rounded-xl bg-muted flex items-center justify-center mx-auto mb-4">
                <Sparkles className="w-7 h-7 text-muted-foreground" />
              </div>
              <h3 className="font-serif text-xl font-semibold text-foreground mb-2">
                More Tutors Coming Soon
              </h3>
              <p className="text-muted-foreground max-w-lg mx-auto">
                Additional AI tutors for advanced topics in agricultural data science,
                multivariate analysis, and specialised research domains are under development.
              </p>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
