import { Layout } from "@/components/layout/Layout";
import { Link } from "react-router-dom";
import {
  Brain,
  BarChart3,
  FlaskConical,
  LineChart,
  Microscope,
  ArrowRight,
  Star,
} from "lucide-react";
import { Button } from "@/components/ui/button";

const topics = [
  {
    icon: Brain,
    title: "AI in Agriculture",
    description:
      "Exploring how artificial intelligence is transforming crop improvement, precision agriculture, and research decision-making.",
  },
  {
    icon: BarChart3,
    title: "Data Science in Agriculture",
    description:
      "Techniques for collecting, managing, and analysing agricultural field data to generate meaningful scientific conclusions.",
  },
  {
    icon: FlaskConical,
    title: "Experimental Design",
    description:
      "Principles of designing rigorous field experiments — from CRD and RCBD to split-plot, factorial, and augmented designs.",
  },
  {
    icon: LineChart,
    title: "Statistical Methods",
    description:
      "ANOVA, regression, correlation, mean separation, and advanced modelling techniques for agricultural research.",
  },
  {
    icon: Microscope,
    title: "Research Analytics",
    description:
      "Interpreting results, building defensible arguments, and translating statistical outputs into scientific narratives.",
  },
];

const featuredArticle = {
  slug: "scaling-agricultural-expertise",
  title:
    "Scaling Agricultural Expertise: How I Guided 200 Students through Plant Genetics using a Custom AI Tutor",
  author: "Dr. Lawrence Stephen Fayeun",
  date: "April 2026",
  categories: ["AI in Agriculture", "Ed-Tech"],
  excerpt:
    "How a 5-layer prompt architecture enabled expert-level mentoring for 200+ students in Quantitative and Biometrical Genetics — with zero hallucinations.",
};

const upcomingArticles = [
  {
    title: "Designing Scientific Guardrails for AI in Crop Improvement Research",
    category: "AI in Agriculture",
  },
  {
    title: "From Field Data to Publishable Tables: Automating Research Outputs with VivaSense",
    category: "Data Science in Agriculture",
  },
  {
    title: "Socratic AI: Teaching Statistical Thinking to Postgraduate Plant Breeders",
    category: "Ed-Tech",
  },
];

export default function TechnicalInsights() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Technical Insights
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Thought leadership and technical perspectives on agricultural data science,
              experimental design, and AI-driven research innovation.
            </p>
          </div>
        </div>
      </section>

      {/* Featured Article */}
      <section className="section-padding">
        <div className="container-wide max-w-4xl">
          <div className="flex items-center gap-2 mb-6">
            <Star className="w-5 h-5 text-accent fill-accent" />
            <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground">
              Featured Article
            </h2>
          </div>

          <Link
            to={`/technical-insights/${featuredArticle.slug}`}
            className="block group"
          >
            <div className="card-elevated p-8 md:p-10 transition-shadow hover:shadow-lg border-l-4 border-l-accent">
              <div className="flex flex-wrap gap-2 mb-3">
                {featuredArticle.categories.map((cat) => (
                  <span
                    key={cat}
                    className="text-xs font-semibold bg-primary/10 text-primary px-3 py-1 rounded-full"
                  >
                    {cat}
                  </span>
                ))}
              </div>
              <h3 className="font-serif text-xl md:text-2xl font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                {featuredArticle.title}
              </h3>
              <p className="text-muted-foreground text-sm leading-relaxed mb-4">
                {featuredArticle.excerpt}
              </p>
              <div className="flex flex-wrap items-center justify-between gap-4">
                <span className="text-xs text-muted-foreground">
                  {featuredArticle.author} · {featuredArticle.date}
                </span>
                <span className="inline-flex items-center gap-1 text-sm font-semibold text-primary group-hover:gap-2 transition-all">
                  Read Article <ArrowRight className="w-4 h-4" />
                </span>
              </div>
            </div>
          </Link>
        </div>
      </section>

      {/* Topics */}
      <section className="section-padding bg-muted/40">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              Areas of Focus
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Technical insights drawn from agricultural research practice, data science,
              and platform development at Field-to-Insight Academy.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {topics.map((topic) => (
              <div key={topic.title} className="card-elevated p-8 text-center">
                <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                  <topic.icon className="w-7 h-7 text-primary" />
                </div>
                <h3 className="font-serif text-xl font-semibold text-foreground mb-3">
                  {topic.title}
                </h3>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {topic.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Upcoming Articles */}
      <section className="section-padding">
        <div className="container-wide max-w-4xl">
          <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-6">
            Upcoming Articles
          </h2>
          <div className="grid gap-4">
            {upcomingArticles.map((item) => (
              <div
                key={item.title}
                className="bg-background border border-border rounded-xl p-6 flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4"
              >
                <span className="text-xs font-semibold text-primary whitespace-nowrap">
                  {item.category}
                </span>
                <h3 className="font-serif text-base font-semibold text-foreground flex-1">
                  {item.title}
                </h3>
                <span className="text-xs text-muted-foreground whitespace-nowrap">Coming Soon</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </Layout>
  );
}
