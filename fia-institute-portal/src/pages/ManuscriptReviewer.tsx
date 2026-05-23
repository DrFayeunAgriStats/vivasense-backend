import { Layout } from "@/components/layout/Layout";
import { Link } from "react-router-dom";
import {
  FileText,
  Search,
  BookOpen,
  CheckCircle2,
  Award,
  GraduationCap,
  Users,
  ClipboardCheck,
  PenTool,
  Newspaper,
  ArrowRight,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: FileText,
    title: "Line-by-Line Academic Review",
    desc: "Paragraph-level language analysis covering grammar, punctuation, style, and academic tone.",
  },
  {
    icon: Search,
    title: "Grammar & Clarity Check",
    desc: "Identifies subject-verb agreement, tense consistency, and readability issues throughout the manuscript.",
  },
  {
    icon: ClipboardCheck,
    title: "Technical & Methodological Critique",
    desc: "Evaluates experimental design, statistical reporting, and scientific rigor against publication standards.",
  },
  {
    icon: BookOpen,
    title: "Citation & Reference Audit",
    desc: "Cross-checks in-text citations against the reference list, flags missing entries and formatting inconsistencies.",
  },
  {
    icon: Award,
    title: "Final Editorial Recommendation",
    desc: "Provides an overall assessment with a prioritized revision list and a clear editorial verdict.",
  },
];

const audience = [
  { icon: PenTool, label: "Authors", desc: "Strengthen your manuscript before journal submission." },
  { icon: GraduationCap, label: "Postgraduate Students", desc: "Get structured feedback on thesis chapters and research papers." },
  { icon: Users, label: "Supervisors", desc: "Pre-screen student manuscripts to focus supervision on substance." },
  { icon: CheckCircle2, label: "Reviewers", desc: "Streamline initial screening with AI-assisted quality checks." },
  { icon: Newspaper, label: "Journal Editors", desc: "Accelerate desk-review decisions with automated consistency audits." },
];

export default function ManuscriptReviewer() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-gradient-to-br from-primary/5 via-background to-accent/10 py-20 md:py-28">
        <div className="container-wide text-center max-w-3xl mx-auto px-4">
          <span className="inline-block text-xs font-semibold tracking-widest uppercase text-primary mb-4">
            Field-to-Insight Academy
          </span>
          <h1 className="font-serif text-4xl md:text-5xl font-bold text-foreground leading-tight mb-5">
            FIA Manuscript Reviewer
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground mb-4">
            AI-assisted manuscript screening for language, technical quality, and citation-reference consistency.
          </p>
          <p className="text-base text-muted-foreground mb-8 max-w-2xl mx-auto">
            Improve your manuscript before journal submission, supervision, or peer review. The FIA Manuscript
            Reviewer provides structured, publication-ready feedback across language, methodology, citations,
            and overall editorial quality.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button asChild size="lg" className="gap-2">
              <Link to="/manuscript-reviewer/upload">
                Start Review <ArrowRight className="w-4 h-4" />
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <a href="#features">See Features</a>
            </Button>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-16 md:py-24 bg-background">
        <div className="container-wide px-4">
          <h2 className="font-serif text-3xl font-bold text-center text-foreground mb-3">
            What the Reviewer Checks
          </h2>
          <p className="text-center text-muted-foreground mb-12 max-w-2xl mx-auto">
            Five layers of automated screening designed for academic manuscripts.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {features.map((f) => (
              <div
                key={f.title}
                className="border border-border rounded-xl p-6 bg-card hover:shadow-md transition-shadow"
              >
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <f.icon className="w-5 h-5 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground mb-2">{f.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Who It's For */}
      <section className="py-16 md:py-24 bg-muted/30">
        <div className="container-wide px-4">
          <h2 className="font-serif text-3xl font-bold text-center text-foreground mb-3">
            Who It's For
          </h2>
          <p className="text-center text-muted-foreground mb-12 max-w-xl mx-auto">
            Designed for every stage of the academic publishing pipeline.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {audience.map((a) => (
              <div
                key={a.label}
                className="flex items-start gap-4 border border-border rounded-xl p-5 bg-card"
              >
                <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                  <a.icon className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground mb-1">{a.label}</h3>
                  <p className="text-sm text-muted-foreground">{a.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Why It Matters */}
      <section className="py-16 md:py-24 bg-background">
        <div className="container-wide px-4 max-w-3xl mx-auto text-center">
          <h2 className="font-serif text-3xl font-bold text-foreground mb-6">Why It Matters</h2>
          <p className="text-muted-foreground leading-relaxed mb-4">
            Manuscript rejection rates remain high across academic journals — often due to preventable issues
            such as language errors, citation inconsistencies, and methodological gaps. The FIA Manuscript
            Reviewer helps authors and institutions catch these issues early, reducing revision cycles and
            improving first-submission quality.
          </p>
          <p className="text-muted-foreground leading-relaxed mb-8">
            By combining AI-assisted screening with structured editorial criteria, this tool bridges the gap
            between initial drafting and publication-ready quality — saving time for authors, supervisors,
            and journal editorial teams alike.
          </p>
          <Button asChild size="lg" className="gap-2">
            <Link to="/manuscript-reviewer/upload">
              Analyze Your Manuscript <ArrowRight className="w-4 h-4" />
            </Link>
          </Button>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="py-10 bg-muted/20 border-t border-border">
        <div className="container-wide px-4 max-w-3xl mx-auto flex items-start gap-3">
          <ShieldCheck className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
          <p className="text-sm text-muted-foreground leading-relaxed">
            <strong>Disclaimer:</strong> This tool supports manuscript improvement and editorial screening.
            Final scholarly judgment should remain human-led. FIA Manuscript Reviewer is intended as a
            quality-assurance aid and does not replace peer review or editorial decision-making.
          </p>
        </div>
      </section>
    </Layout>
  );
}
