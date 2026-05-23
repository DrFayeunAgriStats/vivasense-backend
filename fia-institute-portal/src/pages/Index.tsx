import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  BarChart3,
  Brain,
  GraduationCap,
  Globe,
  Lightbulb,
  Microscope,
  Target,
  Users,
  Briefcase,
  Sprout,
  FileSearch,
  FlaskConical,
  BookOpen,
  Calendar,
  Clock,
  Video,
  Sparkles,
  ClipboardList,
  Sigma,
} from "lucide-react";

const ecosystem = [
  {
    icon: BarChart3,
    title: "VivaSense",
    description:
      "AI-powered agricultural data analysis and scientific interpretation platform.",
    link: "/vivasense",
  },
  {
    icon: Brain,
    title: "AI Tutors",
    description:
      "Intelligent, domain-specific tutors for plant genetics, biometrical genetics, and research writing.",
    link: "/ai-tutors",
  },
  {
    icon: GraduationCap,
    title: "Training & Programmes",
    description:
      "Competence-based training in agricultural data analysis, experimental design, and research methodology.",
    link: "/programmes",
  },
  {
    icon: Lightbulb,
    title: "Technical Insights",
    description:
      "Thought leadership on AI in agriculture, data science, experimental design, and statistical methods.",
    link: "/technical-insights",
  },
  {
    icon: FileSearch,
    title: "Manuscript Review",
    description:
      "AI-assisted academic manuscript evaluation and publication readiness platform.",
    link: "/manuscript-reviewer",
  },
  {
    icon: Sprout,
    title: "Agro-Services",
    description:
      "Professional agricultural consulting including land evaluation, soil analysis, crop suitability assessment, and farm advisory services.",
    link: "/services/agro-services",
  },
  {
    icon: FlaskConical,
    title: "Data Analysis Services",
    description:
      "Professional agricultural data analysis including experimental design, G×E analysis, and heritability studies.",
    link: "/services/data-analysis",
  },
  {
    icon: BookOpen,
    title: "Journal",
    description:
      "Open-access, peer-reviewed academic journal for agricultural science and innovation.",
    link: "/journal",
  },
];

const audiences = [
  {
    icon: GraduationCap,
    label: "Postgraduate Students",
    description: "MSc and PhD candidates in agricultural and biological sciences.",
  },
  {
    icon: Microscope,
    label: "Researchers",
    description: "Scientists conducting field trials and multi-environment experiments.",
  },
  {
    icon: Users,
    label: "Lecturers",
    description: "University faculty teaching statistics and experimental design.",
  },
  {
    icon: Briefcase,
    label: "Professionals",
    description: "Agronomists, breeders, and consultants in industry and NGOs.",
  },
];

export default function Index() {
  return (
    <Layout>
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-primary text-primary-foreground">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAxOGMzLjMxNCAwIDYgMi42ODYgNiA2cy0yLjY4NiA2LTYgNi02LTIuNjg2LTYtNiAyLjY4Ni02IDYtNiIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpIiBzdHJva2Utd2lkdGg9IjIiLz48L2c+PC9zdmc+')] opacity-30"></div>
        <div className="container-wide relative">
          <div className="py-20 md:py-32 lg:py-40">
            <div className="max-w-4xl">
              <h1 className="font-serif text-4xl md:text-5xl lg:text-6xl font-bold leading-tight mb-4 animate-fade-in-up">
                Field-to-Insight Academy
              </h1>

              <p className="text-2xl md:text-3xl text-accent font-semibold mb-4 animate-fade-in-up animation-delay-100">
                Advancing agricultural research through data science, AI, and scientific innovation.
              </p>

              <p className="text-lg text-primary-foreground/80 leading-relaxed mb-6 animate-fade-in-up animation-delay-100">
                An AI-powered academy and ecosystem for agricultural research, training, and data-driven insight.
              </p>

              <p className="text-lg text-primary-foreground/70 font-medium tracking-wider mb-10 animate-fade-in-up animation-delay-100">
                Training • Intelligence • Research • Innovation
              </p>

              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-accent/15 border border-accent/30 text-accent text-xs font-semibold uppercase tracking-[0.18em] mb-5 animate-fade-in-up animation-delay-100">
                <Sparkles className="w-3.5 h-3.5" />
                Now Open · Cohort 3
              </div>

              <div className="flex flex-col sm:flex-row gap-3 animate-fade-in-up animation-delay-200">
                <Button variant="hero" size="lg" asChild>
                  <Link to="/cohort3-registration">
                    Register for Cohort 3
                    <ArrowRight className="w-5 h-5" />
                  </Link>
                </Button>
                <Button variant="hero-outline" size="lg" asChild>
                  <Link to="/about">Explore FIA</Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Programme — Cohort 3 */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-10">
              <span className="inline-block text-xs font-semibold uppercase tracking-[0.22em] text-accent mb-3">
                Featured Programme
              </span>
              <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-3">
                FIA–ADAP Foundations Cohort 3
              </h2>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
                A 6-week foundation programme in Agricultural Data Analytics, Research Thinking,
                Statistical Reasoning, and AI-Assisted Agricultural Research.
              </p>
            </div>

            <div className="card-elevated p-8 md:p-10 border-l-4 border-accent">
              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
                {[
                  { icon: Brain, label: "Research Thinking" },
                  { icon: ClipboardList, label: "Experimental Design" },
                  { icon: Sigma, label: "Statistical Reasoning" },
                  { icon: Sparkles, label: "AI-Assisted Analytics" },
                ].map((p) => (
                  <div key={p.label} className="flex flex-col items-start gap-3 p-4 rounded-lg bg-muted/50">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                      <p.icon className="w-5 h-5 text-primary" />
                    </div>
                    <p className="font-serif text-sm font-semibold text-foreground">{p.label}</p>
                  </div>
                ))}
              </div>

              <div className="grid sm:grid-cols-3 gap-4 mb-8 text-sm">
                <div className="flex items-start gap-3">
                  <Calendar className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-semibold text-foreground">Starts Friday</p>
                    <p className="text-muted-foreground">29 May 2026</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Clock className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-semibold text-foreground">Fridays & Saturdays</p>
                    <p className="text-muted-foreground">7:00 PM – 9:00 PM</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Video className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-semibold text-foreground">Online</p>
                    <p className="text-muted-foreground">Live Zoom Sessions</p>
                  </div>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pt-6 border-t border-border">
                <p className="text-sm font-medium text-muted-foreground">
                  Limited Cohort · Maximum 25 Participants
                </p>
                <Button variant="default" asChild>
                  <Link to="/cohort3-registration">
                    Register Now
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Ecosystem Section */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              The Field-to-Insight Ecosystem
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              An integrated platform for agricultural research, data science, training, and innovation.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {ecosystem.map((item) => (
              <Link
                key={item.title}
                to={item.link}
                className="card-elevated p-8 text-center group hover:border-primary transition-colors"
              >
                <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6 group-hover:bg-primary/20 transition-colors">
                  <item.icon className="w-7 h-7 text-primary" />
                </div>
                <h3 className="font-serif text-xl font-semibold text-foreground mb-3 group-hover:text-primary transition-colors">
                  {item.title}
                </h3>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {item.description}
                </p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* VivaSense Highlight */}
      <section className="section-padding bg-muted">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <div className="w-16 h-16 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <BarChart3 className="w-8 h-8 text-primary" />
            </div>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              VivaSense
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed mb-8">
              AI-Powered Agricultural Data Analysis and Scientific Interpretation.
              Upload your field data and receive instant statistical analysis, interpretation,
              and reporting support.
            </p>
            <Button variant="default" size="lg" asChild>
              <Link to="/vivasense">
                Explore VivaSense
                <ArrowRight className="w-5 h-5" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Mission */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Target className="w-7 h-7 text-primary" />
            </div>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
              Our Mission
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Field-to-Insight Academy exists to bridge the gap between raw
              field data and defensible research insight. Through AI-powered
              tools, competence-based training, and open-access publishing, we
              equip agricultural researchers with the statistical literacy,
              analytical confidence, and decision-making skills needed to
              advance food security and agricultural innovation worldwide.
            </p>
          </div>
        </div>
      </section>

      {/* Who We Serve */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              Who We Serve
            </h2>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto">
            {audiences.map((a) => (
              <div key={a.label} className="card-elevated p-6 text-center">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                  <a.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-serif text-lg font-semibold text-foreground mb-2">
                  {a.label}
                </h3>
                <p className="text-muted-foreground text-sm">{a.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* International Reach */}
      <section className="py-12 bg-primary text-primary-foreground">
        <div className="container-wide">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <Globe className="w-10 h-10 text-accent" />
              <div>
                <h3 className="font-serif text-xl font-semibold">
                  International Participation
                </h3>
                <p className="text-primary-foreground/80">
                  Participants from Nigeria, Niger, Sierra Leone, the United
                  Kingdom, and the United States
                </p>
              </div>
            </div>
            <Button variant="hero-outline" asChild>
              <Link to="/apply">Join the Next Cohort</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl md:text-4xl font-bold mb-6">
            Start Your Journey Today
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            Explore AI tutors, analyse your data, apply to a programme, or
            submit your manuscript to the Journal of Agricultural Innovation.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="hero" size="xl" asChild>
              <Link to="/ai-tutors">
                Explore AI Tutors
                <ArrowRight className="w-5 h-5" />
              </Link>
            </Button>
            <Button variant="hero-outline" size="xl" asChild>
              <Link to="/apply">Apply to Programme</Link>
            </Button>
            <Button variant="hero-outline" size="xl" asChild>
              <Link to="/journal/submit">Submit Manuscript</Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
}
