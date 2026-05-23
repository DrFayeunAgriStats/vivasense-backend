import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  CheckCircle2,
  Clock,
  Calendar,
  Video,
  Users,
  BookOpen,
  Award,
  Target,
  XCircle,
} from "lucide-react";

const programDetails = [
  {
    icon: Clock,
    title: "Duration",
    value: "6 Weeks",
    description: "Intensive, focused learning",
  },
  {
    icon: Video,
    title: "Format",
    value: "12 Live Sessions",
    description: "Interactive Zoom classes",
  },
  {
    icon: Calendar,
    title: "Schedule",
    value: "2x Weekly",
    description: "WAT-friendly timing",
  },
  {
    icon: BookOpen,
    title: "Materials",
    value: "Full Access",
    description: "Recordings & resources",
  },
];

const whoShouldAttend = [
  {
    title: "MSc & PhD Students",
    description: "In Crop Science, Agronomy, Plant Breeding, Horticulture, Soil Science, and other agricultural disciplines who need to analyse experimental data for their thesis.",
  },
  {
    title: "Early-Career Lecturers",
    description: "Building their expertise in teaching and supervising agricultural research methodology.",
  },
  {
    title: "Researchers",
    description: "At universities, research institutes, and agricultural organizations working with field trial data.",
  },
  {
    title: "Agricultural Professionals",
    description: "In NGOs, agribusiness R&D units, and extension services who need data-driven decision making.",
  },
];

const achievements = [
  "Design appropriate experimental layouts (CRD, RCBD, Factorial, Split-plot)",
  "Conduct and interpret One-, Two- and Three-Way ANOVA with correct mean separation tests",
  "Perform simple and multiple linear correlation and regression analysis",
  "Analyse genotype × environment interactions using AMMI and GGE biplot",
  "Apply multivariate techniques (PCA, Cluster Analysis) to complex datasets",
  "Write clear, defensible results sections for your thesis or publications",
  "Confidently present and defend your methodology during supervision",
];

const whatThisProgramIsNot = [
  {
    title: "Not Button-Click Training",
    description: "We don't just show you which buttons to click. You'll understand the statistical logic behind every method.",
  },
  {
    title: "Not Attendance-Based Certification",
    description: "Certification requires demonstrating competence through practical assessments, not just showing up.",
  },
  {
    title: "Not Generic Statistics",
    description: "Every example, dataset, and case study is drawn from real agricultural and biological research contexts.",
  },
];

const pilotOutcomes = [
  "Participants reported improved confidence in data analysis",
  "Participants successfully applied techniques to their own research within weeks",
  "Positive feedback on the agriculture-specific examples and datasets",
  "High engagement during live sessions with practical problem-solving",
];

export default function Program() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-foreground/10 text-sm font-medium mb-6">
              <Award className="w-4 h-4" />
              <span>FIA–ADAP™ Foundations</span>
            </div>
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Program Overview
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
              A comprehensive 6-week program designed to transform how you approach 
              agricultural data analysis—from experimental design to thesis defense.
            </p>
            <p className="text-primary-foreground/70 text-sm italic">
              FIA-ADAP™ Foundations is the entry-level programme within the FIA-ADAP™ 
              programme family at Field-to-Insight Academy.
            </p>
          </div>
        </div>
      </section>

      {/* Program Details Grid */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
            {programDetails.map((detail) => (
              <div key={detail.title} className="card-elevated p-6 text-center">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                  <detail.icon className="w-6 h-6 text-primary" />
                </div>
                <div className="text-sm text-muted-foreground mb-1">{detail.title}</div>
                <div className="text-2xl font-bold text-foreground mb-1">{detail.value}</div>
                <div className="text-sm text-muted-foreground">{detail.description}</div>
              </div>
            ))}
          </div>

          <div className="grid lg:grid-cols-2 gap-12">
            {/* Program Description */}
            <div>
              <h2 className="font-serif text-3xl font-bold text-foreground mb-6">
                About FIA–ADAP™ Foundations
              </h2>
              <div className="prose prose-lg text-muted-foreground space-y-4">
                <p>
                  The Agricultural Data Analysis Program (ADAP) Foundations is the flagship 
                  training program of Field-to-Insight Academy. It addresses a critical gap 
                  in postgraduate agricultural education: the disconnect between running 
                  statistical analyses and truly understanding the methodology.
                </p>
                <p>
                  Unlike generic statistics courses, every concept is taught using 
                  agricultural examples. You'll work with real datasets from crop trials, 
                  plant breeding programs, and agronomic experiments—the exact type of 
                  data you'll encounter in your own research.
                </p>
                <p>
                  The program emphasizes competence over attendance. You won't just listen 
                  to lectures; you'll actively apply each technique, receive feedback, and 
                  demonstrate your understanding through practical assessments.
                </p>
              </div>
            </div>

            {/* Format Details */}
            <div className="bg-muted rounded-2xl p-8">
              <h3 className="font-serif text-2xl font-bold text-foreground mb-6">
                Program Format
              </h3>
              <div className="space-y-6">
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Live Sessions</h4>
                  <p className="text-muted-foreground">
                    12 interactive Zoom sessions (approximately 2 hours each), scheduled 
                    twice weekly at times suitable for West African Time zones.
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Recordings</h4>
                  <p className="text-muted-foreground">
                    All sessions are recorded and made available within 24 hours. 
                    Perfect for revision or if you miss a class.
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Practical Work</h4>
                  <p className="text-muted-foreground">
                    Weekly practical exercises using R Studio, SAS Studio, GRAPES, PAST, 
                    OPSTAT, and Excel. Work with provided datasets or apply to your own research.
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Assessment-Led Certification</h4>
                  <p className="text-muted-foreground">
                    Certificate of Competence requires demonstrating practical ability 
                    through assessments, not just attendance.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* What This Program Is NOT */}
      <section className="section-padding bg-destructive/5">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              What This Program Is NOT
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              FIA–ADAP™ Foundations is a serious, competence-based training program. 
              We believe in clarity about what we offer.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            {whatThisProgramIsNot.map((item) => (
              <div key={item.title} className="bg-card rounded-xl p-6 border border-border">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-lg bg-destructive/10 flex items-center justify-center">
                    <XCircle className="w-5 h-5 text-destructive" />
                  </div>
                  <h3 className="font-semibold text-foreground">{item.title}</h3>
                </div>
                <p className="text-muted-foreground text-sm">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Who Should Attend */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              Who Should Attend?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              FIA–ADAP™ Foundations is designed for anyone who works with 
              agricultural and biological experimental data.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            {whoShouldAttend.map((item) => (
              <div key={item.title} className="bg-card rounded-xl p-6 border border-border">
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <Users className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground mb-2">{item.title}</h3>
                    <p className="text-muted-foreground text-sm">{item.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* What You'll Achieve */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
                What You'll Achieve
              </h2>
              <p className="text-lg text-muted-foreground mb-8">
                By the end of the program, you will be able to confidently:
              </p>
              <div className="space-y-4">
                {achievements.map((item, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-foreground">{item}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="bg-primary rounded-2xl p-8 text-primary-foreground">
              <div className="flex items-center gap-3 mb-6">
                <Target className="w-8 h-8 text-accent" />
                <h3 className="font-serif text-2xl font-bold">Pilot Training Outcomes</h3>
              </div>
              <p className="text-primary-foreground/85 mb-6">
                Results from our initial cohorts demonstrate the program's effectiveness:
              </p>
              <div className="space-y-4">
                {pilotOutcomes.map((outcome, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-accent flex-shrink-0 mt-0.5" />
                    <span className="text-primary-foreground/90">{outcome}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-muted">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            Review our detailed curriculum or apply now to secure your place 
            in the next cohort.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="default" size="lg" asChild>
              <Link to="/curriculum">
                View Curriculum
                <ArrowRight className="w-4 h-4" />
              </Link>
            </Button>
            <Button variant="gold" size="lg" asChild>
              <Link to="/apply">Apply Now</Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
}
