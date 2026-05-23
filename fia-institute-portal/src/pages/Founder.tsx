import { Layout } from "@/components/layout/Layout";
import {
  GraduationCap,
  Microscope,
  Brain,
  Eye,
  BookOpen,
  Sprout,
  BarChart3,
  Lightbulb,
  Users,
  FileSearch,
  FlaskConical,
} from "lucide-react";

const academicBackground = [
  "PhD in Plant Breeding and Genetics",
  "MSc in Crop Production and Protection",
  "BSc in Crop Science and Horticulture",
  "Postdoctoral Research in Quantitative Genetics",
];

const researchInterests = [
  "Plant Breeding and Genetics",
  "Quantitative and Biometrical Genetics",
  "Genotype × Environment Interaction",
  "AI-Assisted Agricultural Research",
  "Experimental Design and Data Analysis",
  "Agricultural Data Science",
];

const platforms = [
  {
    icon: BarChart3,
    name: "VivaSense",
    description: "AI-powered agricultural data analysis and scientific interpretation platform.",
  },
  {
    icon: Brain,
    name: "AI Tutors",
    description: "Intelligent tutoring systems for plant genetics, biometrical genetics, and research writing.",
  },
  {
    icon: BookOpen,
    name: "Journal of Agricultural Innovation",
    description: "Open-access, peer-reviewed academic journal for agricultural science.",
  },
  {
    icon: Sprout,
    name: "Agro-Services",
    description: "Professional consulting for farm establishment, land evaluation, and agribusiness planning.",
  },
  {
    icon: FileSearch,
    name: "Manuscript Review System",
    description: "AI-assisted academic manuscript evaluation and publication readiness platform.",
  },
  {
    icon: FlaskConical,
    name: "Data Analysis Services",
    description: "Professional agricultural data analysis including experimental design, G×E analysis, and heritability studies.",
  },
];

export default function Founder() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="grid lg:grid-cols-5 gap-12 items-center">
            <div className="lg:col-span-3">
              <h1 className="font-serif text-4xl md:text-5xl font-bold mb-4">
                Dr. Fayeun Lawrence Stephen
              </h1>
              <p className="text-lg text-accent font-semibold mb-4">
                Plant Breeder • Data Scientist • AI Systems Developer • Founder
              </p>
              <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
                Designing intelligent platforms for agricultural research, data science,
                and scientific decision-making.
              </p>
              <p className="text-primary-foreground/70 leading-relaxed">
                Founder of Field-to-Insight Academy, building AI-powered systems that
                transform agricultural data into defensible scientific insight.
              </p>
            </div>
            <div className="lg:col-span-2 flex justify-center">
              <div className="w-64 h-64 md:w-80 md:h-80 rounded-2xl bg-primary-foreground/10 border border-primary-foreground/20 flex items-center justify-center">
                <GraduationCap className="w-24 h-24 text-primary-foreground/30" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Academic Background */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center gap-4 mb-8">
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                <GraduationCap className="w-6 h-6 text-primary" />
              </div>
              <h2 className="font-serif text-3xl font-bold text-foreground">
                Academic Background
              </h2>
            </div>
            <div className="space-y-3">
              {academicBackground.map((item, i) => (
                <div key={i} className="flex items-start gap-4 bg-muted rounded-xl p-5 border border-border">
                  <div className="w-2 h-2 rounded-full bg-primary mt-2.5 flex-shrink-0" />
                  <p className="text-foreground">{item}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Research Interests */}
      <section className="section-padding bg-muted">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center gap-4 mb-8">
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                <Microscope className="w-6 h-6 text-primary" />
              </div>
              <h2 className="font-serif text-3xl font-bold text-foreground">
                Research Interests
              </h2>
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              {researchInterests.map((item, i) => (
                <div key={i} className="flex items-start gap-4 bg-card rounded-xl p-5 border border-border">
                  <div className="w-2 h-2 rounded-full bg-primary mt-2.5 flex-shrink-0" />
                  <p className="text-foreground">{item}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* AI Platform Development */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-10">
              <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                <Brain className="w-7 h-7 text-primary" />
              </div>
              <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
                AI Platform Development
              </h2>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Platforms designed and built under Field-to-Insight Academy to support
                agricultural research, education, and innovation.
              </p>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {platforms.map((p) => (
                <div key={p.name} className="card-elevated p-6">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <p.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="font-serif text-xl font-semibold text-foreground mb-2">{p.name}</h3>
                  <p className="text-muted-foreground text-sm leading-relaxed">{p.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Leadership Vision */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Eye className="w-7 h-7 text-primary" />
            </div>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
              Leadership Vision
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed mb-4">
              To build a global ecosystem where agricultural researchers can design rigorous
              experiments, analyse data with confidence, and defend their findings with clarity
              — powered by intelligent tools and competence-based education.
            </p>
            <p className="text-muted-foreground leading-relaxed">
              Field-to-Insight Academy represents a vision where technology amplifies scientific
              rigour, not replaces it. Every platform, tutor, and programme is designed with one
              goal: to move researchers from data to defensible insight.
            </p>
          </div>
        </div>
      </section>

      {/* Publications Placeholder */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <BookOpen className="w-7 h-7 text-primary" />
            </div>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
              Publications
            </h2>
            <div className="bg-muted rounded-xl p-8 border border-border">
              <p className="text-muted-foreground">
                A curated list of peer-reviewed publications and conference proceedings
                will be available here soon.
              </p>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
