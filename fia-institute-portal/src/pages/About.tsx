import { Layout } from "@/components/layout/Layout";
import {
  Target,
  Layers,
  Lightbulb,
  Users,
  Telescope,
  GraduationCap,
  Microscope,
  Briefcase,
  Building2,
  FlaskConical,
} from "lucide-react";

const whoWeServe = [
  "Final-year undergraduate students in agricultural and biological sciences",
  "MSc and PhD students",
  "Early-career lecturers and researchers",
  "Plant breeders, agronomists, and crop scientists",
  "Agricultural professionals seeking analytical competence",
  "Institutions building data-driven agricultural programmes",
];

const philosophy = [
  "Concept-first before software",
  "Design before analysis",
  "Interpretation before reporting",
  "Evidence before opinion",
];

const whatFiaIs = [
  "A competence-based agricultural data academy",
  "A digital learning platform with AI-powered tutors",
  "A statistical intelligence innovation hub (VivaSense)",
  "A developing academic publishing platform (Journal of Agricultural Innovation – JAI)",
];

export default function About() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              About Field-to-Insight Academy
            </h1>
            <p className="text-2xl md:text-3xl text-accent font-semibold mb-6">
              Advancing agricultural research through data science, AI, and scientific innovation.
            </p>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Field-to-Insight Academy (FIA) is a competence-based institute
              dedicated to strengthening analytical thinking, experimental
              design, and scientific interpretation in agricultural research.
            </p>
          </div>
        </div>
      </section>

      {/* Our Mission */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Target className="w-7 h-7 text-primary" />
            </div>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
              Our Mission
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed mb-6">
              To build analytical competence in agricultural scientists by
              equipping them to design rigorous experiments, analyse data
              correctly, and defend their research with confidence.
            </p>
            <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Telescope className="w-7 h-7 text-primary" />
            </div>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
              Our Vision
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              To build a global ecosystem for agricultural research intelligence and innovation.
            </p>
          </div>
        </div>
      </section>

      {/* What FIA Is Today */}
      <section className="section-padding bg-muted">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-10">
              <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                <Layers className="w-7 h-7 text-primary" />
              </div>
              <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
                What FIA Is Today
              </h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                FIA is a growing digital institute that integrates agricultural
                science, statistical reasoning, and artificial intelligence into
                a structured learning ecosystem.
              </p>
              <p className="text-lg text-muted-foreground leading-relaxed mt-4">
                We are no longer just a training programme. FIA now operates as:
              </p>
            </div>
            <div className="space-y-4">
              {whatFiaIs.map((item, i) => (
                <div
                  key={i}
                  className="flex items-start gap-4 bg-card rounded-xl p-5 border border-border"
                >
                  <div className="w-2 h-2 rounded-full bg-primary mt-2.5 flex-shrink-0" />
                  <p className="text-foreground">{item}</p>
                </div>
              ))}
            </div>
            <p className="text-lg text-muted-foreground leading-relaxed mt-8 text-center">
              FIA is positioned as a one-stop intellectual support system for
              agricultural researchers.
            </p>
          </div>
        </div>
      </section>

      {/* Educational Philosophy */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-10">
              <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                <Lightbulb className="w-7 h-7 text-primary" />
              </div>
              <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
                Our Educational Philosophy
              </h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                We believe that running software is not the same as
                understanding science.
              </p>
              <p className="text-lg text-muted-foreground leading-relaxed mt-2">
                Our approach is:
              </p>
            </div>
            <div className="grid sm:grid-cols-2 gap-4">
              {philosophy.map((item, i) => (
                <div
                  key={i}
                  className="bg-muted rounded-xl p-6 text-center border border-border"
                >
                  <p className="text-foreground font-semibold text-lg">
                    {item}
                  </p>
                </div>
              ))}
            </div>
            <p className="text-lg text-muted-foreground leading-relaxed mt-8 text-center">
              We train researchers to think statistically, not just compute
              results.
            </p>
          </div>
        </div>
      </section>

      {/* Who We Serve */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-10">
              <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                <Users className="w-7 h-7 text-primary" />
              </div>
              <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
                Who We Serve
              </h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Field-to-Insight Academy serves:
              </p>
            </div>
            <div className="space-y-3">
              {whoWeServe.map((item, i) => (
                <div
                  key={i}
                  className="flex items-start gap-4 bg-card rounded-xl p-5 border border-border"
                >
                  <div className="w-2 h-2 rounded-full bg-primary mt-2.5 flex-shrink-0" />
                  <p className="text-foreground">{item}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Long-Term Vision */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Telescope className="w-7 h-7 text-primary" />
            </div>
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-6">
              Our Long-Term Vision
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              To become Africa's leading competence-based institute for
              agricultural data science, experimental design, and AI-supported
              research education.
            </p>
            <p className="text-lg text-muted-foreground leading-relaxed mt-4">
              We envision a generation of researchers who can confidently defend
              their work before supervisors, journal reviewers, funding
              agencies, and international audiences.
            </p>
          </div>
        </div>
      </section>
    </Layout>
  );
}
