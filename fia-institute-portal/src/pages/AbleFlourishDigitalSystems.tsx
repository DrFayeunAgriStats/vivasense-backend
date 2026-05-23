import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  Users,
  CalendarDays,
  Globe,
  LayoutDashboard,
  CheckCircle2,
  Building2,
  GraduationCap,
  Landmark,
  Briefcase,
  Shield,
} from "lucide-react";

const solutions = [
  {
    icon: Users,
    title: "Digital Staff Directory",
    description:
      "Searchable staff directory for organizations with departments, contacts, and key offices.",
  },
  {
    icon: CalendarDays,
    title: "Lecture Timetable Platform",
    description:
      "University timetable system showing courses, lecturers, venues, and personalized student timetable.",
  },
  {
    icon: Globe,
    title: "Internal Web Applications",
    description:
      "Secure internal tools for organizational workflows and operations.",
  },
  {
    icon: LayoutDashboard,
    title: "Dashboards & Data Tools",
    description:
      "Administrative dashboards and data visibility tools for decision making.",
  },
];

const reasons = [
  "Simple and easy to use",
  "Secure and scalable",
  "Designed for institutions",
  "Mobile friendly",
  "Customizable",
];

const audiences = [
  { icon: GraduationCap, label: "Universities" },
  { icon: Shield, label: "NGOs" },
  { icon: Building2, label: "Research institutions" },
  { icon: Briefcase, label: "Corporate organizations" },
  { icon: Landmark, label: "Government agencies" },
];

export default function AbleFlourishDigitalSystems() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-4">
              Able-Flourish Digital Systems
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
              Secure, scalable systems that help organizations flourish
            </p>
            <p className="text-primary-foreground/70 leading-relaxed mb-8">
              Able-Flourish Digital Systems builds practical digital tools for
              universities, NGOs, and organizations. Our solutions improve
              efficiency, communication, and access to information.
            </p>
            <div className="flex flex-wrap gap-4">
              <Button variant="hero" size="xl" asChild>
                <a href="#solutions">
                  Explore Solutions
                  <ArrowRight className="w-5 h-5" />
                </a>
              </Button>
              <Button
                variant="outline"
                size="xl"
                className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10"
                asChild
              >
                <Link to="/contact">Request Demo</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* What We Build */}
      <section id="solutions" className="section-padding">
        <div className="container-wide">
          <div className="text-center mb-12">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              What We Build
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Practical digital tools designed to meet real institutional needs.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto">
            {solutions.map((s) => (
              <div
                key={s.title}
                className="card-elevated p-6 hover:border-primary transition-colors"
              >
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <s.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-serif text-lg font-semibold text-foreground mb-2">
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

      {/* Why Organizations Choose Us */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-8 text-center">
              Why Organizations Choose Us
            </h2>
            <div className="grid sm:grid-cols-2 gap-4">
              {reasons.map((r) => (
                <div key={r} className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                  <span className="text-foreground">{r}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Featured Solution */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto">
            <div className="card-elevated overflow-hidden md:flex">
              <div className="md:w-1/2 bg-muted flex items-center justify-center p-8 min-h-[240px]">
                <div className="text-center text-muted-foreground">
                  <LayoutDashboard className="w-16 h-16 mx-auto mb-3 opacity-40" />
                  <p className="text-sm italic">Demo screenshot</p>
                </div>
              </div>
              <div className="md:w-1/2 p-8 flex flex-col justify-center">
                <h3 className="font-serif text-2xl font-bold text-foreground mb-3">
                  Digital Staff Directory
                </h3>
                <p className="text-muted-foreground leading-relaxed mb-6">
                  A secure internal directory that helps organizations quickly
                  find staff contacts, departments, and office holders.
                </p>
                <div>
                  <Button variant="default" asChild>
                    <Link to="/contact">
                      View Demo
                      <ArrowRight className="w-4 h-4" />
                    </Link>
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Who We Serve */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="text-center mb-10">
            <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
              Who We Serve
            </h2>
          </div>
          <div className="flex flex-wrap justify-center gap-6 max-w-3xl mx-auto">
            {audiences.map((a) => (
              <div
                key={a.label}
                className="flex flex-col items-center gap-2 w-32"
              >
                <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center">
                  <a.icon className="w-6 h-6 text-primary" />
                </div>
                <span className="text-sm font-medium text-foreground text-center">
                  {a.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold mb-6">
            Need a Digital Solution for Your Organization?
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            Get in touch to discuss how Able-Flourish Digital Systems can
            support your institution with practical, secure digital tools.
          </p>
          <Button variant="hero" size="xl" asChild>
            <Link to="/contact">
              Request Consultation
              <ArrowRight className="w-5 h-5" />
            </Link>
          </Button>
        </div>
      </section>
    </Layout>
  );
}
