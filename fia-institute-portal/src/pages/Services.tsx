import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  ArrowRight,
  BookOpen,
  Monitor,
  Bot,
  BarChart3,
  Users,
  Sprout,
} from "lucide-react";

const services = [
  {
    icon: BookOpen,
    title: "Curriculum Development & Review",
    description:
      "We support institutions in designing, reviewing, and upgrading undergraduate and postgraduate curricula to ensure alignment with modern scientific methods, experimental design principles, and data analysis requirements.",
    items: [
      "Curriculum design and review",
      "Course outline development",
      "Learning outcomes mapping",
      "Practical and laboratory activity design",
      "Integration of data analysis and AI components into curricula",
    ],
  },
  {
    icon: Monitor,
    title: "Course Platform Development",
    description:
      "We design and develop custom digital learning platforms for individual courses and academic programmes, similar to the CSP409 course platform.",
    subtitle: "These platforms may include:",
    items: [
      "Course-specific AI tutors",
      "Interactive learning modules",
      "Embedded datasets and exercises",
      "Guided practice environments",
    ],
    footer:
      "Designed to complement classroom teaching and support independent learning.",
  },
  {
    icon: Bot,
    title: "Custom AI Tutors for Courses and Programmes",
    description:
      "We build customized AI tutors trained on course-specific content to support students with explanations, examples, and guided learning in agricultural and biological sciences.",
    subtitle: "Examples include:",
    items: [
      "Plant Genetics Mastery Tutor",
      "Course-based AI tutors",
      "Programme-level AI learning assistants",
    ],
  },
  {
    icon: BarChart3,
    title: "Research & Data Analysis Systems",
    description:
      "We develop and deploy digital tools for agricultural data analysis, automated reporting, and statistical interpretation.",
    subtitle: "Example:",
    items: ["VivaSense Statistical Intelligence Engine"],
  },
  {
    icon: Users,
    title: "Training & Capacity Building",
    description:
      "We provide tailored training for staff and students in:",
    items: [
      "Experimental design",
      "Statistical data analysis",
      "Agricultural data interpretation",
      "AI-assisted research workflows",
    ],
  },
];

export default function Services() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Our Services
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Academic, Digital, and AI-Powered Solutions for Agricultural
              Education and Research
            </p>
          </div>
        </div>
      </section>

      {/* Introduction */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto space-y-4">
            <p className="text-lg text-muted-foreground leading-relaxed">
              Field-to-Insight Academy (FIA) provides specialized academic and
              digital services to strengthen teaching, research, and data-driven
              decision-making in agricultural and biological sciences.
            </p>
            <p className="text-lg text-muted-foreground leading-relaxed">
              We work with individuals, departments, and institutions to design
              learning systems, develop digital course platforms, and deploy
              AI-powered tools for research support.
            </p>
          </div>
        </div>
      </section>

      {/* Service Sections */}
      {services.map((service, index) => (
        <section
          key={service.title}
          className={`section-padding ${index % 2 === 1 ? "bg-secondary" : ""}`}
        >
          <div className="container-wide">
            <div className="max-w-3xl mx-auto">
              <div className="flex items-start gap-4 mb-6">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <service.icon className="w-6 h-6 text-primary" />
                </div>
                <h2 className="font-serif text-3xl font-bold text-foreground">
                  {service.title}
                </h2>
              </div>

              <p className="text-muted-foreground leading-relaxed mb-6">
                {service.description}
              </p>

              {service.subtitle && (
                <p className="font-medium text-foreground mb-3">
                  {service.subtitle}
                </p>
              )}

              <ul className="space-y-3 mb-6">
                {service.items.map((item) => (
                  <li key={item} className="flex items-start gap-3">
                    <div className="w-2 h-2 rounded-full bg-primary flex-shrink-0 mt-2" />
                    <span className="text-muted-foreground">{item}</span>
                  </li>
                ))}
              </ul>

              {service.footer && (
                <p className="text-muted-foreground italic">
                  {service.footer}
                </p>
              )}
            </div>
          </div>
        </section>
      ))}

      {/* Agro-Services & Digital Systems Links */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto space-y-6">
            <Card className="overflow-hidden">
              <CardContent className="p-8 flex flex-col sm:flex-row items-start sm:items-center gap-6">
                <div className="w-14 h-14 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Sprout className="w-7 h-7 text-primary" />
                </div>
                <div className="flex-1">
                  <h3 className="font-serif text-2xl font-bold text-foreground mb-2">
                    Agro-Services
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Professional agricultural consulting, farm establishment, and agribusiness
                    advisory services by Able-Flourish Agro-Services Ltd.
                  </p>
                </div>
                <Button variant="default" asChild>
                  <Link to="/services/agro-services">
                    Learn More
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="overflow-hidden">
              <CardContent className="p-8 flex flex-col sm:flex-row items-start sm:items-center gap-6">
                <div className="w-14 h-14 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Monitor className="w-7 h-7 text-primary" />
                </div>
                <div className="flex-1">
                  <h3 className="font-serif text-2xl font-bold text-foreground mb-2">
                    Able-Flourish Digital Systems
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    We design and deploy secure, scalable digital systems for organizations,
                    including staff directories, lecture timetable platforms, dashboards, and
                    internal web applications.
                  </p>
                </div>
                <Button variant="default" asChild>
                  <Link to="/services/able-flourish-digital-systems">
                    Explore Digital Systems
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold mb-6">
            Interested in Our Services?
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            Get in touch to discuss how FIA can support your institution,
            department, or research programme.
          </p>
          <Button variant="hero" size="xl" asChild>
            <Link to="/contact">
              Request a Service
              <ArrowRight className="w-5 h-5" />
            </Link>
          </Button>
        </div>
      </section>
    </Layout>
  );
}
