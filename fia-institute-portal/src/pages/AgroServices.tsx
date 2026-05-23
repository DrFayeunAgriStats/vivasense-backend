import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ArrowRight,
  Sprout,
  TreePine,
  TrendingUp,
  MapPin,
  Landmark,
  Leaf,
  FlaskConical,
  Warehouse,
  Package,
  BarChart3,
  ClipboardList,
  CreditCard,
  CheckCircle2,
  Ban,
  ShieldCheck,
  FileText,
  MessageSquare,
  Search,
  CreditCard as CreditCardIcon,
  Handshake,
} from "lucide-react";

const CONSULTATION_LINK = "#book-consultation";

const services = [
  { icon: Sprout, title: "General Farm Advisory" },
  { icon: ClipboardList, title: "Farm Establishment Planning" },
  { icon: TreePine, title: "Plantation Development", subtitle: "Oil Palm, Cocoa, Plantain, Banana, Horticultural Crops" },
  { icon: Landmark, title: "Farmland Acquisition Support" },
  { icon: MapPin, title: "Land Suitability Assessment" },
  { icon: Leaf, title: "Environmental Impact Assessment", subtitle: "EIA" },
  { icon: FlaskConical, title: "Soil Testing & Interpretation" },
  { icon: Package, title: "Farm Input Recommendation" },
  { icon: BarChart3, title: "Agribusiness Planning & Feasibility Studies" },
  { icon: Warehouse, title: "Greenhouse Production Advisory" },
  { icon: ClipboardList, title: "Data Analysis & Research Advisory" },
];

const steps = [
  { icon: FileText, label: "Client submits service request using the form" },
  { icon: Search, label: "Our team conducts free initial screening" },
  { icon: MessageSquare, label: "Client receives feedback and service recommendation" },
  { icon: CreditCardIcon, label: "Payment details are shared (if professional service is required)" },
  { icon: Handshake, label: "Consultation or project engagement begins" },
];

const fees = [
  { service: "General Farm Advisory", fee: "₦10,000" },
  { service: "Farm Establishment Planning", fee: "₦20,000" },
  { service: "Plantation Development Advisory", fee: "₦25,000" },
  { service: "Land Suitability Assessment", fee: "₦20,000" },
  { service: "Farm Expansion & Optimization", fee: "₦20,000" },
  { service: "Farm Input Recommendation", fee: "₦10,000" },
  { service: "Data Analysis & Research Advisory", fee: "₦15,000" },
  { service: "Agribusiness Planning", fee: "₦20,000" },
  { service: "Greenhouse Production Advisory", fee: "₦20,000" },
  { service: "Environmental Impact Assessment (Preliminary)", fee: "₦30,000" },
];

const consultationPolicies = [
  "All professional consultations are non-refundable",
  "Fees cover expert review, advisory input, and documentation",
  "Payment is required before formal engagement",
  "Free initial screening is available for every request",
];

const paymentPolicies = [
  { icon: CheckCircle2, text: "Initial screening is free" },
  { icon: ShieldCheck, text: "Fees apply only to professional consulting and implementation services" },
  { icon: CreditCard, text: "Payment must be confirmed before service begins" },
  { icon: Ban, text: "All payments are non-refundable once engagement starts" },
];

const whyWorkWithUs = [
  "Evidence-based agricultural solutions",
  "Experienced academic and field professionals",
  "Structured project planning and execution",
  "Transparent process and pricing",
  "Professional documentation and reporting",
];

export default function AgroServices() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-4">
              Agro-Services
            </h1>
            <p className="text-xl font-semibold text-primary-foreground/90 mb-4">
              Professional Agricultural Consulting, Farm Establishment &amp; Agribusiness Support
            </p>
            <p className="text-lg text-primary-foreground/80 leading-relaxed mb-3">
              Agro-Services provides structured and professional agricultural consulting, farm
              establishment planning, land suitability evaluation, environmental impact assessment,
              production optimization, and agribusiness advisory services for farmers, investors,
              researchers, and agripreneurs.
            </p>
            <p className="text-lg text-primary-foreground/80 leading-relaxed mb-8">
              We combine scientific expertise, field experience, and data-driven decision making
              to help clients move from idea to profitable implementation.
            </p>
            <div className="flex flex-wrap gap-4">
              <Button variant="hero" size="xl" asChild>
                <a href={CONSULTATION_LINK}>
                  Book Consultation
                  <ArrowRight className="w-5 h-5" />
                </a>
              </Button>
              <Button variant="hero-outline" size="xl" asChild>
                <a href={CONSULTATION_LINK}>
                  Request Agro-Service
                  <ArrowRight className="w-5 h-5" />
                </a>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Who We Are */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto space-y-6">
            <h2 className="font-serif text-3xl font-bold text-foreground">Who We Are</h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Agro-Services is the professional consulting and implementation outreach unit of
              Field-to-Insight Academy (FIA), operated under Able-Flourish Agro-Services Ltd.
            </p>
            <p className="text-lg text-muted-foreground leading-relaxed">
              We bridge research, innovation, and real-world farm implementation by delivering
              practical, science-based agricultural solutions.
            </p>
          </div>
        </div>
      </section>

      {/* What We Do */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <h2 className="font-serif text-3xl font-bold text-foreground text-center mb-12">
            What We Do
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {services.map((s, i) => (
              <Card key={`${s.title}-${i}`} className="border-border/60 hover:shadow-md transition-shadow">
                <CardContent className="p-6 flex items-start gap-4">
                  <div className="w-11 h-11 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <s.icon className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground leading-tight">{s.title}</h3>
                    {s.subtitle && (
                      <p className="text-sm text-muted-foreground mt-1">{s.subtitle}</p>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="section-padding">
        <div className="container-wide">
          <h2 className="font-serif text-3xl font-bold text-foreground text-center mb-12">
            How It Works
          </h2>
          <div className="max-w-4xl mx-auto grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {steps.map((step, i) => (
              <div key={step.label} className="flex flex-col items-center text-center gap-3">
                <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center relative">
                  <step.icon className="w-6 h-6 text-primary" />
                  <span className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center">
                    {i + 1}
                  </span>
                </div>
                <p className="text-sm font-medium text-foreground">{step.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Free Initial Screening */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto space-y-4">
            <h2 className="font-serif text-3xl font-bold text-foreground">
              Free Initial Screening
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              We offer a free initial screening of all requests submitted through the form.
              This helps us understand your needs and determine whether professional consultation
              or implementation support is required.
            </p>
            <p className="text-lg text-muted-foreground leading-relaxed">
              If professional service is needed, you will receive payment instructions before engagement.
            </p>
          </div>
        </div>
      </section>

      {/* Consultation Fee Policy */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto space-y-6">
            <h2 className="font-serif text-3xl font-bold text-foreground">
              Consultation Fee Policy
            </h2>
            <ul className="space-y-3">
              {consultationPolicies.map((item) => (
                <li key={item} className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary flex-shrink-0" />
                  <span className="text-muted-foreground">{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* Professional Service Fees */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-8">
              Professional Service Fees
            </h2>
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="font-semibold">Service</TableHead>
                    <TableHead className="font-semibold text-right">Fee</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {fees.map((row) => (
                    <TableRow key={row.service}>
                      <TableCell className="text-foreground">{row.service}</TableCell>
                      <TableCell className="text-right font-medium text-foreground">
                        {row.fee}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
            <p className="text-sm text-muted-foreground mt-4 italic">
              Large projects (farm establishment, plantation development, feasibility studies, and
              long-term advisory) are quoted based on scope after screening.
            </p>
          </div>
        </div>
      </section>

      {/* Request for Consultation — Google Form */}
      <section id="book-consultation" className="section-padding">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-4 text-center">
              Request for Agro-Services / Consultation
            </h2>
            <p className="text-center text-muted-foreground mb-8">
              Please complete the form below to request agro-services or consultation. Our team
              will review your submission and respond within 24–48 hours.
            </p>
            <div className="w-full flex justify-center">
              <iframe
                src="https://docs.google.com/forms/d/e/1FAIpQLSekL_A9CAzBNUpn6qFKHkbvnAW6UKQvBFsJIuQYE7Zn5Wy27Q/viewform?embedded=true"
                className="w-full border-0 rounded-lg"
                style={{ minHeight: "900px" }}
                title="Request for Agro-Services / Consultation"
                allowFullScreen
              >
                Loading…
              </iframe>
            </div>
            <div className="mt-8 p-6 bg-primary/5 rounded-lg border border-primary/10 text-center">
              <p className="text-foreground font-semibold mb-2">After Form Submission</p>
              <p className="text-muted-foreground">
                Thank you for submitting your request. Our team will review your information and
                contact you within 24–48 hours with next steps and payment details (if applicable).
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Payment Policy */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-8">
              Payment Policy
            </h2>
            <div className="space-y-4">
              {paymentPolicies.map((p) => (
                <div key={p.text} className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <p.icon className="w-5 h-5 text-primary" />
                  </div>
                  <p className="font-medium text-foreground">{p.text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Why Work With Us */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-8">
              Why Work With Us?
            </h2>
            <div className="grid sm:grid-cols-2 gap-5">
              {whyWorkWithUs.map((item) => (
                <div key={item} className="flex items-center gap-3">
                  <div className="w-2.5 h-2.5 rounded-full bg-primary flex-shrink-0" />
                  <span className="text-foreground font-medium">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold mb-6">
            Ready to Get Started?
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            Book your consultation today and let our experts guide your agricultural project
            from planning to execution.
          </p>
          <Button variant="hero" size="xl" asChild>
            <a href={CONSULTATION_LINK}>
              Book Consultation
              <ArrowRight className="w-5 h-5" />
            </a>
          </Button>
        </div>
      </section>
    </Layout>
  );
}
