import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  CheckCircle2,
  Award,
  Video,
  BookOpen,
  Users,
  Clock,
  FileCheck,
} from "lucide-react";

const included = [
  {
    icon: Video,
    title: "12 Live Zoom Sessions",
    description: "Interactive classes with real-time Q&A (approximately 2 hours each)",
  },
  {
    icon: BookOpen,
    title: "Session Recordings",
    description: "Full access to recordings for revision or if you miss a class",
  },
  {
    icon: FileCheck,
    title: "Practical Exercises",
    description: "Weekly hands-on exercises with agricultural datasets",
  },
  {
    icon: Users,
    title: "Q&A Support",
    description: "Direct access to ask questions during live sessions",
  },
  {
    icon: Clock,
    title: "6 Weeks of Training",
    description: "Intensive, focused learning with structured progression",
  },
  {
    icon: Award,
    title: "Certificate of Competence",
    description: "Assessment-based certification upon successful completion",
  },
];

const valuePoints = [
  "Learn from an experienced university lecturer and plant breeding researcher",
  "Agriculture-specific examples and datasets—not generic statistics",
  "Competence-based program—you'll actually know what you're doing",
  "WAT-friendly scheduling for African participants",
  "Practical skills you can apply immediately to your research",
  "Join a community of agricultural researchers across Africa",
];

export default function Pricing() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Investment
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
              An investment in your research competence that pays dividends 
              throughout your academic and professional career.
            </p>
            <p className="text-primary-foreground/70 text-sm italic">
              FIA-ADAP™ Foundations is the entry-level programme within the FIA-ADAP™ 
              programme family at Field-to-Insight Academy.
            </p>
          </div>
        </div>
      </section>

      {/* Pricing Card */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto">
            <div className="card-elevated overflow-hidden">
              {/* Header */}
              <div className="bg-primary text-primary-foreground p-8 text-center">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-foreground/10 text-sm font-medium mb-4">
                  <Award className="w-4 h-4" />
                  <span>Flagship Program</span>
                </div>
                <h2 className="font-serif text-3xl font-bold mb-2">
                  FIA–ADAP™ Foundations
                </h2>
                <p className="text-primary-foreground/80">
                  Agricultural Data Analysis Program
                </p>
              </div>

              {/* Content */}
              <div className="p-8">
                {/* Pricing Tiers */}
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                  {/* Nigerian Pricing */}
                  <div className="bg-muted rounded-xl p-6 text-center">
                    <p className="text-sm text-muted-foreground mb-2">For Nigerian Participants</p>
                    <p className="text-4xl font-bold text-foreground mb-2">₦150,000</p>
                    <div className="mt-4 p-3 bg-accent/50 rounded-lg">
                      <p className="text-sm font-semibold text-accent-foreground">
                        🎓 Student Discount (50%)
                      </p>
                      <p className="text-2xl font-bold text-primary">₦75,000</p>
                      <p className="text-xs text-muted-foreground">For postgraduate students</p>
                    </div>
                  </div>
                  
                  {/* International Pricing */}
                  <div className="bg-muted rounded-xl p-6 text-center">
                    <p className="text-sm text-muted-foreground mb-2">For International Participants</p>
                    <p className="text-4xl font-bold text-foreground mb-2">$150 USD</p>
                    <p className="text-xs text-muted-foreground mt-2">UK, USA, and other countries</p>
                  </div>
                </div>

                {/* Bank Details */}
                <div className="bg-card border border-border rounded-xl p-6 mb-8">
                  <h3 className="font-semibold text-foreground mb-3 text-center">Payment Details</h3>
                  <div className="grid md:grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-sm text-muted-foreground">Bank</p>
                      <p className="font-semibold text-foreground">WEMA Bank</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Account Name</p>
                      <p className="font-semibold text-foreground">Able-Flourish Agro-Services Ltd</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Account Number</p>
                      <p className="font-bold text-primary text-lg">0126669398</p>
                    </div>
                  </div>
                </div>

                {/* What's Included */}
                <div className="mb-8">
                  <h3 className="font-serif text-xl font-semibold text-foreground mb-6 text-center">
                    What's Included
                  </h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    {included.map((item) => (
                      <div key={item.title} className="flex items-start gap-3 p-4 bg-muted rounded-lg">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                          <item.icon className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-foreground">{item.title}</h4>
                          <p className="text-muted-foreground text-sm">{item.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* CTA */}
                <div className="text-center">
                  <Button variant="gold" size="xl" className="w-full sm:w-auto" asChild>
                    <Link to="/apply">
                      Apply & Pay
                      <ArrowRight className="w-5 h-5" />
                    </Link>
                  </Button>
                  <p className="text-muted-foreground text-sm mt-4">
                    Secure payment via Paystack (Cards, Bank Transfer)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Value Proposition */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
                The Value You Receive
              </h2>
              <p className="text-lg text-muted-foreground">
                This is not a discount course or a quick certification mill. 
                FIA–ADAP™ Foundations is a serious investment in your research capability.
              </p>
            </div>

            <div className="space-y-4">
              {valuePoints.map((point) => (
                <div key={point} className="flex items-start gap-3 p-4 bg-card rounded-lg border border-border">
                  <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                  <span className="text-foreground">{point}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ROI Section */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-6">
              Consider the Alternative
            </h2>
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div className="bg-destructive/5 rounded-xl p-6 border border-destructive/20">
                <h3 className="font-semibold text-foreground mb-4">Without Proper Training</h3>
                <ul className="text-left space-y-2 text-muted-foreground text-sm">
                  <li>• Months of struggling with analysis</li>
                  <li>• Thesis chapters rejected for methodology issues</li>
                  <li>• Anxiety during supervision meetings</li>
                  <li>• Delayed graduation</li>
                  <li>• Publications rejected for statistical errors</li>
                </ul>
              </div>
              <div className="bg-primary/5 rounded-xl p-6 border border-primary/20">
                <h3 className="font-semibold text-foreground mb-4">With FIA–ADAP™</h3>
                <ul className="text-left space-y-2 text-muted-foreground text-sm">
                  <li>• Confident, competent data analysis</li>
                  <li>• Methodology approved first time</li>
                  <li>• Impressive supervision presentations</li>
                  <li>• Timely thesis completion</li>
                  <li>• Publication-ready results</li>
                </ul>
              </div>
            </div>
            <p className="text-muted-foreground">
              The cost of <strong>not</strong> knowing proper data analysis far exceeds 
              the investment in learning it right.
            </p>
          </div>
        </div>
      </section>

      {/* FAQ Teaser */}
      <section className="section-padding bg-muted">
        <div className="container-wide text-center">
          <h2 className="font-serif text-2xl font-bold text-foreground mb-4">
            Have Questions?
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            Check our frequently asked questions or reach out directly.
          </p>
          <Button variant="outline" size="lg" asChild>
            <Link to="/faq">View FAQ</Link>
          </Button>
        </div>
      </section>

      {/* Final CTA */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold mb-6">
            Ready to Invest in Your Research Future?
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            Secure your place in the next cohort of FIA–ADAP™ Foundations.
          </p>
          <Button variant="hero" size="xl" asChild>
            <Link to="/apply">
              Apply & Pay Now
              <ArrowRight className="w-5 h-5" />
            </Link>
          </Button>
        </div>
      </section>
    </Layout>
  );
}
