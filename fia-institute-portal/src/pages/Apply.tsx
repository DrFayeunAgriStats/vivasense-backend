import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { CheckCircle2, ArrowRight, Shield, Clock, Users } from "lucide-react";

export default function Apply() {
  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Apply to FIA–ADAP™ Foundations
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed mb-4">
              Take the first step toward mastering agricultural data analysis.
              Complete the application form below.
            </p>
            <p className="text-primary-foreground/70 text-sm italic">
              FIA-ADAP™ Foundations is the entry-level programme within the FIA-ADAP™
              programme family at Field-to-Insight Academy.
            </p>
          </div>
        </div>
      </section>

      {/* Application Section */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="grid lg:grid-cols-3 gap-12">
            {/* Google Form */}
            <div className="lg:col-span-2">
              <div className="card-elevated p-8">
                <h2 className="font-serif text-2xl font-bold text-foreground mb-6">
                  Application Form
                </h2>

                <p className="text-muted-foreground mb-6">
                  Applications are collected securely via Google Forms.
                  Please complete all required fields carefully.
                </p>

                <iframe
                  src="https://docs.google.com/forms/d/e/1FAIpQLSeyZvwDqH_KNq3Agv-53FIGGUdcvhOjLItY_2c86BFd-3A5oQ/viewform?embedded=true"
                  width="100%"
                  height="1200"
                  frameBorder="0"
                  marginHeight={0}
                  marginWidth={0}
                >
                  Loading…
                </iframe>

                <p className="text-muted-foreground text-sm text-center mt-6">
                  After submitting the application, successful applicants will
                  receive payment instructions by email.
                </p>
              </div>
            </div>

            {/* Sidebar */}
            <div className="lg:col-span-1">
              <div className="sticky top-24 space-y-6">
                {/* Why Apply */}
                <div className="bg-muted rounded-xl p-6">
                  <h3 className="font-serif text-lg font-semibold text-foreground mb-4">
                    Application-Led Entry
                  </h3>
                  <p className="text-muted-foreground text-sm mb-4">
                    FIA–ADAP™ Foundations uses an application-based entry process
                    to ensure participants are committed and well-suited to the program.
                  </p>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="w-4 h-4 text-primary" />
                      <span className="text-sm text-foreground">Ensures commitment</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="w-4 h-4 text-primary" />
                      <span className="text-sm text-foreground">Helps us tailor content</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="w-4 h-4 text-primary" />
                      <span className="text-sm text-foreground">Builds engaged cohorts</span>
                    </div>
                  </div>
                </div>

                {/* Payment Info */}
                <div className="bg-card rounded-xl p-6 border border-border">
                  <h3 className="font-serif text-lg font-semibold text-foreground mb-4">
                    Program Investment
                  </h3>

                  {/* Nigerian Pricing */}
                  <div className="mb-4 pb-4 border-b border-border">
                    <p className="text-sm text-muted-foreground mb-2">
                      For Nigerian Participants
                    </p>
                    <p className="text-2xl font-bold text-foreground">₦150,000</p>
                    <div className="mt-2 p-2 bg-accent/50 rounded-lg">
                      <p className="text-sm font-semibold text-accent-foreground">
                        🎓 Student Discount (50%)
                      </p>
                      <p className="text-xl font-bold text-primary">₦75,000</p>
                      <p className="text-xs text-muted-foreground">
                        For postgraduate students
                      </p>
                    </div>
                  </div>

                  {/* International Pricing */}
                  <div className="mb-4 pb-4 border-b border-border">
                    <p className="text-sm text-muted-foreground mb-2">
                      For International Participants
                    </p>
                    <p className="text-2xl font-bold text-foreground">$150 USD</p>
                    <p className="text-xs text-muted-foreground">
                      UK, USA, and other countries
                    </p>
                  </div>

                  {/* Bank Details */}
                  <div className="bg-muted rounded-lg p-4 mb-4">
                    <p className="text-sm font-semibold text-foreground mb-2">
                      Bank Transfer Details
                    </p>
                    <div className="space-y-1 text-sm">
                      <p>
                        <span className="text-muted-foreground">Bank:</span>{" "}
                        <span className="font-medium">WEMA Bank</span>
                      </p>
                      <p>
                        <span className="text-muted-foreground">Account Name:</span>{" "}
                        <span className="font-medium">
                          Able-Flourish Agro-Services Ltd
                        </span>
                      </p>
                      <p>
                        <span className="text-muted-foreground">Account Number:</span>{" "}
                        <span className="font-bold text-primary">0126669398</span>
                      </p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <Shield className="w-5 h-5 text-primary" />
                      <span className="text-sm text-foreground">Secure payment</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <Clock className="w-5 h-5 text-primary" />
                      <span className="text-sm text-foreground">
                        Payment after application review
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <Users className="w-5 h-5 text-primary" />
                      <span className="text-sm text-foreground">
                        Limited cohort size
                      </span>
                    </div>
                  </div>
                </div>

                {/* Contact */}
                <div className="bg-secondary rounded-xl p-6">
                  <h3 className="font-serif text-lg font-semibold text-foreground mb-2">
                    Need Help?
                  </h3>
                  <p className="text-muted-foreground text-sm">
                    Contact us at{" "}
                    <a
                      href="mailto:info@fieldtoinsightacademy.com.ng"
                      className="text-primary hover:underline"
                    >
                      info@fieldtoinsightacademy.com.ng
                    </a>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
