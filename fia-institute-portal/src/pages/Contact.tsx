import { useState } from "react";
import { Link } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ArrowRight,
  Mail,
  MapPin,
  Building,
  CreditCard,
  Shield,
  Award,
  Send,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function Contact() {
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [form, setForm] = useState({
    name: "",
    email: "",
    inquiryType: "",
    message: "",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    await new Promise((r) => setTimeout(r, 1000));
    toast({
      title: "Message Sent",
      description: "Thank you for reaching out. We will respond within 2–3 working days.",
    });
    setForm({ name: "", email: "", inquiryType: "", message: "" });
    setIsSubmitting(false);
  };

  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Contact & Verification
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Get in touch with Field-to-Insight Academy or verify payments and certificates.
            </p>
          </div>
        </div>
      </section>

      {/* Contact Information */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="grid lg:grid-cols-2 gap-12">
            {/* Contact Details */}
            <div>
              <h2 className="font-serif text-3xl font-bold text-foreground mb-8">
                Get in Touch
              </h2>
              
              <div className="space-y-6">
                <div className="flex items-start gap-4 p-6 bg-muted rounded-xl">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <Mail className="w-6 h-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground mb-1">Email</h3>
                    <a 
                      href="mailto:info@fieldtoinsightacademy.com.ng" 
                      className="text-primary hover:underline"
                    >
                      info@fieldtoinsightacademy.com.ng
                    </a>
                    <p className="text-muted-foreground text-sm mt-1">
                      For general inquiries, applications, and support
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4 p-6 bg-muted rounded-xl">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <MapPin className="w-6 h-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground mb-1">Physical Address</h3>
                    <p className="text-foreground">
                      No. 2 Ajayi Layout, FUTA South Gate<br />
                      Akure, Ondo State, Nigeria
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Contact Form */}
            <div>
              <h2 className="font-serif text-3xl font-bold text-foreground mb-8">
                Send a Message
              </h2>
              <form onSubmit={handleSubmit} className="card-elevated p-8 space-y-5">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name *</Label>
                  <Input
                    id="name"
                    required
                    value={form.name}
                    onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))}
                    placeholder="Your name"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email *</Label>
                  <Input
                    id="email"
                    type="email"
                    required
                    value={form.email}
                    onChange={(e) => setForm((p) => ({ ...p, email: e.target.value }))}
                    placeholder="your@email.com"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="inquiryType">Inquiry Type</Label>
                  <Select
                    value={form.inquiryType}
                    onValueChange={(v) => setForm((p) => ({ ...p, inquiryType: v }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="general">General Inquiry</SelectItem>
                      <SelectItem value="collaboration">Collaboration</SelectItem>
                      <SelectItem value="consulting">Consulting</SelectItem>
                      <SelectItem value="training">Training & Programmes</SelectItem>
                      <SelectItem value="support">Technical Support</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="message">Message *</Label>
                  <Textarea
                    id="message"
                    required
                    rows={5}
                    value={form.message}
                    onChange={(e) => setForm((p) => ({ ...p, message: e.target.value }))}
                    placeholder="How can we help?"
                  />
                </div>
                <Button type="submit" className="w-full" disabled={isSubmitting}>
                  {isSubmitting ? "Sending..." : "Send Message"}
                  <Send className="w-4 h-4" />
                </Button>
              </form>
            </div>

            {/* Corporate Information */}
            <div>
              <h2 className="font-serif text-3xl font-bold text-foreground mb-8">
                Corporate Information
              </h2>
              
              <div className="card-elevated p-8">
                <div className="flex items-start gap-4 mb-6">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <Building className="w-6 h-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground mb-2">Operating Entity</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      Field-to-Insight Academy (FIA) is an educational and professional 
                      training initiative operated by <strong className="text-foreground">Able-Flourish 
                      Agro-Services Ltd</strong>, a company duly registered in Nigeria.
                    </p>
                  </div>
                </div>

                <div className="bg-muted rounded-lg p-6 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Company Name:</span>
                    <span className="font-semibold text-foreground">Able-Flourish Agro-Services Ltd</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">CAC Registration:</span>
                    <span className="font-semibold text-foreground">RC 7408450</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Registered Under:</span>
                    <span className="font-semibold text-foreground">CAMA 2020</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Payment Verification */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-10">
              <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
                Payment Verification
              </h2>
              <p className="text-lg text-muted-foreground">
                All payments to Field-to-Insight Academy must be made to the official 
                corporate account below.
              </p>
            </div>

            <div className="card-elevated p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 rounded-lg bg-primary flex items-center justify-center">
                  <CreditCard className="w-6 h-6 text-primary-foreground" />
                </div>
                <div>
                  <h3 className="font-serif text-xl font-semibold text-foreground">
                    Official Payment Account
                  </h3>
                  <p className="text-muted-foreground text-sm">
                    For program fees and all official transactions
                  </p>
                </div>
              </div>

              <div className="bg-primary/5 rounded-xl p-6 border border-primary/20">
                <div className="grid md:grid-cols-3 gap-6">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Bank Name</p>
                    <p className="text-lg font-bold text-foreground">WEMA Bank</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Account Name</p>
                    <p className="text-lg font-bold text-foreground">Able-Flourish Agro-Services Ltd</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Account Number</p>
                    <p className="text-lg font-bold text-primary">0126669398</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 p-4 bg-destructive/5 rounded-lg border border-destructive/20">
                <div className="flex items-start gap-3">
                  <Shield className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-foreground mb-1">Important Notice</p>
                    <p className="text-sm text-muted-foreground">
                      FIA does not accept payments to personal accounts or third-party accounts. 
                      All program fees are payable exclusively to Able-Flourish Agro-Services Ltd 
                      via its designated WEMA Bank account or approved payment gateway (Paystack).
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Certificate Verification */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <Award className="w-8 h-8 text-primary" />
            </div>
            <h2 className="font-serif text-3xl font-bold text-foreground mb-4">
              Certificate Verification
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              Need to verify the authenticity of an FIA Certificate of Competence? 
              Use our online verification tool or contact us directly.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button variant="default" size="lg" asChild>
                <Link to="/verify-certificate">
                  Verify Online
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
              <Button variant="outline" size="lg" asChild>
                <a href="mailto:info@fieldtoinsightacademy.com.ng?subject=Certificate Verification Request">
                  Contact Us
                </a>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold mb-6">
            Ready to Join FIA?
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            Apply to the next cohort of FIA–ADAP™ Foundations and transform your 
            approach to agricultural data analysis.
          </p>
          <Button variant="hero" size="xl" asChild>
            <Link to="/apply">
              Apply Now
              <ArrowRight className="w-5 h-5" />
            </Link>
          </Button>
        </div>
      </section>
    </Layout>
  );
}
