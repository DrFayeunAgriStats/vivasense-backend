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
import { Checkbox } from "@/components/ui/checkbox";
import {
  ArrowRight,
  CheckCircle2,
  Clock,
  GraduationCap,
  Building2,
  Award,
  Users,
  Briefcase,
  Send,
  Calendar,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const programmes = [
  {
    id: "foundations",
    name: "FIA-ADAP™ Foundations",
    status: "LIVE",
    statusColor: "bg-green-500",
    description: "Entry-level programme for MSc, PhD students, and early-career researchers. Build foundational competence in agricultural data analysis and prediction.",
    features: [
      "6-week intensive programme",
      "12 live Zoom sessions",
      "Competence-based certification",
      "Agriculture-specific curriculum",
    ],
    icon: GraduationCap,
    cta: "View Details",
    ctaLink: "/program",
    hasFullDetails: true,
  },
  {
    id: "advanced",
    name: "FIA-ADAP™ Advanced",
    status: "Coming 2026",
    statusColor: "bg-amber-500",
    description: "Advanced modelling, multivariate analysis, and predictive decision support for experienced researchers and professionals.",
    features: [
      "Advanced analytical techniques",
      "Machine learning applications",
      "Predictive modelling",
      "Research-level methodology",
    ],
    icon: Award,
    cta: "Join Waitlist",
    ctaLink: null,
    hasFullDetails: false,
  },
  {
    id: "professional",
    name: "FIA-ADAP™ Professional Certification",
    status: "Coming 2026",
    statusColor: "bg-amber-500",
    description: "Assessment-based certification pathway for experienced professionals seeking formal recognition of analytical competence.",
    features: [
      "Portfolio-based assessment",
      "Professional recognition",
      "Industry-standard certification",
      "Flexible timeline",
    ],
    icon: Briefcase,
    cta: "Join Waitlist",
    ctaLink: null,
    hasFullDetails: false,
  },
  {
    id: "corporates",
    name: "FIA-ADAP™ Corporates",
    status: "On Request",
    statusColor: "bg-primary",
    description: "Customised training and analytical support for organisations, research institutes, NGOs, and corporate teams.",
    features: [
      "Tailored curriculum",
      "On-site or virtual delivery",
      "Team capacity building",
      "Organisational needs assessment",
    ],
    icon: Building2,
    cta: "Inquire Now",
    ctaLink: "#corporate-inquiry",
    hasFullDetails: false,
  },
];

const organisationTypes = [
  "Research Institute",
  "University / Academic Department",
  "Agribusiness / Corporate",
  "NGO / Development Organisation",
  "Other",
];

const areasOfInterest = [
  { id: "experimental-design", label: "Experimental design & analysis" },
  { id: "data-analysis", label: "Agricultural data analysis & prediction" },
  { id: "multivariate", label: "Multivariate & G×E analysis" },
  { id: "decision-support", label: "Data interpretation & decision support" },
  { id: "custom", label: "Custom / Organisation-specific" },
];

const trainingModes = [
  "Online",
  "Hybrid",
  "On-site (location-dependent)",
];

export default function Programmes() {
  const { toast } = useToast();
  const [waitlistEmail, setWaitlistEmail] = useState("");
  const [waitlistProgramme, setWaitlistProgramme] = useState("");
  const [isSubmittingWaitlist, setIsSubmittingWaitlist] = useState(false);
  const [isSubmittingInquiry, setIsSubmittingInquiry] = useState(false);
  const [selectedAreas, setSelectedAreas] = useState<string[]>([]);
  
  const [corporateForm, setCorporateForm] = useState({
    organisationName: "",
    organisationType: "",
    country: "",
    contactName: "",
    contactRole: "",
    contactEmail: "",
    contactPhone: "",
    trainingMode: "",
    projectDescription: "",
  });

  const handleWaitlistSubmit = async (programme: string) => {
    if (!waitlistEmail) {
      toast({
        title: "Email Required",
        description: "Please enter your email address to join the waitlist.",
        variant: "destructive",
      });
      return;
    }
    
    setIsSubmittingWaitlist(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    
    toast({
      title: "Added to Waitlist!",
      description: `You'll be notified when ${programme} becomes available.`,
    });
    
    setWaitlistEmail("");
    setWaitlistProgramme("");
    setIsSubmittingWaitlist(false);
  };

  const handleCorporateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmittingInquiry(true);
    
    await new Promise((resolve) => setTimeout(resolve, 1500));
    
    toast({
      title: "Inquiry Received",
      description: "Thank you for contacting Field-to-Insight Academy. A member of our team will contact you within 2–3 working days.",
    });
    
    setCorporateForm({
      organisationName: "",
      organisationType: "",
      country: "",
      contactName: "",
      contactRole: "",
      contactEmail: "",
      contactPhone: "",
      trainingMode: "",
      projectDescription: "",
    });
    setSelectedAreas([]);
    setIsSubmittingInquiry(false);
  };

  const handleAreaToggle = (areaId: string) => {
    setSelectedAreas((prev) =>
      prev.includes(areaId)
        ? prev.filter((id) => id !== areaId)
        : [...prev, areaId]
    );
  };

  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-20 md:py-28">
        <div className="container-wide">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-foreground/10 text-sm font-medium mb-6">
              <GraduationCap className="w-4 h-4" />
              <span>FIA-ADAP™ Programme Family</span>
            </div>
            <h1 className="font-serif text-4xl md:text-5xl font-bold mb-6">
              Programmes
            </h1>
            <p className="text-xl text-primary-foreground/85 leading-relaxed">
              Field-to-Insight Academy offers a structured pathway of programmes 
              designed for different stages of your analytical journey—from foundational 
              training to advanced specialisation and institutional support.
            </p>
          </div>
        </div>
      </section>

      {/* Cohort 3 Featured Registration */}
      <section className="section-padding bg-gradient-to-br from-primary/5 via-background to-accent/5">
        <div className="container-wide">
          <div className="max-w-5xl mx-auto">
            <span className="inline-block text-xs font-semibold uppercase tracking-[0.22em] text-accent mb-2">
              Now Open · Registration Live
            </span>
            <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-6">
              Featured Cohort
            </h2>
            <div className="card-elevated p-8 md:p-10 flex flex-col md:flex-row items-start gap-6 border-l-4 border-primary">
              <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0">
                <GraduationCap className="w-7 h-7 text-primary" />
              </div>
              <div className="flex-1">
                <span className="inline-block text-xs font-semibold uppercase tracking-wide text-primary mb-2">
                  6-Week Foundation Programme · Cohort 3
                </span>
                <h3 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-3">
                  FIA–ADAP Foundations Cohort 3
                </h3>
                <p className="text-muted-foreground mb-6">
                  Registration is now open for the third cohort of our flagship foundation programme in
                  agricultural data analytics, research thinking, and AI-assisted research. Limited seats
                  available for students and professionals.
                </p>
                <div className="flex flex-col sm:flex-row gap-3">
                  <Button variant="default" asChild>
                    <Link to="/cohort3-registration">
                      Register for Cohort 3
                      <ArrowRight className="w-4 h-4" />
                    </Link>
                  </Button>
                  <Button variant="outline" asChild>
                    <Link to="/cohort3-registration#pricing">View Pricing</Link>
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Past Workshops */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-5xl mx-auto mb-6">
            <span className="inline-block text-xs font-semibold uppercase tracking-[0.22em] text-accent mb-2">
              Events & Workshops
            </span>
            <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground">
              Past Workshops
            </h2>
          </div>
          <div className="card-elevated p-8 md:p-10 max-w-5xl mx-auto flex flex-col md:flex-row items-start gap-6 border-l-4 border-accent">
            <div className="w-14 h-14 rounded-xl bg-accent/10 flex items-center justify-center flex-shrink-0">
              <Calendar className="w-7 h-7 text-accent" />
            </div>
            <div className="flex-1">
              <span className="inline-block text-xs font-semibold uppercase tracking-wide text-accent mb-2">
                Free Online Workshop · 8–9 May 2026
              </span>
              <h3 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-3">
                VivaSense Data Clinic
              </h3>
              <p className="text-muted-foreground mb-6">
                Turn your field trial data into publication-ready results in minutes — not weeks.
                Two live evening sessions covering ANOVA, G×E, heritability, and AI-powered interpretation.
              </p>
              <Button variant="default" asChild>
                <Link to="/programmes/data-clinic">
                  View Workshop Details
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Programme Cards */}
      <section className="section-padding">
        <div className="container-wide">
          <div className="grid md:grid-cols-2 gap-8">
            {programmes.map((programme) => (
              <div
                key={programme.id}
                className={`card-elevated p-8 relative ${
                  programme.id === "foundations" ? "ring-2 ring-primary" : ""
                }`}
              >
                {/* Status Badge */}
                <div className="flex items-center justify-between mb-6">
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                    <programme.icon className="w-6 h-6 text-primary" />
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-semibold text-white ${programme.statusColor}`}
                  >
                    {programme.status}
                  </span>
                </div>

                <h3 className="font-serif text-2xl font-bold text-foreground mb-3">
                  {programme.name}
                </h3>
                <p className="text-muted-foreground mb-6">
                  {programme.description}
                </p>

                {/* Features */}
                <ul className="space-y-2 mb-8">
                  {programme.features.map((feature) => (
                    <li key={feature} className="flex items-center gap-2 text-sm">
                      <CheckCircle2 className="w-4 h-4 text-primary flex-shrink-0" />
                      <span className="text-foreground">{feature}</span>
                    </li>
                  ))}
                </ul>

                {/* CTA */}
                {programme.hasFullDetails ? (
                  <div className="space-y-3">
                    <Button variant="default" className="w-full" asChild>
                      <Link to={programme.ctaLink || "#"}>
                        {programme.cta}
                        <ArrowRight className="w-4 h-4" />
                      </Link>
                    </Button>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" className="flex-1" asChild>
                        <Link to="/curriculum">Curriculum</Link>
                      </Button>
                      <Button variant="outline" size="sm" className="flex-1" asChild>
                        <Link to="/pricing">Investment</Link>
                      </Button>
                      <Button variant="gold" size="sm" className="flex-1" asChild>
                        <Link to="/apply">Apply</Link>
                      </Button>
                    </div>
                  </div>
                ) : programme.ctaLink === "#corporate-inquiry" ? (
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() => {
                      document.getElementById("corporate-inquiry")?.scrollIntoView({ behavior: "smooth" });
                    }}
                  >
                    {programme.cta}
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                ) : (
                  <div className="space-y-3">
                    <div className="flex gap-2">
                      <Input
                        type="email"
                        placeholder="Enter your email"
                        value={waitlistProgramme === programme.name ? waitlistEmail : ""}
                        onChange={(e) => {
                          setWaitlistEmail(e.target.value);
                          setWaitlistProgramme(programme.name);
                        }}
                        className="flex-1"
                      />
                      <Button
                        variant="outline"
                        disabled={isSubmittingWaitlist && waitlistProgramme === programme.name}
                        onClick={() => handleWaitlistSubmit(programme.name)}
                      >
                        {isSubmittingWaitlist && waitlistProgramme === programme.name ? (
                          <Clock className="w-4 h-4 animate-spin" />
                        ) : (
                          "Join"
                        )}
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground text-center">
                      Be notified when this programme launches
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About FIA-ADAP */}
      <section className="section-padding bg-secondary">
        <div className="container-wide">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground mb-6">
              About the FIA-ADAP™ Programme Family
            </h2>
            <p className="text-lg text-muted-foreground mb-6">
              FIA-ADAP™ (Agricultural Data Analysis & Prediction) is the flagship 
              programme family of Field-to-Insight Academy. Each programme is designed 
              to build competence at different levels—from entry-level foundations 
              through advanced specialisation to institutional capacity building.
            </p>
            <p className="text-muted-foreground">
              All FIA-ADAP™ programmes share a commitment to competence-based 
              assessment, agriculture-specific content, and practical application 
              over passive learning.
            </p>
          </div>
        </div>
      </section>

      {/* Corporate Inquiry Form */}
      <section id="corporate-inquiry" className="section-padding">
        <div className="container-wide">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
                <Building2 className="w-4 h-4" />
                <span>FIA-ADAP™ Corporates (On Request)</span>
              </div>
              <h2 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
                Corporate & Institutional Training Inquiry
              </h2>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                FIA-ADAP™ Corporates provides customised training and analytical support 
                for organisations, research institutes, NGOs, and corporate teams. 
                Please complete the form below and a member of the FIA team will contact 
                you to discuss your needs.
              </p>
            </div>

            <div className="card-elevated p-8">
              <form onSubmit={handleCorporateSubmit} className="space-y-8">
                {/* Organisation Details */}
                <div>
                  <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                    <Building2 className="w-5 h-5 text-primary" />
                    Organisation Details
                  </h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="organisationName">Organisation / Institution Name *</Label>
                      <Input
                        id="organisationName"
                        required
                        value={corporateForm.organisationName}
                        onChange={(e) =>
                          setCorporateForm((prev) => ({ ...prev, organisationName: e.target.value }))
                        }
                        placeholder="Enter organisation name"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="organisationType">Organisation Type *</Label>
                      <Select
                        value={corporateForm.organisationType}
                        onValueChange={(value) =>
                          setCorporateForm((prev) => ({ ...prev, organisationType: value }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select type" />
                        </SelectTrigger>
                        <SelectContent>
                          {organisationTypes.map((type) => (
                            <SelectItem key={type} value={type}>
                              {type}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="mt-4 space-y-2">
                    <Label htmlFor="country">Country *</Label>
                    <Input
                      id="country"
                      required
                      value={corporateForm.country}
                      onChange={(e) =>
                        setCorporateForm((prev) => ({ ...prev, country: e.target.value }))
                      }
                      placeholder="Enter country"
                    />
                  </div>
                </div>

                {/* Contact Person */}
                <div className="pt-6 border-t border-border">
                  <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                    <Users className="w-5 h-5 text-primary" />
                    Contact Person
                  </h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="contactName">Full Name *</Label>
                      <Input
                        id="contactName"
                        required
                        value={corporateForm.contactName}
                        onChange={(e) =>
                          setCorporateForm((prev) => ({ ...prev, contactName: e.target.value }))
                        }
                        placeholder="Enter full name"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="contactRole">Role / Position</Label>
                      <Input
                        id="contactRole"
                        value={corporateForm.contactRole}
                        onChange={(e) =>
                          setCorporateForm((prev) => ({ ...prev, contactRole: e.target.value }))
                        }
                        placeholder="Enter role or position"
                      />
                    </div>
                  </div>
                  <div className="grid md:grid-cols-2 gap-4 mt-4">
                    <div className="space-y-2">
                      <Label htmlFor="contactEmail">Official Email Address *</Label>
                      <Input
                        id="contactEmail"
                        type="email"
                        required
                        value={corporateForm.contactEmail}
                        onChange={(e) =>
                          setCorporateForm((prev) => ({ ...prev, contactEmail: e.target.value }))
                        }
                        placeholder="email@organisation.com"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="contactPhone">Phone / WhatsApp (Optional)</Label>
                      <Input
                        id="contactPhone"
                        type="tel"
                        value={corporateForm.contactPhone}
                        onChange={(e) =>
                          setCorporateForm((prev) => ({ ...prev, contactPhone: e.target.value }))
                        }
                        placeholder="+234 xxx xxx xxxx"
                      />
                    </div>
                  </div>
                </div>

                {/* Training Needs */}
                <div className="pt-6 border-t border-border">
                  <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                    <GraduationCap className="w-5 h-5 text-primary" />
                    Training Needs
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <Label className="mb-3 block">Area of Interest (select all that apply)</Label>
                      <div className="grid md:grid-cols-2 gap-3">
                        {areasOfInterest.map((area) => (
                          <div key={area.id} className="flex items-center space-x-2">
                            <Checkbox
                              id={area.id}
                              checked={selectedAreas.includes(area.id)}
                              onCheckedChange={() => handleAreaToggle(area.id)}
                            />
                            <Label htmlFor={area.id} className="text-sm font-normal cursor-pointer">
                              {area.label}
                            </Label>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="trainingMode">Preferred Training Mode</Label>
                      <Select
                        value={corporateForm.trainingMode}
                        onValueChange={(value) =>
                          setCorporateForm((prev) => ({ ...prev, trainingMode: value }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select preferred mode" />
                        </SelectTrigger>
                        <SelectContent>
                          {trainingModes.map((mode) => (
                            <SelectItem key={mode} value={mode}>
                              {mode}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                {/* Project Description */}
                <div className="pt-6 border-t border-border">
                  <div className="space-y-2">
                    <Label htmlFor="projectDescription">Brief Description of Needs</Label>
                    <Textarea
                      id="projectDescription"
                      value={corporateForm.projectDescription}
                      onChange={(e) =>
                        setCorporateForm((prev) => ({ ...prev, projectDescription: e.target.value }))
                      }
                      placeholder="Please describe your training needs, including datasets you work with, team size, objectives, and any specific timelines or requirements."
                      rows={5}
                    />
                  </div>
                </div>

                {/* Submit */}
                <div className="pt-4">
                  <Button
                    type="submit"
                    variant="default"
                    size="xl"
                    className="w-full"
                    disabled={isSubmittingInquiry}
                  >
                    {isSubmittingInquiry ? (
                      "Submitting..."
                    ) : (
                      <>
                        <Send className="w-5 h-5" />
                        Submit Inquiry
                      </>
                    )}
                  </Button>
                  <p className="text-muted-foreground text-sm text-center mt-4">
                    A member of our team will contact you within 2–3 working days.
                  </p>
                </div>
              </form>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-padding bg-primary text-primary-foreground">
        <div className="container-wide text-center">
          <h2 className="font-serif text-3xl font-bold mb-6">
            Ready to Begin Your Journey?
          </h2>
          <p className="text-xl text-primary-foreground/85 mb-10 max-w-2xl mx-auto">
            FIA-ADAP™ Foundations is currently enrolling. Start building your 
            competence in agricultural data analysis today.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="hero" size="xl" asChild>
              <Link to="/apply">
                Apply to Foundations
                <ArrowRight className="w-5 h-5" />
              </Link>
            </Button>
            <Button variant="hero-outline" size="xl" asChild>
              <Link to="/program">Learn More</Link>
            </Button>
          </div>
        </div>
      </section>
    </Layout>
  );
}
