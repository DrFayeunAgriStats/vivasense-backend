import { Link } from "react-router-dom";
import { GraduationCap, Mail, MapPin, Phone, Building } from "lucide-react";

const footerLinks = {
  program: [
    { name: "Cohort 3 Registration", href: "/cohort3-registration" },
    { name: "Programmes", href: "/programmes" },
    { name: "Curriculum", href: "/curriculum" },
    { name: "Learning Outcomes", href: "/outcomes" },
    { name: "Tools & Framework", href: "/tools" },
  ],
  resources: [
    { name: "Testimonials", href: "/testimonials" },
    { name: "FAQ", href: "/faq" },
    { name: "Investment", href: "/pricing" },
    { name: "AI Tutor", href: "/tutor" },
    { name: "Plant Genetics Mastery Tutor", href: "/plant-genetics-mastery-tutor" },
    { name: "Apply Now", href: "/apply" },
  ],
  about: [
    { name: "About FIA", href: "/program" },
    { name: "Faculty", href: "/faculty" },
    { name: "Journal (JAI)", href: "/journal" },
    { name: "Contact", href: "/contact" },
    { name: "Services", href: "/services" },
    { name: "Privacy Policy", href: "/privacy" },
  ],
};

interface FooterProps {
  variant?: "default" | "minimal-vivasense";
}

export function Footer({ variant = "default" }: FooterProps) {
  if (variant === "minimal-vivasense") {
    return (
      <footer className="bg-[#153D1D] text-primary-foreground">
        <div className="container-wide py-10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 border-b border-primary-foreground/15 pb-8">
            <div>
              <h4 className="font-serif text-xl font-semibold">VivaSense</h4>
              <p className="text-sm text-primary-foreground/80 mt-2 leading-relaxed">
                Scientific analytics platform for agricultural research workflows.
              </p>
            </div>
            <div>
              <h5 className="font-semibold">Field-to-Insight Academy</h5>
              <p className="text-sm text-primary-foreground/80 mt-2 leading-relaxed">
                Built and maintained as long-term academic research infrastructure.
              </p>
              <a
                href="mailto:info@fieldtoinsightacademy.com.ng"
                className="inline-flex items-center gap-2 mt-3 text-sm text-primary-foreground/90 hover:text-primary-foreground"
              >
                <Mail className="w-4 h-4" />
                info@fieldtoinsightacademy.com.ng
              </a>
            </div>
            <div>
              <h5 className="font-semibold">Institutional Access</h5>
              <p className="text-sm text-primary-foreground/80 mt-2 leading-relaxed">
                Department, faculty, and institution-level licensing is available.
              </p>
              <div className="mt-3 flex flex-wrap gap-4 text-sm">
                <Link to="/contact" className="text-primary-foreground/90 hover:text-primary-foreground">
                  Contact
                </Link>
                <a
                  href="https://fieldtoinsightacademy.com.ng"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-foreground/90 hover:text-primary-foreground"
                >
                  Institution Site
                </a>
              </div>
            </div>
          </div>
          <div className="pt-5 text-xs text-primary-foreground/70 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <p>© {new Date().getFullYear()} VivaSense · Field-to-Insight Academy</p>
            <p>Documentation links will be published in upcoming releases.</p>
          </div>
        </div>
      </footer>
    );
  }

  return (
    <footer className="bg-primary text-primary-foreground">
      <div className="container-wide section-padding">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
          {/* Brand */}
          <div className="lg:col-span-1">
            <Link to="/" className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-lg bg-primary-foreground/10 flex items-center justify-center">
                <GraduationCap className="w-6 h-6 text-primary-foreground" />
              </div>
              <div className="flex flex-col">
                <span className="font-serif font-bold text-lg leading-tight">
                  Field-to-Insight
                </span>
                <span className="text-xs text-primary-foreground/70 font-medium tracking-wider uppercase">
                  Academy
                </span>
              </div>
            </Link>
            <p className="text-primary-foreground/80 text-sm leading-relaxed mb-6">
              Transforming agricultural researchers from running analyses blindly
              to designing, analysing, interpreting, and defending data with confidence.
            </p>
            <div className="space-y-3 text-sm text-primary-foreground/70">
              <div className="flex items-center gap-3">
                <Mail className="w-4 h-4 flex-shrink-0" />
                <span>info@fieldtoinsightacademy.com.ng</span>
              </div>
              <div className="flex items-center gap-3">
                <Phone className="w-4 h-4 flex-shrink-0" />
                <a href="tel:+2349022158026" className="hover:text-primary-foreground transition-colors">+234 902 215 8026</a>
              </div>
              <div className="flex items-start gap-3">
                <MapPin className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>No. 2 Ajayi Layout, FUTA South Gate, Akure, Ondo State, Nigeria</span>
              </div>
            </div>
          </div>

          {/* Program Links */}
          <div>
            <h4 className="font-serif font-semibold text-lg mb-6">Program</h4>
            <ul className="space-y-3">
              {footerLinks.program.map((link) => (
                <li key={link.name}>
                  <Link
                    to={link.href}
                    className="text-primary-foreground/70 hover:text-primary-foreground transition-colors text-sm"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources Links */}
          <div>
            <h4 className="font-serif font-semibold text-lg mb-6">Resources</h4>
            <ul className="space-y-3">
              {footerLinks.resources.map((link) => (
                <li key={link.name}>
                  <Link
                    to={link.href}
                    className="text-primary-foreground/70 hover:text-primary-foreground transition-colors text-sm"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* About Links */}
          <div>
            <h4 className="font-serif font-semibold text-lg mb-6">About</h4>
            <ul className="space-y-3">
              {footerLinks.about.map((link) => (
                <li key={link.name}>
                  <Link
                    to={link.href}
                    className="text-primary-foreground/70 hover:text-primary-foreground transition-colors text-sm"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Corporate Information */}
        <div className="mt-12 pt-8 border-t border-primary-foreground/10">
          <div className="bg-primary-foreground/5 rounded-xl p-6">
            <div className="flex items-start gap-3 mb-4">
              <Building className="w-5 h-5 text-primary-foreground/70 flex-shrink-0 mt-0.5" />
              <div>
                <h5 className="font-semibold text-primary-foreground mb-2">Operating Entity</h5>
                <p className="text-primary-foreground/80 text-sm leading-relaxed">
                  Field-to-Insight Academy (FIA) is an educational and professional training 
                  initiative operated by <strong>Able-Flourish Agro-Services Ltd</strong> (RC 7408450), 
                  a company duly registered in Nigeria under the Companies and Allied Matters Act (CAMA) 2020.
                </p>
              </div>
            </div>
            <div className="pl-8 text-sm text-primary-foreground/70">
              <p>All program fees are payable exclusively to Able-Flourish Agro-Services Ltd 
              via its designated WEMA Bank account or approved payment gateway.</p>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-8 pt-8 border-t border-primary-foreground/10">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-primary-foreground/60 text-sm">
              © {new Date().getFullYear()} Field-to-Insight Academy. All rights reserved.
            </p>
            <p className="text-primary-foreground/60 text-sm">
              FIA–ADAP™ is a trademark of Field-to-Insight Academy
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
