import { useState, useRef, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { Menu, X, GraduationCap, ChevronDown } from "lucide-react";

const tutorLinks = [
  { name: "FIA-ADAP Foundation Tutor", href: "/adap-tutor" },
  { name: "Plant Genetics Tutor", href: "/plant-genetics-mastery-tutor" },
  { name: "Biometrical Genetics Tutor", href: "/biometrical-genetics-mastery-tutor" },
  { name: "Thesis Mentor", href: "/thesis-mentor" },
  { name: "Research Writing System", href: "/research-writing", badge: "Login Required" },
];

const servicesLinks = [
  { name: "Manuscript Review", href: "/manuscript-reviewer" },
  { name: "Agro-Services", href: "/services/agro-services" },
  { name: "Data Analysis Services", href: "/services/data-analysis" },
  { name: "Able-Flourish Digital Systems", href: "/services/able-flourish-digital-systems" },
];

type NavChild = { name: string; href: string; comingSoon?: boolean; badge?: string };
type NavItem = { name: string; href: string } | { name: string; children: NavChild[] };

const navigation: NavItem[] = [
  { name: "Home", href: "/" },
  { name: "About", href: "/about" },
  { name: "Founder", href: "/founder" },
  { name: "VivaSense", href: "/vivasense" },
  { name: "AI Tutors", children: tutorLinks },
  { name: "Services", children: servicesLinks },
  { name: "Training & Programmes", href: "/programmes" },
  { name: "Cohort 3", href: "/cohort3-registration" },
  { name: "Technical Insights", href: "/technical-insights" },
  { name: "Journal", href: "/journal" },
  { name: "Contact", href: "/contact" },
];

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const [mobileOpenDropdown, setMobileOpenDropdown] = useState<string | null>(null);
  const navRef = useRef<HTMLDivElement>(null);
  const location = useLocation();

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (navRef.current && !navRef.current.contains(e.target as Node)) {
        setOpenDropdown(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const isChildActive = (children: NavChild[]) =>
    children.some((l) => location.pathname === l.href);

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-md border-b border-border">
      <nav className="container-wide flex items-center justify-between py-4">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center group-hover:bg-primary/90 transition-colors">
            <GraduationCap className="w-6 h-6 text-primary-foreground" />
          </div>
          <div className="flex flex-col">
            <span className="font-serif font-bold text-lg text-foreground leading-tight">
              Field-to-Insight
            </span>
            <span className="text-xs text-muted-foreground font-medium tracking-wider uppercase">
              Academy
            </span>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden xl:flex items-center gap-0.5" ref={navRef}>
          {navigation.map((item) =>
            "children" in item ? (
              <div key={item.name} className="relative">
                <button
                  onClick={() =>
                    setOpenDropdown((prev) => (prev === item.name ? null : item.name))
                  }
                  className={`flex items-center gap-1 px-2.5 py-2 text-sm font-medium rounded-lg transition-colors ${
                    isChildActive(item.children)
                      ? "text-primary bg-muted"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  }`}
                >
                  {item.name}
                  <ChevronDown
                    className={`w-3.5 h-3.5 transition-transform ${
                      openDropdown === item.name ? "rotate-180" : ""
                    }`}
                  />
                </button>
                {openDropdown === item.name && (
                  <div className="absolute top-full left-0 mt-1 w-72 rounded-lg border border-border bg-popover shadow-md py-1 z-50">
                    {item.children.map((child) => (
                      <Link
                        key={child.name}
                        to={child.href}
                        onClick={() => setOpenDropdown(null)}
                        className={`block px-4 py-2.5 text-sm transition-colors ${
                          child.comingSoon
                            ? "text-muted-foreground italic cursor-default"
                            : location.pathname === child.href
                            ? "text-primary bg-muted"
                            : "text-popover-foreground hover:bg-muted"
                        }`}
                      >
                        {child.name}
                        {child.comingSoon && (
                          <span className="ml-2 text-xs bg-muted px-1.5 py-0.5 rounded text-muted-foreground">
                            Soon
                          </span>
                        )}
                        {child.badge && (
                          <span className="ml-2 text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded font-medium">
                            {child.badge}
                          </span>
                        )}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <Link
                key={item.name}
                to={(item as any).href}
                className={`px-2.5 py-2 text-sm font-medium rounded-lg transition-colors ${
                  location.pathname === (item as any).href
                    ? "text-primary bg-muted"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
              >
                {item.name}
              </Link>
            )
          )}
        </div>

        {/* Spacer for layout balance */}
        <div className="hidden xl:block" />

        {/* Mobile menu button */}
        <button
          type="button"
          className="xl:hidden p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </nav>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="xl:hidden bg-background border-b border-border">
          <div className="container-wide py-4 space-y-1">
            {navigation.map((item) =>
              "children" in item ? (
                <div key={item.name}>
                  <button
                    onClick={() =>
                      setMobileOpenDropdown((prev) =>
                        prev === item.name ? null : item.name
                      )
                    }
                    className={`flex items-center justify-between w-full px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                      isChildActive(item.children)
                        ? "text-primary bg-muted"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                    }`}
                  >
                    {item.name}
                    <ChevronDown
                      className={`w-3.5 h-3.5 transition-transform ${
                        mobileOpenDropdown === item.name ? "rotate-180" : ""
                      }`}
                    />
                  </button>
                  {mobileOpenDropdown === item.name && (
                    <div className="ml-4 mt-1 space-y-1">
                      {item.children.map((child) => (
                        <Link
                          key={child.name}
                          to={child.href}
                          className={`block px-4 py-2.5 text-sm rounded-lg transition-colors ${
                            child.comingSoon
                              ? "text-muted-foreground italic"
                              : location.pathname === child.href
                              ? "text-primary bg-muted"
                              : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                          }`}
                          onClick={() => !child.comingSoon && setMobileMenuOpen(false)}
                        >
                          {child.name}
                          {child.comingSoon && (
                            <span className="ml-2 text-xs bg-muted px-1.5 py-0.5 rounded">
                              Soon
                            </span>
                          )}
                          {child.badge && (
                            <span className="ml-2 text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded font-medium">
                              {child.badge}
                            </span>
                          )}
                        </Link>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <Link
                  key={item.name}
                  to={(item as any).href}
                  className={`block px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                    location.pathname === (item as any).href
                      ? "text-primary bg-muted"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  }`}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  {item.name}
                </Link>
              )
            )}
          </div>
        </div>
      )}
    </header>
  );
}
