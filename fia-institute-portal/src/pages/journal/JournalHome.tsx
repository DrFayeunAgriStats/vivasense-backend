import { Link } from "react-router-dom";
import { JournalLayout } from "@/components/journal/JournalLayout";
import { Button } from "@/components/ui/button";
import { BookOpen, Target, Users, FileText, Upload, Newspaper, Archive } from "lucide-react";

const heroButtons = [
  { label: "About the Journal", href: "/journal/about", icon: BookOpen },
  { label: "Aims & Scope", href: "/journal/aims-scope", icon: Target },
  { label: "Editorial Board", href: "/journal/editorial-board", icon: Users },
  { label: "Author Guidelines", href: "/journal/author-guidelines", icon: FileText },
  { label: "Submit Manuscript", href: "/journal/submit", icon: Upload },
  { label: "Current Issue", href: "/journal/current-issue", icon: Newspaper },
  { label: "Archive", href: "/journal/archive", icon: Archive },
];

const scopeAreas = [
  "Agricultural sciences",
  "Plant breeding and genetics",
  "Crop science",
  "Soil science",
  "Agronomy",
  "Horticulture",
  "Agricultural biotechnology",
  "Data-driven agriculture",
  "Climate-smart agriculture",
  "Related fields",
];

export default function JournalHome() {
  return (
    <JournalLayout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-16 md:py-24">
        <div className="container-wide text-center">
          <h1 className="font-serif text-3xl md:text-5xl font-bold mb-4">
            Journal of Agricultural Innovation (JAI)
          </h1>
          <p className="text-primary-foreground/80 text-lg md:text-xl max-w-2xl mx-auto mb-3">
            Advancing research and innovation in agriculture
          </p>
          <p className="text-primary-foreground/60 text-sm mb-8">
            A peer-reviewed open-access journal published by Field-to-Insight Academy
          </p>
          <p className="text-primary-foreground/50 text-xs mb-10 font-mono">
            ISSN (Online): Pending
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            {heroButtons.map((btn) => (
              <Button key={btn.label} variant="hero-outline" size="sm" asChild>
                <Link to={btn.href}>
                  <btn.icon className="w-4 h-4 mr-1" />
                  {btn.label}
                </Link>
              </Button>
            ))}
          </div>
        </div>
      </section>

      {/* About */}
      <section id="about" className="section-padding bg-background">
        <div className="container-wide max-w-3xl">
          <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-6">About the Journal</h2>
          <div className="prose prose-lg text-foreground/80 space-y-4">
            <p>
              The <strong>Journal of Agricultural Innovation (JAI)</strong> is a peer-reviewed, open-access journal
              published by Field-to-Insight Academy. JAI is committed to disseminating high-quality research
              that advances knowledge and practice in agriculture and related disciplines.
            </p>
            <p>
              All submitted manuscripts undergo a rigorous double-blind peer-review process to ensure
              scientific integrity, originality, and relevance. As an open-access publication, all accepted
              articles are freely available to readers worldwide, promoting the broad dissemination of
              agricultural knowledge.
            </p>
            <p>
              JAI welcomes original research articles, review papers, and short communications from
              researchers, academics, and practitioners across the agricultural sciences.
            </p>
          </div>
        </div>
      </section>

      {/* Aims & Scope */}
      <section id="aims-scope" className="section-padding bg-muted/30">
        <div className="container-wide max-w-3xl">
          <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-6">Aims & Scope</h2>
          <p className="text-foreground/80 mb-6">
            JAI publishes original research articles, review papers, and short communications in the following areas:
          </p>
          <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {scopeAreas.map((area) => (
              <li key={area} className="flex items-start gap-3 bg-background rounded-lg p-4 border border-border">
                <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span className="text-foreground/80">{area}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Editorial Board */}
      <section id="editorial-board" className="section-padding bg-background">
        <div className="container-wide max-w-3xl">
          <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-6">Editorial Board</h2>
          <div className="bg-muted/50 rounded-xl p-8 text-center border border-border">
            <Users className="w-10 h-10 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">Editorial Board details will be updated.</p>
          </div>
        </div>
      </section>

      {/* Author Guidelines */}
      <section id="author-guidelines" className="section-padding bg-muted/30">
        <div className="container-wide max-w-3xl">
          <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-6">Author Guidelines</h2>
          <p className="text-foreground/80 mb-6">
            Authors are encouraged to carefully review the full guidelines before submitting their manuscripts.
            Submissions must adhere to the journal's formatting and ethical standards.
          </p>
          <Button variant="outline" asChild>
            <Link to="/journal/author-guidelines">Read Full Author Guidelines</Link>
          </Button>
        </div>
      </section>

      {/* Submit */}
      <section id="submit" className="section-padding bg-background">
        <div className="container-wide max-w-3xl text-center">
          <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-4">Submit a Manuscript</h2>
          <p className="text-foreground/80 mb-6">
            Ready to share your research? Submit your manuscript through our online submission portal.
          </p>
          <Button variant="gold" size="lg" asChild>
            <Link to="/journal/submit">Go to Submission Portal</Link>
          </Button>
        </div>
      </section>

      {/* Current Issue */}
      <section id="current-issue" className="section-padding bg-muted/30">
        <div className="container-wide max-w-3xl">
          <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-6">Current Issue</h2>
          <div className="bg-background rounded-xl p-8 text-center border border-border">
            <Newspaper className="w-10 h-10 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No articles published yet.</p>
          </div>
        </div>
      </section>

      {/* Archive */}
      <section id="archive" className="section-padding bg-background">
        <div className="container-wide max-w-3xl">
          <h2 className="font-serif text-2xl md:text-3xl font-bold text-foreground mb-6">Archive</h2>
          <div className="bg-muted/50 rounded-xl p-8 text-center border border-border">
            <Archive className="w-10 h-10 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">Archive will be available after first issue.</p>
          </div>
        </div>
      </section>
    </JournalLayout>
  );
}
