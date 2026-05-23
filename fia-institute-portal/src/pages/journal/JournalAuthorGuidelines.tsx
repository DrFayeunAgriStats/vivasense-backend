import { JournalLayout } from "@/components/journal/JournalLayout";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

export default function JournalAuthorGuidelines() {
  return (
    <JournalLayout>
      <section className="section-padding bg-background">
        <div className="container-wide max-w-3xl">
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-8">
            Author Guidelines
          </h1>

          <div className="space-y-6 text-foreground/80 leading-relaxed">
            <h2 className="font-serif text-xl font-semibold text-foreground">Manuscript Types</h2>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong className="text-foreground">Original Research Articles</strong> – Full-length papers reporting original research findings.</li>
              <li><strong className="text-foreground">Review Papers</strong> – Comprehensive reviews of a specific topic or field.</li>
              <li><strong className="text-foreground">Short Communications</strong> – Brief reports of significant preliminary findings.</li>
            </ul>

            <h2 className="font-serif text-xl font-semibold text-foreground pt-4">Formatting Requirements</h2>
            <ul className="list-disc pl-6 space-y-2">
              <li>Manuscripts must be written in English.</li>
              <li>Use Times New Roman, 12pt font, double-spaced.</li>
              <li>Include title, abstract (max 250 words), keywords, introduction, materials and methods, results, discussion, conclusion, and references.</li>
              <li>References should follow APA 7th edition format.</li>
              <li>Figures and tables should be embedded within the text at appropriate locations.</li>
            </ul>

            <h2 className="font-serif text-xl font-semibold text-foreground pt-4">Submission Format</h2>
            <ul className="list-disc pl-6 space-y-2">
              <li>Manuscripts should be submitted in PDF or Microsoft Word format.</li>
              <li>Supplementary materials may be uploaded separately.</li>
            </ul>

            <h2 className="font-serif text-xl font-semibold text-foreground pt-4">Ethical Standards</h2>
            <p>
              All submissions must adhere to ethical research standards. Authors must declare any
              conflicts of interest and confirm that the work is original and has not been published
              or submitted elsewhere.
            </p>

            <h2 className="font-serif text-xl font-semibold text-foreground pt-4">Review Process</h2>
            <p>
              All manuscripts undergo double-blind peer review. Authors can expect an initial
              decision within 4–6 weeks of submission.
            </p>

            <div className="pt-8">
              <Button variant="gold" size="lg" asChild>
                <Link to="/journal/submit">Submit Your Manuscript</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </JournalLayout>
  );
}
