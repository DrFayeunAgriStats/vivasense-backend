import { JournalLayout } from "@/components/journal/JournalLayout";

export default function JournalAbout() {
  return (
    <JournalLayout>
      <section className="section-padding bg-background">
        <div className="container-wide max-w-3xl">
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-8">
            About the Journal
          </h1>

          <div className="space-y-6 text-foreground/80 leading-relaxed">
            <p>
              The <strong className="text-foreground">Journal of Agricultural Innovation (JAI)</strong> is
              a peer-reviewed, open-access academic journal published by Field-to-Insight Academy. The journal
              serves as a platform for the dissemination of original and impactful research in agriculture
              and allied disciplines.
            </p>

            <h2 className="font-serif text-xl font-semibold text-foreground pt-4">Open Access Policy</h2>
            <p>
              JAI is fully open access. All published articles are freely available online immediately upon
              publication, ensuring that research findings reach the widest possible audience without
              subscription barriers.
            </p>

            <h2 className="font-serif text-xl font-semibold text-foreground pt-4">Peer Review Process</h2>
            <p>
              All manuscripts submitted to JAI undergo a rigorous double-blind peer-review process.
              Reviewers are selected based on their expertise in the subject area of the manuscript.
              The editorial decision is based on the originality, significance, methodological rigour,
              and clarity of the submission.
            </p>

            <h2 className="font-serif text-xl font-semibold text-foreground pt-4">Publisher</h2>
            <p>
              JAI is published by <strong className="text-foreground">Field-to-Insight Academy</strong>,
              an educational and professional training initiative operated by Able-Flourish Agro-Services Ltd
              (RC 7408450), Nigeria.
            </p>

            <h2 className="font-serif text-xl font-semibold text-foreground pt-4">Article Hosting</h2>
            <p>
              Published articles are hosted on Zenodo, an open-access repository operated by CERN.
              Each article receives a Digital Object Identifier (DOI), ensuring permanent accessibility
              and citability.
            </p>

            <div className="bg-muted/50 rounded-xl p-6 border border-border mt-8">
              <p className="text-sm text-muted-foreground font-mono">ISSN (Online): Pending</p>
            </div>
          </div>
        </div>
      </section>
    </JournalLayout>
  );
}
