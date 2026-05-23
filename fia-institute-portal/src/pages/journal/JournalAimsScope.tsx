import { JournalLayout } from "@/components/journal/JournalLayout";

const scopeAreas = [
  { title: "Agricultural Sciences", description: "Broad research across agricultural disciplines." },
  { title: "Plant Breeding and Genetics", description: "Genetic improvement of crops, molecular breeding, and genomics." },
  { title: "Crop Science", description: "Crop production, physiology, and management." },
  { title: "Soil Science", description: "Soil fertility, soil health, and land management." },
  { title: "Agronomy", description: "Field crop production systems and sustainable practices." },
  { title: "Horticulture", description: "Fruit, vegetable, and ornamental crop research." },
  { title: "Agricultural Biotechnology", description: "Application of biotechnology tools in agriculture." },
  { title: "Data-Driven Agriculture", description: "Precision agriculture, remote sensing, and agricultural informatics." },
  { title: "Climate-Smart Agriculture", description: "Adaptation and mitigation strategies for agricultural systems." },
  { title: "Related Fields", description: "Interdisciplinary research connecting agriculture with other sciences." },
];

export default function JournalAimsScope() {
  return (
    <JournalLayout>
      <section className="section-padding bg-background">
        <div className="container-wide max-w-3xl">
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-4">
            Aims & Scope
          </h1>
          <p className="text-foreground/80 leading-relaxed mb-8">
            The Journal of Agricultural Innovation (JAI) publishes original research articles,
            review papers, and short communications that contribute to the advancement of
            agricultural knowledge and practice. The journal covers a wide range of subject areas
            including but not limited to:
          </p>

          <div className="space-y-4">
            {scopeAreas.map((area) => (
              <div key={area.title} className="bg-muted/30 rounded-xl p-5 border border-border">
                <h3 className="font-semibold text-foreground mb-1">{area.title}</h3>
                <p className="text-sm text-muted-foreground">{area.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </JournalLayout>
  );
}
