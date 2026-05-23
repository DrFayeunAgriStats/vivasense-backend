import { Sprout, FlaskConical, GraduationCap, Briefcase } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const audience = [
  {
    title: "Plant Breeders",
    description: "Multi-environment trial analysis, stability assessment, and genotype evaluation.",
    icon: Sprout,
  },
  {
    title: "Agronomists",
    description: "Field experiment analysis and treatment comparison workflows.",
    icon: FlaskConical,
  },
  {
    title: "Postgraduate Researchers",
    description: "Publication-ready outputs and AI-assisted interpretation support.",
    icon: GraduationCap,
  },
  {
    title: "Agricultural Consultants",
    description: "Decision-ready analytics and field research reporting.",
    icon: Briefcase,
  },
];

export function VivaSenseAudience() {
  return (
    <section className="py-20 bg-background">
      <div className="container-wide">
        <div className="text-center mb-12">
          <h2 className="font-serif text-4xl lg:text-5xl font-bold text-foreground mb-4">
            Who VivaSense Is For
          </h2>
          <p className="text-muted-foreground text-base lg:text-lg max-w-3xl mx-auto">
            Built for serious agricultural research workflows across academic and professional environments.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
          {audience.map((item) => (
            <Card
              key={item.title}
              className="border-border/70 bg-background hover:border-primary/30 hover:shadow-md transition-all duration-300"
            >
              <CardContent className="p-6 text-center h-full">
                <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                  <item.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">{item.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{item.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
