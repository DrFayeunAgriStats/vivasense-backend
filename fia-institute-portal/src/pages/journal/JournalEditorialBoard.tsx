import { Users } from "lucide-react";
import { JournalLayout } from "@/components/journal/JournalLayout";

export default function JournalEditorialBoard() {
  return (
    <JournalLayout>
      <section className="section-padding bg-background">
        <div className="container-wide max-w-3xl">
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-8">
            Editorial Board
          </h1>
          <div className="bg-muted/50 rounded-xl p-12 text-center border border-border">
            <Users className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground text-lg">
              Editorial Board details will be updated.
            </p>
          </div>
        </div>
      </section>
    </JournalLayout>
  );
}
