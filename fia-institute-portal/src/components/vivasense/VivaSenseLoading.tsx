import { Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

interface VivaSenseLoadingProps {
  isLoading: boolean;
  retryMessage?: string | null;
}

export function VivaSenseLoading({ isLoading, retryMessage }: VivaSenseLoadingProps) {
  if (!isLoading) return null;

  return (
    <section className="py-8">
      <div className="container-wide">
        <div className="max-w-3xl mx-auto">
          <Card className="border-primary/30 bg-primary/5">
            <CardContent className="p-8">
              <div className="flex flex-col items-center text-center gap-4">
                <Loader2 className="w-12 h-12 text-primary animate-spin" />
                <div>
                  <h3 className="font-semibold text-foreground mb-2">
                    Running Analysis...
                  </h3>
                  <p className="text-muted-foreground">
                    {retryMessage || "Backend waking up… first run may take up to 60 seconds."}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
