import { AlertCircle, XCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface VivaSenseErrorProps {
  error: string | null;
  onDismiss: () => void;
}

export function VivaSenseError({ error, onDismiss }: VivaSenseErrorProps) {
  if (!error) return null;

  return (
    <section className="py-8">
      <div className="container-wide">
        <div className="max-w-3xl mx-auto">
          <Card className="border-destructive/50 bg-destructive/5">
            <CardContent className="p-6">
              <div className="flex items-start gap-4">
                <AlertCircle className="w-6 h-6 text-destructive flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-destructive mb-2">Analysis Failed</h3>
                  <p className="text-destructive/90 text-sm leading-relaxed whitespace-pre-wrap break-words">
                    {error}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onDismiss}
                  className="flex-shrink-0 text-destructive hover:text-destructive hover:bg-destructive/10"
                >
                  <XCircle className="w-5 h-5" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
