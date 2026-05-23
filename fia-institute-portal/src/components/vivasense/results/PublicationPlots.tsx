import { Card, CardContent } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import { FigureDownloadMenu } from "./FigureDownloadMenu";

interface Props {
  plots: Record<string, string>;
}

function formatCaption(plotName: string, index: number): string {
  const name = plotName
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  return `Figure ${index + 1}. ${name}`;
}

/** Check if a string looks like valid base64 image data */
function isValidBase64(str: string): boolean {
  if (!str || str.length < 100) return false;
  return /^[A-Za-z0-9+/\n\r]+=*$/.test(str.slice(0, 200));
}

export function PublicationPlots({ plots }: Props) {
  const entries = Object.entries(plots).filter(([, b64]) => b64 && isValidBase64(b64));
  if (entries.length === 0) return null;

  return (
    <div className="space-y-6">
      <h3 className="font-serif text-xl font-bold text-foreground flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-primary" />
        Figures
      </h3>
      {entries.map(([plotName, base64], index) => (
        <Card key={plotName}>
          <CardContent className="pt-6">
            <div className="flex justify-center">
              <div className="bg-white rounded border border-border p-6 inline-block">
                <img
                  src={`data:image/png;base64,${base64}`}
                  alt={plotName.replace(/_/g, " ")}
                  className="max-w-full"
                  style={{ maxHeight: "480px" }}
                />
              </div>
            </div>
            <div className="flex items-center justify-between mt-4">
              <p className="text-sm text-muted-foreground italic">
                {formatCaption(plotName, index)}
              </p>
              <FigureDownloadMenu title={plotName} base64={base64} />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
