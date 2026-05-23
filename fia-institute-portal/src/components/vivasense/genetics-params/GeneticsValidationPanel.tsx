import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
import type { GeneticsValidationResult } from "@/types/genetics";

interface Props {
  result: GeneticsValidationResult | null;
}

export function GeneticsValidationPanel({ result }: Props) {
  if (!result) return null;

  return (
    <Card className="border border-primary/20">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          {result.valid ? (
            <CheckCircle2 className="h-5 w-5 text-green-600" />
          ) : (
            <XCircle className="h-5 w-5 text-destructive" />
          )}
          Input Validation
          <Badge variant={result.valid ? "default" : "destructive"} className="ml-auto text-xs">
            {result.valid ? "Passed" : "Failed"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {result.errors && result.errors.length > 0 && (
          <div className="space-y-1">
            {result.errors.map((err, i) => (
              <p key={i} className="text-sm text-destructive flex items-start gap-1.5">
                <XCircle className="h-4 w-4 mt-0.5 shrink-0" />
                {err}
              </p>
            ))}
          </div>
        )}
        {result.warnings && result.warnings.length > 0 && (
          <div className="space-y-1">
            {result.warnings.map((w, i) => (
              <p key={i} className="text-sm text-amber-700 flex items-start gap-1.5">
                <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
                {w}
              </p>
            ))}
          </div>
        )}
        {result.valid && (!result.warnings || result.warnings.length === 0) && (
          <p className="text-sm text-muted-foreground">All inputs are within acceptable ranges.</p>
        )}
      </CardContent>
    </Card>
  );
}
