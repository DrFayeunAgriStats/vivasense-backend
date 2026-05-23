import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, Copy, Check, FlaskConical, Send } from "lucide-react";
import { MODULE_CONTENT, MODULE_NAMES } from "@/data/bgmModuleContent";
import { useToast } from "@/hooks/use-toast";
import type { BgmStudent } from "@/hooks/useBgmSession";

type Props = {
  student: BgmStudent;
  onBack: () => void;
  onApproved: () => void;
};

export function BgmRPracticeLab({ student, onBack, onApproved }: Props) {
  const mod = student.current_module;
  const content = MODULE_CONTENT[mod];
  const rScript = content?.rScript;
  const { toast } = useToast();

  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState("");
  const [interpretation, setInterpretation] = useState("");
  const [status, setStatus] = useState<"pending" | "approved" | "rejected">("pending");
  const [feedback, setFeedback] = useState<string[]>([]);

  const handleCopy = async () => {
    if (rScript) {
      await navigator.clipboard.writeText(rScript.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleSubmit = () => {
    if (!rScript) return;

    const outputLower = output.toLowerCase();
    const interpLower = interpretation.toLowerCase();
    const missing: string[] = [];

    // Check expected concepts in output or interpretation
    const foundConcepts = rScript.expectedConcepts.filter(
      (c) => outputLower.includes(c.toLowerCase()) || interpLower.includes(c.toLowerCase())
    );

    if (output.trim().length < 50) {
      missing.push("R output is too short. Paste the full console output.");
    }
    if (interpretation.trim().length < 30) {
      missing.push("Provide a more detailed interpretation of the results.");
    }
    if (foundConcepts.length < Math.ceil(rScript.expectedConcepts.length * 0.4)) {
      missing.push(`Your output/interpretation should reference key concepts. Expected some of: ${rScript.expectedConcepts.join(", ")}.`);
    }

    if (missing.length > 0) {
      setStatus("rejected");
      setFeedback(missing);
      toast({ title: "R Practice: Not Approved ❌", description: "See corrections below.", variant: "destructive" });
    } else {
      setStatus("approved");
      setFeedback([]);
      toast({ title: "R Practice: Approved ✅", description: "You may now proceed to the assessment." });
      onApproved();
    }
  };

  if (!rScript) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-muted-foreground">No R script available for this module.</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="bg-primary text-primary-foreground py-3">
        <div className="container-wide flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onBack}
            className="text-primary-foreground hover:bg-primary-foreground/10">
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <FlaskConical className="w-5 h-5" />
          <div>
            <h1 className="font-serif text-lg font-bold">R Practice Lab</h1>
            <p className="text-primary-foreground/70 text-[11px]">
              Module {mod}: {MODULE_NAMES[mod - 1]}
            </p>
          </div>
        </div>
      </div>

      <div className="container-wide py-6 max-w-3xl mx-auto space-y-4">
        {/* Instructions */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">{rScript.title}</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p>1. Copy the R script below and run it in RStudio or R console.</p>
            <p>2. Paste the complete output below.</p>
            <p>3. Write your interpretation of the results.</p>
            <p>4. Submit for approval.</p>
          </CardContent>
        </Card>

        {/* R Script */}
        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-mono">R Script</CardTitle>
            <Button size="sm" variant="outline" onClick={handleCopy}>
              {copied ? <Check className="w-3 h-3 mr-1" /> : <Copy className="w-3 h-3 mr-1" />}
              {copied ? "Copied!" : "Copy"}
            </Button>
          </CardHeader>
          <CardContent>
            <pre className="bg-muted rounded-lg p-4 text-xs overflow-x-auto font-mono leading-relaxed whitespace-pre-wrap">
              {rScript.code}
            </pre>
          </CardContent>
        </Card>

        {/* Paste Output */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Paste R Output Here</CardTitle>
          </CardHeader>
          <CardContent>
            <Textarea
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              placeholder="Paste your complete R console output here..."
              rows={8}
              className="font-mono text-xs"
              disabled={status === "approved"}
            />
          </CardContent>
        </Card>

        {/* Interpretation */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Interpret Output Here</CardTitle>
          </CardHeader>
          <CardContent>
            <Textarea
              value={interpretation}
              onChange={(e) => setInterpretation(e.target.value)}
              placeholder="Write your biological and statistical interpretation of the results..."
              rows={5}
              className="text-sm"
              disabled={status === "approved"}
            />
          </CardContent>
        </Card>

        {/* Feedback */}
        {status === "rejected" && feedback.length > 0 && (
          <Card className="border-destructive">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-destructive">❌ Not Approved — Corrections Required</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="list-disc list-inside text-sm space-y-1 text-muted-foreground">
                {feedback.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </CardContent>
          </Card>
        )}

        {status === "approved" && (
          <Card className="border-primary">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-primary">✅ R Practice: Approved</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              You may now proceed to the module assessment.
            </CardContent>
          </Card>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          <Button onClick={onBack} variant="outline" className="flex-1">
            Back to Dashboard
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={status === "approved" || !output.trim() || !interpretation.trim()}
            className="flex-1"
          >
            <Send className="w-4 h-4 mr-2" />
            {status === "approved" ? "Approved ✅" : "Submit for Review"}
          </Button>
        </div>
      </div>
    </div>
  );
}
