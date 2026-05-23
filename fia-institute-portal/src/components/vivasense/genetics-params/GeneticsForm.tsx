import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Play, Loader2, ShieldCheck, FlaskConical } from "lucide-react";
import type { GeneticsMode, GeneticsInputData, GeneticsRequestBody } from "@/types/genetics";

const SAMPLE_SINGLE: { mode: GeneticsMode; data: GeneticsInputData } = {
  mode: "single_environment",
  data: { MSG: 25.4, MSE: 6.2, MSGE: 0, replications: 3, num_environments: 1, trait_mean: 42.5, selection_intensity: 2.06 },
};

const SAMPLE_MULTI: { mode: GeneticsMode; data: GeneticsInputData } = {
  mode: "multi_environment",
  data: { MSG: 32.5, MSE: 5.6, MSGE: 14.2, replications: 3, num_environments: 4, trait_mean: 48.3, selection_intensity: 2.06 },
};

interface GeneticsFormProps {
  onValidate: (body: GeneticsRequestBody) => void;
  onAnalyze: (body: GeneticsRequestBody) => void;
  isLoading: boolean;
  isValidating: boolean;
}

const FIELD_DEFS: { key: keyof GeneticsInputData; label: string; tooltip: string; step?: string }[] = [
  { key: "MSG", label: "MSG (Mean Square Genotype)", tooltip: "Mean square for genotype from ANOVA table" },
  { key: "MSE", label: "MSE (Mean Square Error)", tooltip: "Error mean square from ANOVA table" },
  { key: "MSGE", label: "MSGE (Mean Square G×E)", tooltip: "Genotype × Environment interaction mean square (0 for single-env)" },
  { key: "replications", label: "Replications (r)", tooltip: "Number of replications per environment", step: "1" },
  { key: "num_environments", label: "Number of Environments", tooltip: "Number of environments / locations (1 for single-env)", step: "1" },
  { key: "trait_mean", label: "Trait Grand Mean", tooltip: "Overall mean of the trait across all observations" },
  { key: "selection_intensity", label: "Selection Intensity (k)", tooltip: "Standardised selection differential (e.g. 2.06 for 5%)", step: "0.01" },
];

const DEFAULTS: GeneticsInputData = {
  MSG: 0,
  MSE: 0,
  MSGE: 0,
  replications: 3,
  num_environments: 1,
  trait_mean: 0,
  selection_intensity: 2.06,
};

export function GeneticsForm({ onValidate, onAnalyze, isLoading, isValidating }: GeneticsFormProps) {
  const [mode, setMode] = useState<GeneticsMode>("auto");
  const [data, setData] = useState<GeneticsInputData>({ ...DEFAULTS });

  const update = (key: keyof GeneticsInputData, raw: string) => {
    setData((prev) => ({ ...prev, [key]: raw === "" ? 0 : Number(raw) }));
  };

  const buildBody = (): GeneticsRequestBody => ({ mode, validate_only: false, data });

  return (
    <Card className="border-2 border-primary/20 shadow-lg">
      <CardContent className="p-6 space-y-6">
        {/* Mode selector */}
        <div className="space-y-2">
          <Label className="text-base font-medium">
            Analysis Mode <span className="text-destructive">*</span>
          </Label>
          <Select value={mode} onValueChange={(v) => setMode(v as GeneticsMode)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto-detect</SelectItem>
              <SelectItem value="single_environment">Single Environment</SelectItem>
              <SelectItem value="multi_environment">Multi-Environment</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Auto-detect infers from your input whether single or multi-environment estimation applies.
          </p>
        </div>

        {/* Input fields */}
        <div className="grid gap-4 sm:grid-cols-2">
          {FIELD_DEFS.map(({ key, label, tooltip, step }) => (
            <div key={key} className="space-y-1.5">
              <Label htmlFor={key} className="text-sm font-medium">
                {label}
              </Label>
              <Input
                id={key}
                type="number"
                step={step ?? "any"}
                min={0}
                value={data[key] || ""}
                onChange={(e) => update(key, e.target.value)}
                placeholder={tooltip}
              />
              <p className="text-xs text-muted-foreground">{tooltip}</p>
            </div>
          ))}
        </div>

        {/* Sample data buttons */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-xs font-medium text-muted-foreground mr-1">Load sample:</span>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => { setMode(SAMPLE_SINGLE.mode); setData({ ...SAMPLE_SINGLE.data }); }}
          >
            <FlaskConical className="h-3.5 w-3.5" />
            Single-Environment
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => { setMode(SAMPLE_MULTI.mode); setData({ ...SAMPLE_MULTI.data }); }}
          >
            <FlaskConical className="h-3.5 w-3.5" />
            Multi-Environment
          </Button>
        </div>

        {/* Action buttons */}
        <div className="flex flex-wrap gap-3 pt-2">
          <Button
            variant="outline"
            onClick={() => onValidate(buildBody())}
            disabled={isValidating || isLoading}
            className="gap-2"
          >
            {isValidating ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
            Validate Input
          </Button>
          <Button
            onClick={() => onAnalyze(buildBody())}
            disabled={isLoading}
            className="gap-2"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Genetic Parameter Estimation
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
