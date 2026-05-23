import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { GraduationCap, ArrowRight, CheckCircle, BookOpen, FlaskConical, Target } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const TRACKS = [
  { value: "undergraduate_project", label: "Undergraduate Final-Year Project", duration: "Flexible" },
  { value: "msc_thesis", label: "MSc Thesis Writing", duration: "1 month" },
  { value: "phd_research", label: "PhD Research Writing and Defense", duration: "2 months" },
  { value: "research_paper", label: "Research Paper Writing", duration: "2 weeks" },
];

const DISCIPLINES = [
  "Plant Breeding", "Agronomy", "Crop Science", "Molecular Genetics",
  "Soil Science", "Horticulture", "Animal Science", "General Agricultural Science", "Other",
];

const STAGES = [
  { value: "topic_proposal", label: "Topic / Proposal", icon: Target },
  { value: "literature_review", label: "Literature Review", icon: BookOpen },
  { value: "methodology", label: "Methodology", icon: FlaskConical },
  { value: "data_analysis", label: "Data Analysis", icon: Target },
  { value: "results_writing", label: "Results Writing", icon: BookOpen },
  { value: "discussion", label: "Discussion", icon: FlaskConical },
  { value: "defense_preparation", label: "Defense Preparation", icon: GraduationCap },
];

export default function RWSOnboarding() {
  const navigate = useNavigate();
  const { user, profile, refreshProfile } = useAuth();
  const { toast } = useToast();
  const [step, setStep] = useState(1);
  const [track, setTrack] = useState(profile?.academic_track || "");
  const [discipline, setDiscipline] = useState(profile?.discipline || "");
  const [stage, setStage] = useState(profile?.current_research_stage || "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!user) navigate("/research-writing/signin");
    if (profile?.onboarding_completed) navigate("/research-writing/dashboard");
  }, [user, profile]);

  useEffect(() => {
    if (profile) {
      setTrack(profile.academic_track || "");
      setDiscipline(profile.discipline || "");
      setStage(profile.current_research_stage || "");
    }
  }, [profile]);

  const handleNext = async () => {
    if (step < 3) {
      setStep(step + 1);
      return;
    }
    // Final step - save and proceed to diagnostic
    setSaving(true);
    try {
      const { error } = await supabase.from("profiles").update({
        academic_track: track as any,
        discipline,
        current_research_stage: stage as any,
      }).eq("id", user!.id);
      if (error) throw error;
      await refreshProfile();
      navigate("/research-writing/diagnostic");
    } catch (err: any) {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  const canProceed = step === 1 ? !!track : step === 2 ? !!discipline : !!stage;

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-lg space-y-6">
        <div className="text-center">
          <div className="mx-auto w-12 h-12 rounded-lg bg-primary flex items-center justify-center mb-3">
            <GraduationCap className="w-6 h-6 text-primary-foreground" />
          </div>
          <h1 className="font-serif text-2xl font-bold text-foreground">Welcome to FIA</h1>
          <p className="text-muted-foreground text-sm mt-1">Let's personalise your research journey</p>
        </div>

        <Progress value={(step / 3) * 100} className="h-2" />
        <p className="text-xs text-muted-foreground text-center">Step {step} of 3</p>

        <Card>
          {step === 1 && (
            <>
              <CardHeader>
                <CardTitle className="text-lg">Confirm Your Academic Track</CardTitle>
                <CardDescription>This determines your learning pathway and content.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {TRACKS.map((t) => (
                  <button
                    key={t.value}
                    onClick={() => setTrack(t.value)}
                    className={`w-full text-left p-4 rounded-lg border transition-colors ${
                      track === t.value
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-primary/50"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium text-sm text-foreground">{t.label}</p>
                        <p className="text-xs text-muted-foreground mt-0.5">Duration: {t.duration}</p>
                      </div>
                      {track === t.value && <CheckCircle className="w-5 h-5 text-primary" />}
                    </div>
                  </button>
                ))}
              </CardContent>
            </>
          )}

          {step === 2 && (
            <>
              <CardHeader>
                <CardTitle className="text-lg">Confirm Your Discipline</CardTitle>
                <CardDescription>We'll tailor examples and guidance to your field.</CardDescription>
              </CardHeader>
              <CardContent>
                <Select value={discipline} onValueChange={setDiscipline}>
                  <SelectTrigger><SelectValue placeholder="Select discipline" /></SelectTrigger>
                  <SelectContent>
                    {DISCIPLINES.map((d) => (
                      <SelectItem key={d} value={d}>{d}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </>
          )}

          {step === 3 && (
            <>
              <CardHeader>
                <CardTitle className="text-lg">Current Research Stage</CardTitle>
                <CardDescription>Where are you in your research journey?</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {STAGES.map((s) => (
                  <button
                    key={s.value}
                    onClick={() => setStage(s.value)}
                    className={`w-full text-left p-3 rounded-lg border transition-colors flex items-center gap-3 ${
                      stage === s.value
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-primary/50"
                    }`}
                  >
                    <s.icon className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm font-medium text-foreground">{s.label}</span>
                    {stage === s.value && <CheckCircle className="w-4 h-4 text-primary ml-auto" />}
                  </button>
                ))}
              </CardContent>
            </>
          )}
        </Card>

        <div className="flex justify-between">
          {step > 1 && (
            <Button variant="outline" onClick={() => setStep(step - 1)}>Back</Button>
          )}
          <Button
            className="ml-auto gap-2"
            onClick={handleNext}
            disabled={!canProceed || saving}
          >
            {step === 3 ? "Start Diagnostic" : "Continue"}
            <ArrowRight className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
