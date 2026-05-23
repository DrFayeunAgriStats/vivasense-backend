import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft, Award, CheckCircle, Circle, Loader2, Shield, AlertTriangle,
} from "lucide-react";
import { toast } from "@/hooks/use-toast";

const PASS_THRESHOLDS: Record<string, number> = {
  undergraduate_project: 70,
  msc_thesis: 75,
  phd_research: 80,
  research_paper: 75,
};

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  not_eligible: { label: "Not Eligible", color: "text-muted-foreground" },
  pending_requirements: { label: "Pending Requirements", color: "text-orange-600" },
  pending_review: { label: "Pending Review", color: "text-blue-600" },
  approved: { label: "Approved", color: "text-green-600" },
  issued: { label: "Certificate Issued", color: "text-primary" },
};

export default function RWSCertificateGate() {
  const navigate = useNavigate();
  const { user, profile, loading } = useAuth();
  const [eligibility, setEligibility] = useState<any>(null);
  const [defenseScore, setDefenseScore] = useState<number | null>(null);
  const [milestoneCount, setMilestoneCount] = useState(0);
  const [integrityAccepted, setIntegrityAccepted] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!loading && !user) navigate("/research-writing/signin");
  }, [loading, user]);

  useEffect(() => {
    if (!user) return;
    const fetchData = async () => {
      const [eligRes, defenseRes, milestoneRes] = await Promise.all([
        supabase.from("certificate_eligibility").select("*").eq("user_id", user.id).single(),
        supabase.from("defense_simulator_attempts").select("total_score").eq("user_id", user.id).not("completed_at", "is", null).order("total_score", { ascending: false }).limit(1),
        supabase.from("milestones").select("id", { count: "exact" }).eq("user_id", user.id).eq("is_completed", true),
      ]);
      if (eligRes.data) {
        setEligibility(eligRes.data);
        setIntegrityAccepted(eligRes.data.integrity_accepted);
      }
      if (defenseRes.data?.[0]) setDefenseScore(defenseRes.data[0].total_score);
      if (milestoneRes.count !== null) setMilestoneCount(milestoneRes.count);
      setFetching(false);
    };
    fetchData();
  }, [user]);

  const track = profile?.academic_track || "msc_thesis";
  const threshold = PASS_THRESHOLDS[track] || 75;
  const defenseCompleted = defenseScore !== null && defenseScore >= threshold;
  const milestonesCompleted = milestoneCount >= 3;

  const requirements = [
    { key: "milestones", label: "Complete at least 3 research milestones", met: milestonesCompleted },
    { key: "defense", label: `Complete defense simulation with score ≥ ${threshold}%`, met: defenseCompleted },
    { key: "integrity", label: "Accept academic integrity declaration", met: integrityAccepted },
  ];

  const allMet = requirements.every((r) => r.met);

  const handleIntegrityChange = async (checked: boolean) => {
    setIntegrityAccepted(checked);
    if (user) {
      await supabase.from("certificate_eligibility").upsert({
        user_id: user.id,
        integrity_accepted: checked,
        modules_completed: true,
        milestones_completed: milestonesCompleted,
        defense_completed: defenseCompleted,
        defense_score: defenseScore,
        certificate_status: allMet && checked ? "pending_review" : "pending_requirements",
        updated_at: new Date().toISOString(),
      }, { onConflict: "user_id" });
    }
  };

  const submitForReview = async () => {
    if (!allMet) return;
    setSubmitting(true);
    await supabase.from("certificate_eligibility").upsert({
      user_id: user!.id,
      modules_completed: true,
      milestones_completed: milestonesCompleted,
      defense_completed: defenseCompleted,
      defense_score: defenseScore,
      integrity_accepted: true,
      certificate_status: "pending_review",
      updated_at: new Date().toISOString(),
    }, { onConflict: "user_id" });
    toast({ title: "Submitted", description: "Your certificate application has been submitted for review." });
    setSubmitting(false);
    // Refresh
    const { data } = await supabase.from("certificate_eligibility").select("*").eq("user_id", user!.id).single();
    if (data) setEligibility(data);
  };

  if (loading || fetching) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const currentStatus = eligibility?.certificate_status || "not_eligible";
  const statusMeta = STATUS_LABELS[currentStatus] || STATUS_LABELS.not_eligible;

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-primary text-primary-foreground py-6">
        <div className="container max-w-4xl flex items-center justify-between">
          <div>
            <p className="text-primary-foreground/70 text-xs uppercase tracking-wider mb-1">FIA Research Writing System</p>
            <h1 className="font-serif text-2xl font-bold flex items-center gap-2">
              <Award className="w-6 h-6" /> Certificate of Competence
            </h1>
          </div>
          <Link to="/research-writing/dashboard">
            <Button variant="secondary" size="sm" className="gap-1"><ArrowLeft className="w-4 h-4" /> Dashboard</Button>
          </Link>
        </div>
      </header>

      <div className="container max-w-3xl py-8 space-y-6">
        <Card>
          <CardContent className="pt-6 text-center space-y-3">
            <Award className="w-12 h-12 mx-auto text-primary" />
            <h2 className="font-serif text-xl font-bold">Certificate Status</h2>
            <p className={`font-medium ${statusMeta.color}`}>{statusMeta.label}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-lg">Eligibility Checklist</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {requirements.map((req) => (
              <div key={req.key} className="flex items-start gap-3 p-3 rounded-lg border border-border">
                {req.met ? (
                  <CheckCircle className="w-5 h-5 text-green-600 shrink-0 mt-0.5" />
                ) : (
                  <Circle className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
                )}
                <div>
                  <p className={`text-sm ${req.met ? "text-foreground" : "text-muted-foreground"}`}>{req.label}</p>
                  {req.key === "defense" && defenseScore !== null && (
                    <p className="text-xs text-muted-foreground mt-0.5">Your best score: {defenseScore}/100</p>
                  )}
                  {req.key === "milestones" && (
                    <p className="text-xs text-muted-foreground mt-0.5">{milestoneCount} milestone{milestoneCount !== 1 ? "s" : ""} completed</p>
                  )}
                </div>
              </div>
            ))}

            {!integrityAccepted && (
              <div className="bg-muted/50 p-4 rounded-lg space-y-3">
                <div className="flex items-start gap-2">
                  <Shield className="w-5 h-5 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium">Academic Integrity Declaration</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      I declare that all work submitted through this platform is my own. I have used the AI tools
                      for guidance and learning, not for generating submission-ready text. I understand that the
                      certificate represents demonstrated competence in research methodology and scientific writing.
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    checked={integrityAccepted}
                    onCheckedChange={(v) => handleIntegrityChange(!!v)}
                  />
                  <label className="text-sm">I accept the academic integrity declaration</label>
                </div>
              </div>
            )}

            {allMet && currentStatus !== "pending_review" && currentStatus !== "approved" && currentStatus !== "issued" && (
              <Button onClick={submitForReview} disabled={submitting} className="w-full gap-2">
                {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Award className="w-4 h-4" />}
                Submit for Review
              </Button>
            )}

            {currentStatus === "pending_review" && (
              <div className="bg-blue-50 dark:bg-blue-950/20 p-4 rounded-lg text-center">
                <p className="text-sm text-blue-600">Your application is under review. You will be notified when a decision is made.</p>
              </div>
            )}

            {currentStatus === "issued" && (
              <div className="bg-green-50 dark:bg-green-950/20 p-4 rounded-lg text-center">
                <CheckCircle className="w-8 h-8 mx-auto text-green-600 mb-2" />
                <p className="text-sm text-green-600 font-medium">Your certificate has been issued. Congratulations!</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
