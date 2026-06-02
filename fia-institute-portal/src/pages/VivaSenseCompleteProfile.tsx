import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { BarChart3, Loader2 } from "lucide-react";

const REGISTRATION_SOURCES = [
  "Select one…",
  "Google Search",
  "WhatsApp / Telegram",
  "FIA Newsletter",
  "Colleague / Supervisor",
  "Social Media",
  "FIA Workshop or Training",
  "Other",
];

interface ExistingProfile {
  full_name: string | null;
  institution: string | null;
  position: string | null;
  research_area: string | null;
}

export default function VivaSenseCompleteProfile() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { toast } = useToast();

  const next = searchParams.get("next") || "/vivasense/anova";

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  const [profileExists, setProfileExists] = useState(false);

  // Form fields
  const [fullName, setFullName] = useState("");
  const [institution, setInstitution] = useState("");
  const [position, setPosition] = useState("");
  const [researchArea, setResearchArea] = useState("");
  const [registrationSource, setRegistrationSource] = useState(REGISTRATION_SOURCES[0]);

  // Terms checkboxes
  const [termsAccepted, setTermsAccepted] = useState(false);
  const [privacyAccepted, setPrivacyAccepted] = useState(false);
  const [dataUsageAccepted, setDataUsageAccepted] = useState(false);

  const allTermsChecked = termsAccepted && privacyAccepted && dataUsageAccepted;
  const validSource = registrationSource !== REGISTRATION_SOURCES[0];

  useEffect(() => {
    const init = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session?.user) {
        navigate(
          `/vivasense/auth?next=${encodeURIComponent(`/vivasense/complete-profile?next=${encodeURIComponent(next)}`)}`,
          { replace: true }
        );
        return;
      }

      setUserId(session.user.id);

      const { data: profile } = await supabase
        .from("profiles")
        .select("full_name, institution, position, research_area")
        .eq("id", session.user.id)
        .maybeSingle();

      if (profile) {
        setProfileExists(true);
        // Pre-populate only if the field has a value — never overwrite with blank
        if (profile.full_name)    setFullName(profile.full_name);
        if (profile.institution)  setInstitution(profile.institution);
        if (profile.position)     setPosition(profile.position);
        if (profile.research_area) setResearchArea(profile.research_area);
      } else {
        setProfileExists(false);
        // Pre-populate full_name from auth metadata if available
        const meta = session.user.user_metadata;
        if (meta?.full_name) setFullName(meta.full_name as string);
      }

      setLoading(false);
    };

    void init();
  }, [navigate, next]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!fullName.trim()) {
      toast({ title: "Full name is required", variant: "destructive" });
      return;
    }
    if (!allTermsChecked) {
      toast({ title: "Please accept all three terms to continue", variant: "destructive" });
      return;
    }
    if (!validSource) {
      toast({ title: "Please select how you heard about VivaSense", variant: "destructive" });
      return;
    }
    if (!userId) return;

    setSaving(true);
    try {
      const now = new Date().toISOString();

      // Build update payload — only include fields that are non-empty
      // to avoid overwriting existing data with blanks.
      const payload: Record<string, string | boolean | null> = {
        platform_source: "vivasense",
        terms_accepted_at: now,
        registration_source: registrationSource,
        updated_at: now,
      };
      if (fullName.trim())     payload.full_name     = fullName.trim();
      if (institution.trim())  payload.institution   = institution.trim();
      if (position.trim())     payload.position      = position.trim();
      if (researchArea.trim()) payload.research_area = researchArea.trim();

      if (profileExists) {
        const { error } = await supabase
          .from("profiles")
          .update(payload)
          .eq("id", userId);
        if (error) throw error;
      } else {
        // No profile row exists at all — insert a minimal record
        const { error } = await supabase
          .from("profiles")
          .insert({
            id: userId,
            email: (await supabase.auth.getSession()).data.session?.user.email ?? "",
            full_name: fullName.trim(),
            onboarding_completed: false,
            ...payload,
          });
        if (error) throw error;
      }

      toast({ title: "Profile saved. Welcome to VivaSense!" });
      navigate(next, { replace: true });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      toast({ title: "Could not save profile", description: message, variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0b1d14] flex flex-col items-center justify-center gap-4">
        <div className="w-12 h-12 border-4 border-emerald-400 border-t-transparent rounded-full animate-spin" />
        <p className="text-emerald-300 font-medium">Loading your profile…</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0b1d14] flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-lg">

        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center mb-4">
            <BarChart3 className="w-7 h-7 text-emerald-400" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">VivaSense</h1>
          <p className="text-emerald-400/70 text-sm mt-1">Complete your researcher profile</p>
        </div>

        <div className="rounded-2xl border border-emerald-400/20 bg-white/[0.04] backdrop-blur-sm p-8">
          <p className="text-emerald-200/70 text-sm mb-6 leading-relaxed">
            Before accessing VivaSense, please confirm your details and accept the current terms.
            {profileExists && " We have pre-filled any information already on file."}
          </p>

          <form onSubmit={handleSubmit} className="space-y-5">

            {/* Full name */}
            <div className="space-y-1.5">
              <Label className="text-emerald-200 text-sm">Full name <span className="text-red-400">*</span></Label>
              <Input
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Dr. Jane Okonkwo"
                required
                className="bg-white/5 border-emerald-400/20 text-white placeholder:text-white/30 focus:border-emerald-400"
              />
            </div>

            {/* Institution */}
            <div className="space-y-1.5">
              <Label className="text-emerald-200 text-sm">Institution</Label>
              <Input
                value={institution}
                onChange={(e) => setInstitution(e.target.value)}
                placeholder="University of Agriculture, Abeokuta"
                className="bg-white/5 border-emerald-400/20 text-white placeholder:text-white/30 focus:border-emerald-400"
              />
            </div>

            {/* Position */}
            <div className="space-y-1.5">
              <Label className="text-emerald-200 text-sm">Position / role</Label>
              <Input
                value={position}
                onChange={(e) => setPosition(e.target.value)}
                placeholder="e.g. MSc Student, PhD Researcher, Lecturer"
                className="bg-white/5 border-emerald-400/20 text-white placeholder:text-white/30 focus:border-emerald-400"
              />
            </div>

            {/* Research area */}
            <div className="space-y-1.5">
              <Label className="text-emerald-200 text-sm">Research area</Label>
              <Input
                value={researchArea}
                onChange={(e) => setResearchArea(e.target.value)}
                placeholder="e.g. Crop Improvement, Plant Breeding, Agronomy"
                className="bg-white/5 border-emerald-400/20 text-white placeholder:text-white/30 focus:border-emerald-400"
              />
            </div>

            {/* Registration source */}
            <div className="space-y-1.5">
              <Label className="text-emerald-200 text-sm">
                How did you hear about VivaSense? <span className="text-red-400">*</span>
              </Label>
              <select
                value={registrationSource}
                onChange={(e) => setRegistrationSource(e.target.value)}
                required
                className="w-full rounded-md border border-emerald-400/20 bg-white/5 px-3 py-2 text-sm text-white focus:border-emerald-400 focus:outline-none"
              >
                {REGISTRATION_SOURCES.map((s) => (
                  <option key={s} value={s} className="bg-[#0b1d14]">{s}</option>
                ))}
              </select>
            </div>

            {/* Terms */}
            <div className="rounded-xl border border-emerald-400/20 bg-white/[0.03] p-4 space-y-3">
              <p className="text-xs text-emerald-300/70 uppercase tracking-wide font-semibold mb-1">
                Terms &amp; conditions
              </p>

              <label className="flex items-start gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={termsAccepted}
                  onChange={(e) => setTermsAccepted(e.target.checked)}
                  className="mt-0.5 h-4 w-4 accent-emerald-500 shrink-0"
                />
                <span className="text-sm text-emerald-200/80 group-hover:text-emerald-200 leading-relaxed">
                  I have read and accept the{" "}
                  <a href="/privacy" target="_blank" rel="noopener noreferrer" className="underline text-emerald-400">
                    Terms of Service
                  </a>
                  .
                </span>
              </label>

              <label className="flex items-start gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={privacyAccepted}
                  onChange={(e) => setPrivacyAccepted(e.target.checked)}
                  className="mt-0.5 h-4 w-4 accent-emerald-500 shrink-0"
                />
                <span className="text-sm text-emerald-200/80 group-hover:text-emerald-200 leading-relaxed">
                  I have read and accept the{" "}
                  <a href="/privacy" target="_blank" rel="noopener noreferrer" className="underline text-emerald-400">
                    Privacy Policy
                  </a>
                  .
                </span>
              </label>

              <label className="flex items-start gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={dataUsageAccepted}
                  onChange={(e) => setDataUsageAccepted(e.target.checked)}
                  className="mt-0.5 h-4 w-4 accent-emerald-500 shrink-0"
                />
                <span className="text-sm text-emerald-200/80 group-hover:text-emerald-200 leading-relaxed">
                  I understand that my analysis data may be used in aggregate, anonymised form
                  to improve VivaSense services.
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={saving || !allTermsChecked || !validSource || !fullName.trim()}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50 transition-colors"
            >
              {saving ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Saving…
                </>
              ) : (
                "Complete Profile & Enter VivaSense"
              )}
            </button>

          </form>
        </div>
      </div>
    </div>
  );
}
