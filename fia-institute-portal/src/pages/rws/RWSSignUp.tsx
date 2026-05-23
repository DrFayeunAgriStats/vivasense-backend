import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { GraduationCap, Loader2 } from "lucide-react";

const DISCIPLINES = [
  "Plant Breeding",
  "Agronomy",
  "Crop Science",
  "Molecular Genetics",
  "Soil Science",
  "Horticulture",
  "Animal Science",
  "General Agricultural Science",
  "Other",
];

const TRACKS = [
  { value: "undergraduate_project", label: "Undergraduate Final-Year Project" },
  { value: "msc_thesis", label: "MSc Thesis Writing" },
  { value: "phd_research", label: "PhD Research Writing and Defense" },
  { value: "research_paper", label: "Research Paper Writing" },
];

export default function RWSSignUp() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    fullName: "",
    email: "",
    password: "",
    role: "student",
    track: "",
    discipline: "",
    institution: "",
    country: "",
  });

  const update = (field: string, value: string) => setForm((p) => ({ ...p, [field]: value }));

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.fullName || !form.email || !form.password || !form.track || !form.discipline) {
      toast({ title: "Missing fields", description: "Please fill in all required fields.", variant: "destructive" });
      return;
    }
    if (form.password.length < 6) {
      toast({ title: "Weak password", description: "Password must be at least 6 characters.", variant: "destructive" });
      return;
    }

    setLoading(true);
    try {
      const { data, error } = await supabase.auth.signUp({
        email: form.email,
        password: form.password,
        options: {
          data: { full_name: form.fullName },
          emailRedirectTo: window.location.origin + "/research-writing/onboarding",
        },
      });

      if (error) throw error;

      if (data.user) {
        // Update profile with additional data
        await supabase.from("profiles").update({
          full_name: form.fullName,
          academic_track: form.track as any,
          discipline: form.discipline,
          institution: form.institution || null,
          country: form.country || null,
        }).eq("id", data.user.id);
      }

      toast({
        title: "Account created",
        description: "Please check your email to verify your account before signing in.",
      });
      navigate("/research-writing/signin");
    } catch (err: any) {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Card className="w-full max-w-lg">
        <CardHeader className="text-center">
          <div className="mx-auto w-12 h-12 rounded-lg bg-primary flex items-center justify-center mb-3">
            <GraduationCap className="w-6 h-6 text-primary-foreground" />
          </div>
          <CardTitle className="font-serif text-2xl">Create Your Account</CardTitle>
          <CardDescription>Join the FIA Research Writing System</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSignUp} className="space-y-4">
            <div className="space-y-2">
              <Label>Full Name *</Label>
              <Input value={form.fullName} onChange={(e) => update("fullName", e.target.value)} placeholder="Your full name" />
            </div>
            <div className="space-y-2">
              <Label>Email *</Label>
              <Input type="email" value={form.email} onChange={(e) => update("email", e.target.value)} placeholder="your@email.com" />
            </div>
            <div className="space-y-2">
              <Label>Password *</Label>
              <Input type="password" value={form.password} onChange={(e) => update("password", e.target.value)} placeholder="Min. 6 characters" />
            </div>
            <div className="space-y-2">
              <Label>Academic Track *</Label>
              <Select value={form.track} onValueChange={(v) => update("track", v)}>
                <SelectTrigger><SelectValue placeholder="Select your track" /></SelectTrigger>
                <SelectContent>
                  {TRACKS.map((t) => (
                    <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Discipline *</Label>
              <Select value={form.discipline} onValueChange={(v) => update("discipline", v)}>
                <SelectTrigger><SelectValue placeholder="Select your discipline" /></SelectTrigger>
                <SelectContent>
                  {DISCIPLINES.map((d) => (
                    <SelectItem key={d} value={d}>{d}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Institution</Label>
                <Input value={form.institution} onChange={(e) => update("institution", e.target.value)} placeholder="University name" />
              </div>
              <div className="space-y-2">
                <Label>Country</Label>
                <Input value={form.country} onChange={(e) => update("country", e.target.value)} placeholder="Country" />
              </div>
            </div>
            <Button type="submit" className="w-full" disabled={loading}>
              {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Create Account
            </Button>
            <p className="text-center text-sm text-muted-foreground">
              Already have an account?{" "}
              <Link to="/research-writing/signin" className="text-primary hover:underline">Sign in</Link>
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
