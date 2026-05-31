import { useState } from "react";
import { useNavigate, useSearchParams, Link } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { BarChart3, Loader2, ArrowLeft } from "lucide-react";

type Mode = "signin" | "signup";

export default function VivaSenseAuth() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { toast } = useToast();

  const next = searchParams.get("next") || "/vivasense/anova";

  const [mode, setMode] = useState<Mode>("signin");
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (mode === "signup" && password !== confirmPassword) {
      toast({ title: "Passwords do not match", variant: "destructive" });
      return;
    }

    setLoading(true);
    try {
      if (mode === "signin") {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        navigate(next, { replace: true });
      } else {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        toast({
          title: "Account created",
          description: "Check your email to confirm your account, then sign in.",
        });
        setMode("signin");
        setPassword("");
        setConfirmPassword("");
      }
    } catch (err: any) {
      toast({
        title: mode === "signin" ? "Sign in failed" : "Sign up failed",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b1d14] flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-md">

        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center mb-4">
            <BarChart3 className="w-7 h-7 text-emerald-400" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">VivaSense</h1>
          <p className="text-emerald-400/70 text-sm mt-1">Statistical Analysis Platform</p>
        </div>

        {/* Card */}
        <div className="rounded-2xl border border-emerald-400/20 bg-white/[0.04] backdrop-blur-sm p-8">

          {/* Tab switcher */}
          <div className="flex rounded-lg bg-white/5 p-1 mb-6">
            <button
              type="button"
              onClick={() => setMode("signin")}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
                mode === "signin"
                  ? "bg-emerald-500 text-white"
                  : "text-emerald-300/70 hover:text-emerald-300"
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => setMode("signup")}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${
                mode === "signup"
                  ? "bg-emerald-500 text-white"
                  : "text-emerald-300/70 hover:text-emerald-300"
              }`}
            >
              Create Account
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <Label className="text-emerald-200 text-sm">Email</Label>
              <Input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                required
                className="bg-white/5 border-emerald-400/20 text-white placeholder:text-white/30 focus:border-emerald-400"
              />
            </div>

            <div className="space-y-1.5">
              <Label className="text-emerald-200 text-sm">Password</Label>
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                className="bg-white/5 border-emerald-400/20 text-white placeholder:text-white/30 focus:border-emerald-400"
              />
            </div>

            {mode === "signup" && (
              <div className="space-y-1.5">
                <Label className="text-emerald-200 text-sm">Confirm Password</Label>
                <Input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  className="bg-white/5 border-emerald-400/20 text-white placeholder:text-white/30 focus:border-emerald-400"
                />
              </div>
            )}

            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-emerald-500 hover:bg-emerald-400 text-white font-medium mt-2"
            >
              {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              {mode === "signin" ? "Sign In" : "Create Account"}
            </Button>
          </form>

          {mode === "signin" && (
            <p className="text-center text-xs text-emerald-400/50 mt-4">
              Forgot your password?{" "}
              <Link to="/research-writing/reset-password" className="text-emerald-400 hover:underline">
                Reset it here
              </Link>
            </p>
          )}
        </div>

        {/* Back link */}
        <div className="flex justify-center mt-6">
          <Link
            to="/vivasense"
            className="flex items-center gap-1.5 text-sm text-emerald-400/60 hover:text-emerald-400 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Back to VivaSense
          </Link>
        </div>

      </div>
    </div>
  );
}
