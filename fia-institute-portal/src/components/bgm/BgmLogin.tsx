import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dna, Loader2, AlertCircle } from "lucide-react";

type Props = {
  onLogin: (code: string, name: string, regId: string) => Promise<void>;
};

export function BgmLogin({ onLogin }: Props) {
  const [code, setCode] = useState("");
  const [name, setName] = useState("");
  const [regId, setRegId] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!code.trim() || !name.trim() || !regId.trim()) {
      setError("All fields are required.");
      return;
    }
    if (regId.trim().length < 6) {
      setError("Registration ID must be at least 6 characters.");
      return;
    }

    setLoading(true);
    try {
      await onLogin(code.trim(), name.trim(), regId.trim());
    } catch (err: any) {
      setError(err.message || "Login failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        {/* Branding */}
        <div className="text-center space-y-2">
          <div className="mx-auto w-16 h-16 rounded-full bg-primary flex items-center justify-center">
            <Dna className="w-8 h-8 text-primary-foreground" />
          </div>
          <h1 className="font-serif text-2xl font-bold text-foreground">
            Biometrical Genetics Mastery Tutor
          </h1>
          <p className="text-sm text-muted-foreground">
            Advanced Quantitative Genetics & Breeding
          </p>
          <p className="text-xs text-muted-foreground">
            Field-to-Insight Academy
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Student Login</CardTitle>
            <CardDescription>
              Enter your student code and details to access the tutor.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="code">Student Code</Label>
                <Input
                  id="code"
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="e.g., FIA-BGM-001"
                  disabled={loading}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <Input
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Enter your full name"
                  disabled={loading}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="regId">Matric Number / Registration ID</Label>
                <Input
                  id="regId"
                  value={regId}
                  onChange={(e) => setRegId(e.target.value)}
                  placeholder="Minimum 6 characters"
                  disabled={loading}
                />
              </div>

              {error && (
                <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 p-3 rounded-md">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              )}

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
                Access Tutor
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
