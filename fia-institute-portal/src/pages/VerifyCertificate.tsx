import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { supabase } from "@/integrations/supabase/client";
import { Award, CheckCircle2, XCircle, Search, ShieldCheck } from "lucide-react";

type VerificationResult = {
  found: boolean;
  full_name?: string;
  registration_id?: string;
  completion_token?: string;
  completed_at?: string;
};

export default function VerifyCertificate() {
  const [searchParams] = useSearchParams();
  const tokenFromUrl = searchParams.get("token") || "";
  const [token, setToken] = useState(tokenFromUrl);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  // Auto-verify if token came from URL
  useState(() => {
    if (tokenFromUrl) {
      handleVerify(tokenFromUrl);
    }
  });

  async function handleVerify(t?: string) {
    const searchToken = (t || token).trim();
    if (!searchToken) return;
    setLoading(true);
    setSearched(true);

    const { data } = await supabase
      .from("bgm_students")
      .select("full_name, registration_id, completion_token, updated_at")
      .eq("completion_token", searchToken)
      .eq("token_status", "Generated")
      .maybeSingle();

    if (data) {
      setResult({
        found: true,
        full_name: data.full_name,
        registration_id: data.registration_id,
        completion_token: data.completion_token!,
        completed_at: data.updated_at,
      });
    } else {
      setResult({ found: false });
    }
    setLoading(false);
  }

  return (
    <Layout>
      <section className="bg-primary text-primary-foreground py-16 md:py-24">
        <div className="container-wide">
          <div className="max-w-3xl">
            <div className="flex items-center gap-3 mb-4">
              <ShieldCheck className="w-8 h-8" />
              <h1 className="font-serif text-3xl md:text-4xl font-bold">
                Certificate Verification
              </h1>
            </div>
            <p className="text-lg text-primary-foreground/85">
              Verify the authenticity of a Field-to-Insight Academy Certificate of Competence.
            </p>
          </div>
        </div>
      </section>

      <section className="section-padding">
        <div className="container-wide max-w-2xl mx-auto">
          {/* Search form */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Search className="w-5 h-5 text-primary" />
                Enter Certificate Token
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex gap-3">
                <Input
                  placeholder="e.g. BGM-ADV-123456"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleVerify()}
                  className="font-mono"
                />
                <Button onClick={() => handleVerify()} disabled={loading || !token.trim()}>
                  {loading ? "Checking…" : "Verify"}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Result */}
          {searched && result && (
            <Card className={result.found ? "border-primary/30" : "border-destructive/30"}>
              <CardContent className="pt-8 pb-8">
                {result.found ? (
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                      <CheckCircle2 className="w-8 h-8 text-primary" />
                    </div>
                    <h2 className="font-serif text-2xl font-bold text-foreground">
                      Certificate Verified ✓
                    </h2>
                    <p className="text-muted-foreground">
                      This is an authentic FIA Certificate of Competence.
                    </p>
                    <div className="bg-muted rounded-xl p-6 text-left space-y-3 max-w-sm mx-auto">
                      <div>
                        <p className="text-xs text-muted-foreground">Certificate Holder</p>
                        <p className="font-semibold text-foreground">{result.full_name}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Registration ID</p>
                        <p className="font-semibold text-foreground">{result.registration_id}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Token</p>
                        <p className="font-mono text-sm text-primary">{result.completion_token}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Programme</p>
                        <p className="font-semibold text-foreground">Biometrical Genetics Mastery</p>
                      </div>
                      {result.completed_at && (
                        <div>
                          <p className="text-xs text-muted-foreground">Completed</p>
                          <p className="font-semibold text-foreground">
                            {new Date(result.completed_at).toLocaleDateString("en-GB", {
                              day: "numeric", month: "long", year: "numeric",
                            })}
                          </p>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground pt-2">
                      <Award className="w-4 h-4" />
                      <span>Issued by Field-to-Insight Academy</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 rounded-full bg-destructive/10 flex items-center justify-center mx-auto">
                      <XCircle className="w-8 h-8 text-destructive" />
                    </div>
                    <h2 className="font-serif text-2xl font-bold text-foreground">
                      Certificate Not Found
                    </h2>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      No valid certificate was found with this token. Please check the token
                      and try again, or contact{" "}
                      <a href="mailto:info@fieldtoinsightacademy.com.ng" className="text-primary hover:underline">
                        info@fieldtoinsightacademy.com.ng
                      </a>{" "}
                      for assistance.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </section>
    </Layout>
  );
}
