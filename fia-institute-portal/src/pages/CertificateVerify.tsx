import { useState, useEffect } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { QRCodeSVG } from "qrcode.react";
import {
  CheckCircle2,
  XCircle,
  Search,
  ShieldCheck,
  Award,
  Loader2,
} from "lucide-react";
import type { Certificate } from "@/types/certificate";

export default function CertificateVerify() {
  const [searchParams] = useSearchParams();
  const idFromUrl = searchParams.get("id") || "";
  const [certId, setCertId] = useState(idFromUrl);
  const [result, setResult] = useState<Certificate | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  useEffect(() => {
    if (idFromUrl) {
      handleVerify(idFromUrl);
    }
  }, []);

  async function handleVerify(id?: string) {
    const searchId = (id || certId).trim().toUpperCase();
    if (!searchId) return;
    setLoading(true);
    setSearched(true);
    setNotFound(false);
    setResult(null);

    const { data, error } = await supabase
      .from("certificates")
      .select("*")
      .eq("certificate_id", searchId)
      .maybeSingle();

    if (error) {
      console.error("Verification error:", error);
    }

    if (data) {
      setResult(data as Certificate);
    } else {
      setNotFound(true);
    }
    setLoading(false);
  }

  const formatDate = (d: string | null) => {
    if (!d) return "";
    return new Date(d).toLocaleDateString("en-GB", {
      day: "numeric",
      month: "long",
      year: "numeric",
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-orange-50">
      {/* Header */}
      <header className="bg-white border-b shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold" style={{ color: "#2E7D32" }}>
              Field-to-Insight Academy
            </h1>
            <p className="text-xs text-muted-foreground">Certificate Verification Portal</p>
          </div>
          <ShieldCheck className="w-8 h-8" style={{ color: "#1565C0" }} />
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-10">
        {/* Hero */}
        <div className="text-center mb-8">
          <div
            className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
            style={{ backgroundColor: "rgba(21, 101, 192, 0.1)" }}
          >
            <ShieldCheck className="w-8 h-8" style={{ color: "#1565C0" }} />
          </div>
          <h2 className="text-2xl font-bold text-foreground mb-2">
            Verify Certificate Authenticity
          </h2>
          <p className="text-muted-foreground">
            Enter a certificate ID or scan the QR code on any FIA certificate to verify its
            authenticity.
          </p>
        </div>

        {/* Search */}
        <div className="bg-white rounded-xl shadow-md border p-6 mb-8">
          <label className="text-sm font-medium text-foreground mb-2 block">
            Certificate ID
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              placeholder="e.g. FIA-ADAP-PILOT-2026-001"
              value={certId}
              onChange={(e) => setCertId(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleVerify()}
              className="flex-1 h-11 rounded-lg border border-input bg-background px-4 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <button
              onClick={() => handleVerify()}
              disabled={loading || !certId.trim()}
              className="h-11 px-6 rounded-lg text-white font-medium text-sm disabled:opacity-50 flex items-center gap-2"
              style={{ backgroundColor: "#1565C0" }}
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              Verify
            </button>
          </div>
        </div>

        {/* Results */}
        {searched && !loading && (
          <>
            {result ? (
              <div className="bg-white rounded-xl shadow-md border p-8">
                <div className="text-center mb-6">
                  <div
                    className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3"
                    style={{
                      backgroundColor:
                        result.status === "Active"
                          ? "rgba(46, 125, 50, 0.1)"
                          : "rgba(211, 47, 47, 0.1)",
                    }}
                  >
                    {result.status === "Active" ? (
                      <CheckCircle2 className="w-8 h-8" style={{ color: "#2E7D32" }} />
                    ) : (
                      <XCircle className="w-8 h-8 text-destructive" />
                    )}
                  </div>
                  <h3 className="text-xl font-bold text-foreground">
                    {result.status === "Active"
                      ? "✓ Certificate is Valid"
                      : "✗ Certificate Revoked"}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    {result.status === "Active"
                      ? "This is an authentic Field-to-Insight Academy certificate."
                      : "This certificate has been revoked and is no longer valid."}
                  </p>
                </div>

                <div className="bg-muted rounded-xl p-6 space-y-4 max-w-md mx-auto">
                  <div>
                    <p className="text-xs text-muted-foreground">Certificate Holder</p>
                    <p className="font-semibold text-foreground text-lg">
                      {result.recipient_name}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Programme</p>
                    <p className="font-semibold text-foreground">{result.program_name}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Certificate Type</p>
                    <p className="font-semibold text-foreground">{result.certificate_type}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Certificate ID</p>
                    <p className="font-mono text-sm" style={{ color: "#1565C0" }}>
                      {result.certificate_id}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Date Issued</p>
                    <p className="font-semibold text-foreground">
                      {formatDate(result.issue_date)}
                    </p>
                  </div>
                  {result.training_start_date && result.training_end_date && (
                    <div>
                      <p className="text-xs text-muted-foreground">Training Period</p>
                      <p className="font-semibold text-foreground">
                        {formatDate(result.training_start_date)} –{" "}
                        {formatDate(result.training_end_date)}
                      </p>
                    </div>
                  )}
                  <div>
                    <p className="text-xs text-muted-foreground">Status</p>
                    <span
                      className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                      style={{
                        backgroundColor:
                          result.status === "Active"
                            ? "rgba(46, 125, 50, 0.1)"
                            : "rgba(211, 47, 47, 0.1)",
                        color: result.status === "Active" ? "#2E7D32" : "#d32f2f",
                      }}
                    >
                      {result.status}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground pt-6">
                  <Award className="w-4 h-4" />
                  <span>Issued by Field-to-Insight Academy</span>
                </div>
              </div>
            ) : notFound ? (
              <div className="bg-white rounded-xl shadow-md border p-8 text-center">
                <div
                  className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3"
                  style={{ backgroundColor: "rgba(211, 47, 47, 0.1)" }}
                >
                  <XCircle className="w-8 h-8 text-destructive" />
                </div>
                <h3 className="text-xl font-bold text-foreground mb-2">
                  Certificate Not Found
                </h3>
                <p className="text-muted-foreground max-w-md mx-auto">
                  No valid certificate was found with this ID. Please check the ID and try
                  again, or contact{" "}
                  <a
                    href="mailto:info@fieldtoinsightacademy.com.ng"
                    className="underline"
                    style={{ color: "#1565C0" }}
                  >
                    info@fieldtoinsightacademy.com.ng
                  </a>{" "}
                  for assistance.
                </p>
              </div>
            ) : null}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-xs text-muted-foreground border-t mt-auto">
        <p>© {new Date().getFullYear()} Field-to-Insight Academy</p>
        <p className="mt-1">An Institute for Agricultural Research, Data Analytics & AI Innovation</p>
        <p className="mt-1">Operated by Able-Flourish Agro-Services Ltd, Akure, Nigeria</p>
      </footer>
    </div>
  );
}
