import { useState, useCallback, useRef } from "react";
import { supabase } from "@/integrations/supabase/client";
import {
  ShieldCheck,
  Plus,
  Search,
  Download,
  Ban,
  RotateCcw,
  Loader2,
  LogOut,
  Eye,
  X,
  Upload,
  Pencil,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CertificatePreview } from "@/components/certificates/CertificatePreview";
import type { Certificate } from "@/types/certificate";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { toast } from "sonner";

const FULL_CERTIFICATE_ID_PATTERN = /-\d{4}-\d{3}$/;

function getCertificateIdPreview(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return `FIA-ADAP-PILOT-${new Date().getFullYear()}-001`;
  return FULL_CERTIFICATE_ID_PATTERN.test(trimmed)
    ? trimmed
    : `${trimmed}-${new Date().getFullYear()}-001`;
}

function getErrorMessage(err: unknown) {
  return err instanceof Error ? err.message : "An unexpected error occurred";
}

export default function CertificateAdmin() {
  const [password, setPassword] = useState("");
  const [authed, setAuthed] = useState(false);
  const [adminPassword, setAdminPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  const [certificates, setCertificates] = useState<Certificate[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [previewCert, setPreviewCert] = useState<Certificate | null>(null);
  const [formLoading, setFormLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState<string | null>(null);
  const [editingCert, setEditingCert] = useState<Certificate | null>(null);
  const certPreviewRef = useRef<HTMLDivElement>(null);

  // Form state
  const [formData, setFormData] = useState({
    recipient_name: "",
    program_name: "FIA-ADAP Pilot Training Program",
    certificate_type: "Competence",
    training_start_date: "",
    training_end_date: "",
    issue_date: new Date().toISOString().split("T")[0],
    competencies: "",
    appreciation_text: "",
    certify_phrase: "This is to certify that",
    completion_phrase: "has successfully completed the",
    certificate_id_prefix: "FIA-ADAP-PILOT",
  });
  const [signatureFile, setSignatureFile] = useState<File | null>(null);
  const [signaturePreview, setSignaturePreview] = useState<string | null>(null);

  const adminAction = useCallback(
    async (action: string, data: Record<string, unknown> = {}) => {
      const { data: result, error } = await supabase.functions.invoke(
        "certificate-admin",
        { body: { action, password: adminPassword, ...data } }
      );
      if (error) throw new Error(error.message);
      if (result?.error) throw new Error(result.error);
      return result;
    },
    [adminPassword]
  );

  function handleSignatureChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload an image file (PNG/JPG)");
      return;
    }
    if (file.size > 2 * 1024 * 1024) {
      toast.error("Signature image must be less than 2MB");
      return;
    }
    setSignatureFile(file);
    const reader = new FileReader();
    reader.onload = (ev) => setSignaturePreview(ev.target?.result as string);
    reader.readAsDataURL(file);
  }

  async function handleLogin() {
    setAuthLoading(true);
    try {
      const result = await supabase.functions.invoke("certificate-admin", {
        body: { action: "list", password },
      });
      if (result.error || result.data?.error) {
        toast.error("Invalid password");
        setAuthLoading(false);
        return;
      }
      setAdminPassword(password);
      setCertificates(result.data || []);
      setAuthed(true);
    } catch {
      toast.error("Authentication failed");
    }
    setAuthLoading(false);
  }

  async function loadCertificates() {
    setLoading(true);
    try {
      const data = await adminAction("list");
      setCertificates(data || []);
    } catch (err) {
      toast.error(getErrorMessage(err));
    }
    setLoading(false);
  }

  async function handleCreate() {
    setFormLoading(true);
    try {
      const competencies = formData.competencies
        .split(",")
        .map((s: string) => s.trim())
        .filter(Boolean);

      const fd = new FormData();
      fd.append("action", "create");
      fd.append("password", adminPassword);
      fd.append("recipient_name", formData.recipient_name);
      fd.append("program_name", formData.program_name);
      fd.append("certificate_type", formData.certificate_type);
      fd.append("training_start_date", formData.training_start_date || "");
      fd.append("training_end_date", formData.training_end_date || "");
      fd.append("issue_date", formData.issue_date);
      fd.append("competencies", JSON.stringify(competencies));
      fd.append("appreciation_text", formData.appreciation_text || "");
      fd.append("certify_phrase", formData.certify_phrase);
      fd.append("completion_phrase", formData.completion_phrase);
      fd.append("certificate_id_prefix", formData.certificate_id_prefix || "FIA-ADAP-PILOT");
      fd.append("competencies", JSON.stringify(competencies));
      if (signatureFile) {
        fd.append("signature_file", signatureFile);
      }

      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
      const anonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

      const response = await fetch(
        `${supabaseUrl}/functions/v1/certificate-admin`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${anonKey}`,
            apikey: anonKey,
          },
          body: fd,
        }
      );

      const result = await response.json();
      if (result.error) throw new Error(result.error);

      toast.success("Certificate issued successfully!");
      setShowForm(false);
      setFormData({
        recipient_name: "",
        program_name: "FIA-ADAP Pilot Training Program",
        certificate_type: "Competence",
        training_start_date: "",
        training_end_date: "",
        issue_date: new Date().toISOString().split("T")[0],
        competencies: "",
        appreciation_text: "",
        certify_phrase: "This is to certify that",
        completion_phrase: "has successfully completed the",
        certificate_id_prefix: "FIA-ADAP-PILOT",
      });
      setSignatureFile(null);
      setSignaturePreview(null);
      loadCertificates();
    } catch (err) {
      toast.error(getErrorMessage(err));
    }
    setFormLoading(false);
  }

  function startEditing(cert: Certificate) {
    setEditingCert(cert);
    setFormData({
      recipient_name: cert.recipient_name,
      program_name: cert.program_name,
      certificate_type: cert.certificate_type,
      training_start_date: cert.training_start_date || "",
      training_end_date: cert.training_end_date || "",
      issue_date: cert.issue_date,
      competencies: cert.competencies?.join(", ") || "",
      appreciation_text: cert.appreciation_text || "",
      certify_phrase: cert.certify_phrase || "This is to certify that",
      completion_phrase: cert.completion_phrase || "has successfully completed the",
      certificate_id_prefix: cert.certificate_id.replace(/(-\d{4}-\d{3})+$/, "") || "FIA-ADAP-PILOT",
    });
    setSignatureFile(null);
    setSignaturePreview(cert.signature_url || null);
    setShowForm(true);
  }

  async function handleUpdate() {
    if (!editingCert) return;
    setFormLoading(true);
    try {
      const competencies = formData.competencies
        .split(",")
        .map((s: string) => s.trim())
        .filter(Boolean);

      if (signatureFile) {
        const fd = new FormData();
        fd.append("action", "update");
        fd.append("password", adminPassword);
        fd.append("id", editingCert.id);
        fd.append("recipient_name", formData.recipient_name);
        fd.append("program_name", formData.program_name);
        fd.append("certificate_type", formData.certificate_type);
        fd.append("training_start_date", formData.training_start_date || "");
        fd.append("training_end_date", formData.training_end_date || "");
        fd.append("issue_date", formData.issue_date);
        fd.append("competencies", JSON.stringify(competencies));
        fd.append("appreciation_text", formData.appreciation_text || "");
        fd.append("certify_phrase", formData.certify_phrase);
        fd.append("completion_phrase", formData.completion_phrase);
        fd.append("certificate_id_prefix", formData.certificate_id_prefix || "");
        fd.append("signature_file", signatureFile);

        const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
        const anonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;
        const response = await fetch(
          `${supabaseUrl}/functions/v1/certificate-admin`,
          {
            method: "POST",
            headers: { Authorization: `Bearer ${anonKey}`, apikey: anonKey },
            body: fd,
          }
        );
        const result = await response.json();
        if (result.error) throw new Error(result.error);
      } else {
        await adminAction("update", {
          id: editingCert.id,
          recipient_name: formData.recipient_name,
          program_name: formData.program_name,
          certificate_type: formData.certificate_type,
          training_start_date: formData.training_start_date,
          training_end_date: formData.training_end_date,
          issue_date: formData.issue_date,
          competencies,
          appreciation_text: formData.appreciation_text || "",
          certify_phrase: formData.certify_phrase,
          completion_phrase: formData.completion_phrase,
          certificate_id_prefix: formData.certificate_id_prefix || "",
        });
      }

      toast.success("Certificate updated successfully!");
      setShowForm(false);
      setEditingCert(null);
      resetForm();
      loadCertificates();
    } catch (err) {
      toast.error(getErrorMessage(err));
    }
    setFormLoading(false);
  }

  function resetForm() {
    setFormData({
      recipient_name: "",
      program_name: "FIA-ADAP Pilot Training Program",
      certificate_type: "Competence",
      training_start_date: "",
      training_end_date: "",
      issue_date: new Date().toISOString().split("T")[0],
      competencies: "",
      appreciation_text: "",
      certify_phrase: "This is to certify that",
      completion_phrase: "has successfully completed the",
      certificate_id_prefix: "FIA-ADAP-PILOT",
    });
    setSignatureFile(null);
    setSignaturePreview(null);
  }

  async function handleRevoke(id: string) {
    if (!confirm("Are you sure you want to revoke this certificate?")) return;
    try {
      await adminAction("revoke", { id });
      toast.success("Certificate revoked");
      loadCertificates();
    } catch (err) {
      toast.error(getErrorMessage(err));
    }
  }

  async function handleReinstate(id: string) {
    try {
      await adminAction("reinstate", { id });
      toast.success("Certificate reinstated");
      loadCertificates();
    } catch (err) {
      toast.error(getErrorMessage(err));
    }
  }

  async function handleDownloadPDF(cert: Certificate) {
    setPdfLoading(cert.id);
    setPreviewCert(cert);
    // Wait for render + fonts to load
    await new Promise((r) => setTimeout(r, 1500));
    try {
      const el = certPreviewRef.current;
      if (!el) throw new Error("Preview not ready");

      const canvas = await html2canvas(el, {
        scale: 4,
        useCORS: true,
        allowTaint: true,
        backgroundColor: "#ffffff",
        width: 842,
        height: 595,
        windowWidth: 842,
        windowHeight: 595,
        logging: false,
        onclone: (clonedDoc, clonedEl) => {
          // Fix border-image which html2canvas doesn't support
          const borderEl = clonedEl.querySelector('div[style*="border-image"]') as HTMLElement;
          if (borderEl) {
            borderEl.style.borderImage = "none";
            borderEl.style.border = "8px solid #2E7D32";
          }
        },
      });

      const pdf = new jsPDF({ orientation: "landscape", unit: "mm", format: "a4" });
      const imgData = canvas.toDataURL("image/png", 1.0);
      pdf.addImage(imgData, "PNG", 0, 0, 297, 210);
      pdf.save(`${cert.certificate_id}.pdf`);
      toast.success("PDF downloaded");
    } catch (err) {
      toast.error("Failed to generate PDF: " + getErrorMessage(err));
    }
    setPreviewCert(null);
    setPdfLoading(null);
  }

  const filtered = certificates.filter(
    (c) =>
      c.certificate_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.recipient_name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatDate = (d: string | null) => {
    if (!d) return "—";
    return new Date(d).toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  };

  // Login screen
  if (!authed) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-orange-50 flex items-center justify-center p-4">
        <Card className="w-full max-w-sm">
          <CardHeader className="text-center">
            <ShieldCheck className="w-10 h-10 mx-auto mb-2" style={{ color: "#2E7D32" }} />
            <CardTitle className="text-lg">Certificate Admin</CardTitle>
            <p className="text-sm text-muted-foreground">
              Enter admin password to continue
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Input
                type="password"
                placeholder="Admin password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleLogin()}
              />
              <Button
                onClick={handleLogin}
                disabled={authLoading || !password}
                className="w-full"
                style={{ backgroundColor: "#2E7D32" }}
              >
                {authLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                ) : null}
                Sign In
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-orange-50">
      {/* Hidden certificate preview for PDF generation */}
      {previewCert && pdfLoading && (
        <div style={{ position: "fixed", left: -9999, top: 0, width: 842, height: 595 }}>
          <CertificatePreview ref={certPreviewRef} certificate={previewCert} scale={1} />
        </div>
      )}

      {/* Header */}
      <header className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold" style={{ color: "#2E7D32" }}>
              Certificate Admin Dashboard
            </h1>
            <p className="text-xs text-muted-foreground">
              Field-to-Insight Academy
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setAuthed(false);
              setAdminPassword("");
              setPassword("");
            }}
          >
            <LogOut className="w-4 h-4 mr-2" />
            Sign Out
          </Button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <Card>
            <CardContent className="pt-6">
              <p className="text-sm text-muted-foreground">Total Certificates</p>
              <p className="text-3xl font-bold text-foreground">{certificates.length}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <p className="text-sm text-muted-foreground">Active</p>
              <p className="text-3xl font-bold" style={{ color: "#2E7D32" }}>
                {certificates.filter((c) => c.status === "Active").length}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <p className="text-sm text-muted-foreground">Revoked</p>
              <p className="text-3xl font-bold text-destructive">
                {certificates.filter((c) => c.status === "Revoked").length}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-3 mb-6">
           <Button
            onClick={() => {
              setEditingCert(null);
              resetForm();
              setShowForm(!showForm);
            }}
            style={{ backgroundColor: "#2E7D32" }}
          >
            <Plus className="w-4 h-4 mr-2" />
            Issue New Certificate
          </Button>
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search by ID or name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>

        {/* Issue form */}
        {showForm && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="text-lg flex items-center justify-between">
                {editingCert ? "Edit Certificate" : "Issue New Certificate"}
                <Button variant="ghost" size="sm" onClick={() => { setShowForm(false); setEditingCert(null); }}>
                  <X className="w-4 h-4" />
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Recipient Name *</label>
                  <Input
                    value={formData.recipient_name}
                    onChange={(e) =>
                      setFormData({ ...formData, recipient_name: e.target.value })
                    }
                    placeholder="Full name"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Programme Name</label>
                  <Input
                    value={formData.program_name}
                    onChange={(e) =>
                      setFormData({ ...formData, program_name: e.target.value })
                    }
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Certificate Type</label>
                  <select
                    value={formData.certificate_type}
                    onChange={(e) =>
                      setFormData({ ...formData, certificate_type: e.target.value })
                    }
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="Competence">Competence</option>
                    <option value="Appreciation">Appreciation</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">Certify Phrase</label>
                  <select
                    value={formData.certify_phrase}
                    onChange={(e) =>
                      setFormData({ ...formData, certify_phrase: e.target.value })
                    }
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="This is to certify that">This is to certify that</option>
                    <option value="This certificate is awarded to">This certificate is awarded to</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">Completion Phrase</label>
                  <select
                    value={formData.completion_phrase}
                    onChange={(e) =>
                      setFormData({ ...formData, completion_phrase: e.target.value })
                    }
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="has successfully completed the">has successfully completed the</option>
                    <option value="for the success of">for the success of</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">Issue Date</label>
                  <Input
                    type="date"
                    value={formData.issue_date}
                    onChange={(e) =>
                      setFormData({ ...formData, issue_date: e.target.value })
                    }
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Training Start Date</label>
                  <Input
                    type="date"
                    value={formData.training_start_date}
                    onChange={(e) =>
                      setFormData({ ...formData, training_start_date: e.target.value })
                    }
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Training End Date</label>
                  <Input
                    type="date"
                    value={formData.training_end_date}
                    onChange={(e) =>
                      setFormData({ ...formData, training_end_date: e.target.value })
                    }
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="text-sm font-medium">
                    Competencies (comma-separated)
                  </label>
                  <Input
                    value={formData.competencies}
                    onChange={(e) =>
                      setFormData({ ...formData, competencies: e.target.value })
                    }
                    placeholder="e.g. Data Collection, Statistical Analysis, Report Writing"
                  />
                </div>
                {formData.certificate_type === "Appreciation" && (
                  <div className="md:col-span-2">
                    <label className="text-sm font-medium">
                      Appreciation Text (comma-separated lines)
                    </label>
                    <Input
                      value={formData.appreciation_text}
                      onChange={(e) =>
                        setFormData({ ...formData, appreciation_text: e.target.value })
                      }
                      placeholder="e.g. in recognition of exceptional contribution as R Programming Specialist, to the successful delivery of the DATA ANALYTICS Training Program"
                    />
                  </div>
                )}
                <div>
                  <label className="text-sm font-medium">Certificate ID Prefix</label>
                  <Input
                    value={formData.certificate_id_prefix}
                    onChange={(e) =>
                      setFormData({ ...formData, certificate_id_prefix: e.target.value })
                    }
                    placeholder="e.g. FIA-ADAP-PILOT"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Final ID will be: {getCertificateIdPreview(formData.certificate_id_prefix)}
                  </p>
                </div>
                <div className="md:col-span-2">
                  <label className="text-sm font-medium">
                    Upload Signature Image (PNG/JPG)
                  </label>
                  <div className="flex items-center gap-4 mt-1">
                    <label className="flex items-center gap-2 px-4 py-2 border border-input rounded-md cursor-pointer hover:bg-muted transition-colors text-sm">
                      <Upload className="w-4 h-4" />
                      {signatureFile ? signatureFile.name : "Choose file..."}
                      <input
                        type="file"
                        accept="image/png,image/jpeg,image/jpg"
                        onChange={handleSignatureChange}
                        className="hidden"
                      />
                    </label>
                    {signaturePreview && (
                      <div className="flex items-center gap-2">
                        <img
                          src={signaturePreview}
                          alt="Signature preview"
                          className="h-12 border rounded"
                          style={{ opacity: 0.85 }}
                        />
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setSignatureFile(null);
                            setSignaturePreview(null);
                          }}
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Max 2MB. Will appear above signature line on the certificate.
                  </p>
                </div>
              </div>
              <div className="mt-6 flex gap-3">
                <Button
                  onClick={editingCert ? handleUpdate : handleCreate}
                  disabled={formLoading || !formData.recipient_name.trim()}
                  style={{ backgroundColor: editingCert ? "#1565C0" : "#1565C0" }}
                >
                  {formLoading && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
                  {editingCert ? "Update Certificate" : "Issue Certificate"}
                </Button>
                <Button variant="outline" onClick={() => { setShowForm(false); setEditingCert(null); }}>
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Certificate preview modal */}
        {previewCert && !pdfLoading && (
          <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-xl p-4 max-w-[900px] w-full overflow-auto">
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold">Certificate Preview</h3>
                <Button variant="ghost" size="sm" onClick={() => setPreviewCert(null)}>
                  <X className="w-4 h-4" />
                </Button>
              </div>
              <div className="overflow-auto">
                <CertificatePreview
                  ref={certPreviewRef}
                  certificate={previewCert}
                  scale={0.85}
                />
              </div>
            </div>
          </div>
        )}

        {/* Certificates table */}
        <Card>
          <CardContent className="pt-6">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
              </div>
            ) : filtered.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                {certificates.length === 0
                  ? "No certificates issued yet."
                  : "No certificates match your search."}
              </div>
            ) : (
              <div className="overflow-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Certificate ID</TableHead>
                      <TableHead>Recipient</TableHead>
                      <TableHead className="hidden md:table-cell">Type</TableHead>
                      <TableHead className="hidden md:table-cell">Issued</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filtered.map((cert) => (
                      <TableRow key={cert.id}>
                        <TableCell className="font-mono text-xs">
                          {cert.certificate_id}
                        </TableCell>
                        <TableCell className="font-medium">
                          {cert.recipient_name}
                        </TableCell>
                        <TableCell className="hidden md:table-cell">
                          {cert.certificate_type}
                        </TableCell>
                        <TableCell className="hidden md:table-cell">
                          {formatDate(cert.issue_date)}
                        </TableCell>
                        <TableCell>
                          <span
                            className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium"
                            style={{
                              backgroundColor:
                                cert.status === "Active"
                                  ? "rgba(46, 125, 50, 0.1)"
                                  : "rgba(211, 47, 47, 0.1)",
                              color:
                                cert.status === "Active" ? "#2E7D32" : "#d32f2f",
                            }}
                          >
                            {cert.status}
                          </span>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex gap-1 justify-end">
                            <Button
                              variant="ghost"
                              size="sm"
                              title="Preview"
                              onClick={() => setPreviewCert(cert)}
                            >
                              <Eye className="w-4 h-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              title="Edit"
                              onClick={() => startEditing(cert)}
                            >
                              <Pencil className="w-4 h-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              title="Download PDF"
                              disabled={pdfLoading === cert.id}
                              onClick={() => handleDownloadPDF(cert)}
                            >
                              {pdfLoading === cert.id ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                              ) : (
                                <Download className="w-4 h-4" />
                              )}
                            </Button>
                            {cert.status === "Active" ? (
                              <Button
                                variant="ghost"
                                size="sm"
                                title="Revoke"
                                onClick={() => handleRevoke(cert.id)}
                              >
                                <Ban className="w-4 h-4 text-destructive" />
                              </Button>
                            ) : (
                              <Button
                                variant="ghost"
                                size="sm"
                                title="Reinstate"
                                onClick={() => handleReinstate(cert.id)}
                              >
                                <RotateCcw className="w-4 h-4" style={{ color: "#2E7D32" }} />
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
