import { useState, useRef, useCallback, useEffect } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useNavigate } from "react-router-dom";
import { Upload, FileText, X, Loader2, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { MANUSCRIPT_REVIEWER_BASE } from "@/config/manuscriptReviewer";
import { transformReviewResponse } from "@/lib/transformReviewResponse";

export default function ManuscriptReviewerUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [subjectArea, setSubjectArea] = useState("");
  const [instructions, setInstructions] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [backendDown, setBackendDown] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // Health check on mount
  useEffect(() => {
    fetch(`${MANUSCRIPT_REVIEWER_BASE}/health`, { method: "GET" })
      .then((res) => {
        if (!res.ok) setBackendDown(true);
      })
      .catch(() => setBackendDown(true));
  }, []);

  const accept = ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

  const handleFile = (f: File) => {
    const validTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ];
    if (!validTypes.includes(f.type) && !f.name.endsWith(".pdf") && !f.name.endsWith(".docx")) {
      toast.error("Please upload a PDF or DOCX file.");
      return;
    }
    if (f.size > 20 * 1024 * 1024) {
      toast.error("File size must be under 20 MB.");
      return;
    }
    setFile(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  }, []);

  const submit = async () => {
    if (!file) {
      toast.error("Please upload a manuscript file.");
      return;
    }
    setLoading(true);

    try {
      const fd = new FormData();
      fd.append("file", file);
      if (subjectArea.trim()) fd.append("subject_area", subjectArea.trim());
      if (instructions.trim()) fd.append("instructions", instructions.trim());

      console.log(`[ManuscriptReviewer] POST ${MANUSCRIPT_REVIEWER_BASE}/review/full`);

      const res = await fetch(`${MANUSCRIPT_REVIEWER_BASE}/review/full`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const errText = await res.text().catch(() => "");
        throw new Error(`Server responded ${res.status}${errText ? `: ${errText}` : ""}`);
      }

      const rawData = await res.json();
      const data = transformReviewResponse(rawData);
      sessionStorage.setItem("manuscript_review_result", JSON.stringify(data));
      navigate("/manuscript-reviewer/results");
    } catch (err: any) {
      console.error("[ManuscriptReviewer] Error:", err);
      toast.error(
        err?.message?.includes("Failed to fetch") || err?.message?.includes("NetworkError")
          ? "Could not reach the review server. Please check your connection and try again."
          : `Review failed: ${err?.message || "Unknown error. Please try again."}`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <section className="py-16 md:py-24 bg-background">
        <div className="container-wide px-4 max-w-2xl mx-auto">
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-2 text-center">
            Upload Your Manuscript
          </h1>
          <p className="text-center text-muted-foreground mb-10">
            Upload a DOCX or PDF file and receive a comprehensive AI-assisted review.
          </p>

          {backendDown && (
            <div className="flex items-start gap-3 p-4 mb-6 rounded-lg border border-destructive/30 bg-destructive/5 text-sm text-destructive">
              <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Review server is currently unreachable</p>
                <p className="text-xs mt-1 opacity-80">
                  The backend at {MANUSCRIPT_REVIEWER_BASE} did not respond. You can still upload, but analysis may fail. Please try again later.
                </p>
              </div>
            </div>
          )}

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors mb-6 ${
              dragActive
                ? "border-primary bg-primary/5"
                : file
                ? "border-primary/40 bg-primary/5"
                : "border-border hover:border-primary/50 bg-muted/20"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={accept}
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            {file ? (
              <div className="flex items-center justify-center gap-3">
                <FileText className="w-8 h-8 text-primary" />
                <div className="text-left">
                  <p className="font-medium text-foreground">{file.name}</p>
                  <p className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(0)} KB</p>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); setFile(null); }}
                  className="ml-2 p-1 rounded hover:bg-muted"
                >
                  <X className="w-4 h-4 text-muted-foreground" />
                </button>
              </div>
            ) : (
              <>
                <Upload className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                <p className="font-medium text-foreground mb-1">
                  Drag & drop your manuscript here
                </p>
                <p className="text-sm text-muted-foreground">or click to browse · PDF or DOCX · Max 20 MB</p>
              </>
            )}
          </div>

          {/* Optional fields */}
          <div className="space-y-4 mb-8">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">
                Subject Area <span className="text-muted-foreground font-normal">(optional)</span>
              </label>
              <Input
                placeholder="e.g. Agronomy, Plant Pathology, Soil Science"
                value={subjectArea}
                onChange={(e) => setSubjectArea(e.target.value)}
                maxLength={120}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">
                Special Reviewer Instructions <span className="text-muted-foreground font-normal">(optional)</span>
              </label>
              <Textarea
                placeholder="e.g. Focus on statistical methods, check APA 7th edition formatting..."
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                maxLength={500}
                rows={3}
              />
            </div>
          </div>

          <Button onClick={submit} disabled={loading || !file} size="lg" className="w-full gap-2">
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Analyzing Manuscript…
              </>
            ) : (
              "Analyze Manuscript"
            )}
          </Button>
        </div>
      </section>
    </Layout>
  );
}
