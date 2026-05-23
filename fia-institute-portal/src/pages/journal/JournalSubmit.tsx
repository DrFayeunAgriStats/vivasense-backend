import { useState } from "react";
import { JournalLayout } from "@/components/journal/JournalLayout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { CheckCircle, Upload } from "lucide-react";

export default function JournalSubmit() {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [form, setForm] = useState({
    author_name: "",
    email: "",
    manuscript_title: "",
    abstract: "",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!form.author_name.trim() || !form.email.trim() || !form.manuscript_title.trim() || !form.abstract.trim()) {
      toast({ title: "Please fill in all required fields.", variant: "destructive" });
      return;
    }

    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    const ALLOWED_TYPES = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    ];

    if (file) {
      if (file.size > MAX_FILE_SIZE) {
        toast({ title: "File too large", description: "Maximum file size is 10MB.", variant: "destructive" });
        return;
      }
      if (!ALLOWED_TYPES.includes(file.type)) {
        toast({ title: "Invalid file type", description: "Only PDF and Word documents are accepted.", variant: "destructive" });
        return;
      }
    }

    setLoading(true);
    try {
      let filePath: string | null = null;

      if (file) {
        const ext = file.name.split(".").pop();
        const fileName = `${Date.now()}-${Math.random().toString(36).substring(7)}.${ext}`;
        const { error: uploadError } = await supabase.storage
          .from("manuscripts")
          .upload(fileName, file);
        if (uploadError) throw uploadError;
        filePath = fileName;
      }

      const { error } = await supabase.from("journal_submissions").insert({
        author_name: form.author_name.trim(),
        email: form.email.trim(),
        manuscript_title: form.manuscript_title.trim(),
        abstract: form.abstract.trim(),
        file_path: filePath,
      });

      if (error) throw error;

      setSubmitted(true);
    } catch (err: any) {
      console.error('Submission error:', err);
      let userMessage = 'Unable to submit manuscript. Please try again or contact support.';
      if (err.message?.includes('duplicate')) {
        userMessage = 'This submission may have already been received.';
      } else if (err.message?.includes('storage') || err.message?.includes('upload')) {
        userMessage = 'File upload failed. Please check your file and try again.';
      } else if (err.message?.includes('network') || err.message?.includes('fetch')) {
        userMessage = 'Network error. Please check your connection and try again.';
      } else if (err.message?.includes('email_format') || err.message?.includes('email')) {
        userMessage = 'Please enter a valid email address.';
      }
      toast({ title: "Submission failed", description: userMessage, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <JournalLayout>
        <section className="section-padding bg-background">
          <div className="container-wide max-w-xl text-center">
            <CheckCircle className="w-16 h-16 text-primary mx-auto mb-6" />
            <h1 className="font-serif text-3xl font-bold text-foreground mb-4">
              Manuscript Submitted
            </h1>
            <p className="text-foreground/80 leading-relaxed">
              Thank you for your submission. Your manuscript has been received and will be
              reviewed by our editorial team. You will be contacted at <strong>{form.email}</strong> with
              updates on your submission status.
            </p>
          </div>
        </section>
      </JournalLayout>
    );
  }

  return (
    <JournalLayout>
      <section className="section-padding bg-background">
        <div className="container-wide max-w-xl">
          <h1 className="font-serif text-3xl md:text-4xl font-bold text-foreground mb-2">
            Submit Manuscript
          </h1>
          <p className="text-muted-foreground mb-8">
            Complete the form below to submit your manuscript for review.
          </p>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="author_name">Author Name *</Label>
              <Input
                id="author_name"
                name="author_name"
                value={form.author_name}
                onChange={handleChange}
                placeholder="e.g. Dr. Jane Doe"
                maxLength={200}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="email">Email Address *</Label>
              <Input
                id="email"
                name="email"
                type="email"
                value={form.email}
                onChange={handleChange}
                placeholder="you@example.com"
                maxLength={255}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="manuscript_title">Manuscript Title *</Label>
              <Input
                id="manuscript_title"
                name="manuscript_title"
                value={form.manuscript_title}
                onChange={handleChange}
                placeholder="Enter the full title of your manuscript"
                maxLength={500}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="abstract">Abstract *</Label>
              <Textarea
                id="abstract"
                name="abstract"
                value={form.abstract}
                onChange={handleChange}
                placeholder="Provide a concise abstract (max 250 words)"
                rows={6}
                maxLength={3000}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="file">Upload Manuscript (PDF or Word)</Label>
              <Input
                id="file"
                type="file"
                accept=".pdf,.doc,.docx"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              <p className="text-xs text-muted-foreground">Accepted formats: PDF, DOC, DOCX</p>
            </div>

            <Button type="submit" variant="gold" size="lg" className="w-full" disabled={loading}>
              {loading ? (
                <>
                  <Upload className="w-4 h-4 mr-2 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Submit Manuscript
                </>
              )}
            </Button>
          </form>
        </div>
      </section>
    </JournalLayout>
  );
}
