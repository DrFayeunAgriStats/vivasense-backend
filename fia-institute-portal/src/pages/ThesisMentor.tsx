import { useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Download, Loader2, BookOpen, GraduationCap, AlertTriangle, MessageSquare, Lock } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";
import jsPDF from "jspdf";
import { RWSChatPanel } from "@/components/rws/RWSChatPanel";
import { useAuth } from "@/contexts/AuthContext";

interface Profile {
  studentName: string;
  degreeLevel: string;
  discipline: string;
  thesisTitle: string;
  objectives: string;
  studyLocation: string;
  experimentalDesign: string;
  variables: string;
  statisticalSoftware: string;
  referencingStyle: string;
}

interface GuidanceSections {
  section_objective: string;
  section_outline: string;
  writing_prompts: string;
  concept_overview: string;
  key_questions: string;
  application: string;
  tables_figures: string;
  supervisor_notes: string;
}

type Mode = "thesis" | "proposal";

const REFLECTION_QUESTIONS: Record<Mode, Record<string, string>> = {
  thesis: {
    "Chapter 1 Introduction": "What research problem are you addressing and why is it important?",
    "Chapter 2 Literature Review": "What gap in the literature does your study address?",
    "Chapter 3 Materials and Methods": "What research design are you using and why?",
    "Chapter 4 Results": "What is the most important finding in this chapter?",
    "Chapter 5 Discussion": "What does your result mean for the field?",
    "Chapter 6 Conclusion": "What is the main contribution of your research?",
  },
  proposal: {
    "Title refinement": "State your research topic in one sentence. What makes it important?",
    "Background & Problem statement": "What is your research problem and why does it matter in your context?",
    "Aim/Objectives + research questions/hypotheses": "List your objectives and explain how each connects to the problem.",
    "Literature map + gap statement": "What key themes exist in literature? What gap does your study address?",
    "Conceptual/Theoretical framework (optional)": "What theoretical framework guides your study? How do constructs relate?",
    "Methodology plan (design, sampling, variables, instruments)": "Describe your proposed design and why it answers your objectives.",
    "Data analysis plan (models/tests + assumptions)": "List key variables/outcomes and the tests/models you plan to use.",
    "Workplan (timeline)": "Outline your timeline: literature → design → fieldwork → analysis → reporting.",
    "Budget & justification (optional)": "What are your major budget categories? Why is each necessary?",
    "Expected outcomes + significance": "What will you produce? Who benefits? Why is this significant?",
    "References strategy (no fabricated citations)": "How will you organize literature sources? What tools will you use?",
  },
};

const THESIS_SECTIONS = [
  "Chapter 1 Introduction",
  "Chapter 2 Literature Review",
  "Chapter 3 Materials and Methods",
  "Chapter 4 Results",
  "Chapter 5 Discussion",
  "Chapter 6 Conclusion",
];

const PROPOSAL_SECTIONS = [
  "Title refinement",
  "Background & Problem statement",
  "Aim/Objectives + research questions/hypotheses",
  "Literature map + gap statement",
  "Conceptual/Theoretical framework (optional)",
  "Methodology plan (design, sampling, variables, instruments)",
  "Data analysis plan (models/tests + assumptions)",
  "Workplan (timeline)",
  "Budget & justification (optional)",
  "Expected outcomes + significance",
  "References strategy (no fabricated citations)",
];

const CONFIRMATIONS = [
  { key: "conf1", label: "I understand this tool provides guidance only" },
  { key: "conf2", label: "I will not submit AI-generated text as my own work" },
  { key: "conf3", label: "I will write my thesis/proposal in my own words" },
  { key: "conf4", label: "I will inform my supervisor about using this tool" },
  { key: "conf5", label: "My institution permits AI learning tools" },
];

const SECTION_TITLES = [
  { key: "section_objective", title: "1. Section Objective" },
  { key: "section_outline", title: "2. Section Outline" },
  { key: "writing_prompts", title: "3. Writing Prompts" },
  { key: "concept_overview", title: "4. Concept Overview" },
  { key: "key_questions", title: "5. Key Questions for the Student" },
  { key: "application", title: "6. Application to This Study" },
  { key: "tables_figures", title: "7. Suggested Tables, Figures, or Frameworks" },
  { key: "supervisor_notes", title: "8. Supervisor Discussion Notes" },
];

const MIN_REFLECTION_WORDS = 20;

const cleanMarkdown = (text: string) => {
  if (!text) return "";
  return text.replace(/^##+ /gm, "").replace(/^---$/gm, "").replace(/^___$/gm, "").trim();
};

export default function ThesisMentor() {
  const { toast } = useToast();
  const { profile: authProfile } = useAuth();
  const [activeTab, setActiveTab] = useState("proposal");
  const [confirmations, setConfirmations] = useState<Record<string, boolean>>({});
  const allConfirmed = CONFIRMATIONS.every((c) => confirmations[c.key]);

  const [mode, setMode] = useState<Mode>("thesis");
  const [researchProfile, setResearchProfile] = useState<Profile>({
    studentName: "",
    degreeLevel: "MSc",
    discipline: "",
    thesisTitle: "",
    objectives: "",
    studyLocation: "",
    experimentalDesign: "",
    variables: "",
    statisticalSoftware: "",
    referencingStyle: "APA",
  });

  const [chapter, setChapter] = useState("");
  const [reflection, setReflection] = useState("");
  const [postReflection, setPostReflection] = useState("");
  const [draft, setDraft] = useState("");
  const [statsOutput, setStatsOutput] = useState("");
  const [guidance, setGuidance] = useState<GuidanceSections | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState("");

  const wordCount = reflection.trim().split(/\s+/).filter(Boolean).length;
  const sections = mode === "thesis" ? THESIS_SECTIONS : PROPOSAL_SECTIONS;

  const updateProfile = (field: keyof Profile, value: string) => {
    setResearchProfile((p) => ({ ...p, [field]: value }));
  };

  const handleModeChange = (newMode: Mode) => {
    setMode(newMode);
    const defaultSection = (newMode === "thesis" ? THESIS_SECTIONS : PROPOSAL_SECTIONS)[0];
    setChapter(defaultSection);
    setReflection("");
    setPostReflection("");
    setGuidance(null);
  };

  const handleGenerate = async () => {
    if (!chapter) {
      toast({ title: "Error", description: "Please select a section.", variant: "destructive" });
      return;
    }
    if (wordCount < MIN_REFLECTION_WORDS) {
      toast({ title: "Error", description: `Reflection must be at least ${MIN_REFLECTION_WORDS} words (you have ${wordCount}).`, variant: "destructive" });
      return;
    }

    setIsLoading(true);
    setGuidance(null);
    setPostReflection("");

    try {
      const resp = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/thesis-mentor`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: JSON.stringify({
          mode,
          profile: researchProfile,
          chapter,
          proposal_section: mode === "proposal" ? chapter : null,
          reflection,
          draft: draft || null,
          stats_output: statsOutput || null,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || "Failed to generate guidance");
      }

      const data = await resp.json();
      setGuidance(data.sections);
      setSessionId(data.sessionId);

      const sessions = JSON.parse(localStorage.getItem("fia_thesis_sessions") || "[]");
      sessions.push({ sessionId: data.sessionId, timestamp: data.timestamp, mode, chapter, reflection, guidance: data.sections });
      localStorage.setItem("fia_thesis_sessions", JSON.stringify(sessions));
    } catch (err: any) {
      toast({ title: "Error", description: err.message || "Error generating guidance.", variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    if (!guidance || !sessionId) return;

    const doc = new jsPDF("p", "mm", "a4");
    const pageWidth = doc.internal.pageSize.width;
    const pageHeight = doc.internal.pageSize.height;

    const watermarkText = mode === "thesis"
      ? "AI GUIDANCE NOTES\nNOT THESIS TEXT"
      : "AI GUIDANCE NOTES\nNOT PROPOSAL TEXT";

    const addWatermarkAndHeader = () => {
      doc.setTextColor(230, 230, 230);
      doc.setFontSize(48);
      doc.text(watermarkText, pageWidth / 2, pageHeight / 2, { align: "center", angle: 45 });

      doc.setTextColor(180, 0, 0);
      doc.setFontSize(8);
      doc.setFont("helvetica", "bold");
      doc.text("AI GUIDANCE NOTES – NOT " + (mode === "thesis" ? "THESIS" : "PROPOSAL") + " TEXT", pageWidth / 2, 8, { align: "center" });
      doc.setTextColor(100, 100, 100);
      doc.setFontSize(7);
      doc.setFont("helvetica", "normal");
      doc.text(`Session: ${sessionId}  |  Date: ${new Date().toLocaleDateString()}  |  Student: ${researchProfile.studentName || "Anonymous"}`, pageWidth / 2, 13, { align: "center" });
    };

    addWatermarkAndHeader();
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(14);
    doc.text(`FIA AI ${mode === "thesis" ? "Thesis" : "Proposal"} Mentor – Guidance Notes`, 20, 22);
    doc.setFontSize(10);
    doc.text(`${mode === "thesis" ? "Chapter" : "Section"}: ${chapter}`, 20, 30);

    let yPos = 40;

    const addSection = (title: string, content: string) => {
      if (yPos > pageHeight - 30) {
        doc.addPage();
        addWatermarkAndHeader();
        yPos = 22;
      }
      doc.setFontSize(12);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(0, 0, 0);
      doc.text(title, 20, yPos);
      yPos += 8;
      doc.setFontSize(10);
      doc.setFont("helvetica", "normal");
      const cleaned = cleanMarkdown(content || "");
      const lines = doc.splitTextToSize(cleaned, 170);
      for (const line of lines) {
        if (yPos > pageHeight - 20) {
          doc.addPage();
          addWatermarkAndHeader();
          yPos = 22;
        }
        doc.text(line, 20, yPos);
        yPos += 5;
      }
      yPos += 6;
    };

    addSection("Section Objective", guidance.section_objective);
    addSection("Section Outline", guidance.section_outline);
    addSection("Writing Prompts", guidance.writing_prompts);
    addSection("Concept Overview", guidance.concept_overview);
    addSection("Key Questions", guidance.key_questions);
    addSection("Application to Your Study", guidance.application);
    addSection("Suggested Tables/Figures/Frameworks", guidance.tables_figures);
    addSection("Supervisor Discussion Notes", guidance.supervisor_notes);

    if (postReflection.trim()) {
      addSection("Post-Guidance Reflection", postReflection);
    }

    // Footer
    const pageCount = doc.internal.pages.length;
    for (let i = 1; i < pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(8);
      doc.setTextColor(150, 150, 150);
      const footerText = mode === "proposal"
        ? "Use this guidance to draft your proposal in your own words; confirm feasibility with your supervisor."
        : "You must rewrite all content in your own words.";
      doc.text(footerText, 20, pageHeight - 10);
    }

    doc.save(`FIA_${mode === "thesis" ? "Thesis" : "Proposal"}_${sessionId}.pdf`);
  };

  return (
    <Layout>
      {/* Hero */}
      <section className="bg-primary text-primary-foreground py-12 md:py-16">
        <div className="container-wide">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 rounded-full bg-primary-foreground/10 flex items-center justify-center">
              <BookOpen className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-serif text-3xl md:text-4xl font-bold">FIA Research Writing Mentor</h1>
              <p className="text-primary-foreground/70 text-sm mt-1">
                AI-assisted guidance for proposal, thesis, dissertation, and research paper development
              </p>
            </div>
          </div>
          <p className="text-primary-foreground/85 max-w-2xl">
            Prompts, structure, and supervisor discussion notes — not thesis or proposal text.
            This tool teaches academic reasoning without generating copy-paste content.
          </p>
        </div>
      </section>

      <section className="container-wide py-8">
        <div className="max-w-3xl mx-auto space-y-6">
          {/* Tab Navigation */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid grid-cols-5 w-full">
              <TabsTrigger value="proposal">Proposal Development</TabsTrigger>
              <TabsTrigger value="thesis">Thesis Chapter Writing</TabsTrigger>
              <TabsTrigger value="results" disabled className="gap-1">
                Results <Lock className="w-3 h-3" />
              </TabsTrigger>
              <TabsTrigger value="discussion" disabled className="gap-1">
                Discussion <Lock className="w-3 h-3" />
              </TabsTrigger>
              <TabsTrigger value="defense" disabled className="gap-1">
                Defense <Lock className="w-3 h-3" />
              </TabsTrigger>
            </TabsList>

            {/* Proposal & Thesis Tabs — existing functionality */}
            <TabsContent value="proposal" className="space-y-6 mt-6">
              {renderMentorContent()}
            </TabsContent>
            <TabsContent value="thesis" className="space-y-6 mt-6">
              {renderMentorContent()}
            </TabsContent>

            {/* Future tabs */}
            <TabsContent value="results">
              <Card className="mt-6"><CardContent className="pt-6 text-center text-muted-foreground text-sm">Coming soon — Guided Results Interpretation</CardContent></Card>
            </TabsContent>
            <TabsContent value="discussion">
              <Card className="mt-6"><CardContent className="pt-6 text-center text-muted-foreground text-sm">Coming soon — Discussion Builder</CardContent></Card>
            </TabsContent>
            <TabsContent value="defense">
              <Card className="mt-6"><CardContent className="pt-6 text-center text-muted-foreground text-sm">Coming soon — Defense Preparation</CardContent></Card>
            </TabsContent>
          </Tabs>

          {/* AI Chat Panel */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <MessageSquare className="w-5 h-5" />
                AI Research Mentor Chat
              </CardTitle>
              <p className="text-xs text-muted-foreground">
                Use different AI modes to get guidance on your research. Select Guide, Review, Supervisor, or Defense mode.
              </p>
            </CardHeader>
            <CardContent>
              <RWSChatPanel
                context={{
                  track: authProfile?.academic_track || undefined,
                  discipline: authProfile?.discipline || undefined,
                  stage: authProfile?.current_research_stage || undefined,
                  title: authProfile?.institution || undefined,
                }}
              />
            </CardContent>
          </Card>

          {/* Footer attribution */}
          <p className="text-[10px] text-muted-foreground text-center pt-4">
            FIA AI Thesis & Proposal Mentor™ – Field-to-Insight Academy © Dr. Fayeun Lawrence Stephen
          </p>
        </div>
      </section>
    </Layout>
  );

  function renderMentorContent() {
    return (
      <>
          {/* Academic Integrity Confirmation */}
          {!allConfirmed && (
            <Card className="border-l-4 border-l-primary">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <GraduationCap className="w-5 h-5" />
                  Academic Integrity Confirmation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">Please confirm before using this tool:</p>
                <div className="space-y-3">
                  {CONFIRMATIONS.map((item) => (
                    <label key={item.key} className="flex items-start gap-3 cursor-pointer">
                      <Checkbox
                        checked={!!confirmations[item.key]}
                        onCheckedChange={(checked) =>
                          setConfirmations((prev) => ({ ...prev, [item.key]: !!checked }))
                        }
                        className="mt-0.5"
                      />
                      <span className="text-sm">{item.label}</span>
                    </label>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {allConfirmed && (
            <>
              {/* Mode Selector */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Select Mode</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <Button
                      onClick={() => handleModeChange("thesis")}
                      variant={mode === "thesis" ? "default" : "outline"}
                      className="py-6 text-base font-bold"
                    >
                      📖 Thesis Chapters
                    </Button>
                    <Button
                      onClick={() => handleModeChange("proposal")}
                      variant={mode === "proposal" ? "default" : "outline"}
                      className="py-6 text-base font-bold"
                    >
                      📝 Proposal Development
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Research Profile */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Your Research Profile</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <Input placeholder="Student Name (optional)" value={researchProfile.studentName} onChange={(e) => updateProfile("studentName", e.target.value)} />
                    <Select value={researchProfile.degreeLevel} onValueChange={(v) => updateProfile("degreeLevel", v)}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="MSc">MSc</SelectItem>
                        <SelectItem value="PhD">PhD</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid md:grid-cols-2 gap-4">
                    <Input placeholder="Department or Discipline" value={researchProfile.discipline} onChange={(e) => updateProfile("discipline", e.target.value)} />
                    <Input placeholder={mode === "thesis" ? "Thesis Title" : "Project / Proposal Title"} value={researchProfile.thesisTitle} onChange={(e) => updateProfile("thesisTitle", e.target.value)} />
                  </div>
                  <Textarea placeholder="Research Objectives" value={researchProfile.objectives} onChange={(e) => updateProfile("objectives", e.target.value)} rows={3} />
                  <div className="grid md:grid-cols-2 gap-4">
                    <Input placeholder="Study Location / Context" value={researchProfile.studyLocation} onChange={(e) => updateProfile("studyLocation", e.target.value)} />
                    <Select value={researchProfile.experimentalDesign} onValueChange={(v) => updateProfile("experimentalDesign", v)}>
                      <SelectTrigger><SelectValue placeholder="Experimental Design" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="RCBD">RCBD</SelectItem>
                        <SelectItem value="CRD">CRD</SelectItem>
                        <SelectItem value="Factorial">Factorial</SelectItem>
                        <SelectItem value="Survey">Survey</SelectItem>
                        <SelectItem value="Laboratory experiment">Lab Experiment</SelectItem>
                        <SelectItem value="Modeling study">Modeling Study</SelectItem>
                        <SelectItem value="Other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid md:grid-cols-2 gap-4">
                    <Input placeholder="Variables Measured" value={researchProfile.variables} onChange={(e) => updateProfile("variables", e.target.value)} />
                    <Select value={researchProfile.statisticalSoftware} onValueChange={(v) => updateProfile("statisticalSoftware", v)}>
                      <SelectTrigger><SelectValue placeholder="Statistical Software" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="R">R</SelectItem>
                        <SelectItem value="SAS">SAS</SelectItem>
                        <SelectItem value="SPSS">SPSS</SelectItem>
                        <SelectItem value="GenStat">GenStat</SelectItem>
                        <SelectItem value="Excel">Excel</SelectItem>
                        <SelectItem value="Python">Python</SelectItem>
                        <SelectItem value="Other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Select value={researchProfile.referencingStyle} onValueChange={(v) => updateProfile("referencingStyle", v)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="APA">APA</SelectItem>
                      <SelectItem value="Harvard">Harvard</SelectItem>
                      <SelectItem value="Chicago">Chicago</SelectItem>
                      <SelectItem value="Vancouver">Vancouver</SelectItem>
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>

              {/* Section Selector */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">
                    Select {mode === "thesis" ? "Chapter" : "Proposal Section"}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Select value={chapter} onValueChange={setChapter}>
                    <SelectTrigger><SelectValue placeholder={`-- Select ${mode === "thesis" ? "Chapter" : "Proposal Section"} --`} /></SelectTrigger>
                    <SelectContent>
                      {sections.map((s) => (
                        <SelectItem key={s} value={s}>{s}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>

              {/* Reflection */}
              {chapter && (
                <Card className="border-accent bg-accent/5">
                  <CardHeader>
                    <CardTitle className="text-lg">Reflection Question</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground mb-3">
                      {REFLECTION_QUESTIONS[mode]?.[chapter] || "Reflect on your understanding of this section."}
                    </p>
                    <Textarea
                      placeholder="Write a detailed reflection (minimum 20 words)..."
                      value={reflection}
                      onChange={(e) => setReflection(e.target.value)}
                      rows={4}
                    />
                    <p className={`text-xs mt-2 ${wordCount < MIN_REFLECTION_WORDS ? "text-destructive" : "text-muted-foreground"}`}>
                      {wordCount} / {MIN_REFLECTION_WORDS} words minimum
                    </p>
                  </CardContent>
                </Card>
              )}

              {/* Optional Inputs */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Optional: Paste Draft or Data</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea placeholder="Paste your draft text (optional)" value={draft} onChange={(e) => setDraft(e.target.value)} rows={4} />
                  <Textarea placeholder="Paste statistical output (optional)" value={statsOutput} onChange={(e) => setStatsOutput(e.target.value)} rows={3} />
                  <p className="text-xs text-destructive flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" /> Do not paste confidential research data.
                  </p>
                </CardContent>
              </Card>

              {/* Generate Button */}
              <Button
                onClick={handleGenerate}
                disabled={!chapter || wordCount < MIN_REFLECTION_WORDS || isLoading}
                className="w-full py-6 text-lg font-bold"
                size="lg"
              >
                {isLoading ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="w-5 h-5 animate-spin" /> Generating Guidance...
                  </span>
                ) : (
                  "Generate AI Guidance"
                )}
              </Button>

              {/* Guidance Output */}
              {guidance && (
                <>
                  <div className="space-y-4">
                    {SECTION_TITLES.map((s) => {
                      const content = guidance[s.key as keyof GuidanceSections];
                      if (!content || content === "Section not found") return null;
                      return (
                        <Card key={s.key}>
                          <CardHeader>
                            <CardTitle className="text-lg">{s.title}</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="prose prose-sm max-w-none dark:prose-invert">
                              <ReactMarkdown>{cleanMarkdown(content)}</ReactMarkdown>
                            </div>
                          </CardContent>
                        </Card>
                      );
                    })}
                  </div>

                  {/* Post-Guidance Reflection */}
                  <Card className="border-accent bg-accent/5">
                    <CardHeader>
                      <CardTitle className="text-lg">Next Step: Post-Guidance Reflection</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-3">
                        Based on this guidance, what will you write in your next draft? (2–3 sentences)
                      </p>
                      <Textarea
                        placeholder="Write 2-3 sentences about what you will write next..."
                        value={postReflection}
                        onChange={(e) => setPostReflection(e.target.value)}
                        rows={4}
                      />
                    </CardContent>
                  </Card>

                  {/* Disclaimers */}
                  <Card className="border-yellow-300 bg-yellow-50/50 dark:bg-yellow-900/10">
                    <CardContent className="pt-6">
                      <p className="text-sm font-semibold mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" /> Important Disclaimers
                      </p>
                      <ul className="text-xs space-y-2 text-muted-foreground">
                        <li>• All guidance is for your own writing — you must rewrite in your own words.</li>
                        <li>• Verify all factual claims and citations with your supervisor.</li>
                        {mode === "proposal" && (
                          <li>• Use this guidance to draft your proposal in your own words; confirm feasibility with your supervisor.</li>
                        )}
                        {mode === "thesis" && (
                          <li>• Statistical explanations are guidance only. Verify with your supervisor or statistician.</li>
                        )}
                      </ul>
                    </CardContent>
                  </Card>

                  {/* Download PDF */}
                  <Button onClick={handleDownloadPDF} className="w-full py-6 text-lg" variant="secondary" size="lg">
                    <Download className="w-5 h-5 mr-2" /> Download Guidance Notes (PDF)
                  </Button>
                </>
              )}
            </>
          )}
      </>
    );
  }
}
