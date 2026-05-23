import { useState, useRef, useEffect, useCallback } from "react";
import { Layout } from "@/components/layout/Layout";
import { useAdapSession } from "@/hooks/useAdapSession";
import { adapStreamChat, type ChatMessage } from "@/lib/adapStreamChat";
import { WEEK_TITLES, WEEK_TOPICS, QUICK_ACTIONS, R_CODE } from "@/data/adapWeekData";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";
import {
  Send, Bot, User, Loader2, Trophy, CheckCircle, Lock,
  BarChart3, Award, FileText, ChevronDown, ChevronUp,
  Upload, Copy, Check, LogOut, GraduationCap,
} from "lucide-react";
import { WeekQuizPanel } from "@/components/adap/WeekQuizPanel";
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";

// ─── LOGIN SCREEN ───────────────────────────────────────
function AdapLogin({ onLogin, loading }: { onLogin: (id: string, name: string, cohort: string) => void; loading: boolean }) {
  const [studentId, setStudentId] = useState("");
  const [fullName, setFullName] = useState("");
  const [cohort, setCohort] = useState("");
  const [error, setError] = useState("");
  const [validating, setValidating] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!studentId.trim() || !fullName.trim() || !cohort.trim()) return;

    setValidating(true);
    try {
      // Validate student code exists and is active
      const { data, error: fetchErr } = await supabase
        .from("adap_student_codes")
        .select("code, status")
        .eq("code", studentId.trim())
        .maybeSingle();

      if (fetchErr) throw fetchErr;

      if (!data) {
        setError("Invalid Student Code. Please contact admin for a valid code.");
        return;
      }
      if (data.status !== "active") {
        setError("This code has been deactivated. Contact admin.");
        return;
      }

      onLogin(studentId.trim(), fullName.trim(), cohort.trim());
    } catch {
      setError("Could not verify code. Please try again.");
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center px-4 py-12" style={{ background: "#F5F8F6" }}>
      <div className="w-full max-w-md space-y-6">
        {/* Hero Card */}
        <div className="rounded-2xl p-8 text-white text-center" style={{ background: "linear-gradient(135deg, #0D5C3A 0%, #1B7A4E 100%)" }}>
          <div className="flex justify-center mb-4">
            <div className="w-14 h-14 rounded-full bg-white/10 flex items-center justify-center">
              <GraduationCap className="w-7 h-7" />
            </div>
          </div>
          <h1 className="font-serif text-2xl font-bold mb-1">FIA-ADAP Foundations</h1>
          <p className="text-white/80 text-sm mb-4">Field-to-Insight Analytical Development & Practice</p>
          <div className="inline-block px-3 py-1 rounded-full text-xs font-medium" style={{ background: "rgba(232,160,32,0.25)", color: "#E8A020" }}>
            7 Weeks · Agricultural Data Analysis · 1 Certificate
          </div>
          <p className="text-white/60 text-xs italic mt-4">"From button-clicking to analytical reasoning." — Dr. Fayeun</p>
        </div>

        {/* Login Card */}
        <div className="rounded-2xl bg-white p-6 shadow-sm" style={{ border: "1px solid #DDE8E3" }}>
          <h2 className="font-serif text-lg font-bold text-center mb-4" style={{ color: "#0D5C3A" }}>Sign In / Register</h2>
          <form onSubmit={handleSubmit} className="space-y-3">
            <input
              type="text"
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
              placeholder="Student Code (e.g. FIA-ADAP-001)"
              required
              className="w-full px-4 py-3 rounded-lg text-sm border focus:outline-none focus:ring-2"
              style={{ borderColor: "#DDE8E3", background: "#F5F8F6" }}
            />
            <input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="Full Name"
              required
              className="w-full px-4 py-3 rounded-lg text-sm border focus:outline-none focus:ring-2"
              style={{ borderColor: "#DDE8E3", background: "#F5F8F6" }}
            />
            <input
              type="text"
              value={cohort}
              onChange={(e) => setCohort(e.target.value)}
              placeholder="Cohort (e.g. February 2026)"
              required
              className="w-full px-4 py-3 rounded-lg text-sm border focus:outline-none focus:ring-2"
              style={{ borderColor: "#DDE8E3", background: "#F5F8F6" }}
            />

            {error && (
              <div className="flex items-center gap-2 text-sm p-3 rounded-lg" style={{ background: "#FEF2F2", color: "#DC2626" }}>
                <Lock className="w-4 h-4 flex-shrink-0" />
                {error}
              </div>
            )}

            <Button
              type="submit"
              disabled={loading || validating || !studentId.trim() || !fullName.trim() || !cohort.trim()}
              className="w-full text-white font-medium py-3"
              style={{ background: "#1B7A4E" }}
            >
              {(loading || validating) ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
              Enter Tutor →
            </Button>
          </form>
          <p className="text-xs text-center mt-3" style={{ color: "#7A9A8A" }}>
            Your dashboard and progress are saved to your Student Code
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── DASHBOARD CARD ─────────────────────────────────────
function DashboardCard({ student, onLogout }: { student: any; onLogout: () => void }) {
  const completedCount = student.completedWeeks.length;
  const progress = Math.round((completedCount / 7) * 100);
  const allDone = completedCount === 7;

  return (
    <div className="rounded-2xl overflow-hidden shadow-sm" style={{ border: "1px solid #DDE8E3" }}>
      <div className="p-5 text-white" style={{ background: "linear-gradient(135deg, #0D5C3A 0%, #1B7A4E 100%)" }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Trophy className="w-6 h-6 text-amber-300" />
            <div>
              <h3 className="font-serif font-bold text-lg">Student Dashboard</h3>
              <p className="text-white/70 text-xs">Student ID: {student.studentId} · Cohort: {student.cohort}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-white/15">{completedCount}/7 Weeks Complete</span>
            <button onClick={onLogout} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors" title="Logout">
              <LogOut className="w-4 h-4 text-white/70" />
            </button>
          </div>
        </div>
      </div>

      <div className="divide-y" style={{ borderColor: "#DDE8E3", background: "white" }}>
        <DashRow icon="▶️" label="Current Week" value={`Week ${student.currentWeek}`} />
        <DashRow icon="✅" label="Completed" value={
          completedCount > 0
            ? <div className="flex flex-wrap gap-1">{student.completedWeeks.map((w: number) => (
                <span key={w} className="px-2 py-0.5 rounded-full text-xs font-medium text-white" style={{ background: "#1B7A4E" }}>W{w}</span>
              ))}</div>
            : "None yet"
        } />
        <DashRow icon="📈" label="Progress" value={`${progress}%`} />
        <DashRow icon="🎖️" label="Certificate Status" value={
          allDone
            ? <span className="font-mono text-xs font-bold" style={{ color: "#1B7A4E" }}>{student.certificateCode}</span>
            : <span className="flex items-center gap-1 text-xs" style={{ color: "#E8A020" }}><Lock className="w-3 h-3" /> Locked</span>
        } />
        <DashRow icon="📊" label="Last Quiz Score" value={student.lastQuizScore || "—"} />
      </div>

      <div className="px-5 py-3" style={{ background: "white", borderTop: "1px solid #DDE8E3" }}>
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="font-medium" style={{ color: "#0D5C3A" }}>Overall Progress</span>
          <span className="font-bold" style={{ color: "#1B7A4E" }}>{progress}%</span>
        </div>
        <div className="h-2.5 rounded-full overflow-hidden" style={{ background: "#EAF7EF" }}>
          <div className="h-full rounded-full transition-all duration-500" style={{ width: `${progress}%`, background: "#1B7A4E" }} />
        </div>
      </div>
    </div>
  );
}

function DashRow({ icon, label, value }: { icon: string; label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-5 py-2.5 text-sm" style={{ borderColor: "#DDE8E3" }}>
      <span className="flex items-center gap-2" style={{ color: "#4A6B5D" }}>{icon} {label}</span>
      <span className="font-medium" style={{ color: "#0D5C3A" }}>{typeof value === "string" ? value : value}</span>
    </div>
  );
}

// ─── WEEK TABS ──────────────────────────────────────────
function WeekTabs({ activeWeek, completedWeeks, currentWeek, onSelect }: {
  activeWeek: number; completedWeeks: number[]; currentWeek: number;
  onSelect: (w: number) => void;
}) {
  return (
    <div className="flex gap-1.5 overflow-x-auto pb-1 scrollbar-hide">
      {[0, 1, 2, 3, 4, 5, 6].map((w) => {
        const completed = completedWeeks.includes(w);
        // A week is unlocked only if it's week 0, or all previous weeks are completed
        const unlocked = w === 0 || completedWeeks.includes(w - 1) || completed;
        const active = w === activeWeek;
        return (
          <button
            key={w}
            onClick={() => unlocked && onSelect(w)}
            disabled={!unlocked}
            className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all
              ${active ? "text-white shadow-sm" : completed ? "text-white/90" : unlocked ? "hover:bg-opacity-80" : "opacity-40 cursor-not-allowed"}
            `}
            style={{
              background: active ? "#0D5C3A" : completed ? "#1B7A4E" : unlocked ? "#EAF7EF" : "#e5e7eb",
              color: active ? "white" : completed ? "white" : unlocked ? "#0D5C3A" : "#9ca3af",
            }}
          >
            {!unlocked && <Lock className="w-3 h-3" />}
            W{w}
            {completed && <CheckCircle className="w-3.5 h-3.5" />}
          </button>
        );
      })}
    </div>
  );
}

// ─── CERTIFICATE VIEW ───────────────────────────────────
function CertificateView({ student }: { student: any }) {
  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-2xl w-full p-10 text-center relative" id="adap-certificate">
        <div className="mb-6">
          <GraduationCap className="w-12 h-12 mx-auto mb-2" style={{ color: "#0D5C3A" }} />
          <h2 className="font-serif text-sm tracking-widest uppercase" style={{ color: "#0D5C3A" }}>Field-to-Insight Academy</h2>
        </div>
        <h1 className="font-serif text-3xl font-bold mb-1" style={{ color: "#0D5C3A" }}>Certificate of Completion</h1>
        <p className="text-sm mb-8" style={{ color: "#4A6B5D" }}>FIA-ADAP Foundations — Field-to-Insight Analytical Development & Practice</p>
        <div className="w-16 h-px mx-auto mb-8" style={{ background: "#E8A020" }} />
        <p className="text-sm mb-2" style={{ color: "#4A6B5D" }}>This is to certify that</p>
        <h2 className="font-serif text-2xl font-bold mb-2" style={{ color: "#0D5C3A" }}>{student.fullName}</h2>
        <div className="w-48 h-px mx-auto mb-6" style={{ background: "#E8A020" }} />
        <p className="text-sm max-w-md mx-auto mb-8" style={{ color: "#4A6B5D" }}>
          has successfully demonstrated mastery of agricultural data analysis across 7 weeks of competency-based training
        </p>
        <div className="grid grid-cols-3 gap-4 text-xs mb-8" style={{ color: "#4A6B5D" }}>
          <div><span className="font-semibold block">Cohort</span>{student.cohort}</div>
          <div><span className="font-semibold block">Date</span>{new Date().toLocaleDateString()}</div>
          <div><span className="font-semibold block">Certificate Code</span><span className="font-mono">{student.certificateCode}</span></div>
        </div>
        <div className="pt-4 border-t" style={{ borderColor: "#DDE8E3" }}>
          <div className="w-40 h-px mx-auto mb-2" style={{ background: "#0D5C3A" }} />
          <p className="text-xs font-medium" style={{ color: "#0D5C3A" }}>Prof. Lawrence Stephen Fayeun</p>
          <p className="text-xs" style={{ color: "#4A6B5D" }}>Director, Field-to-Insight Academy</p>
        </div>
      </div>
    </div>
  );
}

// ─── MAIN PAGE ──────────────────────────────────────────
export default function AdapTutor() {
  const { student, loading, login, saveProgress, completeWeek, logout } = useAdapSession();
  const [activeWeek, setActiveWeek] = useState(0);
  const [selectedTopic, setSelectedTopic] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [rCodeOpen, setRCodeOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showCert, setShowCert] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Sync state from student
  useEffect(() => {
    if (student) {
      setActiveWeek(student.currentWeek);
      setMessages(student.chatHistory.length > 0 ? student.chatHistory : [
        { role: "assistant", content: `Welcome to FIA-ADAP Foundations, ${student.fullName}! 🌱\n\nI'm **Dr. Fayeun**, your AI tutor for agricultural data analysis. You're currently on **Week ${student.currentWeek}: ${WEEK_TITLES[student.currentWeek]}**.\n\nUse the quick action buttons below, select a topic, or just ask me anything about this week's content.\n\n*Let's build your analytical foundation — step by step.*\n\n— Dr. Fayeun` },
      ]);
    }
  }, [student?.studentId]);

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isStreaming || !student) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setInput("");
    setMessages(prev => [...prev, userMsg]);
    setIsStreaming(true);

    let assistantSoFar = "";
    const upsert = (chunk: string) => {
      assistantSoFar += chunk;
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && prev.length > 1 && prev[prev.length - 2]?.role === "user") {
          return prev.map((m, i) => (i === prev.length - 1 ? { ...m, content: assistantSoFar } : m));
        }
        return [...prev, { role: "assistant", content: assistantSoFar }];
      });
    };

    try {
      const history = messages.filter(m => m.role === "user" || m.role === "assistant").slice(-20);
      await adapStreamChat({
        messages: [...history, userMsg],
        studentName: student.fullName,
        studentId: student.studentId,
        currentWeek: activeWeek,
        onDelta: upsert,
        onDone: () => {
          setIsStreaming(false);
        },
        onError: (err) => {
          toast({ title: "Error", description: err, variant: "destructive" });
          setIsStreaming(false);
        },
      });
    } catch {
      toast({ title: "Error", description: "Connection failed. Please try again.", variant: "destructive" });
      setIsStreaming(false);
    }
  }, [isStreaming, student, activeWeek, messages, completeWeek, toast]);

  // Save chat periodically
  useEffect(() => {
    if (!student || messages.length === 0) return;
    const timer = setTimeout(() => {
      saveProgress({ chatHistory: messages });
    }, 2000);
    return () => clearTimeout(timer);
  }, [messages, student, saveProgress]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const copyCode = () => {
    navigator.clipboard.writeText(R_CODE[activeWeek]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      const lines = text.split("\n").slice(0, 4);
      const preview = lines.join("\n");
      const cols = lines[0] || "";
      sendMessage(`The student has uploaded a dataset with the following structure:\nColumns: ${cols}\nPreview:\n${preview}\n\nHelp them analyse it using the methods from Week ${activeWeek}.`);
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  if (!student) {
    return (
      <Layout>
        <AdapLogin onLogin={login} loading={loading} />
        <AdapFooter />
      </Layout>
    );
  }

  const allDone = student.completedWeeks.length === 7;

  return (
    <Layout>
      {showCert && <CertificateView student={student} />}
      {showCert && (
        <div className="fixed inset-0 z-[60] flex items-end justify-center pb-8">
          <div className="flex gap-3">
            <Button onClick={() => window.print()} variant="outline" className="bg-white">Print / Save as PDF</Button>
            <Button onClick={() => setShowCert(false)} variant="outline" className="bg-white">Close</Button>
          </div>
        </div>
      )}

      <div className="py-4 md:py-6 px-4" style={{ background: "#F5F8F6", minHeight: "calc(100vh - 140px)" }}>
        <div className="max-w-3xl mx-auto space-y-4">
          {/* Dashboard */}
          <DashboardCard student={student} onLogout={logout} />

          {/* Certificate Download */}
          {allDone && (
            <button
              onClick={() => setShowCert(true)}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-white font-medium transition-all hover:opacity-90"
              style={{ background: "linear-gradient(135deg, #0D5C3A, #1B7A4E)" }}
            >
              <Award className="w-5 h-5" /> 🎓 Download Certificate
            </button>
          )}

          {/* Week Tabs */}
          <WeekTabs
            activeWeek={activeWeek}
            completedWeeks={student.completedWeeks}
            currentWeek={student.currentWeek}
            onSelect={setActiveWeek}
          />

          {/* Topic Selector */}
          <select
            value={selectedTopic}
            onChange={(e) => {
              setSelectedTopic(e.target.value);
              if (e.target.value) sendMessage(`Teach me about: ${e.target.value}`);
            }}
            className="w-full px-4 py-3 rounded-xl text-sm border bg-white focus:outline-none focus:ring-2"
            style={{ borderColor: "#DDE8E3" }}
          >
            <option value="">Choose a topic to focus your questions...</option>
            {WEEK_TOPICS[activeWeek]?.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>

          {/* Chat Window */}
          <div className="rounded-2xl overflow-hidden shadow-sm" style={{ border: "1px solid #DDE8E3", background: "white" }}>
            <div ref={scrollRef} className="h-[400px] overflow-y-auto p-4 space-y-3">
              {messages.map((msg, i) => (
                <div key={i} className={`flex gap-2.5 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                  {msg.role === "assistant" && (
                    <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: "#1B7A4E" }}>
                      <Bot className="w-3.5 h-3.5 text-white" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-xl px-4 py-3 text-sm`}
                    style={{
                      background: msg.role === "user" ? "#1B7A4E" : "#EAF7EF",
                      color: msg.role === "user" ? "white" : "#0D5C3A",
                    }}
                  >
                    <div className="prose prose-sm max-w-none [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_pre]:my-1 [&_code]:text-xs">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                  {msg.role === "user" && (
                    <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: "#EAF7EF" }}>
                      <User className="w-3.5 h-3.5" style={{ color: "#1B7A4E" }} />
                    </div>
                  )}
                </div>
              ))}
              {isStreaming && messages[messages.length - 1]?.role === "user" && (
                <div className="flex gap-2.5">
                  <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: "#1B7A4E" }}>
                    <Bot className="w-3.5 h-3.5 text-white" />
                  </div>
                  <div className="rounded-xl px-4 py-3 flex gap-1" style={{ background: "#EAF7EF" }}>
                    <span className="w-2 h-2 rounded-full animate-bounce" style={{ background: "#1B7A4E", animationDelay: "0ms" }} />
                    <span className="w-2 h-2 rounded-full animate-bounce" style={{ background: "#1B7A4E", animationDelay: "150ms" }} />
                    <span className="w-2 h-2 rounded-full animate-bounce" style={{ background: "#1B7A4E", animationDelay: "300ms" }} />
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="p-3" style={{ borderTop: "1px solid #DDE8E3" }}>
              <div className="flex gap-2">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question about agricultural data analysis..."
                  className="flex-1 resize-none min-h-[44px] max-h-[120px] px-3 py-2.5 rounded-xl text-sm border focus:outline-none focus:ring-2"
                  style={{ borderColor: "#DDE8E3", background: "#F5F8F6" }}
                  rows={1}
                  disabled={isStreaming}
                />
                <button
                  onClick={() => sendMessage(input)}
                  disabled={isStreaming || !input.trim()}
                  className="w-10 h-10 rounded-xl flex items-center justify-center text-white disabled:opacity-40 transition-colors"
                  style={{ background: "#0D5C3A" }}
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex flex-wrap gap-2">
            {QUICK_ACTIONS[activeWeek]?.map((action) => (
              <button
                key={action}
                onClick={() => sendMessage(action)}
                disabled={isStreaming}
                className="px-3 py-1.5 rounded-full text-xs font-medium transition-all hover:shadow-sm disabled:opacity-40"
                style={{
                  background: action.startsWith("✅") ? "#E8A020" : "#EAF7EF",
                  color: action.startsWith("✅") ? "white" : "#0D5C3A",
                  border: `1px solid ${action.startsWith("✅") ? "#E8A020" : "#DDE8E3"}`,
                }}
              >
                {action}
              </button>
            ))}
          </div>

          {/* R Code Panel */}
          <div className="rounded-xl overflow-hidden" style={{ border: "1px solid #DDE8E3" }}>
            <button
              onClick={() => setRCodeOpen(!rCodeOpen)}
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium"
              style={{ background: "#EAF7EF", color: "#0D5C3A" }}
            >
              <span>📦 R Code for Week {activeWeek}</span>
              {rCodeOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            {rCodeOpen && (
              <div className="relative">
                <button
                  onClick={copyCode}
                  className="absolute top-2 right-2 p-1.5 rounded-md bg-white/10 hover:bg-white/20 text-white transition-colors"
                >
                  {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </button>
                <pre className="p-4 text-xs overflow-x-auto" style={{ background: "#1a1a2e", color: "#e0e0e0" }}>
                  <code>{R_CODE[activeWeek]}</code>
                </pre>
              </div>
            )}
          </div>

          {/* Weekly Quiz */}
          <WeekQuizPanel
            weekNumber={activeWeek}
            alreadyCompleted={student.completedWeeks.includes(activeWeek)}
            onPass={() => {
              if (!student.completedWeeks.includes(activeWeek)) {
                completeWeek(activeWeek);
                if (activeWeek < 6) setActiveWeek(activeWeek + 1);
              }
            }}
          />

          {/* Dataset Upload */}
          <div className="rounded-xl overflow-hidden" style={{ border: "1px solid #DDE8E3" }}>
            <button
              onClick={() => setUploadOpen(!uploadOpen)}
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium"
              style={{ background: "#EAF7EF", color: "#0D5C3A" }}
            >
              <span>📂 Upload Practice Dataset</span>
              {uploadOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            {uploadOpen && (
              <div className="p-4 bg-white">
                <label className="flex flex-col items-center gap-2 py-6 rounded-xl cursor-pointer border-2 border-dashed transition-colors hover:border-green-400" style={{ borderColor: "#DDE8E3" }}>
                  <Upload className="w-6 h-6" style={{ color: "#1B7A4E" }} />
                  <span className="text-sm" style={{ color: "#4A6B5D" }}>Click to upload a .csv file</span>
                  <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>
            )}
          </div>
        </div>
      </div>
      <AdapFooter />
    </Layout>
  );
}

function AdapFooter() {
  return (
    <div className="text-center py-4 text-xs" style={{ color: "#7A9A8A", background: "#F5F8F6", borderTop: "1px solid #DDE8E3" }}>
      Powered by FIA | AI-assisted learning tool · © Dr. Fayeun Lawrence Stephen
    </div>
  );
}
