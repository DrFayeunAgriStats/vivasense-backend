import { useState, useEffect } from "react";
import { Layout } from "@/components/layout/Layout";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { Search, Download, RotateCcw, Lock, ShieldCheck, Plus, Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button";

interface AdapStudentRow {
  id: string;
  student_id: string;
  full_name: string;
  cohort: string;
  current_week: number;
  completed_weeks: number[];
  last_quiz_score: string | null;
  certificate_code: string | null;
  last_active: string;
}

interface AdapCode {
  id: string;
  code: string;
  is_admin: boolean;
  status: string;
  created_at: string;
}

export default function AdapAdmin() {
  const [authenticated, setAuthenticated] = useState(false);
  const [password, setPassword] = useState("");
  const [students, setStudents] = useState<AdapStudentRow[]>([]);
  const [codes, setCodes] = useState<AdapCode[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"students" | "codes">("students");
  const [codePrefix, setCodePrefix] = useState("FIA-ADAP");
  const [codeCount, setCodeCount] = useState(10);
  const [generating, setGenerating] = useState(false);
  const [copiedCodes, setCopiedCodes] = useState(false);
  const { toast } = useToast();

  const handleLogin = async () => {
    try {
      const resp = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/adap-admin`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: JSON.stringify({ action: "verify", password }),
      });
      const data = await resp.json();
      if (data.success) {
        setAuthenticated(true);
        fetchStudents();
        fetchCodes();
      } else {
        toast({ title: "Access Denied", description: "Invalid password.", variant: "destructive" });
      }
    } catch {
      toast({ title: "Error", description: "Could not verify password.", variant: "destructive" });
    }
  };

  const fetchStudents = async () => {
    setLoading(true);
    const { data } = await supabase
      .from("adap_students")
      .select("*")
      .order("last_active", { ascending: false });
    setStudents((data as AdapStudentRow[]) || []);
    setLoading(false);
  };

  const fetchCodes = async () => {
    const { data } = await supabase
      .from("adap_student_codes")
      .select("*")
      .order("created_at", { ascending: false });
    setCodes((data as AdapCode[]) || []);
  };

  const resetStudent = async (studentId: string) => {
    if (!confirm(`Reset all progress for ${studentId}? This cannot be undone.`)) return;
    await supabase
      .from("adap_students")
      .update({
        current_week: 0,
        completed_weeks: [],
        last_quiz_score: null,
        certificate_code: null,
        chat_history: [],
      })
      .eq("student_id", studentId);
    toast({ title: "Reset", description: `Progress for ${studentId} has been reset.` });
    fetchStudents();
  };

  const exportCSV = () => {
    const header = "Student ID,Name,Cohort,Weeks Completed,Current Week,Last Quiz Score,Certificate Code,Last Active\n";
    const rows = students.map(s =>
      `"${s.student_id}","${s.full_name}","${s.cohort}","${(s.completed_weeks || []).join(";")}",${s.current_week},"${s.last_quiz_score || ""}","${s.certificate_code || ""}","${new Date(s.last_active).toLocaleString()}"`
    ).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `adap-students-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  };

  const generateCodes = async () => {
    if (codeCount < 1 || codeCount > 100) {
      toast({ title: "Invalid", description: "Enter 1–100 codes.", variant: "destructive" });
      return;
    }
    setGenerating(true);
    const newCodes: { code: string; is_admin: boolean; status: string }[] = [];
    for (let i = 0; i < codeCount; i++) {
      const num = String(codes.length + i + 1).padStart(3, "0");
      const code = `${codePrefix}-${num}`;
      newCodes.push({ code, is_admin: false, status: "active" });
    }

    const { error } = await supabase.from("adap_student_codes").insert(newCodes);
    if (error) {
      // Handle duplicates by using random suffix
      const fallbackCodes = [];
      for (let i = 0; i < codeCount; i++) {
        const rand = Math.random().toString(36).substring(2, 6).toUpperCase();
        fallbackCodes.push({ code: `${codePrefix}-${rand}`, is_admin: false, status: "active" });
      }
      await supabase.from("adap_student_codes").insert(fallbackCodes);
    }
    toast({ title: "Generated", description: `${codeCount} student codes created.` });
    fetchCodes();
    setGenerating(false);
  };

  const toggleCodeStatus = async (code: AdapCode) => {
    const newStatus = code.status === "active" ? "revoked" : "active";
    await supabase.from("adap_student_codes").update({ status: newStatus }).eq("id", code.id);
    toast({ title: "Updated", description: `Code ${code.code} is now ${newStatus}.` });
    fetchCodes();
  };

  const copyAllCodes = () => {
    const activeCodes = codes.filter(c => c.status === "active").map(c => c.code).join("\n");
    navigator.clipboard.writeText(activeCodes);
    setCopiedCodes(true);
    setTimeout(() => setCopiedCodes(false), 2000);
  };

  const filtered = students.filter(s =>
    s.student_id.toLowerCase().includes(search.toLowerCase()) ||
    s.full_name.toLowerCase().includes(search.toLowerCase())
  );

  const filteredCodes = codes.filter(c =>
    c.code.toLowerCase().includes(search.toLowerCase())
  );

  if (!authenticated) {
    return (
      <Layout>
        <div className="min-h-[60vh] flex items-center justify-center px-4" style={{ background: "#F5F8F6" }}>
          <div className="w-full max-w-sm bg-white rounded-2xl p-6 shadow-sm" style={{ border: "1px solid #DDE8E3" }}>
            <div className="flex items-center gap-2 mb-4 justify-center" style={{ color: "#0D5C3A" }}>
              <Lock className="w-5 h-5" />
              <h2 className="font-serif font-bold text-lg">Admin Access</h2>
            </div>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLogin()}
              placeholder="Enter admin password"
              className="w-full px-4 py-3 rounded-lg text-sm border mb-3 focus:outline-none focus:ring-2"
              style={{ borderColor: "#DDE8E3" }}
            />
            <Button onClick={handleLogin} className="w-full text-white" style={{ background: "#1B7A4E" }}>
              <ShieldCheck className="w-4 h-4 mr-2" /> Verify
            </Button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="py-6 px-4" style={{ background: "#F5F8F6", minHeight: "calc(100vh - 140px)" }}>
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between mb-4">
            <h1 className="font-serif text-xl font-bold" style={{ color: "#0D5C3A" }}>FIA-ADAP Admin Panel</h1>
            <div className="flex gap-2">
              <Button
                onClick={() => setActiveTab("students")}
                variant={activeTab === "students" ? "default" : "outline"}
                size="sm"
              >
                Students
              </Button>
              <Button
                onClick={() => setActiveTab("codes")}
                variant={activeTab === "codes" ? "default" : "outline"}
                size="sm"
              >
                Codes
              </Button>
            </div>
          </div>

          <div className="mb-4 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: "#7A9A8A" }} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={activeTab === "students" ? "Search by Student ID or Name..." : "Search codes..."}
              className="w-full pl-9 pr-4 py-2.5 rounded-xl text-sm border focus:outline-none focus:ring-2 bg-white"
              style={{ borderColor: "#DDE8E3" }}
            />
          </div>

          {activeTab === "students" && (
            <>
              <div className="flex justify-end mb-3">
                <Button onClick={exportCSV} variant="outline" size="sm" className="gap-1.5">
                  <Download className="w-4 h-4" /> Export CSV
                </Button>
              </div>
              <div className="bg-white rounded-xl overflow-hidden shadow-sm" style={{ border: "1px solid #DDE8E3" }}>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr style={{ background: "#EAF7EF", color: "#0D5C3A" }}>
                        <th className="px-3 py-2.5 text-left font-medium">Student ID</th>
                        <th className="px-3 py-2.5 text-left font-medium">Name</th>
                        <th className="px-3 py-2.5 text-left font-medium">Cohort</th>
                        <th className="px-3 py-2.5 text-center font-medium">Weeks</th>
                        <th className="px-3 py-2.5 text-center font-medium">Current</th>
                        <th className="px-3 py-2.5 text-center font-medium">Quiz</th>
                        <th className="px-3 py-2.5 text-left font-medium">Certificate</th>
                        <th className="px-3 py-2.5 text-left font-medium">Last Active</th>
                        <th className="px-3 py-2.5 text-center font-medium">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((s) => (
                        <tr key={s.id} className="border-t" style={{ borderColor: "#DDE8E3" }}>
                          <td className="px-3 py-2.5 font-mono text-xs">{s.student_id}</td>
                          <td className="px-3 py-2.5">{s.full_name}</td>
                          <td className="px-3 py-2.5">{s.cohort}</td>
                          <td className="px-3 py-2.5 text-center">{(s.completed_weeks || []).length}/7</td>
                          <td className="px-3 py-2.5 text-center">W{s.current_week}</td>
                          <td className="px-3 py-2.5 text-center">{s.last_quiz_score || "—"}</td>
                          <td className="px-3 py-2.5 font-mono text-xs">{s.certificate_code || "—"}</td>
                          <td className="px-3 py-2.5 text-xs">{new Date(s.last_active).toLocaleDateString()}</td>
                          <td className="px-3 py-2.5 text-center">
                            <button onClick={() => resetStudent(s.student_id)} className="p-1 rounded hover:bg-red-50 text-red-500" title="Reset">
                              <RotateCcw className="w-3.5 h-3.5" />
                            </button>
                          </td>
                        </tr>
                      ))}
                      {filtered.length === 0 && (
                        <tr><td colSpan={9} className="px-3 py-8 text-center" style={{ color: "#7A9A8A" }}>No students found</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {activeTab === "codes" && (
            <>
              {/* Code Generation */}
              <div className="bg-white rounded-xl p-4 mb-4 shadow-sm" style={{ border: "1px solid #DDE8E3" }}>
                <h3 className="font-medium text-sm mb-3" style={{ color: "#0D5C3A" }}>Generate Student Codes</h3>
                <div className="flex flex-wrap gap-3 items-end">
                  <div>
                    <label className="text-xs block mb-1" style={{ color: "#4A6B5D" }}>Prefix</label>
                    <input
                      type="text"
                      value={codePrefix}
                      onChange={(e) => setCodePrefix(e.target.value)}
                      className="px-3 py-2 rounded-lg text-sm border w-40"
                      style={{ borderColor: "#DDE8E3" }}
                    />
                  </div>
                  <div>
                    <label className="text-xs block mb-1" style={{ color: "#4A6B5D" }}>Count (1–100)</label>
                    <input
                      type="number"
                      value={codeCount}
                      onChange={(e) => setCodeCount(Number(e.target.value))}
                      min={1}
                      max={100}
                      className="px-3 py-2 rounded-lg text-sm border w-24"
                      style={{ borderColor: "#DDE8E3" }}
                    />
                  </div>
                  <Button
                    onClick={generateCodes}
                    disabled={generating}
                    className="text-white gap-1.5"
                    style={{ background: "#1B7A4E" }}
                  >
                    <Plus className="w-4 h-4" /> Generate
                  </Button>
                  <Button onClick={copyAllCodes} variant="outline" size="sm" className="gap-1.5">
                    {copiedCodes ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                    {copiedCodes ? "Copied!" : "Copy Active Codes"}
                  </Button>
                </div>
              </div>

              {/* Codes Table */}
              <div className="bg-white rounded-xl overflow-hidden shadow-sm" style={{ border: "1px solid #DDE8E3" }}>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr style={{ background: "#EAF7EF", color: "#0D5C3A" }}>
                        <th className="px-3 py-2.5 text-left font-medium">Code</th>
                        <th className="px-3 py-2.5 text-center font-medium">Status</th>
                        <th className="px-3 py-2.5 text-left font-medium">Created</th>
                        <th className="px-3 py-2.5 text-center font-medium">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredCodes.map((c) => (
                        <tr key={c.id} className="border-t" style={{ borderColor: "#DDE8E3" }}>
                          <td className="px-3 py-2.5 font-mono text-xs">{c.code}</td>
                          <td className="px-3 py-2.5 text-center">
                            <span
                              className="px-2 py-0.5 rounded-full text-xs font-medium"
                              style={{
                                background: c.status === "active" ? "#EAF7EF" : "#FEF2F2",
                                color: c.status === "active" ? "#1B7A4E" : "#DC2626",
                              }}
                            >
                              {c.status}
                            </span>
                          </td>
                          <td className="px-3 py-2.5 text-xs">{new Date(c.created_at).toLocaleDateString()}</td>
                          <td className="px-3 py-2.5 text-center">
                            <button
                              onClick={() => toggleCodeStatus(c)}
                              className="text-xs px-2 py-1 rounded-lg border transition-colors"
                              style={{
                                borderColor: c.status === "active" ? "#DC2626" : "#1B7A4E",
                                color: c.status === "active" ? "#DC2626" : "#1B7A4E",
                              }}
                            >
                              {c.status === "active" ? "Revoke" : "Activate"}
                            </button>
                          </td>
                        </tr>
                      ))}
                      {filteredCodes.length === 0 && (
                        <tr><td colSpan={4} className="px-3 py-8 text-center" style={{ color: "#7A9A8A" }}>No codes found</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </Layout>
  );
}
