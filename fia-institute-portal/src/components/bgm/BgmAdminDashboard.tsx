import { useState, useEffect, useMemo } from "react";
import { supabase } from "@/integrations/supabase/client";
import { BgmStudent } from "@/hooks/useBgmSession";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  ArrowLeft, Users, BookOpen, BarChart3, FileText,
  Search, RefreshCw, Download, CheckCircle2, Clock, AlertCircle,
  GraduationCap, TrendingUp, Loader2, Globe, Eye, CalendarIcon, Plus, Copy,
} from "lucide-react";
import { toast } from "sonner";
import { MODULE_NAMES } from "@/data/bgmModuleContent";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { format, subDays, isAfter, isBefore, startOfDay, endOfDay, parseISO } from "date-fns";
import { cn } from "@/lib/utils";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";

type Props = {
  student: BgmStudent;
  onBack: () => void;
};

type StudentRow = {
  id: string;
  code: string;
  full_name: string;
  registration_id: string;
  current_module: number;
  completed_modules: number[];
  best_scores: Record<string, number>;
  r_practice_status: Record<string, string>;
  progress_percent: number;
  token_status: string;
  completion_token: string | null;
  created_at: string;
  updated_at: string;
};

type SubmissionRow = {
  id: string;
  author_name: string;
  email: string;
  manuscript_title: string;
  abstract: string;
  file_path: string | null;
  status: string | null;
  created_at: string;
};

type PageVisitSummary = {
  page_path: string;
  count: number;
};

function generateCode(prefix = "FIA-BGM") {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  let suffix = "";
  for (let i = 0; i < 6; i++) suffix += chars[Math.floor(Math.random() * chars.length)];
  return `${prefix}-${suffix}`;
}

function CodeGenerator({ onGenerated }: { onGenerated: () => void }) {
  const [count, setCount] = useState(1);
  const [prefix, setPrefix] = useState("FIA-BGM");
  const [generating, setGenerating] = useState(false);
  const [generated, setGenerated] = useState<string[]>([]);

  const handleGenerate = async () => {
    setGenerating(true);
    const newCodes: string[] = [];
    for (let i = 0; i < count; i++) newCodes.push(generateCode(prefix));

    const { error } = await supabase
      .from("student_codes")
      .insert(newCodes.map(code => ({ code, status: "active", is_admin: false })));

    if (error) {
      toast.error("Failed to generate codes: " + error.message);
    } else {
      setGenerated(newCodes);
      toast.success(`${newCodes.length} code(s) generated successfully!`);
      onGenerated();
    }
    setGenerating(false);
  };

  const copyAll = () => {
    navigator.clipboard.writeText(generated.join("\n"));
    toast.success("Codes copied to clipboard!");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Plus className="w-5 h-5 text-primary" />
          Generate Student Codes
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Prefix</label>
            <Input value={prefix} onChange={e => setPrefix(e.target.value)} className="w-32" />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">Count</label>
            <Input type="number" min={1} max={100} value={count}
              onChange={e => setCount(Math.max(1, Math.min(100, Number(e.target.value))))} className="w-20" />
          </div>
          <Button onClick={handleGenerate} disabled={generating}>
            {generating ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : <Plus className="w-4 h-4 mr-1" />}
            Generate
          </Button>
        </div>
        {generated.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Generated Codes:</span>
              <Button variant="outline" size="sm" onClick={copyAll}>
                <Copy className="w-3 h-3 mr-1" /> Copy All
              </Button>
            </div>
            <div className="bg-muted rounded-md p-3 max-h-40 overflow-auto font-mono text-xs space-y-1">
              {generated.map(c => <div key={c}>{c}</div>)}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function BgmAdminDashboard({ student, onBack }: Props) {
  const [students, setStudents] = useState<StudentRow[]>([]);
  const [submissions, setSubmissions] = useState<SubmissionRow[]>([]);
  const [codes, setCodes] = useState<{ code: string; status: string; is_admin: boolean }[]>([]);
  const [rawVisits, setRawVisits] = useState<{ page_path: string; visited_at: string }[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState("students");

  // Date filter state
  const [visitFilter, setVisitFilter] = useState<"7d" | "30d" | "all" | "custom">("all");
  const [customFrom, setCustomFrom] = useState<Date | undefined>(undefined);
  const [customTo, setCustomTo] = useState<Date | undefined>(undefined);

  const fetchAll = async () => {
    setLoading(true);
    const [studentsRes, codesRes, subsRes, visitsRes] = await Promise.all([
      supabase.from("bgm_students").select("*").order("created_at", { ascending: false }),
      supabase.from("student_codes").select("code, status, is_admin"),
      supabase.from("journal_articles").select("*").order("created_at", { ascending: false }),
      supabase.from("page_visits").select("page_path, visited_at"),
    ]);

    if (studentsRes.data) {
      setStudents(studentsRes.data.map(s => ({
        ...s,
        best_scores: (s.best_scores as Record<string, number>) || {},
        r_practice_status: (s.r_practice_status as Record<string, string>) || {},
      })));
    }
    if (codesRes.data) setCodes(codesRes.data);
    if (subsRes.data) setSubmissions(subsRes.data as any);

    // Store raw visits
    if (visitsRes.data) {
      setRawVisits(visitsRes.data);
    }

    setLoading(false);
  };

  useEffect(() => { fetchAll(); }, []);

  // Derived filtered visits data
  const { pageVisits, totalVisits, todayVisits, filteredTotal, dailyTrend } = useMemo(() => {
    let filtered = rawVisits;

    if (visitFilter === "7d") {
      const cutoff = startOfDay(subDays(new Date(), 7));
      filtered = rawVisits.filter(v => isAfter(new Date(v.visited_at), cutoff));
    } else if (visitFilter === "30d") {
      const cutoff = startOfDay(subDays(new Date(), 30));
      filtered = rawVisits.filter(v => isAfter(new Date(v.visited_at), cutoff));
    } else if (visitFilter === "custom") {
      filtered = rawVisits.filter(v => {
        const d = new Date(v.visited_at);
        if (customFrom && isBefore(d, startOfDay(customFrom))) return false;
        if (customTo && isAfter(d, endOfDay(customTo))) return false;
        return true;
      });
    }

    const today = new Date().toDateString();
    const todayCount = filtered.filter(v => new Date(v.visited_at).toDateString() === today).length;

    const countMap: Record<string, number> = {};
    filtered.forEach(v => {
      countMap[v.page_path] = (countMap[v.page_path] || 0) + 1;
    });
    const sorted = Object.entries(countMap)
      .map(([page_path, count]) => ({ page_path, count }))
      .sort((a, b) => b.count - a.count);

    // Daily trend aggregation
    const dayMap: Record<string, number> = {};
    filtered.forEach(v => {
      const day = format(new Date(v.visited_at), "yyyy-MM-dd");
      dayMap[day] = (dayMap[day] || 0) + 1;
    });
    const trend = Object.entries(dayMap)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([day, visits]) => ({ date: format(parseISO(day), "MMM dd"), visits }));

    return {
      pageVisits: sorted,
      totalVisits: rawVisits.length,
      todayVisits: todayCount,
      filteredTotal: filtered.length,
      dailyTrend: trend,
    };
  }, [rawVisits, visitFilter, customFrom, customTo]);

  // Analytics
  const totalStudents = students.length;
  const avgProgress = totalStudents > 0
    ? Math.round(students.reduce((a, s) => a + s.progress_percent, 0) / totalStudents)
    : 0;
  const completedAll = students.filter(s => s.completed_modules.length >= 11).length;
  const activeToday = students.filter(s => {
    const d = new Date(s.updated_at);
    const now = new Date();
    return d.toDateString() === now.toDateString();
  }).length;

  // Module completion distribution
  const moduleDistribution = Array.from({ length: 11 }, (_, i) => {
    const mod = i + 1;
    const count = students.filter(s => s.completed_modules.includes(mod)).length;
    return { module: mod, name: MODULE_NAMES[i], count, pct: totalStudents > 0 ? Math.round((count / totalStudents) * 100) : 0 };
  });

  const filteredStudents = students.filter(s =>
    s.full_name.toLowerCase().includes(search.toLowerCase()) ||
    s.code.toLowerCase().includes(search.toLowerCase()) ||
    s.registration_id.toLowerCase().includes(search.toLowerCase())
  );

  const exportCSV = () => {
    const header = "Name,Code,Registration ID,Current Module,Completed,Progress %,Token Status\n";
    const rows = students.map(s =>
      `"${s.full_name}","${s.code}","${s.registration_id}",${s.current_module},"${s.completed_modules.join(";")}",${s.progress_percent},"${s.token_status}"`
    ).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bgm_students_export.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-primary text-primary-foreground py-4">
        <div className="container-wide flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={onBack}
              className="text-primary-foreground hover:bg-primary-foreground/10">
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h1 className="font-serif text-lg font-bold">Admin Dashboard</h1>
              <p className="text-primary-foreground/70 text-xs">Logged in as {student.full_name}</p>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={fetchAll}
            className="text-primary-foreground border-primary-foreground/30 hover:bg-primary-foreground/10">
            <RefreshCw className="w-4 h-4 mr-1" /> Refresh
          </Button>
        </div>
      </div>

      <div className="container-wide py-6 max-w-6xl mx-auto space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <Users className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Total Students</p>
              <p className="font-bold text-2xl">{totalStudents}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <TrendingUp className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Avg Progress</p>
              <p className="font-bold text-2xl">{avgProgress}%</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <GraduationCap className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Fully Completed</p>
              <p className="font-bold text-2xl">{completedAll}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <Clock className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Active Today</p>
              <p className="font-bold text-2xl">{activeToday}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <Globe className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Total Page Visits</p>
              <p className="font-bold text-2xl">{totalVisits}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <Eye className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Visits Today</p>
              <p className="font-bold text-2xl">{todayVisits}</p>
            </CardContent>
          </Card>
        </div>

        {/* Tabs */}
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="grid grid-cols-5 w-full max-w-2xl">
            <TabsTrigger value="students"><Users className="w-4 h-4 mr-1" /> Students</TabsTrigger>
            <TabsTrigger value="visits"><Globe className="w-4 h-4 mr-1" /> Visits</TabsTrigger>
            <TabsTrigger value="journal"><FileText className="w-4 h-4 mr-1" /> Journal</TabsTrigger>
            <TabsTrigger value="analytics"><BarChart3 className="w-4 h-4 mr-1" /> Analytics</TabsTrigger>
            <TabsTrigger value="codes"><BookOpen className="w-4 h-4 mr-1" /> Codes</TabsTrigger>
          </TabsList>

          {/* Students Tab */}
          <TabsContent value="students" className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="relative flex-1 max-w-sm">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input placeholder="Search students..." value={search} onChange={e => setSearch(e.target.value)}
                  className="pl-9" />
              </div>
              <Button variant="outline" size="sm" onClick={exportCSV}>
                <Download className="w-4 h-4 mr-1" /> Export CSV
              </Button>
            </div>

            <Card>
              <CardContent className="p-0">
                <div className="overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Code</TableHead>
                        <TableHead>Reg ID</TableHead>
                        <TableHead className="text-center">Module</TableHead>
                        <TableHead className="text-center">Completed</TableHead>
                        <TableHead className="text-center">Progress</TableHead>
                        <TableHead className="text-center">Token</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredStudents.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                            No students found
                          </TableCell>
                        </TableRow>
                      ) : filteredStudents.map(s => (
                        <TableRow key={s.id}>
                          <TableCell className="font-medium">{s.full_name}</TableCell>
                          <TableCell className="font-mono text-xs">{s.code}</TableCell>
                          <TableCell className="text-xs">{s.registration_id}</TableCell>
                          <TableCell className="text-center">{s.current_module}</TableCell>
                          <TableCell className="text-center">
                            <Badge variant="outline">{s.completed_modules.length}/11</Badge>
                          </TableCell>
                          <TableCell className="text-center">
                            <div className="flex items-center gap-2">
                              <Progress value={s.progress_percent} className="h-2 w-16" />
                              <span className="text-xs">{s.progress_percent}%</span>
                            </div>
                          </TableCell>
                          <TableCell className="text-center">
                            {s.token_status === "Locked" ? (
                              <Badge variant="outline" className="text-xs">Locked</Badge>
                            ) : (
                              <Badge className="text-xs bg-primary">Generated</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Visits Tab */}
          <TabsContent value="visits" className="space-y-4">
            {/* Date Filter Bar */}
            <Card>
              <CardContent className="pt-4 pb-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-muted-foreground mr-1">Period:</span>
                  {(["7d", "30d", "all"] as const).map((key) => (
                    <Button
                      key={key}
                      variant={visitFilter === key ? "default" : "outline"}
                      size="sm"
                      onClick={() => setVisitFilter(key)}
                    >
                      {key === "7d" ? "Last 7 Days" : key === "30d" ? "Last 30 Days" : "All Time"}
                    </Button>
                  ))}
                  <Button
                    variant={visitFilter === "custom" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setVisitFilter("custom")}
                  >
                    <CalendarIcon className="w-4 h-4 mr-1" /> Custom
                  </Button>

                  {visitFilter === "custom" && (
                    <div className="flex items-center gap-2 ml-2">
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button variant="outline" size="sm" className={cn("justify-start text-left font-normal", !customFrom && "text-muted-foreground")}>
                            <CalendarIcon className="w-4 h-4 mr-1" />
                            {customFrom ? format(customFrom, "MMM dd, yyyy") : "From"}
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-auto p-0" align="start">
                          <Calendar mode="single" selected={customFrom} onSelect={setCustomFrom} initialFocus className="p-3 pointer-events-auto" />
                        </PopoverContent>
                      </Popover>
                      <span className="text-muted-foreground text-sm">–</span>
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button variant="outline" size="sm" className={cn("justify-start text-left font-normal", !customTo && "text-muted-foreground")}>
                            <CalendarIcon className="w-4 h-4 mr-1" />
                            {customTo ? format(customTo, "MMM dd, yyyy") : "To"}
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-auto p-0" align="start">
                          <Calendar mode="single" selected={customTo} onSelect={setCustomTo} initialFocus className="p-3 pointer-events-auto" />
                        </PopoverContent>
                      </Popover>
                    </div>
                  )}

                  <span className="ml-auto text-xs text-muted-foreground">
                    Showing <strong>{filteredTotal}</strong> of {totalVisits} visits
                  </span>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Globe className="w-5 h-5 text-primary" />
                    Page Visit Summary
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {pageVisits.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No visits recorded yet.</p>
                  ) : pageVisits.map((pv) => {
                    const maxCount = pageVisits[0]?.count || 1;
                    const pct = Math.round((pv.count / maxCount) * 100);
                    return (
                      <div key={pv.page_path} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground font-mono truncate max-w-[200px]">
                            {pv.page_path}
                          </span>
                          <span className="font-medium">{pv.count} visits</span>
                        </div>
                        <Progress value={pct} className="h-2" />
                      </div>
                    );
                  })}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Eye className="w-5 h-5 text-primary" />
                    Quick Stats
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Filtered Views</span>
                    <span className="font-bold text-lg">{filteredTotal}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Total Page Views</span>
                    <span className="font-bold text-lg">{totalVisits}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Today's Views</span>
                    <span className="font-bold text-lg">{todayVisits}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Unique Pages</span>
                    <span className="font-bold text-lg">{pageVisits.length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Most Visited</span>
                    <span className="font-mono text-xs truncate max-w-[150px]">
                      {pageVisits[0]?.page_path || "—"}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Visit Trends Chart */}
            {dailyTrend.length > 1 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-primary" />
                    Visit Trends
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={{ visits: { label: "Visits", color: "hsl(var(--primary))" } }} className="h-[300px] w-full">
                    <LineChart data={dailyTrend} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-border/50" />
                      <XAxis dataKey="date" tick={{ fontSize: 12 }} className="fill-muted-foreground" />
                      <YAxis allowDecimals={false} tick={{ fontSize: 12 }} className="fill-muted-foreground" />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Line type="monotone" dataKey="visits" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                    </LineChart>
                  </ChartContainer>
                </CardContent>
              </Card>
            )}

            {/* Full table */}
            <Card>
              <CardContent className="p-0">
                <div className="overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Page Path</TableHead>
                        <TableHead className="text-center">Visits</TableHead>
                        <TableHead className="text-center">% of Filtered</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pageVisits.map(pv => (
                        <TableRow key={pv.page_path}>
                          <TableCell className="font-mono text-sm">{pv.page_path}</TableCell>
                          <TableCell className="text-center font-medium">{pv.count}</TableCell>
                          <TableCell className="text-center text-sm">
                            {filteredTotal > 0 ? Math.round((pv.count / filteredTotal) * 100) : 0}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Journal Tab */}
          <TabsContent value="journal" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <FileText className="w-5 h-5 text-primary" />
                  Published Articles ({submissions.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Title</TableHead>
                        <TableHead>Authors</TableHead>
                        <TableHead className="text-center">Vol/Issue</TableHead>
                        <TableHead className="text-center">Current</TableHead>
                        <TableHead>Published</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {submissions.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={5} className="text-center text-muted-foreground py-8">
                            No articles yet
                          </TableCell>
                        </TableRow>
                      ) : submissions.map((a: any) => (
                        <TableRow key={a.id}>
                          <TableCell className="font-medium max-w-xs truncate">{a.title || a.manuscript_title}</TableCell>
                          <TableCell className="text-sm">{a.authors || a.author_name}</TableCell>
                          <TableCell className="text-center text-sm">
                            {a.volume && a.issue ? `${a.volume}(${a.issue})` : "—"}
                          </TableCell>
                          <TableCell className="text-center">
                            {a.is_current_issue ? <CheckCircle2 className="w-4 h-4 text-primary mx-auto" /> : "—"}
                          </TableCell>
                          <TableCell className="text-sm">
                            {a.published_at ? new Date(a.published_at).toLocaleDateString() : a.created_at ? new Date(a.created_at).toLocaleDateString() : "—"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-primary" />
                  Module Completion Distribution
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {moduleDistribution.map(m => (
                  <div key={m.module} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">
                        M{m.module}: {m.name}
                      </span>
                      <span className="font-medium">{m.count} ({m.pct}%)</span>
                    </div>
                    <Progress value={m.pct} className="h-2" />
                  </div>
                ))}
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Score Distribution</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {students.filter(s => Object.keys(s.best_scores).length > 0).length === 0 ? (
                    <p className="text-sm text-muted-foreground">No assessment scores yet.</p>
                  ) : students.filter(s => Object.keys(s.best_scores).length > 0).slice(0, 10).map(s => (
                    <div key={s.id} className="flex items-center justify-between text-sm">
                      <span className="truncate max-w-[150px]">{s.full_name}</span>
                      <div className="flex gap-1 flex-wrap">
                        {Object.entries(s.best_scores).map(([mod, score]) => (
                          <Badge key={mod} variant={Number(score) >= 7 ? "default" : "outline"} className="text-[10px]">
                            M{mod}: {score}/10
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Recent Activity</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {students.slice(0, 8).map(s => (
                    <div key={s.id} className="flex items-center justify-between text-sm">
                      <span className="truncate max-w-[150px]">{s.full_name}</span>
                      <span className="text-xs text-muted-foreground">
                        {new Date(s.updated_at).toLocaleDateString()}
                      </span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Codes Tab */}
          <TabsContent value="codes" className="space-y-4">
            <CodeGenerator onGenerated={fetchAll} />
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-primary" />
                  Student Codes ({codes.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Code</TableHead>
                        <TableHead className="text-center">Status</TableHead>
                        <TableHead className="text-center">Admin</TableHead>
                        <TableHead className="text-center">Registered</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {codes.map(c => {
                        const registered = students.find(s => s.code === c.code);
                        return (
                          <TableRow key={c.code}>
                            <TableCell className="font-mono text-sm">{c.code}</TableCell>
                            <TableCell className="text-center">
                              <Badge variant={c.status === "active" ? "default" : "outline"} className="text-xs">
                                {c.status}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-center">
                              {c.is_admin ? <CheckCircle2 className="w-4 h-4 text-primary mx-auto" /> : <span className="text-muted-foreground">—</span>}
                            </TableCell>
                            <TableCell className="text-center">
                              {registered ? (
                                <span className="text-xs">{registered.full_name}</span>
                              ) : (
                                <Badge variant="outline" className="text-xs text-muted-foreground">Unused</Badge>
                              )}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
