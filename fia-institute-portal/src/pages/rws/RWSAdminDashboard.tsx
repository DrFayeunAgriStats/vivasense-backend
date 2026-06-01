import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Users, Brain, Shield, Award, BarChart3, TrendingUp, Activity, MessageSquare, ClipboardList, Sparkles,
} from "lucide-react";

interface Stats {
  totalUsers: number;
  trackCounts: Record<string, number>;
  avgDiagnostic: number;
  totalConversations: number;
  modeDistribution: Record<string, number>;
  defenseAvgScore: number;
  defenseAttempts: number;
  certsIssued: number;
  certsPending: number;
  drillsCompleted: number;
  activeStreaks: number;
  proUsers: number;
  freeUsers: number;
  proInterestUsers: number;
  totalVivaSenseAnalyses: number;
  successfulVivaSenseAnalyses: number;
  failedVivaSenseAnalyses: number;
  vivaSenseByType: Record<string, number>;
}

interface VivaSenseAnalysisLog {
  id: string;
  user_id: string;
  analysis_type: string;
  design_type: string | null;
  success: boolean;
  created_at: string;
}

export default function RWSAdminDashboard() {
  const navigate = useNavigate();
  const { user, loading } = useAuth();
  const [isAdmin, setIsAdmin] = useState(false);
  const [checking, setChecking] = useState(true);
  const [stats, setStats] = useState<Stats | null>(null);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);
  const [vivaSenseLogs, setVivaSenseLogs] = useState<VivaSenseAnalysisLog[]>([]);
  const [analysisTypeFilter, setAnalysisTypeFilter] = useState<string>("all");
  const [timeWindow, setTimeWindow] = useState<"all" | "7d" | "30d" | "custom">("30d");
  const [customFrom, setCustomFrom] = useState<string>("");
  const [customTo, setCustomTo] = useState<string>("");

  useEffect(() => {
    if (!loading && !user) navigate("/research-writing/signin");
  }, [loading, user]);

  useEffect(() => {
    if (!user) return;
    const checkAdmin = async () => {
      const { data } = await supabase.from("user_roles").select("role").eq("user_id", user.id).eq("role", "admin").single();
      setIsAdmin(!!data);
      setChecking(false);
    };
    checkAdmin();
  }, [user]);

  useEffect(() => {
    if (!isAdmin) return;
    const fetchStats = async () => {
      const [profilesRes, convsRes, defenseRes, certsRes, drillsRes, streaksRes, logsRes, vivaSenseLogsRes] = await Promise.all([
        supabase.from("profiles").select("academic_track, diagnostic_score, plan, pro_interest"),
        supabase.from("ai_conversations").select("mode"),
        supabase.from("defense_simulator_attempts").select("total_score").not("completed_at", "is", null),
        supabase.from("certificate_eligibility").select("certificate_status"),
        supabase.from("daily_drill_attempts").select("id", { count: "exact" }),
        supabase.from("user_streaks").select("current_streak").gt("current_streak", 0),
        supabase.from("audit_logs").select("*").order("created_at", { ascending: false }).limit(50),
        supabase
          .from("analysis_logs")
          .select("id, user_id, analysis_type, design_type, success, created_at")
          .order("created_at", { ascending: false })
          .limit(100),
      ]);

      const profiles = profilesRes.data || [];
      const trackCounts: Record<string, number> = {};
      let diagSum = 0, diagCount = 0;
      let proUsers = 0;
      let freeUsers = 0;
      let proInterestUsers = 0;
      profiles.forEach((p) => {
        if (p.academic_track) trackCounts[p.academic_track] = (trackCounts[p.academic_track] || 0) + 1;
        if (p.diagnostic_score !== null) { diagSum += p.diagnostic_score; diagCount++; }
        const plan = (p.plan || "free").toLowerCase();
        if (plan === "pro" || plan === "institutional") {
          proUsers += 1;
        } else {
          freeUsers += 1;
        }
        if (p.pro_interest) {
          proInterestUsers += 1;
        }
      });

      const convs = convsRes.data || [];
      const modeDistribution: Record<string, number> = {};
      convs.forEach((c) => { modeDistribution[c.mode] = (modeDistribution[c.mode] || 0) + 1; });

      const analysisLogs = (vivaSenseLogsRes.data || []) as VivaSenseAnalysisLog[];
      const vivaSenseByType: Record<string, number> = {};
      let successfulVivaSenseAnalyses = 0;
      let failedVivaSenseAnalyses = 0;
      analysisLogs.forEach((log) => {
        vivaSenseByType[log.analysis_type] = (vivaSenseByType[log.analysis_type] || 0) + 1;
        if (log.success) {
          successfulVivaSenseAnalyses += 1;
        } else {
          failedVivaSenseAnalyses += 1;
        }
      });

      const defenses = defenseRes.data || [];
      const defenseAvg = defenses.length ? defenses.reduce((s, d) => s + (d.total_score || 0), 0) / defenses.length : 0;

      const certs = certsRes.data || [];

      setStats({
        totalUsers: profiles.length,
        trackCounts,
        avgDiagnostic: diagCount ? Math.round(diagSum / diagCount) : 0,
        totalConversations: convs.length,
        modeDistribution,
        defenseAvgScore: Math.round(defenseAvg),
        defenseAttempts: defenses.length,
        certsIssued: certs.filter((c) => c.certificate_status === "issued").length,
        certsPending: certs.filter((c) => c.certificate_status === "pending_review").length,
        drillsCompleted: drillsRes.count || 0,
        activeStreaks: streaksRes.data?.length || 0,
        proUsers,
        freeUsers,
        proInterestUsers,
        totalVivaSenseAnalyses: analysisLogs.length,
        successfulVivaSenseAnalyses,
        failedVivaSenseAnalyses,
        vivaSenseByType,
      });

      setAuditLogs(logsRes.data || []);
      setVivaSenseLogs(analysisLogs);
    };
    fetchStats();
  }, [isAdmin]);

  if (loading || checking) return <div className="min-h-screen bg-background flex items-center justify-center"><div className="animate-pulse text-muted-foreground">Loading...</div></div>;
  if (!isAdmin) return (
    <Layout>
      <section className="container max-w-3xl py-12 text-center">
        <p className="text-muted-foreground">You do not have admin access.</p>
      </section>
    </Layout>
  );

  const TRACK_LABELS: Record<string, string> = {
    undergraduate_project: "Undergraduate",
    msc_thesis: "MSc",
    phd_research: "PhD",
    research_paper: "Research Paper",
  };

  const uniqueAnalysisTypes = useMemo(
    () => Array.from(new Set(vivaSenseLogs.map((log) => log.analysis_type))).sort(),
    [vivaSenseLogs]
  );

  const filteredVivaSenseLogs = useMemo(() => {
    const now = Date.now();

    return vivaSenseLogs.filter((log) => {
      if (analysisTypeFilter !== "all" && log.analysis_type !== analysisTypeFilter) {
        return false;
      }

      const createdAt = new Date(log.created_at).getTime();
      if (Number.isNaN(createdAt)) {
        return false;
      }

      if (timeWindow === "7d") {
        return createdAt >= now - 7 * 24 * 60 * 60 * 1000;
      }

      if (timeWindow === "30d") {
        return createdAt >= now - 30 * 24 * 60 * 60 * 1000;
      }

      if (timeWindow === "custom") {
        if (!customFrom && !customTo) {
          return true;
        }

        const fromBound = customFrom ? new Date(`${customFrom}T00:00:00`).getTime() : -Infinity;
        const toBound = customTo ? new Date(`${customTo}T23:59:59`).getTime() : Infinity;
        return createdAt >= fromBound && createdAt <= toBound;
      }

      return true;
    });
  }, [vivaSenseLogs, analysisTypeFilter, timeWindow, customFrom, customTo]);

  const filteredVivaSenseByType = useMemo(() => {
    const byType: Record<string, number> = {};
    filteredVivaSenseLogs.forEach((log) => {
      byType[log.analysis_type] = (byType[log.analysis_type] || 0) + 1;
    });
    return byType;
  }, [filteredVivaSenseLogs]);

  const filteredSuccessCount = useMemo(
    () => filteredVivaSenseLogs.filter((log) => log.success).length,
    [filteredVivaSenseLogs]
  );

  const filteredFailureCount = useMemo(
    () => filteredVivaSenseLogs.filter((log) => !log.success).length,
    [filteredVivaSenseLogs]
  );

  return (
    <Layout>
      <section className="bg-primary text-primary-foreground py-8">
        <div className="container max-w-5xl">
          <h1 className="font-serif text-2xl md:text-3xl font-bold">Admin Analytics Dashboard</h1>
          <p className="text-primary-foreground/70 text-sm mt-1">FIA Research Writing System</p>
        </div>
      </section>

      <div className="container max-w-5xl py-8 space-y-6">
        {stats && (
          <Tabs defaultValue="users">
            <TabsList className="w-full grid grid-cols-2 md:grid-cols-7 mb-6">
              <TabsTrigger value="users" className="text-xs gap-1"><Users className="w-3 h-3" />Users</TabsTrigger>
              <TabsTrigger value="learning" className="text-xs gap-1"><Brain className="w-3 h-3" />Learning</TabsTrigger>
              <TabsTrigger value="ai" className="text-xs gap-1"><MessageSquare className="w-3 h-3" />AI Usage</TabsTrigger>
              <TabsTrigger value="defense" className="text-xs gap-1"><Shield className="w-3 h-3" />Defense</TabsTrigger>
              <TabsTrigger value="certs" className="text-xs gap-1"><Award className="w-3 h-3" />Certificates</TabsTrigger>
              <TabsTrigger value="vivasense" className="text-xs gap-1"><Sparkles className="w-3 h-3" />VivaSense</TabsTrigger>
              <TabsTrigger value="audit" className="text-xs gap-1"><ClipboardList className="w-3 h-3" />Audit Log</TabsTrigger>
            </TabsList>

            <TabsContent value="users">
              <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard icon={Users} label="Total Users" value={stats.totalUsers} />
                <StatCard icon={Activity} label="Active Streaks" value={stats.activeStreaks} />
                <StatCard icon={BarChart3} label="Avg Diagnostic" value={`${stats.avgDiagnostic}%`} />
                <StatCard icon={TrendingUp} label="Total Drills" value={stats.drillsCompleted} />
              </div>
              <Card className="mt-4">
                <CardHeader><CardTitle className="text-base">Users by Track</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(stats.trackCounts).map(([track, count]) => (
                      <div key={track} className="flex items-center justify-between p-3 rounded-lg border">
                        <span className="text-sm text-foreground">{TRACK_LABELS[track] || track}</span>
                        <Badge variant="secondary">{count}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="learning">
              <div className="grid sm:grid-cols-3 gap-4">
                <StatCard icon={BarChart3} label="Avg Diagnostic Score" value={`${stats.avgDiagnostic}%`} />
                <StatCard icon={TrendingUp} label="Total Drills Done" value={stats.drillsCompleted} />
                <StatCard icon={Activity} label="Active Streaks" value={stats.activeStreaks} />
              </div>
            </TabsContent>

            <TabsContent value="ai">
              <div className="grid sm:grid-cols-2 gap-4 mb-4">
                <StatCard icon={MessageSquare} label="Total AI Sessions" value={stats.totalConversations} />
              </div>
              <Card>
                <CardHeader><CardTitle className="text-base">AI Mode Distribution</CardTitle></CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {Object.entries(stats.modeDistribution).sort((a, b) => b[1] - a[1]).map(([mode, count]) => (
                      <div key={mode} className="flex items-center justify-between p-2 rounded border">
                        <span className="text-sm text-foreground capitalize">{mode.replace(/_/g, " ")}</span>
                        <Badge variant="outline">{count}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="defense">
              <div className="grid sm:grid-cols-2 gap-4">
                <StatCard icon={Shield} label="Total Attempts" value={stats.defenseAttempts} />
                <StatCard icon={TrendingUp} label="Avg Score" value={`${stats.defenseAvgScore}/100`} />
              </div>
            </TabsContent>

            <TabsContent value="certs">
              <div className="grid sm:grid-cols-2 gap-4">
                <StatCard icon={Award} label="Certificates Issued" value={stats.certsIssued} />
                <StatCard icon={Award} label="Pending Review" value={stats.certsPending} />
              </div>
            </TabsContent>

            <TabsContent value="vivasense" className="space-y-4">
              <Card>
                <CardHeader><CardTitle className="text-base">Filter VivaSense Runs</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                    <div>
                      <label className="text-xs text-muted-foreground uppercase tracking-wider">Analysis Type</label>
                      <select
                        className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                        value={analysisTypeFilter}
                        onChange={(e) => setAnalysisTypeFilter(e.target.value)}
                      >
                        <option value="all">All Types</option>
                        {uniqueAnalysisTypes.map((analysisType) => (
                          <option key={analysisType} value={analysisType}>
                            {analysisType.replace(/_/g, " ")}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="text-xs text-muted-foreground uppercase tracking-wider">Time Window</label>
                      <select
                        className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                        value={timeWindow}
                        onChange={(e) => setTimeWindow(e.target.value as "all" | "7d" | "30d" | "custom")}
                      >
                        <option value="30d">Last 30 days</option>
                        <option value="7d">Last 7 days</option>
                        <option value="all">All loaded logs</option>
                        <option value="custom">Custom range</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-xs text-muted-foreground uppercase tracking-wider">From</label>
                      <input
                        type="date"
                        className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                        value={customFrom}
                        onChange={(e) => setCustomFrom(e.target.value)}
                        disabled={timeWindow !== "custom"}
                      />
                    </div>

                    <div>
                      <label className="text-xs text-muted-foreground uppercase tracking-wider">To</label>
                      <input
                        type="date"
                        className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                        value={customTo}
                        onChange={(e) => setCustomTo(e.target.value)}
                        disabled={timeWindow !== "custom"}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4">
                <StatCard icon={BarChart3} label="Filtered Analyses" value={filteredVivaSenseLogs.length} />
                <StatCard icon={TrendingUp} label="Filtered Successful" value={filteredSuccessCount} />
                <StatCard icon={Activity} label="Filtered Failed" value={filteredFailureCount} />
              </div>

              <div className="grid sm:grid-cols-3 gap-4">
                <StatCard icon={Award} label="Pro Users" value={stats.proUsers} />
                <StatCard icon={Users} label="Free Users" value={stats.freeUsers} />
                <StatCard icon={Sparkles} label="Pro Interested" value={stats.proInterestUsers} />
              </div>

              <Card>
                <CardHeader><CardTitle className="text-base">Analysis Type Mix</CardTitle></CardHeader>
                <CardContent>
                  {Object.keys(filteredVivaSenseByType).length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-4">No VivaSense analysis logs for the selected filters.</p>
                  ) : (
                    <div className="space-y-2">
                      {Object.entries(filteredVivaSenseByType)
                        .sort((a, b) => b[1] - a[1])
                        .map(([analysisType, count]) => (
                          <div key={analysisType} className="flex items-center justify-between p-2 rounded border">
                            <span className="text-sm text-foreground capitalize">{analysisType.replace(/_/g, " ")}</span>
                            <Badge variant="outline">{count}</Badge>
                          </div>
                        ))}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle className="text-base">Recent VivaSense Runs</CardTitle></CardHeader>
                <CardContent>
                  {filteredVivaSenseLogs.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-4">No analysis runs match the selected filters.</p>
                  ) : (
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                      {filteredVivaSenseLogs.map((log) => (
                        <div key={log.id} className="flex items-start gap-3 p-3 rounded border border-border text-xs">
                          <div className="shrink-0 pt-0.5 flex gap-1">
                            <Badge variant="outline" className="text-[10px] capitalize">{log.analysis_type.replace(/_/g, " ")}</Badge>
                            <Badge variant={log.success ? "secondary" : "destructive"} className="text-[10px]">
                              {log.success ? "Success" : "Failed"}
                            </Badge>
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-foreground">
                              User: <span className="font-mono">{log.user_id}</span>
                              {log.design_type ? <span> · Design: {log.design_type}</span> : null}
                            </p>
                            <p className="text-muted-foreground mt-0.5">{new Date(log.created_at).toLocaleString()}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="audit">
              <Card>
                <CardHeader><CardTitle className="text-base">Recent Audit History</CardTitle></CardHeader>
                <CardContent>
                  {auditLogs.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-4">No audit events recorded yet.</p>
                  ) : (
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                      {auditLogs.map((log) => (
                        <div key={log.id} className="flex items-start gap-3 p-3 rounded border border-border text-xs">
                          <div className="shrink-0 pt-0.5">
                            <Badge variant="outline" className="text-[10px]">{log.action_type.replace(/_/g, " ")}</Badge>
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-foreground">{log.action_description}</p>
                            <p className="text-muted-foreground mt-0.5">{new Date(log.created_at).toLocaleString()}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}
      </div>
    </Layout>
  );
}

function StatCard({ icon: Icon, label, value }: { icon: any; label: string; value: string | number }) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center gap-2 mb-2">
          <Icon className="w-4 h-4 text-primary" />
          <p className="text-xs text-muted-foreground uppercase tracking-wider">{label}</p>
        </div>
        <p className="text-2xl font-bold text-foreground">{value}</p>
      </CardContent>
    </Card>
  );
}
