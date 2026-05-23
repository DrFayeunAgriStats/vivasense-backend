import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { Layout } from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Users, Brain, Shield, Award, BarChart3, TrendingUp, Activity, MessageSquare, ClipboardList,
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
}

export default function RWSAdminDashboard() {
  const navigate = useNavigate();
  const { user, loading } = useAuth();
  const [isAdmin, setIsAdmin] = useState(false);
  const [checking, setChecking] = useState(true);
  const [stats, setStats] = useState<Stats | null>(null);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);

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
      const [profilesRes, convsRes, defenseRes, certsRes, drillsRes, streaksRes, logsRes] = await Promise.all([
        supabase.from("profiles").select("academic_track, diagnostic_score"),
        supabase.from("ai_conversations").select("mode"),
        supabase.from("defense_simulator_attempts").select("total_score").not("completed_at", "is", null),
        supabase.from("certificate_eligibility").select("certificate_status"),
        supabase.from("daily_drill_attempts").select("id", { count: "exact" }),
        supabase.from("user_streaks").select("current_streak").gt("current_streak", 0),
        supabase.from("audit_logs").select("*").order("created_at", { ascending: false }).limit(50),
      ]);

      const profiles = profilesRes.data || [];
      const trackCounts: Record<string, number> = {};
      let diagSum = 0, diagCount = 0;
      profiles.forEach((p) => {
        if (p.academic_track) trackCounts[p.academic_track] = (trackCounts[p.academic_track] || 0) + 1;
        if (p.diagnostic_score !== null) { diagSum += p.diagnostic_score; diagCount++; }
      });

      const convs = convsRes.data || [];
      const modeDistribution: Record<string, number> = {};
      convs.forEach((c) => { modeDistribution[c.mode] = (modeDistribution[c.mode] || 0) + 1; });

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
      });

      setAuditLogs(logsRes.data || []);
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
            <TabsList className="w-full grid grid-cols-2 md:grid-cols-6 mb-6">
              <TabsTrigger value="users" className="text-xs gap-1"><Users className="w-3 h-3" />Users</TabsTrigger>
              <TabsTrigger value="learning" className="text-xs gap-1"><Brain className="w-3 h-3" />Learning</TabsTrigger>
              <TabsTrigger value="ai" className="text-xs gap-1"><MessageSquare className="w-3 h-3" />AI Usage</TabsTrigger>
              <TabsTrigger value="defense" className="text-xs gap-1"><Shield className="w-3 h-3" />Defense</TabsTrigger>
              <TabsTrigger value="certs" className="text-xs gap-1"><Award className="w-3 h-3" />Certificates</TabsTrigger>
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
