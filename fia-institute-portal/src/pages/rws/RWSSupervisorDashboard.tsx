import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Users, Calendar, AlertTriangle, CheckCircle, Clock, ArrowLeft, Loader2,
  BarChart3, GraduationCap, Eye, Flag, XCircle, MessageSquare, RotateCcw,
} from "lucide-react";
import { toast } from "@/hooks/use-toast";

const TRACK_LABELS: Record<string, string> = {
  undergraduate_project: "UG Project",
  msc_thesis: "MSc Thesis",
  phd_research: "PhD Research",
  research_paper: "Research Paper",
};

const STAGE_LABELS: Record<string, string> = {
  topic_proposal: "Proposal",
  literature_review: "Lit Review",
  methodology: "Methodology",
  data_analysis: "Data Analysis",
  results_writing: "Results",
  discussion: "Discussion",
  defense_preparation: "Defense",
};

const LEVEL_LABELS: Record<string, string> = {
  beginner: "Beginner",
  developing: "Developing",
  advanced: "Advanced",
};

export default function RWSSupervisorDashboard() {
  const navigate = useNavigate();
  const { user, loading } = useAuth();
  const [students, setStudents] = useState<any[]>([]);
  const [bookings, setBookings] = useState<any[]>([]);
  const [flags, setFlags] = useState<any[]>([]);
  const [selectedStudent, setSelectedStudent] = useState<string | null>(null);
  const [studentDetail, setStudentDetail] = useState<any>(null);
  const [isSupervisor, setIsSupervisor] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [supervisorNotes, setSupervisorNotes] = useState<Record<string, string>>({});
  const [overrideComment, setOverrideComment] = useState("");

  useEffect(() => {
    if (!loading && !user) navigate("/research-writing/signin");
  }, [loading, user]);

  useEffect(() => {
    if (!user) return;
    const fetchData = async () => {
      const { data: roles } = await supabase
        .from("user_roles")
        .select("role")
        .eq("user_id", user.id);
      const hasSupervisorRole = roles?.some((r) => r.role === "supervisor" || r.role === "admin");
      setIsSupervisor(!!hasSupervisorRole);
      if (!hasSupervisorRole) { setFetching(false); return; }

      const { data: assignments } = await supabase
        .from("supervisor_assignments")
        .select("student_id")
        .eq("supervisor_id", user.id)
        .eq("assignment_status", "active");

      const studentIds = assignments?.map((a) => a.student_id) || [];
      if (studentIds.length > 0) {
        const { data: profiles } = await supabase.from("profiles").select("*").in("id", studentIds);
        if (profiles) setStudents(profiles);
      }

      const { data: bks } = await supabase
        .from("booking_requests").select("*").eq("supervisor_id", user.id).order("created_at", { ascending: false });
      if (bks) setBookings(bks);

      const { data: flgs } = await supabase
        .from("supervisor_flags").select("*").eq("supervisor_id", user.id).eq("is_resolved", false);
      if (flgs) setFlags(flgs);

      setFetching(false);
    };
    fetchData();
  }, [user]);

  const handleBookingAction = async (bookingId: string, status: string) => {
    const note = supervisorNotes[bookingId] || null;
    const update: any = { booking_status: status, supervisor_note: note, updated_at: new Date().toISOString() };
    if (status === "approved") update.approved_datetime = new Date().toISOString();

    await supabase.from("booking_requests").update(update).eq("id", bookingId);

    // Log audit
    if (user) {
      const booking = bookings.find((b) => b.id === bookingId);
      await supabase.from("audit_logs").insert({
        user_id: user.id,
        action_type: `booking_${status}`,
        action_description: `Booking ${status} for milestone: ${booking?.milestone_type || "unknown"}`,
        target_user_id: booking?.student_id,
      });
    }

    toast({ title: "Updated", description: `Booking ${status}.` });
    setBookings((prev) => prev.map((b) => b.id === bookingId ? { ...b, ...update } : b));
  };

  const handleMilestoneOverride = async (
    milestoneId: string,
    studentId: string,
    overrideType: "approve" | "request_revision" | "override_feedback",
    milestoneTitle: string
  ) => {
    if (!user) return;
    const comment = overrideComment.trim();

    if (overrideType === "approve") {
      await supabase.from("milestones").update({ is_completed: true, completed_at: new Date().toISOString() }).eq("id", milestoneId);
    }

    await supabase.from("supervisor_overrides").insert({
      supervisor_id: user.id,
      student_id: studentId,
      milestone_id: milestoneId,
      override_type: overrideType,
      override_reason: overrideType === "approve" ? "Supervisor approved" : overrideType === "request_revision" ? "Revision requested" : "AI feedback overridden",
      supervisor_comment: comment || null,
    });

    await supabase.from("audit_logs").insert({
      user_id: user.id,
      action_type: `milestone_${overrideType}`,
      action_description: `${overrideType.replace(/_/g, " ")} milestone: ${milestoneTitle}`,
      target_user_id: studentId,
    });

    // Create notification for student
    await supabase.from("notifications").insert({
      user_id: studentId,
      title: overrideType === "approve" ? "Milestone Approved" : overrideType === "request_revision" ? "Revision Requested" : "Supervisor Feedback",
      message: comment || `Your milestone "${milestoneTitle}" has been ${overrideType.replace(/_/g, " ")}.`,
      type: overrideType === "approve" ? "success" : "info",
      link: "/research-writing/dashboard",
    });

    setOverrideComment("");
    toast({ title: "Action recorded", description: `Milestone ${overrideType.replace(/_/g, " ")} successfully.` });

    // Refresh detail
    if (selectedStudent) viewStudentDetail(selectedStudent);
  };

  const handleAddFlag = async (studentId: string, flagType: string, description: string) => {
    if (!user) return;
    await supabase.from("supervisor_flags").insert({
      supervisor_id: user.id,
      student_id: studentId,
      flag_type: flagType,
      description,
    });

    await supabase.from("audit_logs").insert({
      user_id: user.id,
      action_type: "supervisor_flag",
      action_description: `Flag: ${flagType} - ${description}`,
      target_user_id: studentId,
    });

    toast({ title: "Flag added" });
  };

  const viewStudentDetail = async (studentId: string) => {
    setSelectedStudent(studentId);
    const [profRes, mileRes, defRes, convRes, certRes, overRes] = await Promise.all([
      supabase.from("profiles").select("*").eq("id", studentId).single(),
      supabase.from("milestones").select("*").eq("user_id", studentId).order("created_at"),
      supabase.from("defense_simulator_attempts").select("*").eq("user_id", studentId).not("completed_at", "is", null).order("total_score", { ascending: false }).limit(1),
      supabase.from("ai_conversations").select("id", { count: "exact" }).eq("user_id", studentId),
      supabase.from("certificate_eligibility").select("*").eq("user_id", studentId).single(),
      supabase.from("supervisor_overrides").select("*").eq("student_id", studentId).order("created_at", { ascending: false }).limit(10),
    ]);
    setStudentDetail({
      profile: profRes.data,
      milestones: mileRes.data || [],
      bestDefense: defRes.data?.[0] || null,
      conversationCount: convRes.count || 0,
      cert: certRes.data || null,
      overrides: overRes.data || [],
    });
  };

  if (loading || fetching) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!isSupervisor) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="pt-6 text-center space-y-3">
            <AlertTriangle className="w-8 h-8 mx-auto text-orange-600" />
            <p className="text-sm text-muted-foreground">You do not have supervisor access. Contact your administrator.</p>
            <Link to="/research-writing/dashboard"><Button variant="outline" size="sm">Back to Dashboard</Button></Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const pendingBookings = bookings.filter((b) => b.booking_status === "pending");
  const upcomingBookings = bookings.filter((b) => b.booking_status === "approved" && new Date(b.approved_datetime) > new Date());

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-primary text-primary-foreground py-6">
        <div className="container max-w-5xl flex items-center justify-between">
          <div>
            <p className="text-primary-foreground/70 text-xs uppercase tracking-wider mb-1">FIA Research Writing System</p>
            <h1 className="font-serif text-2xl font-bold flex items-center gap-2">
              <GraduationCap className="w-6 h-6" /> Supervisor Dashboard
            </h1>
          </div>
          <Link to="/research-writing/dashboard">
            <Button variant="secondary" size="sm" className="gap-1"><ArrowLeft className="w-4 h-4" /> Dashboard</Button>
          </Link>
        </div>
      </header>

      <div className="container max-w-5xl py-8 space-y-6">
        {/* Overview Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card><CardContent className="pt-6 text-center">
            <Users className="w-6 h-6 mx-auto text-primary mb-1" />
            <p className="text-2xl font-bold">{students.length}</p>
            <p className="text-xs text-muted-foreground">Students</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6 text-center">
            <Clock className="w-6 h-6 mx-auto text-orange-600 mb-1" />
            <p className="text-2xl font-bold">{pendingBookings.length}</p>
            <p className="text-xs text-muted-foreground">Pending Requests</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6 text-center">
            <Calendar className="w-6 h-6 mx-auto text-green-600 mb-1" />
            <p className="text-2xl font-bold">{upcomingBookings.length}</p>
            <p className="text-xs text-muted-foreground">Upcoming Sessions</p>
          </CardContent></Card>
          <Card><CardContent className="pt-6 text-center">
            <Flag className="w-6 h-6 mx-auto text-destructive mb-1" />
            <p className="text-2xl font-bold">{flags.length}</p>
            <p className="text-xs text-muted-foreground">Active Flags</p>
          </CardContent></Card>
        </div>

        {selectedStudent && studentDetail ? (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{studentDetail.profile?.full_name}</CardTitle>
                <Button variant="ghost" size="sm" onClick={() => setSelectedStudent(null)}>← Back</Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid sm:grid-cols-3 gap-3 text-sm">
                <div><span className="text-muted-foreground">Track:</span> {TRACK_LABELS[studentDetail.profile?.academic_track] || "—"}</div>
                <div><span className="text-muted-foreground">Discipline:</span> {studentDetail.profile?.discipline || "—"}</div>
                <div><span className="text-muted-foreground">Stage:</span> {STAGE_LABELS[studentDetail.profile?.current_research_stage] || "—"}</div>
                <div><span className="text-muted-foreground">Level:</span> {LEVEL_LABELS[studentDetail.profile?.diagnostic_level] || "—"}</div>
                <div><span className="text-muted-foreground">AI Sessions:</span> {studentDetail.conversationCount}</div>
                <div><span className="text-muted-foreground">Defense:</span> {studentDetail.bestDefense ? `${studentDetail.bestDefense.total_score}/100` : "Not attempted"}</div>
              </div>

              {/* Milestones with override actions */}
              <div>
                <h4 className="text-sm font-medium mb-2">Milestones ({studentDetail.milestones.length})</h4>
                {studentDetail.milestones.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No milestones yet.</p>
                ) : (
                  <div className="space-y-2">
                    {studentDetail.milestones.map((m: any) => (
                      <div key={m.id} className="p-3 rounded border border-border space-y-2">
                        <div className="flex items-center gap-2 text-xs">
                          {m.is_completed ? <CheckCircle className="w-3 h-3 text-green-600" /> : <Clock className="w-3 h-3 text-muted-foreground" />}
                          <span className="font-medium">{m.title}</span>
                          <span className="text-muted-foreground ml-auto">{m.stage}</span>
                        </div>
                        {!m.is_completed && (
                          <div className="space-y-2 pt-1">
                            <Textarea
                              placeholder="Supervisor comment (optional)"
                              className="resize-none text-xs min-h-[50px]"
                              value={overrideComment}
                              onChange={(e) => setOverrideComment(e.target.value)}
                            />
                            <div className="flex flex-wrap gap-2">
                              <Button size="sm" variant="default" className="gap-1 text-xs h-7"
                                onClick={() => handleMilestoneOverride(m.id, selectedStudent, "approve", m.title)}>
                                <CheckCircle className="w-3 h-3" /> Approve
                              </Button>
                              <Button size="sm" variant="outline" className="gap-1 text-xs h-7"
                                onClick={() => handleMilestoneOverride(m.id, selectedStudent, "request_revision", m.title)}>
                                <RotateCcw className="w-3 h-3" /> Request Revision
                              </Button>
                              <Button size="sm" variant="secondary" className="gap-1 text-xs h-7"
                                onClick={() => handleMilestoneOverride(m.id, selectedStudent, "override_feedback", m.title)}>
                                <MessageSquare className="w-3 h-3" /> Add Comment
                              </Button>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Recent Overrides */}
              {studentDetail.overrides?.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2">Recent Supervisor Actions</h4>
                  <div className="space-y-1">
                    {studentDetail.overrides.map((o: any) => (
                      <div key={o.id} className="text-xs p-2 rounded border border-border">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-[10px]">{o.override_type.replace(/_/g, " ")}</Badge>
                          <span className="text-muted-foreground">{new Date(o.created_at).toLocaleDateString()}</span>
                        </div>
                        {o.supervisor_comment && <p className="mt-1 text-muted-foreground">{o.supervisor_comment}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {studentDetail.cert && (
                <div>
                  <h4 className="text-sm font-medium mb-1">Certificate Status</h4>
                  <Badge variant={studentDetail.cert.certificate_status === "issued" ? "default" : "secondary"}>
                    {studentDetail.cert.certificate_status.replace(/_/g, " ")}
                  </Badge>
                </div>
              )}
            </CardContent>
          </Card>
        ) : (
          <Tabs defaultValue="students">
            <TabsList>
              <TabsTrigger value="students">Students</TabsTrigger>
              <TabsTrigger value="bookings">Bookings {pendingBookings.length > 0 && <Badge variant="destructive" className="ml-1 text-[10px] px-1">{pendingBookings.length}</Badge>}</TabsTrigger>
            </TabsList>

            <TabsContent value="students" className="space-y-3 mt-4">
              {students.length === 0 ? (
                <Card><CardContent className="pt-6 text-center text-sm text-muted-foreground">No students assigned yet.</CardContent></Card>
              ) : (
                students.map((s) => (
                  <Card key={s.id}>
                    <CardContent className="pt-4 pb-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <p className="text-sm font-medium">{s.full_name}</p>
                          <p className="text-xs text-muted-foreground">
                            {TRACK_LABELS[s.academic_track] || "—"} • {STAGE_LABELS[s.current_research_stage] || "—"} • {LEVEL_LABELS[s.diagnostic_level] || "—"}
                          </p>
                          {s.discipline && <p className="text-xs text-muted-foreground">{s.discipline}</p>}
                        </div>
                        <Button variant="outline" size="sm" className="gap-1" onClick={() => viewStudentDetail(s.id)}>
                          <Eye className="w-3 h-3" /> View
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </TabsContent>

            <TabsContent value="bookings" className="space-y-3 mt-4">
              {bookings.length === 0 ? (
                <Card><CardContent className="pt-6 text-center text-sm text-muted-foreground">No booking requests.</CardContent></Card>
              ) : (
                bookings.map((b) => {
                  const student = students.find((s) => s.id === b.student_id);
                  return (
                    <Card key={b.id}>
                      <CardContent className="pt-4 pb-4 space-y-2">
                        <div className="flex items-start justify-between">
                          <div>
                            <p className="text-sm font-medium">{student?.full_name || "Student"}</p>
                            <p className="text-xs text-muted-foreground">{b.milestone_type.replace(/_/g, " ")} • {new Date(b.requested_date).toLocaleDateString()}</p>
                            {b.student_note && <p className="text-xs text-muted-foreground mt-1">{b.student_note}</p>}
                          </div>
                          <Badge variant={b.booking_status === "pending" ? "secondary" : b.booking_status === "approved" ? "default" : "destructive"}>
                            {b.booking_status}
                          </Badge>
                        </div>
                        {b.booking_status === "pending" && (
                          <div className="space-y-2 pt-2 border-t border-border">
                            <Textarea
                              placeholder="Add a note (optional)"
                              className="resize-none text-xs min-h-[60px]"
                              value={supervisorNotes[b.id] || ""}
                              onChange={(e) => setSupervisorNotes((p) => ({ ...p, [b.id]: e.target.value }))}
                            />
                            <div className="flex gap-2">
                              <Button size="sm" className="gap-1" onClick={() => handleBookingAction(b.id, "approved")}>
                                <CheckCircle className="w-3 h-3" /> Approve
                              </Button>
                              <Button size="sm" variant="outline" onClick={() => handleBookingAction(b.id, "reschedule_requested")}>
                                <Calendar className="w-3 h-3" /> Reschedule
                              </Button>
                              <Button size="sm" variant="destructive" onClick={() => handleBookingAction(b.id, "declined")}>
                                <XCircle className="w-3 h-3" /> Decline
                              </Button>
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  );
                })
              )}
            </TabsContent>
          </Tabs>
        )}
      </div>
    </div>
  );
}
