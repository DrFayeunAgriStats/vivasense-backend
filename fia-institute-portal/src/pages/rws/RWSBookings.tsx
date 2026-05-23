import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  ArrowLeft, Calendar, CheckCircle, Clock, FileText, Loader2, Plus, Send,
} from "lucide-react";
import { toast } from "@/hooks/use-toast";

const MILESTONE_TYPES = [
  { value: "proposal_draft", label: "Proposal Draft Completed" },
  { value: "literature_review_draft", label: "Literature Review Draft Completed" },
  { value: "methodology_draft", label: "Methodology Draft Completed" },
  { value: "results_draft", label: "Results Draft Completed" },
  { value: "discussion_draft", label: "Discussion Draft Completed" },
  { value: "manuscript_draft", label: "Manuscript Draft Completed" },
  { value: "defense_preparation", label: "Defense Preparation Completed" },
];

const STATUS_BADGE: Record<string, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
  pending: { label: "Pending", variant: "secondary" },
  approved: { label: "Approved", variant: "default" },
  reschedule_requested: { label: "Reschedule Requested", variant: "outline" },
  declined: { label: "Declined", variant: "destructive" },
  completed: { label: "Completed", variant: "default" },
};

export default function RWSBookings() {
  const navigate = useNavigate();
  const { user, profile, loading, session } = useAuth();
  const [bookings, setBookings] = useState<any[]>([]);
  const [supervisorId, setSupervisorId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({ milestone_type: "", student_note: "", requested_date: "", draft_reference: "" });
  const [submitting, setSubmitting] = useState(false);
  const [fetching, setFetching] = useState(true);

  useEffect(() => {
    if (!loading && !user) navigate("/research-writing/signin");
  }, [loading, user]);

  useEffect(() => {
    if (!user) return;
    const fetchData = async () => {
      // Get supervisor assignment
      const { data: assignment } = await supabase
        .from("supervisor_assignments")
        .select("supervisor_id")
        .eq("student_id", user.id)
        .eq("assignment_status", "active")
        .single();
      if (assignment) setSupervisorId(assignment.supervisor_id);

      // Get bookings
      const { data: bks } = await supabase
        .from("booking_requests")
        .select("*")
        .eq("student_id", user.id)
        .order("created_at", { ascending: false });
      if (bks) setBookings(bks);
      setFetching(false);
    };
    fetchData();
  }, [user]);

  const handleSubmit = async () => {
    if (!formData.milestone_type || !formData.requested_date || !supervisorId) return;
    setSubmitting(true);
    const { error } = await supabase.from("booking_requests").insert({
      student_id: user!.id,
      supervisor_id: supervisorId,
      milestone_type: formData.milestone_type,
      student_note: formData.student_note || null,
      draft_reference: formData.draft_reference || null,
      requested_date: formData.requested_date,
    });
    if (error) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } else {
      toast({ title: "Request Sent", description: "Your booking request has been submitted." });
      setShowForm(false);
      setFormData({ milestone_type: "", student_note: "", requested_date: "", draft_reference: "" });
      // Refresh
      const { data: bks } = await supabase.from("booking_requests").select("*").eq("student_id", user!.id).order("created_at", { ascending: false });
      if (bks) setBookings(bks);
    }
    setSubmitting(false);
  };

  if (loading || fetching) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-primary text-primary-foreground py-6">
        <div className="container max-w-4xl flex items-center justify-between">
          <div>
            <p className="text-primary-foreground/70 text-xs uppercase tracking-wider mb-1">FIA Research Writing System</p>
            <h1 className="font-serif text-2xl font-bold flex items-center gap-2">
              <Calendar className="w-6 h-6" /> Supervisor Sessions
            </h1>
          </div>
          <Link to="/research-writing/dashboard">
            <Button variant="secondary" size="sm" className="gap-1"><ArrowLeft className="w-4 h-4" /> Dashboard</Button>
          </Link>
        </div>
      </header>

      <div className="container max-w-4xl py-8 space-y-6">
        {!supervisorId ? (
          <Card>
            <CardContent className="pt-6 text-center space-y-3">
              <p className="text-muted-foreground text-sm">No supervisor has been assigned to you yet.</p>
              <p className="text-xs text-muted-foreground">Your institution admin will assign a supervisor to your account.</p>
            </CardContent>
          </Card>
        ) : (
          <>
            <div className="flex justify-between items-center">
              <h2 className="font-serif text-lg font-bold">Your Booking Requests</h2>
              <Button onClick={() => setShowForm(!showForm)} size="sm" className="gap-1">
                <Plus className="w-4 h-4" /> New Request
              </Button>
            </div>

            {showForm && (
              <Card className="border-l-4 border-l-primary">
                <CardContent className="pt-6 space-y-4">
                  <h3 className="font-medium text-sm">Request Supervision Session</h3>
                  <Select value={formData.milestone_type} onValueChange={(v) => setFormData((p) => ({ ...p, milestone_type: v }))}>
                    <SelectTrigger><SelectValue placeholder="Select milestone..." /></SelectTrigger>
                    <SelectContent>
                      {MILESTONE_TYPES.map((m) => (
                        <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Input
                    type="date"
                    value={formData.requested_date}
                    onChange={(e) => setFormData((p) => ({ ...p, requested_date: e.target.value }))}
                    placeholder="Preferred date"
                  />
                  <Input
                    value={formData.draft_reference}
                    onChange={(e) => setFormData((p) => ({ ...p, draft_reference: e.target.value }))}
                    placeholder="Link to draft document (optional)"
                  />
                  <Textarea
                    value={formData.student_note}
                    onChange={(e) => setFormData((p) => ({ ...p, student_note: e.target.value }))}
                    placeholder="Notes for supervisor (optional)"
                    className="resize-none"
                  />
                  <div className="flex gap-2">
                    <Button onClick={handleSubmit} disabled={submitting || !formData.milestone_type || !formData.requested_date} className="gap-1">
                      {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />} Submit Request
                    </Button>
                    <Button variant="ghost" onClick={() => setShowForm(false)}>Cancel</Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {bookings.length === 0 ? (
              <Card><CardContent className="pt-6 text-center text-sm text-muted-foreground">No booking requests yet.</CardContent></Card>
            ) : (
              <div className="space-y-3">
                {bookings.map((b) => {
                  const status = STATUS_BADGE[b.booking_status] || STATUS_BADGE.pending;
                  const milestone = MILESTONE_TYPES.find((m) => m.value === b.milestone_type);
                  return (
                    <Card key={b.id}>
                      <CardContent className="pt-4 pb-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <FileText className="w-4 h-4 text-primary" />
                              <p className="text-sm font-medium">{milestone?.label || b.milestone_type}</p>
                            </div>
                            <p className="text-xs text-muted-foreground flex items-center gap-1">
                              <Clock className="w-3 h-3" /> Requested: {new Date(b.requested_date).toLocaleDateString()}
                            </p>
                            {b.approved_datetime && (
                              <p className="text-xs text-green-600 flex items-center gap-1">
                                <CheckCircle className="w-3 h-3" /> Approved: {new Date(b.approved_datetime).toLocaleString()}
                              </p>
                            )}
                            {b.student_note && <p className="text-xs text-muted-foreground mt-1">{b.student_note}</p>}
                            {b.supervisor_note && (
                              <p className="text-xs text-primary mt-1 italic">Supervisor: {b.supervisor_note}</p>
                            )}
                          </div>
                          <Badge variant={status.variant}>{status.label}</Badge>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
