import { BgmStudent } from "@/hooks/useBgmSession";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Dna, BookOpen, CheckCircle2, BarChart3, Lock, Unlock,
  LogOut, MessageSquare, ClipboardList, FlaskConical, GraduationCap, AlertCircle
} from "lucide-react";
import { MODULE_NAMES } from "@/data/bgmModuleContent";
import { QRCodeSVG } from "qrcode.react";

type Props = {
  student: BgmStudent;
  workbookDone: boolean;
  rPracticeDone: boolean;
  onContinueLearning: () => void;
  onWorkbook: () => void;
  onRPractice: () => void;
  onAssessment: () => void;
  onLogout: () => void;
  onAdmin?: () => void;
};

export function BgmDashboard({
  student, workbookDone, rPracticeDone,
  onContinueLearning, onWorkbook, onRPractice, onAssessment, onLogout, onAdmin,
}: Props) {
  const completedCount = student.completed_modules.length;
  const tokenLocked = student.token_status === "Locked";
  const mod = student.current_module;
  const rStatus = student.r_practice_status?.[mod] || (rPracticeDone ? "approved" : "pending");
  const assessmentLocked = !workbookDone || (!rPracticeDone && rStatus !== "approved");

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-primary text-primary-foreground py-6">
        <div className="container-wide flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-primary-foreground/10 flex items-center justify-center">
              <Dna className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-serif text-xl font-bold">Biometrical Genetics Mastery Tutor</h1>
              <p className="text-primary-foreground/70 text-xs">Field-to-Insight Academy</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {student.is_admin && onAdmin && (
              <Button variant="outline" size="sm" onClick={onAdmin}
                className="text-primary-foreground border-primary-foreground/30 hover:bg-primary-foreground/10">
                Admin
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={onLogout}
              className="text-primary-foreground/80 hover:text-primary-foreground hover:bg-primary-foreground/10">
              <LogOut className="w-4 h-4 mr-1" /> Logout
            </Button>
          </div>
        </div>
      </div>

      <div className="container-wide py-6 space-y-6 max-w-4xl mx-auto">
        {/* Student info card */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <GraduationCap className="w-5 h-5 text-primary" />
              Student Dashboard
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            <p><span className="text-muted-foreground">Name:</span> <strong>{student.full_name}</strong></p>
            <p><span className="text-muted-foreground">Registration ID:</span> <strong>{student.registration_id}</strong></p>
            <p><span className="text-muted-foreground">Student Code:</span> <strong>{student.code}</strong></p>
            <p><span className="text-muted-foreground">Modules Completed:</span> <strong>{completedCount}/11</strong></p>
          </CardContent>
        </Card>

        {/* Stats cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <BookOpen className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Current Module</p>
              <p className="font-bold text-lg">{mod}</p>
              <p className="text-[10px] text-muted-foreground truncate">{MODULE_NAMES[mod - 1]}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <CheckCircle2 className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Completed</p>
              <p className="font-bold text-lg">{completedCount > 0 ? student.completed_modules.join(", ") : "—"}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              <BarChart3 className="w-6 h-6 text-primary mx-auto mb-1" />
              <p className="text-xs text-muted-foreground">Progress</p>
              <p className="font-bold text-lg">{student.progress_percent}%</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-3 text-center">
              {tokenLocked ? (
                <>
                  <Lock className="w-6 h-6 text-muted-foreground mx-auto mb-1" />
                  <p className="text-xs text-muted-foreground">Token</p>
                  <p className="font-bold text-sm">Locked 🔒</p>
                </>
              ) : (
                <>
                  <Unlock className="w-6 h-6 text-primary mx-auto mb-1" />
                  <p className="text-xs text-muted-foreground">Token</p>
                  <p className="font-bold text-sm">Generated ✅</p>
                  {student.completion_token && (
                    <>
                      <p className="text-[10px] text-primary font-mono mt-1">{student.completion_token}</p>
                      <div className="mt-2 flex justify-center">
                        <QRCodeSVG
                          value={`${window.location.origin}/verify-certificate?token=${encodeURIComponent(student.completion_token)}`}
                          size={80}
                          level="M"
                        />
                      </div>
                      <p className="text-[9px] text-muted-foreground mt-1">Scan to verify</p>
                    </>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Progress bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Overall Progress</span>
            <span>{student.progress_percent}%</span>
          </div>
          <Progress value={student.progress_percent} className="h-3" />
        </div>

        {/* Module flow checklist */}
        {completedCount < 11 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Module {mod} Checklist</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <CheckCircle2 className={`w-4 h-4 ${true ? "text-primary" : "text-muted-foreground"}`} />
                <span>1. Conceptual Revision (Tutor Chat)</span>
                <Badge variant="outline" className="text-[10px]">Always available</Badge>
              </div>
              <div className="flex items-center gap-2">
                {workbookDone ? (
                  <CheckCircle2 className="w-4 h-4 text-primary" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-muted-foreground" />
                )}
                <span>2. Workbook (min 3 answers)</span>
                {workbookDone && <Badge className="text-[10px] bg-primary">Done</Badge>}
              </div>
              <div className="flex items-center gap-2">
                {rPracticeDone || rStatus === "approved" ? (
                  <CheckCircle2 className="w-4 h-4 text-primary" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-muted-foreground" />
                )}
                <span>3. R Practice Lab</span>
                {(rPracticeDone || rStatus === "approved") && <Badge className="text-[10px] bg-primary">Approved</Badge>}
              </div>
              <div className="flex items-center gap-2">
                {student.completed_modules.includes(mod) ? (
                  <CheckCircle2 className="w-4 h-4 text-primary" />
                ) : (
                  <Lock className="w-4 h-4 text-muted-foreground" />
                )}
                <span>4. Assessment (10 questions, ≥7/10)</span>
                {assessmentLocked && <Badge variant="outline" className="text-[10px] text-destructive">Locked</Badge>}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Action buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Button onClick={onContinueLearning} className="w-full" size="lg">
            <MessageSquare className="w-4 h-4 mr-2" /> Continue Learning
          </Button>
          <Button variant="outline" className="w-full" size="lg" onClick={onWorkbook}
            disabled={completedCount >= 11}>
            <ClipboardList className="w-4 h-4 mr-2" /> Workbook Tasks
            {workbookDone && <CheckCircle2 className="w-4 h-4 ml-1 text-primary" />}
          </Button>
          <Button variant="outline" className="w-full" size="lg" onClick={onRPractice}
            disabled={completedCount >= 11}>
            <FlaskConical className="w-4 h-4 mr-2" /> R Practice Lab
            {(rPracticeDone || rStatus === "approved") && <CheckCircle2 className="w-4 h-4 ml-1 text-primary" />}
          </Button>
          <Button
            variant={assessmentLocked ? "outline" : "default"}
            className="w-full"
            size="lg"
            onClick={onAssessment}
            disabled={assessmentLocked || completedCount >= 11}
          >
            <GraduationCap className="w-4 h-4 mr-2" /> Take Module Assessment
            {assessmentLocked && <Lock className="w-3 h-3 ml-1" />}
          </Button>
        </div>

        {assessmentLocked && completedCount < 11 && (
          <p className="text-xs text-muted-foreground text-center">
            Complete the Workbook and R Practice Lab to unlock the assessment.
          </p>
        )}
      </div>
    </div>
  );
}
