import { useState } from "react";
import { useBgmSession } from "@/hooks/useBgmSession";
import { BgmLogin } from "@/components/bgm/BgmLogin";
import { BgmDashboard } from "@/components/bgm/BgmDashboard";
import { BgmTutorChat } from "@/components/bgm/BgmTutorChat";
import { BgmWorkbook } from "@/components/bgm/BgmWorkbook";
import { BgmRPracticeLab } from "@/components/bgm/BgmRPracticeLab";
import { BgmAssessment } from "@/components/bgm/BgmAssessment";
import { BgmAdminDashboard } from "@/components/bgm/BgmAdminDashboard";
import { Loader2 } from "lucide-react";

type View = "dashboard" | "chat" | "workbook" | "rpractice" | "assessment" | "admin";

export default function BiometricalGeneticsTutor() {
  const { student, loading, login, logout, refreshStudent } = useBgmSession();
  const [view, setView] = useState<View>("dashboard");
  const [workbookDone, setWorkbookDone] = useState(false);
  const [rPracticeDone, setRPracticeDone] = useState(false);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!student) {
    return <BgmLogin onLogin={async (code, name, regId) => { await login(code, name, regId); }} />;
  }

  if (view === "chat") {
    return <BgmTutorChat student={student} onBack={() => setView("dashboard")} />;
  }

  if (view === "workbook") {
    return (
      <BgmWorkbook
        student={student}
        onBack={() => setView("dashboard")}
        onComplete={() => { setWorkbookDone(true); setView("dashboard"); }}
      />
    );
  }

  if (view === "rpractice") {
    return (
      <BgmRPracticeLab
        student={student}
        onBack={() => setView("dashboard")}
        onApproved={() => { setRPracticeDone(true); }}
      />
    );
  }

  if (view === "assessment") {
    return (
      <BgmAssessment
        student={student}
        onBack={() => setView("dashboard")}
        onModuleComplete={async () => {
          await refreshStudent();
          setWorkbookDone(false);
          setRPracticeDone(false);
          setView("dashboard");
        }}
      />
    );
  }

  if (view === "admin" && student.is_admin) {
    return <BgmAdminDashboard student={student} onBack={() => setView("dashboard")} />;
  }

  return (
    <BgmDashboard
      student={student}
      workbookDone={workbookDone}
      rPracticeDone={rPracticeDone}
      onContinueLearning={() => setView("chat")}
      onWorkbook={() => setView("workbook")}
      onRPractice={() => setView("rpractice")}
      onAssessment={() => setView("assessment")}
      onLogout={logout}
      onAdmin={student.is_admin ? () => setView("admin") : undefined}
    />
  );
}
