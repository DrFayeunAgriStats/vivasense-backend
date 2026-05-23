import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";

export type BgmStudent = {
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
  is_admin: boolean;
};

const SESSION_KEY = "bgm_session";

export function useBgmSession() {
  const [student, setStudent] = useState<BgmStudent | null>(null);
  const [loading, setLoading] = useState(true);

  // Restore session on mount
  useEffect(() => {
    const saved = localStorage.getItem(SESSION_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setStudent(parsed);
      } catch { /* ignore */ }
    }
    setLoading(false);
  }, []);

  const login = useCallback(async (code: string, fullName: string, registrationId: string) => {
    // 1. Validate code
    const { data: codeRow, error: codeErr } = await supabase
      .from("student_codes")
      .select("code, status, is_admin")
      .eq("code", code)
      .single();

    if (codeErr || !codeRow) throw new Error("Invalid student code.");
    if (codeRow.status !== "active") throw new Error("This code has been deactivated.");

    // 2. Check if student already exists
    const { data: existing } = await supabase
      .from("bgm_students")
      .select("*")
      .eq("code", code)
      .single();

    let studentData: any;

    if (existing) {
      // Update name/reg if changed
      if (existing.full_name !== fullName || existing.registration_id !== registrationId) {
        await supabase
          .from("bgm_students")
          .update({ full_name: fullName, registration_id: registrationId, updated_at: new Date().toISOString() })
          .eq("code", code);
      }
      studentData = { ...existing, full_name: fullName, registration_id: registrationId };
    } else {
      // Create new student
      const { data: newStudent, error: insertErr } = await supabase
        .from("bgm_students")
        .insert({ code, full_name: fullName, registration_id: registrationId })
        .select()
        .single();

      if (insertErr || !newStudent) throw new Error("Failed to create student record.");
      studentData = newStudent;
    }

    const session: BgmStudent = {
      id: studentData.id,
      code: studentData.code,
      full_name: studentData.full_name,
      registration_id: studentData.registration_id,
      current_module: studentData.current_module,
      completed_modules: studentData.completed_modules || [],
      best_scores: (studentData.best_scores as Record<string, number>) || {},
      r_practice_status: (studentData.r_practice_status as Record<string, string>) || {},
      progress_percent: studentData.progress_percent,
      token_status: studentData.token_status,
      completion_token: studentData.completion_token,
      is_admin: codeRow.is_admin,
    };

    localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    setStudent(session);
    return session;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(SESSION_KEY);
    setStudent(null);
  }, []);

  const refreshStudent = useCallback(async () => {
    if (!student) return;
    const { data } = await supabase
      .from("bgm_students")
      .select("*")
      .eq("code", student.code)
      .single();

    if (data) {
      const { data: codeRow } = await supabase
        .from("student_codes")
        .select("is_admin")
        .eq("code", student.code)
        .single();

      const updated: BgmStudent = {
        id: data.id,
        code: data.code,
        full_name: data.full_name,
        registration_id: data.registration_id,
        current_module: data.current_module,
        completed_modules: data.completed_modules || [],
        best_scores: (data.best_scores as Record<string, number>) || {},
        r_practice_status: (data.r_practice_status as Record<string, string>) || {},
        progress_percent: data.progress_percent,
        token_status: data.token_status,
        completion_token: data.completion_token,
        is_admin: codeRow?.is_admin ?? false,
      };
      localStorage.setItem(SESSION_KEY, JSON.stringify(updated));
      setStudent(updated);
    }
  }, [student]);

  return { student, loading, login, logout, refreshStudent };
}
