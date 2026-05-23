import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import type { ChatMessage } from "@/lib/adapStreamChat";

export interface AdapStudent {
  studentId: string;
  fullName: string;
  cohort: string;
  currentWeek: number;
  completedWeeks: number[];
  lastQuizScore: string | null;
  certificateCode: string | null;
  chatHistory: ChatMessage[];
}

const MAX_CHAT_HISTORY = 60; // 30 turns = 60 messages

export function useAdapSession() {
  const [student, setStudent] = useState<AdapStudent | null>(null);
  const [loading, setLoading] = useState(false);

  const login = useCallback(async (studentId: string, fullName: string, cohort: string) => {
    setLoading(true);
    try {
      // Try to find existing student
      const { data: existing } = await supabase
        .from("adap_students")
        .select("*")
        .eq("student_id", studentId)
        .maybeSingle();

      if (existing) {
        const s: AdapStudent = {
          studentId: existing.student_id,
          fullName: existing.full_name,
          cohort: existing.cohort,
          currentWeek: existing.current_week,
          completedWeeks: existing.completed_weeks || [],
          lastQuizScore: existing.last_quiz_score,
          certificateCode: existing.certificate_code,
          chatHistory: (existing.chat_history as any) || [],
        };
        setStudent(s);
        // Update last_active and name/cohort if changed
        await supabase
          .from("adap_students")
          .update({ full_name: fullName, cohort, last_active: new Date().toISOString() })
          .eq("student_id", studentId);
        return s;
      }

      // Create new student
      const { error } = await supabase.from("adap_students").insert({
        student_id: studentId,
        full_name: fullName,
        cohort,
        current_week: 0,
        completed_weeks: [],
        chat_history: [],
      });

      if (error) throw error;

      const s: AdapStudent = {
        studentId,
        fullName,
        cohort,
        currentWeek: 0,
        completedWeeks: [],
        lastQuizScore: null,
        certificateCode: null,
        chatHistory: [],
      };
      setStudent(s);
      return s;
    } finally {
      setLoading(false);
    }
  }, []);

  const saveProgress = useCallback(async (updates: Partial<AdapStudent>) => {
    if (!student) return;
    const merged = { ...student, ...updates };
    setStudent(merged);

    const trimmedHistory = merged.chatHistory.slice(-MAX_CHAT_HISTORY);

    await supabase
      .from("adap_students")
      .update({
        current_week: merged.currentWeek,
        completed_weeks: merged.completedWeeks,
        last_quiz_score: merged.lastQuizScore,
        certificate_code: merged.certificateCode,
        chat_history: trimmedHistory as any,
        last_active: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      .eq("student_id", student.studentId);
  }, [student]);

  const completeWeek = useCallback(async (weekNum: number) => {
    if (!student) return;
    const completed = [...new Set([...student.completedWeeks, weekNum])].sort();
    const nextWeek = Math.min(weekNum + 1, 6);
    let certCode: string | null = student.certificateCode;

    // Check if all 7 weeks complete
    if (completed.length === 7) {
      const randomCode = Math.random().toString(36).substring(2, 8).toUpperCase();
      const sanitizedId = student.studentId.replace(/[^a-zA-Z0-9]/g, "");
      certCode = `FIA-ADAP-${sanitizedId}-${randomCode}`;
    }

    await saveProgress({
      completedWeeks: completed,
      currentWeek: completed.length === 7 ? 6 : nextWeek,
      certificateCode: certCode,
    });
  }, [student, saveProgress]);

  const logout = useCallback(() => {
    setStudent(null);
  }, []);

  return { student, loading, login, saveProgress, completeWeek, logout };
}
