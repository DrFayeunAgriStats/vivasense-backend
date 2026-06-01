import { supabase } from "@/integrations/supabase/client";

interface LogPayload {
  analysis_type: string;
  design_type?: string;
  trait_count?: number;
  dataset_rows?: number;
  success: boolean;
  error_message?: string;
  duration_ms?: number;
}

export const logAnalysis = async (payload: LogPayload) => {
  try {
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (!session?.user) return;

    await supabase.from("analysis_logs").insert({
      user_id: session.user.id,
      ...payload,
    });
  } catch (error) {
    console.error("Failed to log analysis:", error);
  }
};