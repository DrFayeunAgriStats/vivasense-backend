import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

const FULL_CERTIFICATE_ID_PATTERN = /-\d{4}-\d{3}$/;

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const adminPassword = Deno.env.get("CERT_ADMIN_PASSWORD");
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

    const contentType = req.headers.get("content-type") || "";
    let action: string;
    let password: string;
    let data: Record<string, unknown> = {};

    if (contentType.includes("multipart/form-data")) {
      const formData = await req.formData();
      action = formData.get("action") as string;
      password = formData.get("password") as string;
      
      // Extract all non-file fields
      for (const [key, value] of formData.entries()) {
        if (key !== "action" && key !== "password" && key !== "signature_file") {
          data[key] = value;
        }
      }
      
      // Handle competencies as JSON string
      if (typeof data.competencies === "string") {
        try {
          data.competencies = JSON.parse(data.competencies as string);
        } catch {
          data.competencies = [];
        }
      }

      // Handle signature file upload
      const signatureFile = formData.get("signature_file");
      if (signatureFile && typeof signatureFile === "object" && "arrayBuffer" in signatureFile) {
        const supabase = createClient(supabaseUrl, serviceRoleKey);
        const file = signatureFile as File;
        const ext = file.name.split(".").pop() || "png";
        const fileName = `sig_${Date.now()}.${ext}`;
        const arrayBuffer = await file.arrayBuffer();
        const uint8 = new Uint8Array(arrayBuffer);
        
        const { error: uploadError } = await supabase.storage
          .from("certificate-signatures")
          .upload(fileName, uint8, {
            contentType: file.type || "image/png",
            upsert: true,
          });
        
        if (uploadError) {
          return jsonResponse({ error: "Signature upload failed: " + uploadError.message }, 500);
        }
        
        const { data: urlData } = supabase.storage
          .from("certificate-signatures")
          .getPublicUrl(fileName);
        
        data.signature_url = urlData.publicUrl;
      }
    } else {
      const body = await req.json();
      action = body.action;
      password = body.password;
      const { action: _a, password: _p, ...rest } = body;
      data = rest;
    }

    if (!password || password !== adminPassword) {
      return jsonResponse({ error: "Unauthorized" }, 401);
    }

    const supabase = createClient(supabaseUrl, serviceRoleKey);

    switch (action) {
      case "list": {
        const { data: certs, error } = await supabase
          .from("certificates")
          .select("*")
          .order("created_at", { ascending: false });
        if (error) return jsonResponse({ error: error.message }, 500);
        return jsonResponse(certs);
      }

      case "create": {
        const year = new Date().getFullYear();
        const customPrefix = (data.certificate_id_prefix as string)?.trim() || "FIA-ADAP-PILOT";

        if (FULL_CERTIFICATE_ID_PATTERN.test(customPrefix)) {
          if (!data.recipient_name || !(data.recipient_name as string).trim()) {
            return jsonResponse({ error: "Recipient name is required" }, 400);
          }
          if (!["Competence", "Appreciation"].includes(data.certificate_type as string)) {
            return jsonResponse({ error: "Invalid certificate type" }, 400);
          }

          const { data: cert, error } = await supabase
            .from("certificates")
            .insert({
              certificate_id: customPrefix,
              recipient_name: (data.recipient_name as string).trim(),
              program_name: (data.program_name as string)?.trim() || "FIA-ADAP Pilot Training Program",
              training_start_date: data.training_start_date || null,
              training_end_date: data.training_end_date || null,
              issue_date: data.issue_date || new Date().toISOString().split("T")[0],
              certificate_type: data.certificate_type || "Competence",
              competencies: data.competencies || [],
              appreciation_text: data.appreciation_text || null,
              certify_phrase: data.certify_phrase || "This is to certify that",
              completion_phrase: data.completion_phrase || "has successfully completed the",
              signature_url: data.signature_url || null,
            })
            .select()
            .single();

          if (error) return jsonResponse({ error: error.message }, 500);
          return jsonResponse(cert);
        }

        const prefix = `${customPrefix}-${year}-`;

        const { data: existing } = await supabase
          .from("certificates")
          .select("certificate_id")
          .like("certificate_id", `${prefix}%`)
          .order("certificate_id", { ascending: false })
          .limit(1);

        let nextNum = 1;
        if (existing && existing.length > 0) {
          const lastId = existing[0].certificate_id;
          const lastNum = parseInt(lastId.split("-").pop() || "0", 10);
          nextNum = lastNum + 1;
        }

        const certificate_id = `${prefix}${String(nextNum).padStart(3, "0")}`;

        if (!data.recipient_name || !(data.recipient_name as string).trim()) {
          return jsonResponse({ error: "Recipient name is required" }, 400);
        }
        if (!["Competence", "Appreciation"].includes(data.certificate_type as string)) {
          return jsonResponse({ error: "Invalid certificate type" }, 400);
        }

        const { data: cert, error } = await supabase
          .from("certificates")
          .insert({
            certificate_id,
            recipient_name: (data.recipient_name as string).trim(),
            program_name: (data.program_name as string)?.trim() || "FIA-ADAP Pilot Training Program",
            training_start_date: data.training_start_date || null,
            training_end_date: data.training_end_date || null,
            issue_date: data.issue_date || new Date().toISOString().split("T")[0],
            certificate_type: data.certificate_type || "Competence",
            competencies: data.competencies || [],
            appreciation_text: data.appreciation_text || null,
            certify_phrase: data.certify_phrase || "This is to certify that",
            completion_phrase: data.completion_phrase || "has successfully completed the",
            signature_url: data.signature_url || null,
          })
          .select()
          .single();

        if (error) return jsonResponse({ error: error.message }, 500);
        return jsonResponse(cert);
      }

      case "update": {
        if (!data.id) return jsonResponse({ error: "Certificate ID required" }, 400);
        const updateFields: Record<string, unknown> = { updated_at: new Date().toISOString() };
        if (data.recipient_name) updateFields.recipient_name = (data.recipient_name as string).trim();
        if (data.program_name) updateFields.program_name = (data.program_name as string).trim();
        if (data.certificate_type) updateFields.certificate_type = data.certificate_type;
        if (data.issue_date) updateFields.issue_date = data.issue_date;
        if (data.training_start_date !== undefined) updateFields.training_start_date = data.training_start_date || null;
        if (data.training_end_date !== undefined) updateFields.training_end_date = data.training_end_date || null;
        if (data.competencies !== undefined) updateFields.competencies = data.competencies;
        if (data.appreciation_text !== undefined) updateFields.appreciation_text = data.appreciation_text || null;
        if (data.certify_phrase) updateFields.certify_phrase = data.certify_phrase;
        if (data.completion_phrase) updateFields.completion_phrase = data.completion_phrase;
        if (data.signature_url) updateFields.signature_url = data.signature_url;

        // Regenerate certificate_id if prefix changed
        const newPrefix = (data.certificate_id_prefix as string)?.trim();
        if (newPrefix) {
          // Get current cert to check if prefix changed
          const { data: currentCert } = await supabase
            .from("certificates")
            .select("certificate_id")
            .eq("id", data.id)
            .single();
          
          if (currentCert) {
            const oldId = currentCert.certificate_id;
            // Extract the sequence number from old ID (last segment after final dash)
            const lastDash = oldId.lastIndexOf("-");
            const seqNum = oldId.substring(lastDash + 1);
            const year = new Date().getFullYear();
            const newCertId = FULL_CERTIFICATE_ID_PATTERN.test(newPrefix) ? newPrefix : `${newPrefix}-${year}-${seqNum}`;
            if (newCertId !== oldId) {
              updateFields.certificate_id = newCertId;
            }
          }
        }

        const { data: updatedCert, error: updateError } = await supabase
          .from("certificates")
          .update(updateFields)
          .eq("id", data.id)
          .select()
          .single();
        if (updateError) return jsonResponse({ error: updateError.message }, 500);
        return jsonResponse(updatedCert);
      }

      case "revoke": {
        if (!data.id) return jsonResponse({ error: "Certificate ID required" }, 400);
        const { data: cert, error } = await supabase
          .from("certificates")
          .update({ status: "Revoked", updated_at: new Date().toISOString() })
          .eq("id", data.id)
          .select()
          .single();
        if (error) return jsonResponse({ error: error.message }, 500);
        return jsonResponse(cert);
      }

      case "reinstate": {
        if (!data.id) return jsonResponse({ error: "Certificate ID required" }, 400);
        const { data: cert, error } = await supabase
          .from("certificates")
          .update({ status: "Active", updated_at: new Date().toISOString() })
          .eq("id", data.id)
          .select()
          .single();
        if (error) return jsonResponse({ error: error.message }, 500);
        return jsonResponse(cert);
      }

      default:
        return jsonResponse({ error: "Unknown action" }, 400);
    }
  } catch (err) {
    console.error("certificate-admin error:", err);
    return jsonResponse({ error: err instanceof Error ? err.message : "Internal server error" }, 500);
  }
});
