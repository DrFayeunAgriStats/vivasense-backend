export type Certificate = {
  id: string;
  certificate_id: string;
  recipient_name: string;
  program_name: string;
  training_start_date: string | null;
  training_end_date: string | null;
  issue_date: string;
  certificate_type: string;
  status: string;
  competencies: string[] | null;
  appreciation_text: string | null;
  certify_phrase: string;
  completion_phrase: string;
  signature_url: string | null;
  created_at: string;
  updated_at: string;
};
