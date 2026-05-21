
CREATE TABLE public.certificates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  certificate_id text UNIQUE NOT NULL,
  recipient_name text NOT NULL,
  program_name text NOT NULL DEFAULT 'FIA-ADAP Pilot Training Program',
  training_start_date date,
  training_end_date date,
  issue_date date NOT NULL DEFAULT CURRENT_DATE,
  certificate_type text NOT NULL DEFAULT 'Competence',
  status text NOT NULL DEFAULT 'Active',
  competencies text[] DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.certificates ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can verify certificates"
  ON public.certificates FOR SELECT
  USING (true);
