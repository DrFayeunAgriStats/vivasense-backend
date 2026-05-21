
CREATE TABLE public.adap_students (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id TEXT NOT NULL UNIQUE,
  full_name TEXT NOT NULL,
  cohort TEXT NOT NULL,
  current_week INTEGER NOT NULL DEFAULT 0,
  completed_weeks INTEGER[] NOT NULL DEFAULT '{}',
  last_quiz_score TEXT,
  certificate_code TEXT,
  chat_history JSONB NOT NULL DEFAULT '[]'::jsonb,
  last_active TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.adap_students ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read/write for adap_students"
  ON public.adap_students
  FOR ALL
  USING (true)
  WITH CHECK (true);
