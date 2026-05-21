CREATE TABLE public.adap_student_codes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  code text NOT NULL UNIQUE,
  is_admin boolean NOT NULL DEFAULT false,
  status text NOT NULL DEFAULT 'active',
  created_at timestamp with time zone NOT NULL DEFAULT now()
);

ALTER TABLE public.adap_student_codes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can check adap codes" ON public.adap_student_codes FOR SELECT USING (true);
CREATE POLICY "Anyone can insert adap codes" ON public.adap_student_codes FOR INSERT WITH CHECK (true);