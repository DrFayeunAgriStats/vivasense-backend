
-- Student codes table (admin creates these)
CREATE TABLE public.student_codes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  code text UNIQUE NOT NULL,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive')),
  is_admin boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.student_codes ENABLE ROW LEVEL SECURITY;

-- Anyone can read active codes (for login validation)
CREATE POLICY "Anyone can check codes" ON public.student_codes
  FOR SELECT USING (true);

-- Students table
CREATE TABLE public.bgm_students (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  code text NOT NULL REFERENCES public.student_codes(code) ON DELETE CASCADE,
  full_name text NOT NULL,
  registration_id text NOT NULL,
  current_module int NOT NULL DEFAULT 1,
  completed_modules int[] NOT NULL DEFAULT '{}',
  best_scores jsonb NOT NULL DEFAULT '{}',
  r_practice_status jsonb NOT NULL DEFAULT '{}',
  progress_percent int NOT NULL DEFAULT 0,
  token_status text NOT NULL DEFAULT 'Locked',
  completion_token text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(code)
);

ALTER TABLE public.bgm_students ENABLE ROW LEVEL SECURITY;

-- Anyone can read/insert/update students (no auth system, code-based access)
CREATE POLICY "Anyone can read students" ON public.bgm_students
  FOR SELECT USING (true);

CREATE POLICY "Anyone can insert students" ON public.bgm_students
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Anyone can update students" ON public.bgm_students
  FOR UPDATE USING (true) WITH CHECK (true);
