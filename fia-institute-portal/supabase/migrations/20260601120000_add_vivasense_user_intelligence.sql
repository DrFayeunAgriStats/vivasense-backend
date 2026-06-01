ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS plan text DEFAULT 'free',
  ADD COLUMN IF NOT EXISTS plan_activated_at timestamp with time zone,
  ADD COLUMN IF NOT EXISTS plan_notes text,
  ADD COLUMN IF NOT EXISTS pro_interest boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS pro_interest_date timestamp with time zone;

CREATE TABLE IF NOT EXISTS public.analysis_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  analysis_type text NOT NULL,
  design_type text,
  trait_count integer,
  dataset_rows integer,
  success boolean NOT NULL DEFAULT true,
  error_message text,
  duration_ms integer,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);

ALTER TABLE public.analysis_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own analysis logs"
  ON public.analysis_logs
  FOR ALL
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Admins can read all analysis logs"
  ON public.analysis_logs
  FOR SELECT
  TO authenticated
  USING (public.has_role(auth.uid(), 'admin'));

CREATE INDEX IF NOT EXISTS idx_analysis_logs_user_id ON public.analysis_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_logs_analysis_type ON public.analysis_logs(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_logs_created_at ON public.analysis_logs(created_at DESC);
