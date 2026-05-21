
-- Audit logs table
CREATE TABLE public.audit_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  action_type text NOT NULL,
  action_description text NOT NULL,
  target_user_id uuid,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Admins can read all audit logs"
  ON public.audit_logs FOR SELECT TO authenticated
  USING (has_role(auth.uid(), 'admin'));

CREATE POLICY "Authenticated can insert audit logs"
  ON public.audit_logs FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Supervisors can insert audit logs"
  ON public.audit_logs FOR INSERT TO authenticated
  WITH CHECK (has_role(auth.uid(), 'supervisor'));

-- Supervisor overrides table
CREATE TABLE public.supervisor_overrides (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  supervisor_id uuid NOT NULL,
  student_id uuid NOT NULL,
  milestone_id uuid REFERENCES public.milestones(id) ON DELETE CASCADE,
  override_type text NOT NULL,
  override_reason text,
  supervisor_comment text,
  original_status text,
  new_status text,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.supervisor_overrides ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Supervisors can manage own overrides"
  ON public.supervisor_overrides FOR ALL TO authenticated
  USING (auth.uid() = supervisor_id)
  WITH CHECK (auth.uid() = supervisor_id);

CREATE POLICY "Students can view own overrides"
  ON public.supervisor_overrides FOR SELECT TO authenticated
  USING (auth.uid() = student_id);

CREATE POLICY "Admins can view all overrides"
  ON public.supervisor_overrides FOR SELECT TO authenticated
  USING (has_role(auth.uid(), 'admin'));

-- Portfolio privacy settings
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS portfolio_visibility text NOT NULL DEFAULT 'private';
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS portfolio_share_token text UNIQUE;

-- Add index on audit logs
CREATE INDEX idx_audit_logs_user_id ON public.audit_logs(user_id);
CREATE INDEX idx_audit_logs_action_type ON public.audit_logs(action_type);
CREATE INDEX idx_audit_logs_created_at ON public.audit_logs(created_at DESC);
