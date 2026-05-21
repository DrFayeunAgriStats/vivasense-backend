
-- Supervisor assignments
CREATE TABLE public.supervisor_assignments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  supervisor_id uuid NOT NULL,
  student_id uuid NOT NULL,
  assigned_at timestamptz NOT NULL DEFAULT now(),
  assignment_status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(supervisor_id, student_id)
);
ALTER TABLE public.supervisor_assignments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Supervisors can view own assignments" ON public.supervisor_assignments
  FOR SELECT TO authenticated
  USING (auth.uid() = supervisor_id OR auth.uid() = student_id);

CREATE POLICY "Admins can manage assignments" ON public.supervisor_assignments
  FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin'))
  WITH CHECK (public.has_role(auth.uid(), 'admin'));

-- Booking requests
CREATE TABLE public.booking_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id uuid NOT NULL,
  supervisor_id uuid NOT NULL,
  milestone_type text NOT NULL,
  student_note text,
  draft_reference text,
  requested_date date NOT NULL,
  approved_datetime timestamptz,
  booking_status text NOT NULL DEFAULT 'pending',
  supervisor_note text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.booking_requests ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Students can manage own bookings" ON public.booking_requests
  FOR ALL TO authenticated
  USING (auth.uid() = student_id)
  WITH CHECK (auth.uid() = student_id);

CREATE POLICY "Supervisors can view and update assigned bookings" ON public.booking_requests
  FOR ALL TO authenticated
  USING (auth.uid() = supervisor_id)
  WITH CHECK (auth.uid() = supervisor_id);

-- Defense simulator attempts
CREATE TABLE public.defense_simulator_attempts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  track text NOT NULL,
  simulation_type text NOT NULL DEFAULT 'full',
  scores jsonb NOT NULL DEFAULT '{}',
  total_score integer,
  ai_feedback_summary text,
  completed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.defense_simulator_attempts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own defense attempts" ON public.defense_simulator_attempts
  FOR ALL TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Defense simulator messages
CREATE TABLE public.defense_simulator_messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  attempt_id uuid NOT NULL REFERENCES public.defense_simulator_attempts(id) ON DELETE CASCADE,
  role text NOT NULL,
  content text NOT NULL,
  examiner_role text,
  created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.defense_simulator_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own defense messages" ON public.defense_simulator_messages
  FOR ALL TO authenticated
  USING (EXISTS (SELECT 1 FROM public.defense_simulator_attempts a WHERE a.id = defense_simulator_messages.attempt_id AND a.user_id = auth.uid()))
  WITH CHECK (EXISTS (SELECT 1 FROM public.defense_simulator_attempts a WHERE a.id = defense_simulator_messages.attempt_id AND a.user_id = auth.uid()));

-- Certificate eligibility
CREATE TABLE public.certificate_eligibility (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL UNIQUE,
  modules_completed boolean NOT NULL DEFAULT false,
  milestones_completed boolean NOT NULL DEFAULT false,
  defense_completed boolean NOT NULL DEFAULT false,
  defense_score integer,
  integrity_accepted boolean NOT NULL DEFAULT false,
  supervisor_flags_resolved boolean NOT NULL DEFAULT true,
  certificate_status text NOT NULL DEFAULT 'not_eligible',
  reviewed_by uuid,
  reviewed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.certificate_eligibility ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own eligibility" ON public.certificate_eligibility
  FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own eligibility" ON public.certificate_eligibility
  FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own eligibility" ON public.certificate_eligibility
  FOR UPDATE TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Supervisors can view assigned student eligibility" ON public.certificate_eligibility
  FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM public.supervisor_assignments sa WHERE sa.student_id = certificate_eligibility.user_id AND sa.supervisor_id = auth.uid()));

CREATE POLICY "Admins can manage all eligibility" ON public.certificate_eligibility
  FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin'))
  WITH CHECK (public.has_role(auth.uid(), 'admin'));

-- Competency scores
CREATE TABLE public.competency_scores (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  category text NOT NULL,
  score integer NOT NULL DEFAULT 0,
  max_score integer NOT NULL DEFAULT 100,
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(user_id, category)
);
ALTER TABLE public.competency_scores ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own competency scores" ON public.competency_scores
  FOR ALL TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Supervisors can view assigned student scores" ON public.competency_scores
  FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM public.supervisor_assignments sa WHERE sa.student_id = competency_scores.user_id AND sa.supervisor_id = auth.uid()));

-- Supervisor flags
CREATE TABLE public.supervisor_flags (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  supervisor_id uuid NOT NULL,
  student_id uuid NOT NULL,
  flag_type text NOT NULL,
  description text,
  is_resolved boolean NOT NULL DEFAULT false,
  resolved_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.supervisor_flags ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Supervisors can manage own flags" ON public.supervisor_flags
  FOR ALL TO authenticated
  USING (auth.uid() = supervisor_id)
  WITH CHECK (auth.uid() = supervisor_id);

CREATE POLICY "Students can view own flags" ON public.supervisor_flags
  FOR SELECT TO authenticated
  USING (auth.uid() = student_id);

-- Admins can read all profiles for supervisor dashboard
CREATE POLICY "Admins can read all profiles" ON public.profiles
  FOR SELECT TO authenticated
  USING (public.has_role(auth.uid(), 'admin'));

-- Supervisors can read assigned student profiles
CREATE POLICY "Supervisors can read assigned profiles" ON public.profiles
  FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM public.supervisor_assignments sa WHERE sa.student_id = profiles.id AND sa.supervisor_id = auth.uid()));
