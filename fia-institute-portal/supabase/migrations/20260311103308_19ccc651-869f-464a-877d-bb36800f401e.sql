
-- Subscriptions table
CREATE TABLE public.subscriptions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  tier text NOT NULL DEFAULT 'free',
  subscription_status text NOT NULL DEFAULT 'active',
  payment_reference text,
  payment_status text DEFAULT 'pending',
  payment_date timestamp with time zone,
  renewal_date timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  UNIQUE(user_id)
);
ALTER TABLE public.subscriptions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can read own subscription" ON public.subscriptions FOR SELECT TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own subscription" ON public.subscriptions FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own subscription" ON public.subscriptions FOR UPDATE TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Admins can manage all subscriptions" ON public.subscriptions FOR ALL TO authenticated USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));

-- Daily drill attempts
CREATE TABLE public.daily_drill_attempts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  drill_type text NOT NULL,
  drill_content jsonb NOT NULL DEFAULT '{}'::jsonb,
  student_response text,
  ai_feedback text,
  score integer,
  completed_at timestamp with time zone,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);
ALTER TABLE public.daily_drill_attempts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can manage own drill attempts" ON public.daily_drill_attempts FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- User streaks
CREATE TABLE public.user_streaks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  current_streak integer NOT NULL DEFAULT 0,
  longest_streak integer NOT NULL DEFAULT 0,
  last_activity_date date,
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  UNIQUE(user_id)
);
ALTER TABLE public.user_streaks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can manage own streaks" ON public.user_streaks FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Badges
CREATE TABLE public.badges (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  key text NOT NULL UNIQUE,
  label text NOT NULL,
  description text,
  icon text DEFAULT 'award',
  created_at timestamp with time zone NOT NULL DEFAULT now()
);
ALTER TABLE public.badges ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read badges" ON public.badges FOR SELECT TO authenticated USING (true);

-- User badges
CREATE TABLE public.user_badges (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  badge_id uuid NOT NULL REFERENCES public.badges(id) ON DELETE CASCADE,
  earned_at timestamp with time zone NOT NULL DEFAULT now(),
  UNIQUE(user_id, badge_id)
);
ALTER TABLE public.user_badges ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can read own badges" ON public.user_badges FOR SELECT TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own badges" ON public.user_badges FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Supervisors can view assigned student badges" ON public.user_badges FOR SELECT TO authenticated USING (EXISTS (SELECT 1 FROM supervisor_assignments sa WHERE sa.student_id = user_badges.user_id AND sa.supervisor_id = auth.uid()));

-- AI usage logs
CREATE TABLE public.ai_usage_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  mode text NOT NULL,
  token_count integer DEFAULT 0,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);
ALTER TABLE public.ai_usage_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can manage own AI usage" ON public.ai_usage_logs FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Admins can read all AI usage" ON public.ai_usage_logs FOR SELECT TO authenticated USING (public.has_role(auth.uid(), 'admin'));

-- Notifications
CREATE TABLE public.notifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  title text NOT NULL,
  message text NOT NULL,
  type text NOT NULL DEFAULT 'info',
  is_read boolean NOT NULL DEFAULT false,
  link text,
  created_at timestamp with time zone NOT NULL DEFAULT now()
);
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can manage own notifications" ON public.notifications FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Institution accounts
CREATE TABLE public.institution_accounts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  contact_email text NOT NULL,
  country text,
  max_students integer DEFAULT 50,
  subscription_tier text DEFAULT 'institutional',
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now()
);
ALTER TABLE public.institution_accounts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Admins can manage institutions" ON public.institution_accounts FOR ALL TO authenticated USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));

-- Institution students
CREATE TABLE public.institution_students (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  institution_id uuid NOT NULL REFERENCES public.institution_accounts(id) ON DELETE CASCADE,
  user_id uuid NOT NULL,
  enrolled_at timestamp with time zone NOT NULL DEFAULT now(),
  UNIQUE(institution_id, user_id)
);
ALTER TABLE public.institution_students ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Admins can manage institution students" ON public.institution_students FOR ALL TO authenticated USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));
CREATE POLICY "Users can read own enrollment" ON public.institution_students FOR SELECT TO authenticated USING (auth.uid() = user_id);

-- Seed default badges
INSERT INTO public.badges (key, label, description, icon) VALUES
  ('first_analysis', 'First Analysis Completed', 'Completed your first statistical interpretation exercise', 'microscope'),
  ('results_pro', 'Results Interpretation Pro', 'Demonstrated strong results interpretation skills', 'target'),
  ('lit_review_thinker', 'Literature Review Thinker', 'Engaged deeply with literature synthesis exercises', 'book-open'),
  ('defense_ready', 'Defense Ready', 'Achieved competent score in defense simulation', 'shield'),
  ('consistency_5', 'Research Consistency', 'Maintained a 5-day research streak', 'flame'),
  ('consistency_10', 'Research Dedication', 'Maintained a 10-day research streak', 'flame'),
  ('methodology_master', 'Methodology Master', 'Demonstrated strong methodology understanding', 'flask-conical'),
  ('first_drill', 'First Daily Drill', 'Completed your first Daily Research Drill', 'zap');
