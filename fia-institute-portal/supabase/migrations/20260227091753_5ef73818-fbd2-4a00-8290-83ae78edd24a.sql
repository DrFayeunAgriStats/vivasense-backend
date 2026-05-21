CREATE POLICY "Anyone can insert codes"
ON public.student_codes FOR INSERT
WITH CHECK (true);