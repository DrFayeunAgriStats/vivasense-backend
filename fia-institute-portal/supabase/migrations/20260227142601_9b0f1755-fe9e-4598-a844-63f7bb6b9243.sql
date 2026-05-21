
DROP POLICY "Anyone can verify certificates" ON public.certificates;
CREATE POLICY "Anyone can verify certificates"
  ON public.certificates
  FOR SELECT
  TO anon, authenticated
  USING (true);
