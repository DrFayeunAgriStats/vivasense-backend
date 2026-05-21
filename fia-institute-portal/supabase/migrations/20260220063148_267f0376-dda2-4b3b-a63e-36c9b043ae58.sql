-- Add explicit SELECT deny policy for manuscripts storage bucket
CREATE POLICY "Manuscripts are not publicly readable"
ON storage.objects FOR SELECT
USING (bucket_id = 'manuscripts' AND (select auth.role()) = 'service_role');
