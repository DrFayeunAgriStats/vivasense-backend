
-- Add signature_url column to certificates table
ALTER TABLE public.certificates ADD COLUMN IF NOT EXISTS signature_url text DEFAULT NULL;

-- Create storage bucket for certificate signatures
INSERT INTO storage.buckets (id, name, public)
VALUES ('certificate-signatures', 'certificate-signatures', true)
ON CONFLICT (id) DO NOTHING;

-- Allow anyone to read signature files (public bucket)
CREATE POLICY "Public read access for certificate signatures"
ON storage.objects FOR SELECT
USING (bucket_id = 'certificate-signatures');

-- Allow authenticated uploads via edge function (service role handles this)
CREATE POLICY "Service role upload for certificate signatures"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'certificate-signatures');
