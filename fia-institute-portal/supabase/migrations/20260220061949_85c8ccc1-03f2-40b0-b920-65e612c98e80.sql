
-- Add column constraints for server-side input validation
ALTER TABLE public.journal_submissions
  ADD CONSTRAINT author_name_length CHECK (char_length(author_name) > 0 AND char_length(author_name) <= 200),
  ADD CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$'),
  ADD CONSTRAINT email_length CHECK (char_length(email) <= 255),
  ADD CONSTRAINT title_length CHECK (char_length(manuscript_title) > 0 AND char_length(manuscript_title) <= 500),
  ADD CONSTRAINT abstract_length CHECK (char_length(abstract) > 0 AND char_length(abstract) <= 3000);

-- Replace overly permissive INSERT policy with validation-aware one
DROP POLICY IF EXISTS "Anyone can submit manuscripts" ON public.journal_submissions;

CREATE POLICY "Anyone can submit valid manuscripts"
ON public.journal_submissions FOR INSERT
WITH CHECK (
  char_length(author_name) > 0 AND char_length(author_name) <= 200 AND
  email ~* '^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$' AND
  char_length(manuscript_title) > 0 AND char_length(manuscript_title) <= 500 AND
  char_length(abstract) > 0 AND char_length(abstract) <= 3000
);

-- Restrict storage: only service_role can delete manuscripts
CREATE POLICY "Only service role can delete manuscripts"
ON storage.objects FOR DELETE
USING (bucket_id = 'manuscripts' AND (select auth.role()) = 'service_role');
