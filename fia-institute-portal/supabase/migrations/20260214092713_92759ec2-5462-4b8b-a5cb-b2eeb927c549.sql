
-- Articles table for Journal of Agricultural Innovation
CREATE TABLE public.journal_articles (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  title TEXT NOT NULL,
  authors TEXT NOT NULL,
  abstract TEXT,
  doi_url TEXT,
  volume INTEGER,
  issue INTEGER,
  is_current_issue BOOLEAN DEFAULT false,
  published_at DATE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.journal_articles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Articles are publicly readable"
ON public.journal_articles FOR SELECT USING (true);

-- Submissions table
CREATE TABLE public.journal_submissions (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  author_name TEXT NOT NULL,
  email TEXT NOT NULL,
  manuscript_title TEXT NOT NULL,
  abstract TEXT NOT NULL,
  file_path TEXT,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.journal_submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can submit manuscripts"
ON public.journal_submissions FOR INSERT WITH CHECK (true);

-- Storage bucket for manuscript uploads
INSERT INTO storage.buckets (id, name, public) VALUES ('manuscripts', 'manuscripts', false);

CREATE POLICY "Anyone can upload manuscripts"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'manuscripts');
