
CREATE TABLE public.page_visits (
  id uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  page_path text NOT NULL,
  visited_at timestamp with time zone NOT NULL DEFAULT now(),
  user_agent text,
  referrer text
);

ALTER TABLE public.page_visits ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can insert visits" ON public.page_visits
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Anyone can read visits" ON public.page_visits
  FOR SELECT USING (true);

CREATE INDEX idx_page_visits_page_path ON public.page_visits (page_path);
CREATE INDEX idx_page_visits_visited_at ON public.page_visits (visited_at);
