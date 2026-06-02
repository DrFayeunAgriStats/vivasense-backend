-- Add VivaSense-specific profile columns for platform identification and onboarding.
-- platform_source defaults to 'legacy' so all existing rows require completion.

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS platform_source     text                     DEFAULT 'legacy',
  ADD COLUMN IF NOT EXISTS position            text,
  ADD COLUMN IF NOT EXISTS research_area       text,
  ADD COLUMN IF NOT EXISTS registration_source text,
  ADD COLUMN IF NOT EXISTS terms_accepted_at   timestamp with time zone;

-- Index for the guard query (filters by platform_source on every protected page load)
CREATE INDEX IF NOT EXISTS idx_profiles_platform_source ON public.profiles(platform_source);
