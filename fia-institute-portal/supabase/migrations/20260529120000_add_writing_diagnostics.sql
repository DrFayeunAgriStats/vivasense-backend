-- Writing Sample Diagnostic results
-- Replaces the 15-question MCQ approach.
-- profiles.diagnostic_level and all downstream usage are unchanged.

CREATE TABLE IF NOT EXISTS writing_diagnostics (
  id             uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id        uuid        NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  writing_sample text        NOT NULL,
  questions      jsonb       NOT NULL,   -- ["Q1 text", "Q2 text", "Q3 text", "Q4 text", "Q5 text"]
  answers        jsonb       NOT NULL,   -- ["A1 text", "A2 text", "A3 text", "A4 text", "A5 text"]
  assigned_level text        NOT NULL,  -- "FOUNDATION" | "DEVELOPING" | "ADVANCED" (before DB mapping)
  rationale      text,
  strengths      jsonb,                 -- ["str1", "str2", "str3"]
  gaps           jsonb,                 -- ["gap1", "gap2", "gap3"]
  first_priority text,
  created_at     timestamptz DEFAULT now()
);

ALTER TABLE writing_diagnostics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own writing diagnostics"
  ON writing_diagnostics FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own writing diagnostics"
  ON writing_diagnostics FOR INSERT
  WITH CHECK (auth.uid() = user_id);
