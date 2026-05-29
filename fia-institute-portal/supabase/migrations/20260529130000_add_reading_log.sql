-- Reading Accountability Layer
-- Stores papers students log, AI questions, and student answers.
-- Integrates into rws-ai context and supervisor briefings.

CREATE TABLE IF NOT EXISTS reading_log (
  id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid        NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  title            text        NOT NULL,
  authors          text        NOT NULL,
  journal          text        NOT NULL,
  year             integer     NOT NULL,
  relevance_note   text        NOT NULL,
  ai_questions     jsonb,                  -- ["Q1", "Q2", "Q3"] set when paper is logged
  student_answers  jsonb,                  -- ["A1", "A2", "A3"] set when answers submitted
  answer_completed boolean     NOT NULL DEFAULT false,
  created_at       timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE reading_log ENABLE ROW LEVEL SECURITY;

-- Students can read their own entries
CREATE POLICY "Users can read own reading log"
  ON reading_log FOR SELECT
  USING (auth.uid() = user_id);

-- Students can insert their own entries
CREATE POLICY "Users can insert own reading log"
  ON reading_log FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Students can update their own entries (required for saving answers)
CREATE POLICY "Users can update own reading log"
  ON reading_log FOR UPDATE
  USING (auth.uid() = user_id);
