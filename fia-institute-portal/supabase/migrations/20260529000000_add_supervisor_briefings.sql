-- Add thesis_title and supervisor_email to student profiles
ALTER TABLE profiles
  ADD COLUMN IF NOT EXISTS thesis_title text,
  ADD COLUMN IF NOT EXISTS supervisor_email text;

-- Supervisor briefings: generated after AI sessions, student-approved before storage
CREATE TABLE IF NOT EXISTS supervisor_briefings (
  id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid        NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  conversation_id  uuid        REFERENCES ai_conversations(id) ON DELETE SET NULL,
  briefing_text    text        NOT NULL,
  mode             text,
  stage            text,
  topic            text,          -- first ~80 chars of first user message
  exchange_count   integer     DEFAULT 0,
  created_at       timestamptz DEFAULT now()
);

ALTER TABLE supervisor_briefings ENABLE ROW LEVEL SECURITY;

-- Students can read their own briefings
CREATE POLICY "Users can read own briefings"
  ON supervisor_briefings
  FOR SELECT
  USING (auth.uid() = user_id);

-- Students can insert their own briefings (save is user-triggered from modal)
CREATE POLICY "Users can insert own briefings"
  ON supervisor_briefings
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Students can delete their own briefings
CREATE POLICY "Users can delete own briefings"
  ON supervisor_briefings
  FOR DELETE
  USING (auth.uid() = user_id);
