-- Run in the TARGET Supabase project's SQL Editor.
-- Recreates the 4 tables Aria still uses (prefixed with aria_ for
-- easy identification alongside the existing wiki_pages tables).
-- Run this BEFORE importing any CSVs.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS aria_users (
  id BIGINT PRIMARY KEY,
  username TEXT,
  first_name TEXT,
  timezone TEXT DEFAULT 'Asia/Singapore',
  preferences JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS aria_conversations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id BIGINT REFERENCES aria_users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  token_estimate INT DEFAULT 0,
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_aria_conv_user_time ON aria_conversations(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS aria_memories (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id BIGINT REFERENCES aria_users(id) ON DELETE CASCADE,
  category TEXT NOT NULL CHECK (category IN (
    'personal', 'preference', 'goal', 'task', 'relationship',
    'habit', 'work', 'health', 'interest', 'other'
  )),
  content TEXT NOT NULL,
  importance INT DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
  source_message_id UUID REFERENCES aria_conversations(id) ON DELETE SET NULL,
  is_active BOOLEAN DEFAULT TRUE,
  last_referenced_at TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_aria_memories_active ON aria_memories(user_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_aria_memories_category ON aria_memories(user_id, category) WHERE is_active = TRUE;

CREATE TABLE IF NOT EXISTS aria_conversation_summaries (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id BIGINT REFERENCES aria_users(id) ON DELETE CASCADE,
  summary TEXT NOT NULL,
  period_start TIMESTAMPTZ NOT NULL,
  period_end TIMESTAMPTZ NOT NULL,
  message_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_aria_summaries_user ON aria_conversation_summaries(user_id, period_end DESC);

CREATE OR REPLACE FUNCTION aria_update_updated_at() RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_aria_users_updated ON aria_users;
CREATE TRIGGER tr_aria_users_updated BEFORE UPDATE ON aria_users FOR EACH ROW EXECUTE FUNCTION aria_update_updated_at();

DROP TRIGGER IF EXISTS tr_aria_memories_updated ON aria_memories;
CREATE TRIGGER tr_aria_memories_updated BEFORE UPDATE ON aria_memories FOR EACH ROW EXECUTE FUNCTION aria_update_updated_at();
