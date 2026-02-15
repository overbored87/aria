-- ============================================================
-- Aria Bot: Supabase Schema Migration
-- Run this in the Supabase SQL Editor (supabase.com → SQL Editor)
-- ============================================================

-- 1. Users table
CREATE TABLE IF NOT EXISTS users (
  id BIGINT PRIMARY KEY,                        -- Telegram user ID
  username TEXT,
  first_name TEXT,
  timezone TEXT DEFAULT 'Asia/Singapore',
  preferences JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Conversations table - all messages
CREATE TABLE IF NOT EXISTS conversations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  token_estimate INT DEFAULT 0,
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conv_user_time
  ON conversations(user_id, created_at DESC);

-- 3. Memories table - extracted facts and important info
CREATE TABLE IF NOT EXISTS memories (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
  category TEXT NOT NULL CHECK (category IN (
    'personal', 'preference', 'goal', 'task', 'relationship',
    'habit', 'work', 'health', 'interest', 'other'
  )),
  content TEXT NOT NULL,
  importance INT DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
  source_message_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
  is_active BOOLEAN DEFAULT TRUE,
  last_referenced_at TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_active
  ON memories(user_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_memories_category
  ON memories(user_id, category) WHERE is_active = TRUE;

-- 4. Conversation summaries - efficient long-term context
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
  summary TEXT NOT NULL,
  period_start TIMESTAMPTZ NOT NULL,
  period_end TIMESTAMPTZ NOT NULL,
  message_count INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_user
  ON conversation_summaries(user_id, period_end DESC);

-- 5. Scheduled messages - proactive messaging
CREATE TABLE IF NOT EXISTS scheduled_messages (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK (type IN (
    'morning_checkin', 'evening_checkin', 'task_followup',
    'goal_reminder', 'custom', 'affirmation'
  )),
  content TEXT,                                  -- NULL = generate dynamically
  context JSONB DEFAULT '{}'::jsonb,
  scheduled_for TIMESTAMPTZ NOT NULL,
  sent_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scheduled_pending
  ON scheduled_messages(scheduled_for) WHERE sent_at IS NULL;

-- 6. Proactive message daily log
CREATE TABLE IF NOT EXISTS proactive_message_log (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
  message_type TEXT NOT NULL,
  sent_at TIMESTAMPTZ DEFAULT NOW(),
  date DATE DEFAULT CURRENT_DATE
);

CREATE INDEX IF NOT EXISTS idx_proactive_daily
  ON proactive_message_log(user_id, date);

-- ============================================================
-- Auto-update triggers
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_users_updated ON users;
CREATE TRIGGER tr_users_updated
  BEFORE UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS tr_memories_updated ON memories;
CREATE TRIGGER tr_memories_updated
  BEFORE UPDATE ON memories
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();
