-- Run in the TARGET Supabase project's SQL Editor.
-- Renames the already-created tables to use the aria_ prefix.
-- Safe to run even with existing data — RENAME preserves rows,
-- FKs, and indexes' data (indexes/triggers/function renamed separately below).

ALTER TABLE users RENAME TO aria_users;
ALTER TABLE conversations RENAME TO aria_conversations;
ALTER TABLE memories RENAME TO aria_memories;
ALTER TABLE conversation_summaries RENAME TO aria_conversation_summaries;

-- Indexes
ALTER INDEX IF EXISTS idx_conv_user_time RENAME TO idx_aria_conv_user_time;
ALTER INDEX IF EXISTS idx_memories_active RENAME TO idx_aria_memories_active;
ALTER INDEX IF EXISTS idx_memories_category RENAME TO idx_aria_memories_category;
ALTER INDEX IF EXISTS idx_summaries_user RENAME TO idx_aria_summaries_user;

-- Trigger function (renaming is safe — triggers reference it by OID, not name)
ALTER FUNCTION update_updated_at() RENAME TO aria_update_updated_at;

-- Triggers (cosmetic, but keeps names consistent)
ALTER TRIGGER tr_users_updated ON aria_users RENAME TO tr_aria_users_updated;
ALTER TRIGGER tr_memories_updated ON aria_memories RENAME TO tr_aria_memories_updated;
