-- Run in Supabase SQL Editor.
-- Drops tables with no live code path (scheduler.py is a no-op stub;
-- these had writer/reader functions defined but never called).
-- Safe to run: nothing in the kept tables (users, conversations,
-- memories, conversation_summaries) references these via FK.

DROP TABLE IF EXISTS scheduled_messages;
DROP TABLE IF EXISTS proactive_message_log;
DROP TABLE IF EXISTS reminders;
