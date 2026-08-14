CREATE TABLE IF NOT EXISTS uploads (
  id            INTEGER PRIMARY KEY,
  owner_id      INTEGER NOT NULL,
  root_label    TEXT NOT NULL,
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  files_offered INTEGER NOT NULL DEFAULT 0,
  files_sent    INTEGER NOT NULL DEFAULT 0,
  files_failed  INTEGER NOT NULL DEFAULT 0
);

-- Outbox for folder deletions (§3.2c): the request records intent and returns;
-- a worker drains this and does the cascade, so deletes survive restarts and
-- never block the UI. A row here means "still deleting"; gone means done.
CREATE TABLE IF NOT EXISTS folder_deletions (
  id           INTEGER PRIMARY KEY,
  owner_id     INTEGER NOT NULL,
  root_label   TEXT NOT NULL,
  requested_at TEXT NOT NULL,
  UNIQUE(owner_id, root_label)
);

CREATE TABLE IF NOT EXISTS photos (
  id              INTEGER PRIMARY KEY,
  owner_id        INTEGER NOT NULL,
  content_hash    TEXT NOT NULL,
  storage_key     TEXT NOT NULL,
  phash           TEXT,
  bytes           INTEGER,
  width           INTEGER,
  height          INTEGER,
  shot_at         TEXT,
  camera          TEXT,
  lens            TEXT,
  gps_lat         REAL,
  gps_lon         REAL,
  thumb_key       TEXT,
  caption         TEXT,
  caption_model   TEXT,
  ai_title        TEXT,               -- AI-written short title (caption stage)
  ai_description  TEXT,               -- AI-written description (caption stage)
  caption_vec     BLOB,               -- text embedding of the caption (§9 similar)
  embedding_model TEXT,
  exif_json       TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  UNIQUE(owner_id, content_hash)
);
CREATE INDEX IF NOT EXISTS photos_owner_shot ON photos(owner_id, shot_at);

CREATE TABLE IF NOT EXISTS photo_sources (
  id        INTEGER PRIMARY KEY,
  photo_id  INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  upload_id INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  rel_path  TEXT NOT NULL,
  filename  TEXT NOT NULL,
  mtime     REAL,
  UNIQUE(photo_id, rel_path)
);
CREATE INDEX IF NOT EXISTS photo_sources_photo ON photo_sources(photo_id);

CREATE TABLE IF NOT EXISTS photo_facets (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  key        TEXT NOT NULL,
  value_text TEXT,
  value_num  REAL,
  PRIMARY KEY (photo_id, key)
);
CREATE INDEX IF NOT EXISTS photo_facets_lookup ON photo_facets(key, value_text);
CREATE INDEX IF NOT EXISTS photo_facets_range ON photo_facets(key, value_num);

CREATE VIRTUAL TABLE IF NOT EXISTS photo_vec USING vec0(embedding float[1152]);

CREATE TABLE IF NOT EXISTS tags (
  id        INTEGER PRIMARY KEY,
  dimension TEXT NOT NULL,
  label     TEXT NOT NULL,
  UNIQUE(dimension, label)
);

CREATE TABLE IF NOT EXISTS photo_tags (
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  tag_id   INTEGER NOT NULL REFERENCES tags(id),
  score    REAL NOT NULL,
  source   TEXT NOT NULL,
  PRIMARY KEY (photo_id, tag_id, source)
);
CREATE INDEX IF NOT EXISTS photo_tags_tag ON photo_tags(tag_id);

CREATE TABLE IF NOT EXISTS jobs (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  stage      TEXT NOT NULL,
  status     TEXT NOT NULL,
  attempts   INTEGER NOT NULL DEFAULT 0,
  error      TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (photo_id, stage)
);
CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(stage, status);

CREATE TABLE IF NOT EXISTS groups (
  id          INTEGER PRIMARY KEY,
  owner_id    INTEGER NOT NULL,
  kind        TEXT NOT NULL,
  name        TEXT NOT NULL,
  description TEXT,                    -- AI-written story for a memory (§6, §11)
  params      TEXT,
  status      TEXT NOT NULL,
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS group_photos (
  group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  rank     REAL,
  PRIMARY KEY (group_id, photo_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS photo_fts USING fts5(caption, tags_text);

CREATE TABLE IF NOT EXISTS chat_sessions (
  id         INTEGER PRIMARY KEY,
  owner_id   INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_sessions_owner ON chat_sessions(owner_id, id);

CREATE TABLE IF NOT EXISTS chat_messages (
  id         INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  question   TEXT NOT NULL,
  answer     TEXT NOT NULL,
  sources    TEXT,                     -- JSON array of cited photo ids
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_messages_session ON chat_messages(session_id, id);
