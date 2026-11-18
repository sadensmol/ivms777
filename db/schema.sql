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

-- Memories are searchable in chat by their name/description (§10, §11). rowid =
-- groups.id; rebuilt in lockstep with the memories in albums/memory_store.py.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(name, description);

-- One prepared memory feed (design §6, §11.1). `group_id` is the primary key AND a
-- cascading foreign key onto `groups`, so §11's atomic memory swap drops every stale
-- presentation for free — a recycled group id can never inherit an old feed. The
-- WAVs themselves live in `data_dir/audio` and are pruned on the next prepare run.
CREATE TABLE IF NOT EXISTS memory_presentations (
  group_id      INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  language      TEXT NOT NULL DEFAULT 'en',
  signature     TEXT NOT NULL,
  timeline_json TEXT NOT NULL,
  music_key     TEXT,
  narration_key TEXT,
  status        TEXT NOT NULL,
  error         TEXT,
  -- EVERY CHOICE THE BUILD MADE, as JSON (design §11.1): the cast it asked for and the
  -- cast it got, the score's sections and sampling, the seed, the model context, what
  -- the story planned and what was used. Written on success AND on failure. Without it
  -- "which voice read this?" had no answer after the fact, and a bracket the story pass
  -- ignored was indistinguishable from one it obeyed.
  decisions_json TEXT,
  created_at    TEXT NOT NULL,
  PRIMARY KEY (group_id, language)
);

-- What the VIEWER said about their own memory, and WHEN in the film they said it.
-- Paused mid-play, typed a note; the next generation weaves it into the narration at
-- that moment (design §11.3). Cascades with the memory, like a presentation: a note
-- about a memory that no longer exists describes nothing.
CREATE TABLE IF NOT EXISTS memory_notes (
  id         INTEGER PRIMARY KEY,
  group_id   INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  -- Seconds into the film. The user's own sense of WHERE this belongs in the story,
  -- which no label and no model can supply.
  at_seconds REAL NOT NULL,
  -- Any language. It is translated into the film's language when it is used, so the
  -- viewer never has to think about which one the film is in.
  text       TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_notes_by_memory
  ON memory_notes(group_id, at_seconds);

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
  created_at TEXT NOT NULL,
  -- Wall-clock the whole turn took, end to end (§10). Shown in the UI as
  -- "thought for N s" — on the Jetson a turn can include a model swap, so how
  -- long it took is information the user wants, not a detail.
  elapsed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS chat_messages_session ON chat_messages(session_id, id);

-- Global per-owner chat toggles (§10): guardrails (off) restricts chat to the
-- library and refuses off-topic questions; direct_answers (on) keeps the
-- deterministic direct-DB step. Applied across every session, not per-session.
-- Their defaults reproduce today's pipeline exactly.
CREATE TABLE IF NOT EXISTS chat_prefs (
  owner_id       INTEGER PRIMARY KEY,
  guardrails     INTEGER NOT NULL DEFAULT 0,
  direct_answers INTEGER NOT NULL DEFAULT 1
);

-- Owner-level settings the UI can change (§4.1). Today: the four model slots,
-- keys `model_slot.image_embed` / `.text_embed` / `.caption` / `.planner`, each
-- value a catalog key. A MISSING row means "env override, else profile default" —
-- a default is never written out, so an untouched install and a fresh library
-- resolve identically.
CREATE TABLE IF NOT EXISTS app_settings (
  owner_id   INTEGER NOT NULL,
  key        TEXT NOT NULL,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (owner_id, key)
);

-- A SAVED VOICE (design §6, §11.1): the name the user gave a person's voice. User data — never a catalog Speaker.
CREATE TABLE IF NOT EXISTS saved_voices (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,   -- never reused: a delete intent, a stale tab or a narrator id never hits a new voice
  owner_id   INTEGER NOT NULL,
  name       TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(owner_id, name)
);
-- One example clip of a saved voice, in ONE language. The model-independent source of truth.
CREATE TABLE IF NOT EXISTS voice_examples (
  id           INTEGER PRIMARY KEY AUTOINCREMENT, -- never reused, as saved_voices.id
  voice_id     INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  language     TEXT NOT NULL,          -- feed.language.LANGUAGES
  source       TEXT NOT NULL,          -- 'recorded' | 'uploaded'
  transcript   TEXT,                   -- the words speech recognition heard; for a take stored before it, the text it was
                                       -- shown (replaced when the hearing job hears it); NULL = none yet
  storage_key  TEXT NOT NULL,          -- PCM-16 mono 48 kHz WAV in data_dir/voices (content-addressed)
  content_hash TEXT NOT NULL,          -- sha256 of that WAV
  seconds      REAL NOT NULL,
  sample_rate  INTEGER NOT NULL,
  consent_at   TEXT NOT NULL,          -- unused: repeats created_at; no consent statement is asked
  created_at   TEXT NOT NULL,
  transcript_confirmed_at TEXT,        -- unused; kept, because databases already hold it and no column is dropped
  transcript_model TEXT,               -- '<asr key>@<revision>' that heard this take; NULL = not heard yet
  transcript_error TEXT,               -- why speech recognition gave no usable words ('the speech recognition found no
                                       -- words in it', 'it holds more than 1000 characters of words'); NULL otherwise
  kit_id       TEXT,                   -- the recording-kit text the take was shown ('ru-slow-1'), a suggestion; NULL for an
                                       -- upload or a recording with no kit text
  UNIQUE(voice_id, content_hash)
);
-- An existing table gains transcript_confirmed_at, kit_id, transcript_model and transcript_error through _ADDED_COLUMNS
-- (db/connection.py).
-- A take with no heard words (transcript_model NULL) is heard by the hearing job (feed/voice_hearing.py) right after it
-- is saved; one with transcript_error stays, and no film or test speaks from it. Films and tests speak from the newest
-- take of the language with heard words (design §11.1).
CREATE INDEX IF NOT EXISTS voice_examples_by_voice ON voice_examples(voice_id, language);
-- What a cloning model built from one example: an opaque blob, valid only for that model + revision. A cache — rebuilt when it does not match.
-- A film and a test send the stored blobs of their take with it (reference_for); app writes no new rows. Its rows stay
-- until the drain removes them with their take or voice.
CREATE TABLE IF NOT EXISTS voice_conditionings (
  example_id  INTEGER NOT NULL REFERENCES voice_examples(id) ON DELETE CASCADE,
  model_key   TEXT NOT NULL,
  revision    TEXT NOT NULL,
  storage_key TEXT NOT NULL,           -- blob in data_dir/voices
  created_at  TEXT NOT NULL,
  PRIMARY KEY (example_id, model_key, revision)
);
-- The speaker embedding of one take (design §6, §11.1): what POST /voice/embed answered for the take's audio. A cache,
-- valid only for that model, revision and preprocessing version. A deleted take takes its vector with it.
CREATE TABLE IF NOT EXISTS voice_take_embeddings (
  example_id INTEGER NOT NULL REFERENCES voice_examples(id) ON DELETE CASCADE,
  model_key  TEXT NOT NULL,            -- the reply's model ('ecapa-voxceleb')
  revision   TEXT NOT NULL,            -- the pinned Hub commit
  pipeline   INTEGER NOT NULL,         -- audio/speaker_embed.py:PIPELINE, the preprocessing version
  vector     BLOB NOT NULL,            -- the raw embedding: 192 little-endian float32 values
  created_at TEXT NOT NULL,
  PRIMARY KEY (example_id, model_key, revision, pipeline)
);
-- Legacy per-language film-example choices. Films, tests and the page no longer read or write them. Existing rows stay.
-- Deleting their example sets example_id to NULL; deleting their voice removes the rows. This table owns no files.
CREATE TABLE IF NOT EXISTS voice_film_examples (
  voice_id   INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  language   TEXT NOT NULL,                                          -- feed.language.LANGUAGES
  example_id INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL,
  chosen_at  TEXT NOT NULL,
  PRIMARY KEY (voice_id, language)
);
CREATE INDEX IF NOT EXISTS voice_film_examples_by_example ON voice_film_examples(example_id);
-- The last test of one example ("▶ Generated", design §6, §11.1): a WAV in data_dir/voices to compare with the example.
-- Never used by a film. Its file leaves only through the voice outbox drain.
-- Films and tests no longer use it: the one test per voice is voice_tests. Its rows stay until the drain
-- removes them with their take or voice.
CREATE TABLE IF NOT EXISTS voice_previews (
  example_id      INTEGER PRIMARY KEY REFERENCES voice_examples(id) ON DELETE CASCADE,
  text_kind       TEXT NOT NULL CHECK (text_kind IN ('reading', 'sentence', 'custom')),
  text_id         TEXT,               -- the test-sentence id ('EN-2') when text_kind = 'sentence'; else NULL
  text            TEXT NOT NULL,      -- the words the model was asked to say
  model_key       TEXT NOT NULL,      -- the reply's model
  revision        TEXT,               -- the reply's revision
  conditioning    TEXT NOT NULL CHECK (conditioning IN ('built', 'restored')),
  elapsed_seconds REAL NOT NULL,      -- wall time of the /speak call
  seconds         REAL NOT NULL,      -- length of the test audio
  storage_key     TEXT NOT NULL,      -- PCM-16 WAV in data_dir/voices (content-addressed)
  created_at      TEXT NOT NULL
);
-- One bake that `app` started (design §6): the snapshot of the takes it sent, written BEFORE the bake starts and never
-- changed. A reattach after an app restart reads it back; the row goes when the bake has ended and its result is stored
-- or dropped.
CREATE TABLE IF NOT EXISTS voice_bake_runs (
  bake_key            TEXT PRIMARY KEY,      -- a random hex token; the bake's tag carries it
  owner_id            INTEGER NOT NULL,
  voice_id            INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  snapshot_json       TEXT NOT NULL,         -- [{example_id, content_hash, language, seconds, transcript, kit_id, fingerprint}] in send order
  dataset_hash        TEXT NOT NULL,         -- the hash of those takes; equals the models service's dataset_sha256
  cancel_requested_at TEXT,                  -- set by Cancel, a replace or a voice delete before they answer; a run with it
                                             -- set is never stored, also after an app restart
  created_at          TEXT NOT NULL
);
-- The voice model of one saved voice (design §6, §11.1): the LoRA adapter baked from ALL its takes. Films and tests no
-- longer speak through it: its rows stay as data for the bake check (scripts/voice_bake_check.py). One row per voice; a
-- Rebake updated it in place (version + 1). Its file leaves only through the voice outbox drain.
CREATE TABLE IF NOT EXISTS voice_models (
  voice_id      INTEGER PRIMARY KEY REFERENCES saved_voices(id) ON DELETE CASCADE,
  version       INTEGER NOT NULL,      -- 1, then the stored version + 1 at each Rebake
  bake_key      TEXT NOT NULL UNIQUE,  -- the bake it came from: one bake is stored at most once
  model_key     TEXT NOT NULL,         -- the catalog key it was trained for ('voxcpm2')
  revision      TEXT NOT NULL,         -- the pinned Hub commit it was trained on
  lora_json     TEXT NOT NULL,         -- the catalog's LoRA shape (AdapterSpec.lora), sorted JSON
  config_json   TEXT NOT NULL,         -- the training config it ran with
  dataset_hash  TEXT NOT NULL,         -- the hash of the takes it learned from
  takes         INTEGER NOT NULL,
  seconds       REAL NOT NULL,         -- the length of those takes
  storage_key   TEXT NOT NULL,         -- safetensors in data_dir/voices (content-addressed)
  sha256        TEXT NOT NULL,
  bytes         INTEGER NOT NULL,
  metrics_json  TEXT NOT NULL,         -- the models service's bake status: stage seconds, steps, loss, peak memory, as measured
  created_at    TEXT NOT NULL
);
-- What a voice model learned: one row per trained take. No foreign key to voice_examples: the row outlives a deleted
-- take. No page reads it. The bake check does: stored_model takes the model's languages from here, and --longer-bake
-- checks each learned take's fingerprint.
CREATE TABLE IF NOT EXISTS voice_model_takes (
  voice_id    INTEGER NOT NULL REFERENCES voice_models(voice_id) ON DELETE CASCADE,
  example_id  INTEGER NOT NULL,
  language    TEXT NOT NULL,          -- the take's language when the bake started
  seconds     REAL NOT NULL,
  fingerprint TEXT NOT NULL,          -- sha256 of "<content_hash>\t<language>\t<transcript>" when the bake started
  PRIMARY KEY (voice_id, example_id)
);
-- The last test of one saved voice (design §6, §11.1): speech made from the take films of its language speak from
-- (prompt_example_id), to judge by ear. One row per voice; a new test replaces it. Never used by a film. Its file leaves
-- only through the voice outbox drain.
CREATE TABLE IF NOT EXISTS voice_tests (
  voice_id        INTEGER PRIMARY KEY REFERENCES saved_voices(id) ON DELETE CASCADE,
  language        TEXT NOT NULL,
  text_kind       TEXT NOT NULL CHECK (text_kind IN ('sentence', 'take', 'custom')),
  text_id         TEXT,               -- the sentence id ('RU-1') when text_kind = 'sentence'; else NULL
  example_id      INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL,  -- the take whose words were spoken; NULL once it is deleted
  text            TEXT NOT NULL,      -- the words the model was asked to say
  model_version   INTEGER NOT NULL,   -- voice_models.version that spoke; 0 = spoken from a take (prompt_example_id)
  model_sha256    TEXT NOT NULL,      -- '' when spoken from a take
  model_key       TEXT NOT NULL,      -- the reply's model
  revision        TEXT NOT NULL,      -- the reply's revision
  elapsed_seconds REAL NOT NULL,      -- wall time of the /speak call
  seconds         REAL NOT NULL,      -- length of the test audio
  storage_key     TEXT NOT NULL,      -- PCM-16 WAV in data_dir/voices (content-addressed)
  created_at      TEXT NOT NULL,
  likeness_json   TEXT,               -- the speaker-similarity measure, sorted JSON (docs/data-model.md)
  likeness_error  TEXT,               -- why there is no measure; both NULL = never measured, which is not a zero
  prompt_example_id INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL  -- the take that spoke; NULL once it is deleted
);
-- An existing table gains likeness_json, likeness_error and prompt_example_id through _ADDED_COLUMNS (db/connection.py).
-- Who narrates one memory's film in one language, when the user chose a narrator. No row = Automatic.
-- voice_id = a saved voice. preset_speaker = a catalog speaker name (the route checks it with feed.voice.resolve).
-- Both NULL = the chosen saved voice was deleted: the film fails until the user picks again (D33).
CREATE TABLE IF NOT EXISTS memory_narrators (
  group_id       INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  language       TEXT NOT NULL,
  voice_id       INTEGER REFERENCES saved_voices(id) ON DELETE SET NULL,
  preset_speaker TEXT,
  PRIMARY KEY (group_id, language),
  CHECK (voice_id IS NULL OR preset_speaker IS NULL)
);
-- The user's edited narration of one memory's film in one language (design §6, §11.3). No row = the film's own words.
-- Seeded with ALL passages of the ready film on the first change, with every replay input, so a retry never needs that film.
-- A film made from it REBASES it in the transaction of its ready save: the rows take that film's passages, times and script; the
-- words stay. Only "Write a new story", a retelling and the memory's own deletion remove it.
CREATE TABLE IF NOT EXISTS narration_drafts (
  group_id      INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  language      TEXT NOT NULL,                          -- feed.language.LANGUAGES
  draft_token   TEXT NOT NULL,                          -- 32 hex, new at every seeding; a rebase keeps it
  source_token  TEXT NOT NULL,                          -- timeline.script.token of the film whose times the rows carry
  revision      INTEGER NOT NULL CHECK (revision >= 1), -- 1 at seeding, +1 on every change and on every rebase
  script_json   TEXT NOT NULL,                          -- narrator_asked, narrator_used, cards, score_plan, music_prompt
  passages_json TEXT NOT NULL,                          -- [{n, text, start, end, photo, photo_id, voice_asked, voice_used}], telling order, never empty
  updated_at    TEXT NOT NULL,
  PRIMARY KEY (group_id, language)
);
-- Intent to delete a saved voice or ONE of its examples (design §6; the §3.2c outbox pattern). The request only inserts here.
-- No foreign key: the intent outlives its target and ends only after the files are gone (D33).
CREATE TABLE IF NOT EXISTS voice_deletions (
  id           INTEGER PRIMARY KEY,
  owner_id     INTEGER NOT NULL,
  kind         TEXT NOT NULL CHECK (kind IN ('voice', 'example')),
  target_id    INTEGER NOT NULL,       -- saved_voices.id or voice_examples.id
  requested_at TEXT NOT NULL,
  cascaded_at  TEXT,                   -- set in the transaction that deletes the rows; NULL = the rows are still there
  UNIQUE(owner_id, kind, target_id)
);
-- A file in data_dir/voices that may have lost its last row, or may never get one: an example WAV, a conditioning blob,
-- a voice model file or a test WAV. The drain removes it only when no row references it, then deletes every row of that
-- key (the "ack").
CREATE TABLE IF NOT EXISTS voice_file_deletions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  storage_key  TEXT NOT NULL,
  deletion_id  INTEGER REFERENCES voice_deletions(id),  -- NULL = the write-ahead row of a file write (§2.7), or a replaced conditioning blob, voice model file or test WAV
  requested_at TEXT NOT NULL,
  UNIQUE(storage_key, deletion_id)
);
CREATE INDEX IF NOT EXISTS voice_file_deletions_by_deletion ON voice_file_deletions(deletion_id);
-- A face group (design §6): faces the worker judged to be one person. person_id is the saved voice the owner attached it to;
-- NULL = not named yet ("Faces to name"). Deleting the person returns the group to unnamed.
CREATE TABLE IF NOT EXISTS face_groups (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,   -- never reused: a stale tab never acts on a new group
  owner_id   INTEGER NOT NULL,
  person_id  INTEGER REFERENCES saved_voices(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS face_groups_by_owner_person ON face_groups(owner_id, person_id);
-- One face found in a photo by the `faces` stage. The box is in 0..1 of the upright image; vector is the L2-normalised
-- float32 face embedding. pinned = the owner moved it by hand, so clustering never moves it again.
CREATE TABLE IF NOT EXISTS faces (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  group_id   INTEGER REFERENCES face_groups(id) ON DELETE SET NULL,
  box_x      REAL NOT NULL,
  box_y      REAL NOT NULL,
  box_w      REAL NOT NULL,
  box_h      REAL NOT NULL,
  score      REAL NOT NULL,
  vector     BLOB NOT NULL,
  model      TEXT NOT NULL,                       -- '<face model key>@<revision>'
  pinned     INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS faces_photo_id_idx ON faces(photo_id);
CREATE INDEX IF NOT EXISTS faces_group_id_idx ON faces(group_id);
