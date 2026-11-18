# Data model

The exact SQLite schema, indexes, the FTS/vec virtual tables, and the EXIF facet
key set. Design §6 carries the *shape* and *why*; this file is the DDL. Section
references (`§7`, `§9`, …) point into `docs/design.md`.

## Schema

```sql
-- one row per distinct image, keyed by its bytes
photos (
  id              INTEGER PRIMARY KEY,
  owner_id        INTEGER NOT NULL,
  content_hash    TEXT NOT NULL,      -- sha256 of file bytes; the identity
  storage_key     TEXT NOT NULL,      -- where the original lives in Storage
  phash           TEXT,               -- perceptual hash, near-duplicate groups
  bytes           INTEGER,
  width           INTEGER,
  height          INTEGER,
  shot_at         TEXT,               -- EXIF DateTimeOriginal, ISO-8601
  camera          TEXT,
  lens            TEXT,
  gps_lat         REAL,
  gps_lon         REAL,
  thumb_key       TEXT,
  caption         TEXT,
  caption_model   TEXT,
  caption_vec     BLOB,               -- caption text embedding, for §9 similarity
  embedding_model TEXT,               -- the resolved `image_embed` catalog key (§4.1); NULL = no vector
  exif_json       TEXT,               -- full EXIF as captured, for reference
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  UNIQUE(owner_id, content_hash)
);
CREATE INDEX photos_owner_shot ON photos(owner_id, shot_at);

-- every local path these bytes arrived from; >1 row means a duplicate on disk
photo_sources (
  id          INTEGER PRIMARY KEY,
  photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  upload_id   INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  rel_path    TEXT NOT NULL,          -- path relative to the selected root
  filename    TEXT NOT NULL,
  mtime       REAL,
  UNIQUE(photo_id, rel_path)
);
CREATE INDEX photo_sources_photo ON photo_sources(photo_id);

-- EXIF-derived facets: exact, queryable, never model-guessed
photo_facets (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  key        TEXT NOT NULL,           -- camera_make, iso, year, time_of_day, ...
  value_text TEXT,                    -- set for categorical facets
  value_num  REAL,                    -- set for numeric facets, enables ranges
  PRIMARY KEY (photo_id, key)
);
CREATE INDEX photo_facets_lookup ON photo_facets(key, value_text);
CREATE INDEX photo_facets_range  ON photo_facets(key, value_num);

-- sqlite-vec virtual table; rowid joins to photos.id. The width is the SELECTED
-- image_embed model's `dim` (design §4.1) — 1152 for the default SigLIP 2
-- so400m. Switching to a model of a different width DROPS and recreates this
-- table (vectors from two encoders are not comparable) and requeues the embed
-- stage; it is never migrated in place.
CREATE VIRTUAL TABLE photo_vec USING vec0(
  embedding float[1152]
);

tags (
  id        INTEGER PRIMARY KEY,
  dimension TEXT NOT NULL,
  label     TEXT NOT NULL,
  UNIQUE(dimension, label)
);

photo_tags (
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  tag_id   INTEGER NOT NULL REFERENCES tags(id),
  score    REAL NOT NULL,             -- 0..1
  source   TEXT NOT NULL,             -- siglip | exif | pixel | user  (no vlm — caption model writes no tags, §7)
  PRIMARY KEY (photo_id, tag_id, source)
);
CREATE INDEX photo_tags_tag ON photo_tags(tag_id);

jobs (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  stage      TEXT NOT NULL,           -- thumbnail | embed | taxonomy | caption | caption_embed | faces
  status     TEXT NOT NULL,           -- pending | running | done | failed
  attempts   INTEGER NOT NULL DEFAULT 0,
  error      TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (photo_id, stage)
);
CREATE INDEX jobs_pending ON jobs(stage, status);

groups (
  id          INTEGER PRIMARY KEY,
  owner_id    INTEGER NOT NULL,
  kind        TEXT NOT NULL,          -- event | cluster | duplicate | memory
  name        TEXT NOT NULL,          -- AI-written title for a memory
  description TEXT,                   -- AI-written story for a memory
  params      TEXT,                   -- JSON, how it was generated; carries the
                                      -- library signature a memory was built from
  status      TEXT NOT NULL,          -- suggested | accepted | dismissed
  created_at  TEXT NOT NULL
);

group_photos (
  group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  rank     REAL,
  PRIMARY KEY (group_id, photo_id)
);

uploads (
  id            INTEGER PRIMARY KEY,
  owner_id      INTEGER NOT NULL,
  root_label    TEXT NOT NULL,         -- the folder name the user picked
  started_at    TEXT NOT NULL,
  finished_at   TEXT,
  files_offered INTEGER DEFAULT 0,     -- hashes probed
  files_sent    INTEGER DEFAULT 0,     -- bytes actually transferred
  files_failed  INTEGER DEFAULT 0
);

-- Persisted chat transcript (§10). A session groups a conversation; "New
-- session" starts a fresh one. The current session is the owner's latest.
chat_sessions (
  id         INTEGER PRIMARY KEY,
  owner_id   INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

chat_messages (
  id         INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  question   TEXT NOT NULL,
  answer     TEXT NOT NULL,
  sources    TEXT,                     -- JSON array of cited photo ids
  created_at TEXT NOT NULL
);

-- Owner-level settings the UI can change (design §4.1). Today: the four model
-- slots, keys `model_slot.image_embed` / `.text_embed` / `.caption` / `.planner`,
-- each value a catalog key. A MISSING row means "env override, else profile
-- default" — never a written-out default, so an untouched install and a fresh
-- library resolve identically.
app_settings (
  owner_id   INTEGER NOT NULL,
  key        TEXT NOT NULL,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (owner_id, key)
);

-- A SAVED VOICE (design §6, §11.1): the name the user gave a person's voice. User
-- data — never a catalog Speaker. Ids use AUTOINCREMENT and are never reused, so a
-- delete intent, a stale tab or a stored narrator id never hits a new voice.
saved_voices (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  owner_id   INTEGER NOT NULL,
  name       TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(owner_id, name)
);

-- One example clip of a saved voice, in ONE language. The model-independent source
-- of truth.
voice_examples (
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
-- A take with no heard words (transcript_model NULL) is heard by the hearing job after it is saved; one with
-- transcript_error stays, and no film or test speaks from it. Films and tests speak from the newest take of the
-- language with heard words (design §11.1).
CREATE INDEX voice_examples_by_voice ON voice_examples(voice_id, language);

-- What a cloning model built from one example: an opaque blob, valid only for that
-- model + revision. A cache — rebuilt when it does not match.
-- A film and a test send the stored blobs of their take with it (reference_for); app writes no new rows. Its rows stay
-- until the drain removes them with their take or voice.
voice_conditionings (
  example_id  INTEGER NOT NULL REFERENCES voice_examples(id) ON DELETE CASCADE,
  model_key   TEXT NOT NULL,
  revision    TEXT NOT NULL,
  storage_key TEXT NOT NULL,           -- blob in data_dir/voices
  created_at  TEXT NOT NULL,
  PRIMARY KEY (example_id, model_key, revision)
);

-- The speaker embedding of one take (design §6, §11.1): what POST /voice/embed answered for the take's audio. A cache,
-- valid only for that model, revision and preprocessing version. A deleted take takes its vector with it.
voice_take_embeddings (
  example_id INTEGER NOT NULL REFERENCES voice_examples(id) ON DELETE CASCADE,
  model_key  TEXT NOT NULL,            -- the catalog key ('ecapa-voxceleb')
  revision   TEXT NOT NULL,            -- the pinned Hub commit
  pipeline   INTEGER NOT NULL,         -- audio/speaker_embed.py:PIPELINE, the preprocessing version
  vector     BLOB NOT NULL,            -- the raw embedding: 192 little-endian float32 values
  created_at TEXT NOT NULL,
  PRIMARY KEY (example_id, model_key, revision, pipeline)
);

-- Legacy per-language film-example choices. Films, tests and the page no longer read or write them. Existing rows stay.
-- Deleting their example sets example_id to NULL; deleting their voice removes the rows. This table owns no files.
voice_film_examples (
  voice_id   INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  language   TEXT NOT NULL,                                          -- feed.language.LANGUAGES
  example_id INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL,
  chosen_at  TEXT NOT NULL,
  PRIMARY KEY (voice_id, language)
);
CREATE INDEX voice_film_examples_by_example ON voice_film_examples(example_id);

-- The last test of one example ("▶ Generated", design §6, §11.1): a WAV in data_dir/voices to compare with the example.
-- Never used by a film. Its file leaves only through the voice outbox drain.
-- Films and tests no longer use it: the one test per voice is voice_tests. Its rows stay until the drain removes them
-- with their take or voice.
voice_previews (
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
voice_bake_runs (
  bake_key            TEXT PRIMARY KEY,      -- a random hex token; the bake's tag carries it
  owner_id            INTEGER NOT NULL,
  voice_id            INTEGER NOT NULL REFERENCES saved_voices(id) ON DELETE CASCADE,
  snapshot_json       TEXT NOT NULL,         -- [{example_id, content_hash, language, seconds, transcript, kit_id, fingerprint}] in send order;
                                             -- transcript = the words speech recognition heard
  dataset_hash        TEXT NOT NULL,         -- the hash of those takes; equals the models service's dataset_sha256
  cancel_requested_at TEXT,                  -- set by Cancel, a replace or a voice delete before they answer; a run with it
                                             -- set is never stored, also after an app restart
  created_at          TEXT NOT NULL
);

-- The voice model of one saved voice (design §6, §11.1): the LoRA adapter baked from ALL its takes. Films and tests no
-- longer speak through it: only scripts/voice_bake_check.py reads it. One row per voice; a Rebake updated it in place
-- (version + 1). It stays when takes are added or deleted. Its file leaves only through the voice outbox drain.
voice_models (
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
-- A model whose model_key, revision or lora_json no longer matches the catalog's cloning entry is not usable by the
-- bake check (stored_model).

-- What a voice model learned: one row per trained take. No foreign key to voice_examples: the row outlives a deleted
-- take. No page reads it. The bake check does: stored_model takes the model's languages from here, and --longer-bake
-- checks each learned take's fingerprint.
voice_model_takes (
  voice_id    INTEGER NOT NULL REFERENCES voice_models(voice_id) ON DELETE CASCADE,
  example_id  INTEGER NOT NULL,
  language    TEXT NOT NULL,          -- the take's language when the bake started
  seconds     REAL NOT NULL,
  fingerprint TEXT NOT NULL,          -- sha256 of "<content_hash>\t<language>\t<transcript>" when the bake started
  PRIMARY KEY (voice_id, example_id)
);

-- The last test of one saved voice (design §6, §11.1): speech made from the take films of its language speak from, to
-- judge by ear. One row per voice; a new test replaces it. Never used by a film. Its file leaves only through the voice
-- outbox drain.
voice_tests (
  voice_id        INTEGER PRIMARY KEY REFERENCES saved_voices(id) ON DELETE CASCADE,
  language        TEXT NOT NULL,
  text_kind       TEXT NOT NULL CHECK (text_kind IN ('sentence', 'take', 'custom')),
  text_id         TEXT,               -- the sentence id ('RU-1') when text_kind = 'sentence'; else NULL
  example_id      INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL,  -- the take whose words were spoken; NULL once it is deleted
  text            TEXT NOT NULL,      -- the words the model was asked to say
  model_version   INTEGER NOT NULL,   -- voice_models.version that spoke; 0 = spoken from a take (prompt_example_id)
  model_sha256    TEXT NOT NULL,      -- that model's sha256; '' when spoken from a take
  model_key       TEXT NOT NULL,      -- the reply's model
  revision        TEXT NOT NULL,      -- the reply's revision
  elapsed_seconds REAL NOT NULL,      -- wall time of the /speak call
  seconds         REAL NOT NULL,      -- length of the test audio
  storage_key     TEXT NOT NULL,      -- PCM-16 WAV in data_dir/voices (content-addressed)
  created_at      TEXT NOT NULL,
  likeness_json   TEXT,               -- the speaker-similarity measure, sorted JSON: cosine; per take its leave-one-out cosine;
                                      -- own min, median, max (NULL under 2 takes); evaluator key@revision; pipeline and method
                                      -- versions; takes count; the exact example ids and content hashes of the reference takes;
                                      -- skipped ids with reasons; reference hash = sha256 of the sorted content hashes
  likeness_error  TEXT,               -- why there is no measure; a failed measure never fails the test. Both NULL = never measured,
                                      -- which is not a zero
  prompt_example_id INTEGER REFERENCES voice_examples(id) ON DELETE SET NULL  -- the take that spoke the test; NULL once it
                                      -- is deleted, and for a test made through a voice model
);
-- An existing table gains likeness_json, likeness_error and prompt_example_id through _ADDED_COLUMNS (db/connection.py).

-- Who narrates one memory's film in one language, when the user chose a narrator. No row = Automatic.
-- voice_id = a saved voice. preset_speaker = a catalog speaker name (the route checks it with feed.voice.resolve).
-- Both NULL = the chosen saved voice was deleted: the film fails until the user picks again (design §6).
-- An existing table gains preset_speaker through _ADDED_COLUMNS (db/connection.py), without the CHECK;
-- set_narrator refuses a row with both columns on every table.
memory_narrators (
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
narration_drafts (
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

-- Intent to delete a saved voice or ONE of its examples (design §6; the §3.2c outbox
-- pattern). The request only inserts here. No foreign key: the intent outlives its
-- target and ends only after the files are gone.
voice_deletions (
  id           INTEGER PRIMARY KEY,
  owner_id     INTEGER NOT NULL,
  kind         TEXT NOT NULL CHECK (kind IN ('voice', 'example')),
  target_id    INTEGER NOT NULL,       -- saved_voices.id or voice_examples.id
  requested_at TEXT NOT NULL,
  cascaded_at  TEXT,                   -- set in the transaction that deletes the rows; NULL = the rows are still there
  UNIQUE(owner_id, kind, target_id)
);

-- A file in data_dir/voices that may have lost its last row, or may never get one:
-- an example WAV, a conditioning blob, a voice model file or a test WAV. The drain removes it only when no row
-- references it, then deletes every row of that key (the "ack").
voice_file_deletions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  storage_key  TEXT NOT NULL,
  deletion_id  INTEGER REFERENCES voice_deletions(id),  -- NULL = the write-ahead row of a file write, or a replaced conditioning blob, voice model file or test WAV
  requested_at TEXT NOT NULL,
  UNIQUE(storage_key, deletion_id)
);
CREATE INDEX voice_file_deletions_by_deletion ON voice_file_deletions(deletion_id);

-- A face group (design §6): faces the worker judged to be one person. A person IS a saved_voices row.
-- person_id NULL = not named yet ("Faces to name", §13). Deleting the person returns the group to unnamed.
-- The folder-deletion drain deletes a group left with no face (§3.2c).
face_groups (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,                       -- never reused
  owner_id   INTEGER NOT NULL,
  person_id  INTEGER REFERENCES saved_voices(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT
);
CREATE INDEX face_groups_by_owner_person ON face_groups(owner_id, person_id);

-- One face found in a photo by the `faces` stage (design §8). A re-run of the stage replaces the photo's rows.
faces (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,                       -- never reused
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  group_id   INTEGER REFERENCES face_groups(id) ON DELETE SET NULL,
  box_x      REAL NOT NULL,                                           -- the box, in 0..1 of the upright image
  box_y      REAL NOT NULL,
  box_w      REAL NOT NULL,
  box_h      REAL NOT NULL,
  score      REAL NOT NULL,                                           -- the detector score
  vector     BLOB NOT NULL,                                           -- float32, L2-normalised, the entry's dim (512)
  model      TEXT NOT NULL,                                           -- '<face model key>@<revision>'
  pinned     INTEGER NOT NULL DEFAULT 0,                              -- 1 = moved by the owner; grouping never moves it
  created_at TEXT NOT NULL
);
CREATE INDEX faces_photo_id_idx ON faces(photo_id);
CREATE INDEX faces_group_id_idx ON faces(group_id);
```

The schema version is 15: version 15 adds `face_groups` and `faces` and changes no
existing table. A face stores no file; a crop is cut from the detail thumbnail on
request (design §13).

Plus an FTS5 virtual table `photo_fts(caption, tags_text)`, its rows keyed by
`photos.id` and refreshed by the taxonomy and caption stages. Because `tags_text`
is derived from many `photo_tags` rows, the stage rebuilds the row explicitly
(delete-then-insert) rather than through per-row triggers.

`tags` is a shared vocabulary and deliberately has no `owner_id`; ownership comes
from the joined photo. Every user-scoped query filters on `owner_id`, and a
repository-layer helper makes omitting it awkward.

Storing every tag with a `score` and a `source` lets the UI show why a tag is
present, and lets thresholds be tuned per dimension without re-running models.

## EXIF facet keys

Every photo's full EXIF is stored verbatim in `photos.exif_json`. From it, a fixed
set of **facets** is derived into `photo_facets`, each either categorical
(`value_text`) or numeric (`value_num`, so ranges work):

| Group | Facets |
|---|---|
| Camera | `camera_make`, `camera_model`, `lens`, `software` |
| Exposure | `iso`, `aperture`, `shutter_speed`, `focal_length`, `exposure_bias`, `flash`, `exposure_program`, `metering_mode`, `white_balance` |
| Time | `year`, `month`, `weekday`, `hour`, `time_of_day` (night/dawn/morning/afternoon/evening), `is_weekend` |
| Place | `has_gps`, `gps_lat`, `gps_lon`, `place_city`, `place_country` (reverse-geocoded, §11) |
| Image | `megapixels`, `orientation`, `aspect` (portrait/landscape/square) |

`time_of_day` uses the local clock hour from EXIF (what a photographer means by
"evening shots"), not recomputed from GPS and UTC.
