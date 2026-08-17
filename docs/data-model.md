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
  embedding_model TEXT,
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

-- sqlite-vec virtual table; rowid joins to photos.id
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
  stage      TEXT NOT NULL,           -- thumbnail | embed | taxonomy | caption
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
```

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
