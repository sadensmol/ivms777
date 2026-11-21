# PLAN 27 — TECH-29: face recognition (find faces, group them, name them as people)

- **Issue:** TECH-29 — https://linear.app/sadensmol/issue/TECH-29
- **Repo:** `ivms777` only. One PR to `main`.
- **Design:** `docs/design.md` §2, §3.2c, §4.1, §5, §5.1, §6, §8, §8.1, §11.1, §13,
  §13.1, §14; `docs/models.md` "The face model — InsightFace buffalo_l";
  `docs/data-model.md` (`face_groups`, `faces`); `docs/ingest-pipeline.md` stage 6;
  `docs/ui.md` `/voices`, `/faces/{id}`, the Faces tab, the photo People line.

This plan is a snapshot of one unit of work. The design is the current state.

## 1. Goal

Find the faces in every uploaded photo, group the faces of one person, let the owner
name each group as a person, and show those people on the photo page and in the film
shot brief. The model runs locally, on the GPU, in the models service only.

## 2. Decisions

| Id | Decision | Why |
|---|---|---|
| U1 | A person IS a `saved_voices` row. A face group points to it (`face_groups.person_id`, `ON DELETE SET NULL`). No second person table. A face group page can create a new person, on every profile. | The owner: "it's single list, same what we use now to attach voice, from now face is also attacheable." |
| O1 | InsightFace `buffalo_l` (`immich-app/buffalo_l` @ `d09715916a0778919a770c343533641e250b8699`): SCRFD-10G detector + ArcFace R50 (512 values), ONNX converted to torch with `onnx2torch`, run on `mps` / `cuda` in a killable `faces` child. | Best accuracy of the open models; the model Immich uses for this feature; one GPU runtime (torch). Non-commercial weights: allowed for this private library, never in a commercial build (the MusicGen rule). |
| D1 | Grouping: each new face joins the owner's group with the closest mean vector when cosine ≥ `match_threshold` (0.5), else a new group. In the worker, numpy, per face. Pinned faces never move. | Incremental, no model in the worker, never moves a face the owner reviewed. 0.5 is Immich's public default. |
| D2 | `faces` is the last stage, drained after `caption_embed` in its own guarded group. | Reprocess re-runs only faces; on jetson the face unit never needs the caption model at once. |
| D3 | No crop files: a crop is cut from the detail thumbnail on request. "Faces to name" lists a group only from 2 faces (declared, env-overridable). | Nothing for the delete outbox to remove; single faces of strangers would flood the list. |
| D4 | Keep the Voices page, its name and its routes; add faces to it. | The owner asked for one list "same what we use now"; a rename was not requested. |

Rejected: ONNX Runtime GPU (a second runtime, no proven CUDA 13 build for JetPack 7,
partial CoreML), YuNet + SFace (OpenCV DNN on the CPU, forbidden by §3.1),
facenet-pytorch (pins an old torch), DBSCAN/HDBSCAN re-clustering (moves reviewed
faces).

## 3. Steps

Waves: (1) design + contract stubs, (2) QA tests, (3) implementation, (4) QA verify.

1. **Design first.** Update `docs/design.md` and its §5 mermaid diagram, then
   `docs/models.md`, `docs/data-model.md`, `docs/ingest-pipeline.md`, `docs/ui.md`,
   `docs/roadmap.md`, and this plan.
2. **Contract stubs.** Every new name the tests call, returning neutral values:
   `catalog.FACE_MODELS` and `face_model_for`; `FaceNotOffered`, `FaceImageRefused`
   and `detect_faces` on the backends; `POST /faces/detect`;
   `ModelsClient.detect_faces`; `faces` in `ingest/jobs.py:STAGES`; the two tables
   (schema version 15); `ingest/faces.py`, `ingest/people.py`, `web/faces_api.py`;
   `ShotSignal.people`.
3. **The face model child** (`embedding/face_model.py`). Load the pinned snapshot from
   disk only; convert `detection/model.onnx` and `recognition/model.onnx` with
   `onnx2torch`; refuse any device that is not `mps` or `cuda*`; decode SCRFD boxes and
   5 landmarks; drop faces under `score_floor` (0.7) or with a side under `min_face_px`
   (32); align each face to 112×112; return boxes in 0..1, scores and L2-normalised
   512-value vectors. Module scope imports no torch, onnx or `onnx2torch`. Add `onnx`
   and `onnx2torch` to the `models` extra (`onnx2torch` needs `torchvision`, matched to
   the locked torch on every profile) and the snapshot to `scripts/fetch_models.py`.
4. **`POST /faces/detect`.** `SlotManager` registers a `faces` unit once (like
   `speaker_embed`). `CompositeBackend.detect_faces` runs at `Priority.BATCH`, label
   "finding faces", outside the voice gate, and checks the reply's revision and every
   vector's size. 503 with no face model on the profile, 422 for an unreadable image
   or bad base64, 500 for a load or model error. The fake backend returns declared
   faces per image.
5. **Store faces and groups.** `face_groups(id AUTOINCREMENT, owner_id, person_id →
   saved_voices ON DELETE SET NULL, created_at, updated_at)` and `faces(id
   AUTOINCREMENT, photo_id → photos ON DELETE CASCADE, group_id → face_groups ON
   DELETE SET NULL, box_x, box_y, box_w, box_h, score, vector, model, pinned,
   created_at)`, indexes on `faces(photo_id)`, `faces(group_id)` and
   `face_groups(owner_id, person_id)`.
6. **The `faces` stage and grouping** (`ingest/faces.py`, `ingest/pipeline.py`,
   `ingest/folders.py`, `inference/models_client.py`, `config.py`). `backfill_faces`
   queues a job for every photo without one. The handler opens the original upright,
   shrinks it to `input_edge` (640), calls the service, replaces the photo's faces in
   one transaction, and groups each new face (D1). An outage leaves the job pending
   and never fails the pass. Folder deletion removes the faces through the cascade and
   then deletes the owner's groups left with no face.
7. **Review and name** (`ingest/people.py`, `web/faces_api.py`, `web/voices_api.py`,
   templates, `web/static/faces.js`). "Faces to name" on `/voices`; the group leaf
   `/faces/{id}` with Attach, New person (`feed.saved_voices` name rules) and Not this
   person; the Faces tab on `/voices/{id}` with Remove from this person; the crop route
   `/faces/crop/{face_id}`. Every query is owner-scoped.
8. **People on the photo page.** `people_in_photos` (faces → groups → saved voices,
   owner-scoped) feeds a People line of plain-text names on `/photo/{id}`.
9. **People in the film shot brief.** `ShotSignal.people` (empty by default) is filled
   in `read_signals`; `brief()` appends `people=Ann, Bob` only when it is not empty, so
   a shot with no named person keeps its exact old brief.

## 4. Navigation (design §13.1)

- `/faces/{id}` is a leaf one level under the `/voices` grid. A "Faces to name" crop is
  a plain link: the one push.
- Every action on the leaf is a POST answered in place or by `location.replace`; none
  pushes history.
- Close (×, Esc, the "Voices" link): `history.back()` when the page carries
  `data-from-grid` (same-origin referrer with the path `/voices`), else
  `location.replace("/voices")`.
- The Faces tab of `/voices/{id}` links to no group leaf, and the photo page's People
  line links to no person: both would be a leaf inside a leaf.

## 5. Acceptance criteria

The ids are the run's `AC1`–`AC15` (TECH-29 `PLAN.md` §S10), never renumbered:

- **AC1** — the catalog declares `buffalo-l` outside the slots (pinned commit, mac,
  jetson and cloud, 512 values, score floor, match threshold, input edge, estimated
  cost, non-commercial note); `face_model_for(profile)` returns it; not in settings.
- **AC2** — `POST /faces/detect` returns per face a box in 0..1, a score and a
  512-value L2-normalised vector, with model and revision; 503 with no face model; 422
  for an unreadable image; another revision or vector size is an error.
- **AC3** — only the models service imports the face model, torch, onnx or
  `onnx2torch`; the one-model-process gate passes.
- **AC4** — every photo gets a `faces` job; faces are stored; a re-run replaces them; no
  face = no rows; an outage leaves the job pending and the pass does not fail.
- **AC5** — a new face joins the closest group at the threshold, else a new group; a
  pinned face never moves.
- **AC6** — "Faces to name" on `/voices` lists every unnamed group with at least the
  minimum face count, with crops and count, linking to `/faces/{id}`.
- **AC7** — on `/faces/{id}` the owner attaches the group to a person or creates one by
  name; a taken name is refused with its reason; every profile; an attached group
  leaves the list.
- **AC8** — Not this person moves one face to a pinned group of its own.
- **AC9** — the Faces tab of `/voices/{id}` lists the person's groups; Remove returns a
  group to "Faces to name".
- **AC10** — a new face that matches an attached group is that person with no action.
- **AC11** — the photo page lists the people in it, plain text, each once.
- **AC12** — the shot brief names the people (`people=…`); a shot with none is
  unchanged.
- **AC13** — a folder delete removes the deleted photos' faces and the groups left
  empty; a person delete returns their groups to "Faces to name"; a photo kept for
  another folder keeps its faces.
- **AC14** — `/faces/{id}` obeys §13.1; the crop route serves the owner's face and 404
  for another owner's.
- **AC15** — the design, `docs/models.md`, `docs/data-model.md` and this plan describe
  the feature as built.

## 6. Verification

- `uv run pytest -q` and `uv run ruff check .`, with the fake models backend. No test
  loads a model or drives the running app.
- New suites: `tests/test_face_model.py`, `tests/test_faces_stage.py`,
  `tests/test_face_clusters.py`, `tests/test_web_faces.py`,
  `tests/test_design_docs_faces.py`; extended: `test_catalog.py`,
  `test_modelsvc_api.py`, `test_web_photo.py`, `test_feed_signals.py`,
  `test_folder_deletion.py`, `test_saved_voices.py`, `test_migrate.py`,
  `test_one_model_process_gate.py` (exempt `embedding/face_model.py`; forbid `onnx`
  and `onnx2torch` elsewhere).
- One real `POST /faces/detect` on one photo against the running models service must
  return finite 512-value vectors before the feature is called ready (the owner
  confirms on jetson).

## 7. Risks

| Id | Risk | Mitigation |
|---|---|---|
| K1 | `onnx2torch` fails on the SCRFD graph, or runs it wrong on mps | the one real call above; fallback is a new decision (ONNX Runtime for the detector only) |
| K2 | Jetson memory: one more child | an ordinary governor unit, evicted like the others; drains after the caption stage |
| K3 | Upload order splits one person into two groups | attach both groups to the same person; the photo shows one name |
| K4 | A Reprocess of `faces` replaces a photo's faces | the owner's Not this person fix on that photo is lost; documented; Reprocess is manual |
| K5 | The 0.5 threshold is not tuned on this library | declared data with an env override; the owner fixes a face by hand |
| K6 | `onnx2torch` needs `torchvision`, not yet locked | match it to the locked torch on every profile when the lock is updated |

## 8. Rollout

A plain merge. Schema 15 only adds tables. On the first worker pass every photo gets a
`faces` job and the stage fills in over the library, with its own progress row and
Reprocess button on `/upload`. `make models` fetches the pinned snapshot once; until
then each `faces` job fails with "not on disk" and the photos stay usable.

## 9. Not in scope

Search, filters and chat by person; a person photo grid; picking a narrator from the
people in a memory; renaming Voices to People.
