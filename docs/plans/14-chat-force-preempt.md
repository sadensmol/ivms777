# Chat must WIN the models — force-preempt ingest + tracking notes

> Tracking + execution plan. Captures the **current** model-switch behaviour, the
> **chat-question scenarios**, the **resource-bar views**, the **bug** (chat does
> not win the lease), and the **proposed fix** (forcibly abort the in-flight
> caption and reload for chat). Canonical current-state spec stays `docs/design.md`
> §8.1; this plan proposes a change to it.

## 1. How model switching works today (§8.1)

- One decision point: `models/coordinator.py::ModelCoordinator`. A caller declares a
  **workload**; the coordinator RAM-guards it, takes a **cross-process lease** (a
  row in SQLite `model_lease`: holder, workload, priority, heartbeat,
  `preempt_requested`), then loads exactly that model-set and evicts the rest.
- Nothing switches on page navigation. Models switch **only** when code enters
  `with coordinator.require(WORKLOAD):`.

| Workload | Priority | Model-set (`models/workloads.py`) |
|---|---|---|
| `CHAT`, `MEMORY_REBUILD` | interactive (10) | SigLIP + planner `qwen2.5:3b` |
| `SEARCH` | interactive (10) | SigLIP |
| `INGEST_EMBED` | background (1) | SigLIP |
| `INGEST_CAPTION` | background (1) | caption LLM (mac: `qwen2.5vl:7b`, jetson: `:3b`) |

**Preemption (current):** an interactive request finds a background holder, sets
`preempt_requested`, and **polls `try_acquire` every 0.25 s for up to 30 s**
(`_PREEMPT_TIMEOUT_S`). The worker (`ingest/worker.py::drain`) checks
`should_preempt()` **before claiming each photo** and raises `Preempted` to yield —
so it releases the lease at the **next photo boundary**, never mid-photo. Design
§8.1 states the interactive request "waits only for the model swap plus at most one
in-flight photo."

## 2. THE BUG — chat does not win

- The caption call is a single blocking Ollama request bounded by the inference
  client's **120 s** timeout (`inference/client.py::complete`).
- One in-flight caption on the mac `qwen2.5vl:7b` model routinely runs **> 30 s**.
- Chat's preempt wait is capped at **30 s** → it raises `TimeoutError` **before**
  the worker reaches its next-photo yield point.
- `web/app.py::chat_stream` catches it and streams `BUSY_REPLY`
  ("The library is busy processing photos right now — please try again in a
  moment."). The worker keeps the lease, so the resource bar **stays on
  `ingest_caption`** — exactly the symptom observed.

**Root cause:** `_PREEMPT_TIMEOUT_S` (30 s) is shorter than one in-flight caption
(≤ 120 s), so the §8.1 "wait for one in-flight photo" guarantee is not honoured.
The design also forbids mid-flight abort, so even honouring it means a ≤ 120 s hang.

## 3. Chat-question scenarios (all wrapped in `require("CHAT")` today)

`web/app.py::chat_stream` takes the CHAT lease for the **entire** turn, then:

| # | Question kind | Detector | Needs a model? | Today |
|---|---|---|---|---|
| 1 | Off-topic | `is_photo_question` (LLM gate) | planner LLM | Under CHAT lease |
| 2 | Count / aggregate ("how many images?") | `is_aggregate_question` | **No** — pure `SELECT count(*)` for the number; LLM only to phrase it | Under CHAT lease → blocked by ingest ← the reported failure |
| 3 | Photo search | default path | SigLIP + planner | Under CHAT lease |
| 4 | Memory show | `memory_for_show` / `_auto_memory` | planner to phrase | Under CHAT lease |

Scenario 2 is the sharp edge: the **answer datum** ("how many") needs no model, yet
the turn still contends for the caption lease and shows "busy".

## 4. Resource-bar views (`web/static/resources.js`, right of the nav)

`RAM used/total · CPU% · <lease>` where `<lease>` is:

| Shown | Meaning |
|---|---|
| `idle` | no workload holds the lease; models stay as last left |
| `ingest_embed · siglip · 1.6 GB` | worker embedding/taxonomy (background) |
| `ingest_caption · qwen2.5vl:7b · 5.9 GB` | worker captioning (background) — the slow holder |
| `search · siglip · …` | `/library` search / `/photo` paging (interactive) |
| `chat · siglip+qwen2.5:3b · …` | a chat turn holds the models (interactive) |
| `memory_rebuild · siglip+qwen2.5:3b · …` | Organize → Rebuild memories (interactive) |

Expected on Ask: the bar flips from `ingest_caption` → `chat …`. Today it does not,
because chat times out (§2).

## 5. Proposed fix — chat WINS, fast

**A. Forcibly abort the in-flight caption (design change to §8.1).**
Replace "wait for one in-flight photo" with **cooperative cancellation**:
- The caption stage streams its Ollama response and checks `preempt_requested`
  between chunks; on preempt it **closes the stream** (disconnecting Ollama, which
  stops generating) and raises `Preempted`.
- `drain` lets `Preempted` propagate (it must NOT be caught by the per-file
  `except Exception` → `fail()`, which would burn a retry). The aborted photo's
  caption job stays **pending** and is retried on the next pass.
- The `with coordinator.require("INGEST_CAPTION")` exit releases the lease
  immediately; chat's next `try_acquire` succeeds within a chunk-time.
- Net: chat waits only for **one stream chunk + the model swap**, not a whole
  caption. This is the "some way" to forcibly stop ingest without a mid-CUDA kill.

**B. Count/aggregate answers need no lease (scenario 2).**
Answer count/total questions from the DB **before** taking the CHAT lease, so
"how many images" is instant even while captioning holds the models.

**C. Safety net.** Raise `_PREEMPT_TIMEOUT_S` above the caption inference timeout
so that, even if A's abort point is missed, chat still wins after one caption
rather than falsely reporting "busy".

## 6. Tasks (TDD — failing test first each) — DONE (A+B+C)

- [x] **C:** `_PREEMPT_TIMEOUT_S` 30 s → 150 s (safety net above the 120 s caption).
      Test `test_interactive_waits_out_a_slow_caption_past_30s`.
- [x] **B:** `chat/agent.py::auto_answer` — count/aggregate answered straight from
      the DB; `web/app.py::chat_stream` short-circuits it **before** the CHAT lease.
      Tests `test_auto_answer_*`, `test_count_question_answers_without_taking_the_model_lease`.
- [x] **A:** `inference/client.py` cancellable `complete(should_stop=…)` +
      `InferenceCancelled`; `ingest/caption.py` streams and converts cancel →
      `Preempted`; `ingest/worker.py::drain` re-raises `Preempted` and
      `requeue_running`s the photo (pending, not failed); `ingest/pipeline.py`
      passes `should_preempt` into the caption handler. Tests
      `test_complete_aborts_the_stream_when_should_stop_fires`,
      `test_caption_aborts_in_flight_and_requeues_the_photo_on_preempt`.
- [x] **Design:** `docs/design.md` §8.1 "Hard preemption" rewritten (cooperative
      caption cancellation); §10 prose + flow diagram add the pre-lease count
      short-circuit; `web/app.py` busy-comment updated.
- [ ] **Manual:** confirm the resource bar flips `ingest_caption → chat` on Ask.

## 7. Decision — RESOLVED

Full force-preempt (**A+B+C**) implemented. Chat now wins in ~one stream chunk +
model swap; counts answer instantly with no lease; 150 s is only the safety net.
