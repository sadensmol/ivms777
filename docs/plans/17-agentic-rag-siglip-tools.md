# Plan 17 — Agentic-RAG chat: SigLIP-image tools + memory search

**Supersedes** the caption-embedding rerank in chat (plan 10/15's `caption_vec`
seed). Derived from §10 (agentic-RAG chat) and §9.2 (retriever core).

## Why

Two findings, both measured on the real library:

1. **SigLIP image↔text is the right photo retrieval; caption-text embeddings are a
   lossy detour.** SigLIP is trained for image↔text, so embedding the *query text*
   and matching it against each photo's **image** vector ranks the right photo first
   (`text("dogs")` → the dog photos; `text("a man in black")` → photo 2). Embedding
   *captions* and matching query-text↔caption-text is text↔text, which SigLIP was not
   trained for and a dedicated text embedder only partly fixes — and it is muddied by
   caption phrasing (a dog photo whose caption also says "SUV/Smolensk/sign" ranks
   next to the SUV photos). The library already stores image vectors and already does
   SigLIP query-text↔image KNN in `search/retriever.py::candidates()` /
   `search/semantic.py::search_photos()`.

2. **Chat must stay AGENTIC and actually call tools.** The value of the chat loop is
   that the model *searches* — the library (SigLIP) and memories — verifies what it
   finds, and grounds its answer on it. A pass that skips tools (plain top-k, or a
   bare general answer) "stopped searching" and is not what we want.

## What changes

- **Chat is a general assistant that is ALSO agentic over the library.** It is NOT
  limited to photo questions: any question — general knowledge, chit-chat, anything —
  is answered directly from the model's own knowledge, with NO off-topic gate and NO
  forced photo grounding. It is an agentic-RAG tool loop (`chat.agent`, restored as
  the chat path): the planner model drives it, and **only when the question is about
  the user's photos or memories** does it call a tool; otherwise it just answers.
- **Tools (all read-only, all through the core):**
  - `search_library(query)` — **SigLIP image↔text** via `search_photos()` (query text
    vs image vectors). THE photo finder.
  - `search_memories(query)` / `list_memories()` — find/enumerate memories
    (`albums`/`chat.agent` memory helpers) so the loop searches memories too.
  - `similar(photo_id)`, `nearby(photo_id)` — widen from a hit (unchanged, core).
- **Caption-vector rerank is removed from chat.** No `caption_vec` seed, no
  `nomic-embed-text` in the chat path. (`caption_vec`/§9 "similar" may keep it as one
  image+tag+caption signal, out of scope here; chat no longer depends on it.)
- **Grounding & honesty:** the answer is grounded ONLY on tool results and cites
  `[photo:ID]`; when tools return nothing relevant the model says so. Tool-calls are
  schema-constrained; the prompt has NO few-shot "expand first" example (it biased the
  weak model to wander); `temperature=0` on every planner/agent/answer call for
  determinism.

## Steps

- [ ] **Design:** update §10 (agentic-RAG flow + its mermaid) — tools = SigLIP
      `search_library` + memory search + similar/nearby; drop the caption-cosine seed;
      note SigLIP image↔text is the finder. Update §9's "two embedding spaces" note.
- [ ] `search/…`: expose a clean `search_library` (SigLIP image↔text) the tool calls;
      add a `search_memories` helper.
- [ ] `chat/agent.py`: tool set = search_library / search_memories / list_memories /
      similar / nearby; schema + prompt updated (no bad example); no caption seed;
      general-answer path for non-library questions; temp 0.
- [ ] `web/app.py`: chat route calls the agentic loop (restore), keeps direct_answer
      for pure DB counts, drops the off-topic gate.
- [ ] Tests: agent calls `search_library` for a photo query and cites the right photo;
      calls a memory tool for a memory query; answers a general question with no tool;
      honest-empty when tools find nothing.
- [ ] Verify live on the Jetson-style stack: `find me all photos with X`, a memory
      question, and a general-knowledge question.

## Non-goals

- Re-tuning §9 "similar photos" (image KNN + tags [+ caption]) — separate.
- A learned reranker / cross-encoder.
