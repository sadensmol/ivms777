# Plan 10 (LATER PHASE): Agentic RAG + reranking for chat retrieval

> **Status: deferred.** Captured now, built later. Chat currently does one-shot
> fusion retrieval (top-30) + one grounded answer — it dumps loosely-related
> photos and the model confabulates ("a dog on the dashboard" when there is no
> dog). This plan replaces that with reranking + a bounded agent loop so chat
> returns real matches or an honest "nothing found".

> **For agentic workers (when this is picked up):** REQUIRED SUB-SKILL: superpowers:writing-plans to expand each task into bite-sized TDD steps before executing; then superpowers:executing-plans.

**Goal:** Make `/chat` answers precise. "find me a photo with a dog" returns the photos that actually contain a dog (or says there are none), not 30 nearest-neighbour thumbnails narrated into a hallucination.

**Why the current design is wrong here:** §9.1 keeps *interactive* retrieval to a single call for latency, and §10 force-feeds the top-30 to the model. Two failure modes follow: (1) semantic KNN always returns k neighbours, so there is no "nothing matches"; (2) the model is told to narrate and cite, so it invents. The user has accepted the latency cost of an agent loop for chat specifically, so §9.1's "interactive = one call" rule gets a documented exception for chat.

## Approach (three layers, cheapest first)

1. **Relevance floor + rerank (the 80%).** Fusion produces a candidate set with scores. Rerank the candidates against the query and drop everything below a floor. Two viable rerankers, decide at build time via a small bake-off:
   - **Embedding cross-check** — cosine(query_text_embedding, photo_embedding) already available from `photo_vec`; keep a calibrated threshold per the §17 note that SigLIP scores are poorly calibrated (tune on a small dev set).
   - **LLM rerank** — one batched call scoring each candidate's caption+tags for "does this match the query?" (0-3), keep ≥2. More accurate, one extra call.
   If nothing clears the floor → the answer is an honest "I couldn't find photos of X", **no sources dumped**.

2. **Reuse the query planner (already built).** Route the chat question through `search/planner.py::plan` first (tags + keyword + facets), so "a photo with a dog" uses the `subject` tag and the caption FTS — the same precise path `/library` search already uses — before semantic fusion. This alone fixes most "find X" queries.

3. **Bounded agent loop (the last 20%).** For questions a single retrieval cannot answer ("which trip had the most restaurant photos?"), let the planner model drive a few rounds of read-only tools — `search(query)`, `filter_by_tag`, `filter_by_facet`, `photos_near_in_time` — inspecting results and refining before it answers. Modelled on the Memories composer (`albums/compose.py`), capped at a few rounds. Verify-before-answer: the model must confirm each cited photo matches, so citations are trustworthy.

## Tasks (to be expanded into TDD steps at execution)

1. **Rerank module** — `search/rerank.py`: `rerank(conn, query, candidate_ids, *, floor) -> list[int]` (embedding cross-check first; LLM variant behind a flag). Tests: matches rank above non-matches; empty when none clear the floor. Include the calibration/bake-off script over a small hand-labelled dev set (§17).
2. **Chat retriever v2** — `chat/agent.py`: planner → fusion → rerank → floor, returning only verified matches (+ scores). Fall back to today's fusion if the planner/rerank errors. Tests with `FakeInferenceClient`/`FakeEmbedder`.
3. **Bounded agent loop** — extend `chat/agent.py` with the tool set + round cap; verify-before-answer. Deterministic under the fake (queued tool/answer turns), mirroring `test_memory_compose.py`.
4. **Wire into `/chat/stream`** — replace `retrieve()` with the agent retriever; **sources = verified matches only** (cap ~8), honest empty state when none. Update `chat.js`/template if the sources contract changes.
5. **Design updates** — rewrite §10 retrieval, add the §9.1 "chat is the interactive exception (agent loop allowed)" note, and record the rerank floor + bake-off result.

## Non-goals

- Changing `/library` search (already planner-backed and precise).
- Multi-turn conversational memory (still per-question; see §10).
- A learned reranker model — the embedding/LLM rerank is enough at this scale.

## Interim state (until this ships)

Chat stays one-shot fusion. Known-bad: dumps up to 30 loosely-related photos and can narrate a non-existent match. Users wanting precise "find X" should use `/library` search, which is planner-backed.
