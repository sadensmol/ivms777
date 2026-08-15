# Plan 11 (LATER PHASE): Read-only MCP server over the organized library

> **Status: deferred.** Captured now, built later. The counterpart to stage 2
> (`ivms777-sync`, §7): instead of exporting a change plan to reshape the disk,
> expose the *organized* library over the Model Context Protocol so an external
> agent reads the organization's data live. Read-only, single-owner, stdio.

> **For agentic workers (when this is picked up):** REQUIRED SUB-SKILL:
> superpowers:writing-plans to expand each task into bite-sized TDD steps before
> executing; then superpowers:executing-plans. Building an MCP server? Consult the
> `claude-api` skill / MCP SDK docs first.

**Goal:** A local MCP server that lets an agent (Claude Desktop, a local agent)
query the organized library — search, ask, memories, albums, a photo with its
metadata, the export plan — without the app UI and without ever writing anything.

**Why this shape.** The app already *is* the read model (§5: `app` serves reads,
`worker` owns writes, storage/inference reached only through their interfaces).
An MCP server is just **another read entrypoint alongside `web/app.py`**, calling
the same read functions. Nothing new touches the write path, so "source folders
are sacred" (§3.2c) and the layered boundaries (§5) hold by construction.

## Scope (from the design decision)

- **Read-only.** No mutation tools — no create/rename album, no tag edits, no
  rebuild, no folder delete. An agent reads; the human still organizes via the UI.
- **Single-owner, stdio, no auth.** Matches today's single-user model (§3.2). The
  server binds to the configured `owner_id`. Hosted / multi-tenant MCP with
  per-owner tokens waits on the auth work (§18, v02) — explicitly out of scope.
- **Goes through the app read layer**, reusing existing functions; the MCP module
  adds no new SQL that the app doesn't already have.

## Tool / resource surface (all read-only)

Reusing what already exists — no new retrieval logic:

- `search(query, limit=30)` → ranked photos (id, caption, date, tags, thumb URL).
  Reuses the planner-backed reranked retriever (`chat/agent.py::retrieve`, §10).
- `ask(question)` → the grounded answer + cited photo ids. Reuses the agentic
  chat pipeline (`agent_retrieve` + `chat_messages`, §10); returns the finished
  text (not an SSE stream) plus its citations.
- `list_memories()` / `get_memory(id)` → stored `groups(kind='memory')` with their
  photos. Reuses `albums/memory_store.py` (§11).
- `list_albums(by, grain?)` / `get_album(by, key)` → any organizer's albums.
  Reuses `albums/registry.py` organizers (date, camera, place; §11).
- `get_photo(id)` → full metadata: caption, tags, EXIF facets, place, sources.
  Reuses the same reads `chat/context.py` and `/photo` use.
- `get_export_plan()` → the stage-2 manifest as a resource (§7's `/api/manifest`),
  so an agent can *inspect* the proposed disk layout — applying it stays
  `ivms777-sync`'s job, under explicit confirmation (§7).
- Resources: `memories`, `albums` as browsable MCP resources; photos exposed as
  resource URIs resolving to thumbnails.

## Tasks (to be expanded into TDD steps at execution)

1. **Server skeleton** — `mcp/server.py`: an MCP server (Python MCP SDK / FastMCP)
   over stdio, wired to a read-only `AppContext` for the configured owner. A
   `mcp` console entry + compose/profile wiring. Test: server starts, lists the
   declared tools/resources.
2. **Query tools** — `search`, `get_photo`, `list_albums`/`get_album`,
   `list_memories`/`get_memory`, each a thin adapter over the existing read
   function. Tests with `FakeEmbedder`/`FakeInferenceClient` assert the adapter
   returns the same ids/shape the app does.
3. **`ask` tool** — non-streaming wrapper of the §10 chat pipeline returning
   answer + citations. Deterministic under the fakes (queued planner/agent turns).
4. **Export-plan resource** — expose §7's manifest read-only; assert it matches
   `/api/manifest` for the same library and that no write path is reachable.
5. **Design updates** — add a §5/§17.1 note that `mcp/` is a read-only entrypoint
   peer to `web/`, and flip the §18 bullet to done.

## Non-goals

- Any mutation (organize, tag, rebuild, delete) — read-only by design.
- Applying disk changes — that is `ivms777-sync` (§7); MCP only *exposes* the plan.
- Auth / multi-tenant / networked transport — waits on the §18 auth work (v02).
- New retrieval or ranking logic — MCP reuses §9/§10/§11 as-is.

## Interim state (until this ships)

The library is reachable only through the web UI and the chat/search routes. An
external agent cannot query the organized data; there is no MCP endpoint.
