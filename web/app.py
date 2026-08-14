import json
import threading
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from albums.by_date import DEFAULT_GRAIN, GRAIN_LABELS, ByDateOrganizer
from albums.memories import MemoriesOrganizer
from albums.memories_build import build_memories
from albums.memory_store import current_signature, stored_signature
from albums.registry import ORGANIZERS, get_organizer
from chat.context import build_context as build_chat_context
from chat.history import add_message, current_session, new_session, session_messages
from chat.retrieve import is_photo_question, retrieve
from config import Settings
from inference.prompts import chat_messages
from ingest.caption import backfill_captions, caption_handler
from ingest.embed import backfill_embeds, embed_handler
from ingest.facets import backfill_place_facets
from ingest.jobs import STAGES, reprocess, stage_counts
from ingest.taxonomy import backfill_taxonomy, taxonomy_handler
from ingest.thumbs import backfill_thumbnails, thumb_key
from ingest.vocab import load_vocab, seed_tags
from ingest.worker import drain, thumbnail_handler
from search.dates import date_where
from search.facets import (
    SIDEBAR_GROUPS,
    SORTABLE,
    build_where,
    facet_counts,
    parse_filters,
)
from search.fusion import reciprocal_rank_fusion
from search.keyword import keyword_search
from search.planner import plan, spec_to_params
from search.semantic import search_photos, similar_photos
from search.tags import parse_tag_filters, tag_sidebar, tag_where
from web.deps import AppContext, build_context
from web.upload_api import register as register_upload_api

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab.yaml"

# Columns every grid row needs: identity, the caption and camera/date shown in
# the hover popup, whether it is embedded (AI status), and the redundant-copy
# count behind the ×N badge.
SELECT_COLS = (
    "SELECT p.id, p.caption, p.shot_at, p.camera, p.embedding_model,"
    " (SELECT count(*) - 1 FROM photo_sources s WHERE s.photo_id = p.id) AS dupe_count"
    " FROM photos p"
)

BASE_SQL = SELECT_COLS + " WHERE p.owner_id = ? AND p.thumb_key IS NOT NULL"

# Only photos whose bytes were found at more than one local path.
DUPES_ONLY = (
    " AND (SELECT count(*) FROM photo_sources s2 WHERE s2.photo_id = p.id) > 1"
)

DEFAULT_ORDER = " ORDER BY COALESCE(p.shot_at, p.created_at) DESC, p.id DESC"


def _order_clause(sort: str | None) -> tuple[str, list]:
    facet_key = SORTABLE.get(sort or "")
    if facet_key is None:
        return DEFAULT_ORDER, []
    direction = "ASC" if (sort or "").endswith("_asc") else "DESC"
    clause = (
        " ORDER BY (SELECT f.value_num FROM photo_facets f"
        " WHERE f.photo_id = p.id AND f.key = ?) "
        f"{direction}, p.id DESC"
    )
    return clause, [facet_key]


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="ivms777")
    app.state.context = build_context(settings)
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Load the taxonomy vocabulary once and seed the tag ids so the taxonomy stage
    # can reference them; seeding is idempotent.
    vocab = load_vocab(VOCAB_PATH)
    seed_tags(app.state.context.conn, vocab)

    def context() -> AppContext:
        return app.state.context

    # Memories build state. Single-owner (§3.2): one build at a time per process.
    # `inference_override`/`queue_inference`/`await_build` are test seams so the
    # build runs a FakeInferenceClient and is joinable; production uses settings.
    app.state.memories_building = False
    app.state.memories_progress = {"done": 0, "total": 0}
    app.state.inference_override = None
    app.state.memories_build_thread = None

    def queue_inference(responses: list[str]) -> None:
        from inference.fakes import FakeInferenceClient

        app.state.inference_override = FakeInferenceClient(responses)

    def await_build() -> None:
        thread = app.state.memories_build_thread
        if thread is not None:
            thread.join()

    app.state.queue_inference = queue_inference
    app.state.await_build = await_build

    def drain_now() -> None:
        """Build thumbnails and embeddings for whatever has arrived.

        The `worker` container drains continuously in deployment. Doing it here
        too means a single-container run — and every test — still produces a
        usable, searchable grid without waiting on a poll.
        """
        ctx = context()
        embedder, model_name = ctx.settings.build_embedder()
        client, caption_model = ctx.settings.build_inference_client()
        backfill_thumbnails(ctx.conn)
        backfill_embeds(ctx.conn)
        backfill_taxonomy(ctx.conn)
        backfill_place_facets(ctx.conn)
        backfill_captions(ctx.conn)
        drain(
            ctx.conn,
            {
                "thumbnail": thumbnail_handler(
                    ctx.originals,
                    ctx.derived,
                    ctx.settings.thumb_grid_px,
                    ctx.settings.thumb_detail_px,
                ),
                "embed": embed_handler(ctx.originals, embedder, model_name),
                "taxonomy": taxonomy_handler(ctx.derived, embedder, vocab),
                "caption": caption_handler(
                    ctx.derived, client, caption_model,
                    list(vocab.dimensions), ctx.settings.thumb_detail_px,
                ),
            },
        )

    register_upload_api(app, context, drain_now)

    def _filter_where(ctx: AppContext, params: dict[str, str]) -> tuple[str, list]:
        # EXIF facets AND model tags AND date range AND the optional dupes flag.
        facet_where, facet_params = build_where(parse_filters(params))
        tags_where, tags_params = tag_where(
            parse_tag_filters(params), ctx.settings.tag_score_min
        )
        date_frag, date_params = date_where(params)
        where = facet_where + tags_where + date_frag
        if params.get("dupes"):
            where += DUPES_ONLY
        return where, [*facet_params, *tags_params, *date_params]

    def _rows_in_order(ctx: AppContext, page_ids: list[int]) -> list:
        placeholders = ", ".join("?" for _ in page_ids)
        ranking = " ".join(f"WHEN {pid} THEN {rank}" for rank, pid in enumerate(page_ids))
        return list(ctx.conn.execute(
            SELECT_COLS
            + f" WHERE p.owner_id = ? AND p.id IN ({placeholders})"
            + f" ORDER BY CASE p.id {ranking} END",
            (ctx.settings.owner_id, *page_ids),
        ))

    def _search_page(ctx: AppContext, params: dict[str, str], query: str, offset: int) -> list:
        # Semantic + keyword rankings, fused; facet/tag filters narrow the survivors.
        embedder, _ = ctx.settings.build_embedder()
        semantic = search_photos(ctx.conn, embedder, ctx.settings.owner_id, query, k=200)
        keyword = keyword_search(ctx.conn, ctx.settings.owner_id, query, k=200)
        fused = reciprocal_rank_fusion([semantic, keyword])
        if not fused:
            return []
        where, where_params = _filter_where(ctx, params)
        if where:
            placeholders = ", ".join("?" for _ in fused)
            allowed = {
                row["id"] for row in ctx.conn.execute(
                    "SELECT p.id FROM photos p WHERE p.owner_id = ?" + where
                    + f" AND p.id IN ({placeholders})",
                    (ctx.settings.owner_id, *where_params, *fused),
                )
            }
            fused = [pid for pid in fused if pid in allowed]
        page_ids = fused[offset : offset + ctx.settings.page_size]
        return _rows_in_order(ctx, page_ids) if page_ids else []

    def fetch_page(offset: int, params: dict[str, str]) -> list:
        ctx = context()
        query = params.get("q", "").strip()
        if query:
            return _search_page(ctx, params, query, offset)
        where, where_params = _filter_where(ctx, params)
        order, order_params = _order_clause(params.get("sort"))
        sql = BASE_SQL + where + order + " LIMIT ? OFFSET ?"
        bound = [
            ctx.settings.owner_id, *where_params, *order_params,
            ctx.settings.page_size, offset,
        ]
        return list(ctx.conn.execute(sql, bound))

    def _params(request: Request) -> dict[str, str]:
        # Checkboxes share a name and submit repeated params; joining them keeps
        # multi-select working, which dict(request.query_params) would silently drop.
        return {
            key: ",".join(request.query_params.getlist(key)) for key in request.query_params
        }

    KEEP_KEYS = ("sort", "q", "dupes", "date_from", "date_to", "planned")

    def _query_string(params: dict[str, str]) -> str:
        keep = {
            k: v for k, v in params.items()
            if k.startswith(("f_", "n_", "t_")) or k in KEEP_KEYS
        }
        return urlencode(keep)

    def parsed_chips(params: dict[str, str]) -> list[dict]:
        # A removable chip per active predicate. Removing one drops that predicate
        # and keeps everything else (incl. planned=1), so nothing re-plans.
        def without(name: str, value: str | None = None) -> str:
            keep = dict(params)
            if value is not None:  # multi-value f_/t_: drop just this one value
                rest = [v for v in keep[name].split(",") if v and v != value]
                keep[name] = ",".join(rest) if rest else None
                if keep[name] is None:
                    keep.pop(name, None)
            else:
                keep.pop(name, None)
            return "/library?" + urlencode({
                k: v for k, v in keep.items()
                if k.startswith(("f_", "n_", "t_")) or k in KEEP_KEYS
            })

        chips: list[dict] = []
        for name, raw in params.items():
            if name.startswith(("f_", "t_")):
                for value in raw.split(","):
                    if value:
                        chips.append({"label": f"{name[2:]}: {value}",
                                      "remove": without(name, value)})
            elif name.startswith("n_"):
                chips.append({"label": f"{name[2:]}: {raw}", "remove": without(name)})
            elif name in ("date_from", "date_to"):
                chips.append({"label": f"{name}: {raw}", "remove": without(name)})
        return chips

    def _sidebar() -> list[dict]:
        ctx = context()
        groups = []
        for title, keys in SIDEBAR_GROUPS:
            entries = [
                {"key": key, "values": facet_counts(ctx.conn, ctx.settings.owner_id, key)}
                for key in keys
            ]
            groups.append({"title": title, "entries": [e for e in entries if e["values"]]})
        return [group for group in groups if group["entries"]]

    def _tag_sidebar() -> list[dict]:
        ctx = context()
        return tag_sidebar(
            ctx.conn, ctx.settings.owner_id, list(vocab.dimensions),
            ctx.settings.tag_score_min, 12,
        )

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/library")

    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request):
        params = _params(request)
        query = params.get("q", "").strip()
        if query and not params.get("planned"):
            # Plan the free-text query once, then redirect to the materialized
            # filter params (§9.1). From there the page is param-driven and chips
            # are removable without re-planning.
            ctx = context()
            client, _ = ctx.settings.build_inference_client()
            spec = plan(client, ctx.settings.planner_model or "fake", query,
                        list(vocab.dimensions))
            target = spec_to_params(spec, query=query, dimensions=list(vocab.dimensions))
            return RedirectResponse("/library?" + urlencode(target), status_code=303)
        rows = fetch_page(0, params)
        return templates.TemplateResponse(
            request,
            "library.html",
            {
                "photos": rows,
                "next_offset": len(rows),
                "page_size": context().settings.page_size,
                "query": _query_string(params),
                "sidebar": _sidebar(),
                "tag_sidebar": _tag_sidebar(),
                "chips": parsed_chips(params),
                "active": params,
            },
        )

    @app.get("/library/page", response_class=HTMLResponse)
    def library_page(request: Request, offset: int = 0) -> HTMLResponse:
        params = _params(request)
        rows = fetch_page(offset, params)
        return templates.TemplateResponse(
            request,
            "_grid_page.html",
            {
                "photos": rows,
                "next_offset": offset + len(rows),
                "page_size": context().settings.page_size,
                "query": _query_string(params),
            },
        )

    @app.post("/reprocess")
    def reprocess_library(
        from_stage: str = Form("taxonomy"), to_stage: str = Form(""),
    ) -> RedirectResponse:
        # Re-run a range of stages over the existing library (no re-upload). The
        # everyday button stops before `caption` (images are static — no reason to
        # re-caption); a separate, explicit button re-captions. The worker drains
        # the reset jobs on its next poll; the /upload page shows progress.
        ctx = context()
        stage = from_stage if from_stage in STAGES else "taxonomy"
        bound = to_stage if to_stage in STAGES else None
        reprocess(ctx.conn, ctx.settings.owner_id, stage, bound)
        return RedirectResponse("/upload", status_code=303)

    @app.get("/organize", response_class=HTMLResponse)
    def organize(
        request: Request, by: str | None = None, grain: str | None = None
    ) -> HTMLResponse:
        ctx = context()
        organizer = get_organizer(by)
        albums = organizer.organize(ctx.conn, ctx.settings.owner_id, grain)
        # The grain sub-selector only applies to the date organizer.
        is_date = organizer.name == ByDateOrganizer.name
        active_grain = grain if grain in ByDateOrganizer.grains else DEFAULT_GRAIN
        # Memories carries a rebuild control and a stale/building indicator.
        is_memories = organizer.name == MemoriesOrganizer.name
        owner_id = ctx.settings.owner_id
        memories_stale = is_memories and (
            stored_signature(ctx.conn, owner_id) != current_signature(ctx.conn, owner_id)
        )
        return templates.TemplateResponse(
            request,
            "organize.html",
            {
                "organizers": [(o.name, o.label) for o in ORGANIZERS.values()],
                "active": organizer.name,
                "grains": GRAIN_LABELS if is_date else [],
                "active_grain": active_grain if is_date else None,
                "albums": albums,
                "total_photos": sum(a.size for a in albums),
                "memories_stale": memories_stale,
                "memories_building": app.state.memories_building,
            },
        )

    @app.post("/organize/memories/rebuild")
    def memories_rebuild() -> RedirectResponse:
        # One build at a time per process (§3.2). The build makes many model calls,
        # so it runs on a daemon thread; the thread gets its own DB connection via
        # the thread-local AppContext.conn.
        ctx = context()
        if not app.state.memories_building:
            client = app.state.inference_override or ctx.settings.build_inference_client()[0]
            model = ctx.settings.planner_model or "fake"
            owner_id = ctx.settings.owner_id
            app.state.memories_building = True
            app.state.memories_progress = {"done": 0, "total": 0}

            def on_progress(done: int, total: int) -> None:
                app.state.memories_progress = {"done": done, "total": total}

            def run() -> None:
                try:
                    build_memories(ctx.conn, client, model, owner_id,
                                   force=True, progress=on_progress)
                finally:
                    app.state.memories_building = False

            thread = threading.Thread(target=run, daemon=True)
            app.state.memories_build_thread = thread
            thread.start()
        return RedirectResponse("/organize?by=memories", status_code=303)

    @app.get("/organize/memories/progress", response_class=HTMLResponse)
    def memories_progress() -> HTMLResponse:
        # Polled by HTMX while a build runs. When it finishes, reload the page so
        # the freshly-built memories render (they are written atomically at the end).
        if not app.state.memories_building:
            return HTMLResponse("", headers={"HX-Refresh": "true"})
        prog = app.state.memories_progress
        total = prog["total"]
        pct = int(100 * prog["done"] / total) if total else 0
        return HTMLResponse(
            '<span hx-get="/organize/memories/progress" hx-trigger="every 1500ms"'
            ' hx-swap="outerHTML">'
            f"Building memories… {prog['done']}/{total} ({pct}%)</span>"
        )

    @app.get("/thumb/{photo_id}")
    def thumb(photo_id: int, size: str = "grid") -> Response:
        ctx = context()
        row = ctx.conn.execute(
            "SELECT content_hash, thumb_key FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, ctx.settings.owner_id),
        ).fetchone()
        if row is None or row["thumb_key"] is None:
            raise HTTPException(status_code=404)
        px = ctx.settings.thumb_detail_px if size == "detail" else ctx.settings.thumb_grid_px
        key = thumb_key(row["content_hash"], px)
        if not ctx.derived.exists(key):
            raise HTTPException(status_code=404)
        return Response(ctx.derived.read(key), media_type="image/jpeg")

    def _ordered_ids(params: dict[str, str]) -> list[int]:
        # The full ordered id list the library grid would show under these params
        # (search + filters + sort), without pagination — the sequence `/photo`
        # pages through when the collection is the library.
        ctx = context()
        owner = ctx.settings.owner_id
        query = params.get("q", "").strip()
        if query:
            embedder, _ = ctx.settings.build_embedder()
            semantic = search_photos(ctx.conn, embedder, owner, query, k=200)
            keyword = keyword_search(ctx.conn, owner, query, k=200)
            fused = reciprocal_rank_fusion([semantic, keyword])
            where, where_params = _filter_where(ctx, params)
            if where and fused:
                placeholders = ", ".join("?" for _ in fused)
                allowed = {
                    row["id"] for row in ctx.conn.execute(
                        "SELECT p.id FROM photos p WHERE p.owner_id = ?" + where
                        + f" AND p.id IN ({placeholders})",
                        (owner, *where_params, *fused),
                    )
                }
                fused = [pid for pid in fused if pid in allowed]
            return fused
        where, where_params = _filter_where(ctx, params)
        order, order_params = _order_clause(params.get("sort"))
        sql = (
            "SELECT p.id FROM photos p WHERE p.owner_id = ? AND p.thumb_key IS NOT NULL"
            + where + order
        )
        return [row["id"] for row in ctx.conn.execute(sql, (owner, *where_params, *order_params))]

    def _photo_context(ctx_param: str | None, params: dict[str, str]) -> dict:
        # Resolve the collection a photo was opened within: an Organize album
        # (`album:<by>:<grain>:<key>` — covers date/camera/place/memories) or the
        # library (default). Returns its ordered ids, title, and description.
        ctx = context()
        owner = ctx.settings.owner_id
        if ctx_param and ctx_param.startswith("album:"):
            parts = ctx_param.split(":", 3)
            if len(parts) == 4:
                _, by, grain, key = parts
                for album in get_organizer(by).organize(ctx.conn, owner, grain or None):
                    if album.key == key:
                        return {"ids": album.photo_ids, "title": album.title,
                                "description": album.description}
            # The album is gone (e.g. memories rebuilt) — fall back to the library.
        query = params.get("q", "").strip()
        return {
            "ids": _ordered_ids(params),
            "title": f"Search: {query}" if query else "Library",
            "description": None,
        }

    def _origin_url(ctx_param: str | None, params: dict[str, str]) -> str:
        # The top-level grid a photo was opened from — where "close" returns to.
        if ctx_param and ctx_param.startswith("album:"):
            parts = ctx_param.split(":", 3)
            if len(parts) == 4:
                _, by, grain, _key = parts
                return f"/organize?by={by}" + (f"&grain={grain}" if grain else "")
        library_query = _query_string(params)
        return "/library" + (f"?{library_query}" if library_query else "")

    @app.get("/photo/{photo_id}", response_class=HTMLResponse)
    def photo_detail(request: Request, photo_id: int) -> HTMLResponse:
        ctx = context()
        photo = ctx.conn.execute(
            "SELECT * FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, ctx.settings.owner_id),
        ).fetchone()
        if photo is None:
            raise HTTPException(status_code=404)
        sources = list(ctx.conn.execute(
            "SELECT rel_path FROM photo_sources WHERE photo_id = ? ORDER BY rel_path",
            (photo_id,),
        ))
        facets = list(ctx.conn.execute(
            "SELECT key, value_text, value_num FROM photo_facets WHERE photo_id = ? ORDER BY key",
            (photo_id,),
        ))
        wasted = (photo["bytes"] or 0) * max(0, len(sources) - 1)
        tags: dict[str, list[dict]] = {}
        for row in ctx.conn.execute(
            "SELECT t.dimension, t.label, pt.score, pt.source FROM photo_tags pt"
            " JOIN tags t ON t.id = pt.tag_id WHERE pt.photo_id = ?"
            " ORDER BY t.dimension, pt.score DESC",
            (photo_id,),
        ):
            tags.setdefault(row["dimension"], []).append(
                {"label": row["label"], "score": row["score"], "source": row["source"]}
            )
        similar = similar_photos(ctx.conn, ctx.settings.owner_id, photo_id, k=12)

        # Page within the collection this photo was opened from (§13) — never leak
        # into another memory/album. `ctx` names the collection; ids are its order.
        params = _params(request)
        ctx_param = params.get("ctx")
        collection = _photo_context(ctx_param, params)
        ids = collection["ids"]
        if photo_id not in ids:
            # Opened outside any collection (a "similar" jump, a stale link) — fall
            # back to the whole library so paging still works.
            ids = _ordered_ids({})
            collection = {"title": "Library", "description": None, "ids": ids}
            ctx_param = None
        index = ids.index(photo_id) if photo_id in ids else -1
        prev_id = ids[index - 1] if index > 0 else None
        next_id = ids[index + 1] if 0 <= index < len(ids) - 1 else None

        keep = {
            k: v for k, v in params.items()
            if k == "ctx" or k.startswith(("f_", "n_", "t_"))
            or k in ("q", "sort", "dupes", "date_from", "date_to", "planned")
        }
        return templates.TemplateResponse(
            request,
            "photo.html",
            {
                "photo": photo, "sources": sources, "facets": facets,
                "similar": similar, "wasted_bytes": wasted,
                "embedded": photo["embedding_model"] is not None,
                "tags": tags,
                "prev_id": prev_id, "next_id": next_id,
                "ctx_query": urlencode(keep),
                "origin_url": _origin_url(ctx_param, params),
                "collection": collection,
                "position": index + 1 if index >= 0 else None,
                "total": len(ids),
            },
        )

    def progress_payload() -> dict:
        ctx = context()
        failures = list(
            ctx.conn.execute(
                "SELECT j.stage, j.error,"
                " (SELECT s.rel_path FROM photo_sources s WHERE s.photo_id = p.id"
                "  ORDER BY s.id LIMIT 1) AS path"
                " FROM jobs j JOIN photos p ON p.id = j.photo_id"
                " WHERE j.status = 'failed' ORDER BY path LIMIT 50"
            )
        )
        return {
            "stages": [(stage, stage_counts(ctx.conn, stage)) for stage in STAGES],
            "failures": failures,
        }

    OFF_TOPIC_REPLY = (
        "I can only answer questions about your photos — try asking what's in "
        "them, or when and where they were taken."
    )

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page(request: Request) -> HTMLResponse:
        # Render the current session's persisted turns as static history (§10).
        ctx = context()
        session_id = current_session(ctx.conn, ctx.settings.owner_id)
        return templates.TemplateResponse(
            request, "chat.html",
            {"messages": session_messages(ctx.conn, session_id)},
        )

    @app.post("/chat/new")
    def chat_new() -> RedirectResponse:
        ctx = context()
        new_session(ctx.conn, ctx.settings.owner_id)
        return RedirectResponse("/chat", status_code=303)

    @app.get("/chat/stream")
    def chat_stream(q: str = "") -> StreamingResponse:
        # Gate off-topic questions, else retrieve + ground + stream (§10). The
        # retrieved ids go first so the client shows its evidence thumbnails
        # before the answer streams in. The finished turn is persisted.
        ctx = context()
        owner_id = ctx.settings.owner_id
        embedder, _ = ctx.settings.build_embedder()
        client, _ = ctx.settings.build_inference_client()
        model = ctx.settings.planner_model or "fake"
        session_id = current_session(ctx.conn, owner_id)

        def events():
            if not is_photo_question(client, model, q):
                yield f"event: sources\ndata: {json.dumps({'ids': []})}\n\n"
                yield f"data: {json.dumps({'delta': OFF_TOPIC_REPLY})}\n\n"
                add_message(ctx.conn, session_id, q, OFF_TOPIC_REPLY, [])
                yield "event: done\ndata: {}\n\n"
                return
            ids = retrieve(ctx.conn, embedder, owner_id, q, k=30)
            messages = chat_messages(q, build_chat_context(ctx.conn, ids))
            yield f"event: sources\ndata: {json.dumps({'ids': ids})}\n\n"
            parts: list[str] = []
            for delta in client.stream(model, messages):
                parts.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            add_message(ctx.conn, session_id, q, "".join(parts), ids)
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/upload", response_class=HTMLResponse)
    def upload_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "upload.html", progress_payload())

    @app.get("/upload/progress", response_class=HTMLResponse)
    def progress(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "_progress.html", progress_payload())

    return app


def app_factory() -> FastAPI:
    from config import get_settings

    return create_app(get_settings())




