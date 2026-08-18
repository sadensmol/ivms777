import json
import logging
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

logger = logging.getLogger("ivms777.web")

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from albums.by_date import DEFAULT_GRAIN, GRAIN_LABELS, ByDateOrganizer
from albums.memories import MemoriesOrganizer
from albums.memories_build import build_memories
from albums.memory_store import (
    album_key_for_group,
    current_signature,
    stored_signature,
)
from albums.registry import ORGANIZERS, get_organizer
from chat.agent import (
    agentic_gather,
    direct_answer,
    is_app_topic,
    is_photo_show,
    memories_for_show,
    route,
    search_library,
    search_memories,
)
from chat.citations import CitationFilter
from chat.context import build_context as build_chat_context
from chat.history import (
    add_message,
    cited_ids,
    current_session,
    new_session,
    session_messages,
)
from chat.prefs import get_prefs, set_prefs
from config import Settings
from inference.prompts import (
    GUARDRAIL_REFUSAL,
    agentic_answer_messages,
    chat_messages,
    general_chat_messages,
)
from ingest.folders import enqueue_folder_deletion, list_folders
from ingest.jobs import (
    STAGES,
    format_speed,
    reprocess,
    reprocess_one,
    stage_counts,
    stage_speed,
)
from ingest.pipeline import drain_pass
from ingest.thumbs import thumb_key
from ingest.vocab import load_vocab, seed_tags
from models import catalog as model_catalog
from models import slots as model_slots
from models.resources import snapshot
from search import signals
from search.dates import date_where
from search.facets import (
    SIDEBAR_GROUPS,
    SORTABLE,
    build_where,
    facet_counts,
    parse_filters,
)
from search.planner import plan, spec_to_params
from search.retriever import Query, candidates
from search.semantic import similar_photos, similarity_breakdown
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


def _context_photo_ids(messages: list[dict]) -> set[int]:
    """Photo ids the model was actually shown, from the prompt it is about to answer.

    Read back off the built messages rather than threaded through every branch, so
    it cannot drift out of sync with what was really sent — the whole point is that
    the allow-list matches the prompt exactly.
    """
    text = "".join(
        m["content"] for m in messages if isinstance(m.get("content"), str)
    )
    return {int(pid) for pid in re.findall(r"\[photo:(\d+)\]", text)}


def _has_member_collage(ctx_param: str | None) -> bool:
    # A bounded collection — an Organize album, or a memory shown in chat — renders
    # its WHOLE set as a collage on the leaf, and its members are excluded from the
    # "similar" strip. The library/search/similar collections are unbounded, so they
    # get neither (§13).
    return bool(ctx_param) and ctx_param.startswith(("album:", "chat-memory:"))


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


# How many photos the similar strip holds. Anything the STRICT gates did not
# admit lives behind "Show more" (§9), never mixed in with the real matches.
SIMILAR_K = 12


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="ivms777")
    app.state.context = build_context(settings)
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _static_v(name: str) -> str:
        # Cache-bust static assets by file mtime: the URL changes whenever the file
        # does, so a browser never serves a stale app.css / chat.js after a rebuild.
        try:
            return str(int((STATIC_DIR / name).stat().st_mtime))
        except OSError:
            return "0"

    templates.env.globals["static_v"] = _static_v

    # Load the taxonomy vocabulary once and seed the tag ids so the taxonomy stage
    # can reference them; seeding is idempotent.
    vocab = load_vocab(VOCAB_PATH)
    seed_tags(app.state.context.conn, vocab)

    def context() -> AppContext:
        return app.state.context

    def planner_model() -> str:
        """The model the `planner` slot holds RIGHT NOW (§4.1).

        `settings.planner_model` is only the env override that feeds the resolver —
        it is NOT the answer. Reading it directly made the chat header say
        `gemma4-E2B` (the profile default) while the resident model was the
        `qwen3-vl-8b` the user had picked in the settings popup.
        """
        ctx = context()
        return model_slots.resolve_key(ctx.conn, ctx.settings, "planner")

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
        """Run one ingest pass for whatever has arrived — single-process runs only.

        A bare `uvicorn` run and every test have no `worker`, so draining here is
        what produces a usable, searchable grid without waiting on a poll. The pass
        itself (`drain_pass`) is the single shared pipeline used by both, and it
        runs the GPU-free thumbnail stage even when the embedder/inference backend
        is down (§8).

        Wherever a `worker` container DOES run — every compose profile — this is
        off (`inline_drain`, set False for `app` in compose.yaml): a full inline
        pass would hold the upload-finish request open for the whole library and
        race the worker for the GPU. §5: app serves reads, worker owns writes.
        """
        if not context().settings.inline_drain:
            return
        drain_pass(context(), vocab)

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
        # Candidate generation via the retriever core (§9.2); facet/tag filters narrow the survivors.
        # SigLIP load/evict + GPU serialization are owned by the `models` service
        # conveyor (plan 18); a search is just an HTTP embed call through it.
        embedder, _ = ctx.settings.build_embedder()
        fused = candidates(ctx.conn, embedder, ctx.settings.owner_id, Query(text=query, k=200))
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

    @app.get("/api/resources")
    def resources() -> JSONResponse:
        ctx = context()
        client = ctx.settings.build_models_client()
        selected = model_slots.resolve_keys(ctx.conn, ctx.settings)
        snap = snapshot(
            ctx.conn,
            planner_model=selected["planner"],
            caption_model=selected["caption"],
            embed_model=selected["image_embed"],
            text_embed_model=selected["text_embed"],
            models_client=client,
        )
        # The `models` service holds no DB (§4.1): a restart puts it back on the
        # profile defaults. This poll already runs every ~2 s, so it is where the
        # user's stored selection is put back — no second polling loop, no lease.
        _repush_slots(client, snap.get("slots") or {}, selected)
        return JSONResponse(snap)

    def _repush_slots(client, reported: dict, selected: dict) -> None:
        if not reported or reported == selected:
            return
        if not any(model_catalog.is_switchable(slot, settings.profile) for slot in selected):
            return
        try:
            client.set_slots(selected)
        except Exception:  # noqa: BLE001 - best effort; the next poll tries again
            logger.warning("could not push model slots to the models service")

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
            spec = plan(client, planner_model(), query,
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

    # --- the ⚙ settings popup: model slots (design §4.1, §13) ----------------
    # It is an OVERLAY, not a route: these render a fragment into a <dialog>, so
    # nothing here pushes history or changes the URL (§13.1 stays untouched).
    _SLOT_META = {
        "image_embed": (
            "Image embeddings",
            "Search, zero-shot tags and visual similarity.",
        ),
        "text_embed": (
            "Caption text embeddings",
            "How close two captions are in meaning (“similar photos”).",
        ),
        "caption": ("Captions", "The title and description written for each photo."),
        "planner": ("Planner & chat", "Reads your question and writes the answer."),
    }

    def _settings_view(select: str | None) -> dict:
        ctx = context()
        selected = model_slots.resolve_keys(ctx.conn, ctx.settings)
        downloads: dict[tuple[str, str], dict] = {}
        error = None
        try:
            payload = ctx.settings.build_models_client().catalog()
            downloads = {(e["slot"], e["key"]): e["download"] for e in payload["entries"]}
        except Exception as exc:  # noqa: BLE001 - the popup still lists the catalog
            error = f"the models service is unreachable ({type(exc).__name__})"
        chosen_slot, _, chosen_key = (select or "").partition(":")

        sections = []
        for slot in model_catalog.SLOTS:
            title, blurb = _SLOT_META[slot]
            switchable = model_catalog.is_switchable(slot, ctx.settings.profile)
            entries = []
            for entry in model_catalog.entries_for(slot, ctx.settings.profile):
                download = downloads.get(
                    (slot, entry.key), {"state": "unknown", "bytes": 0, "total": 0, "error": None}
                )
                total = download.get("total") or 0
                # What is on disk already. A slot's entry can be PART downloaded
                # because two slots share a file — `caption` and `planner` differ
                # only by the vision projector — so offering the whole download
                # would overstate the cost by several GB (§4.1).
                have_mb = download.get("bytes", 0) / (1024 * 1024)
                partial = download.get("state") == "absent" and have_mb >= 50
                entries.append(
                    {
                        "key": entry.key,
                        "display": entry.display,
                        "note": entry.note,
                        "size_gb": round(entry.size_mb / 1024, 1),
                        "have_gb": round(have_mb / 1024, 1) if partial else None,
                        "left_gb": round(max(entry.size_mb - have_mb, 0) / 1024, 1)
                        if partial
                        else None,
                        "cost_mb": entry.cost_mb,
                        "cost_measured": entry.cost_measured,
                        "current": selected[slot] == entry.key,
                        "selected": switchable and slot == chosen_slot and entry.key == chosen_key,
                        "download": download,
                        "pct": int(download.get("bytes", 0) * 100 / total) if total else 0,
                    }
                )
            sections.append(
                {
                    "slot": slot,
                    "title": title,
                    "blurb": blurb,
                    "switchable": switchable,
                    "entries": entries,
                    "confirm": _switch_confirm(slot, chosen_slot, chosen_key, switchable),
                }
            )
        return {
            "sections": sections,
            "select": select or "",
            "error": error,
            "polling": any(
                e["download"].get("state") == "downloading"
                for section in sections
                for e in section["entries"]
            ),
        }

    def _switch_confirm(slot, chosen_slot, chosen_key, switchable) -> dict | None:
        """What the switch would cost — shown BEFORE the Switch button is offered."""
        ctx = context()
        if not switchable or slot != chosen_slot or not chosen_key:
            return None
        try:
            preview = model_slots.preview(ctx.conn, ctx.settings, slot, chosen_key)
        except ValueError:
            return None
        if not preview.stages and preview.photos_requeued == 0:
            return {"key": chosen_key, "stages": [], "photos": 0, "vectors_dropped": False}
        return {
            "key": chosen_key,
            "stages": list(preview.stages),
            "photos": preview.photos_requeued,
            "vectors_dropped": preview.vectors_dropped,
        }

    def _settings_response(request: Request, select: str | None) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_settings_models.html", _settings_view(select)
        )

    @app.get("/settings/models", response_class=HTMLResponse)
    def settings_models(request: Request, select: str = "") -> HTMLResponse:
        return _settings_response(request, select)

    @app.post("/settings/models", response_class=HTMLResponse)
    def switch_model_slot(
        request: Request, slot: str = Form(...), key: str = Form(...)
    ) -> HTMLResponse:
        ctx = context()
        try:
            model_slots.switch(ctx.conn, ctx.settings, slot, key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # The DB is the source of truth; a service that refuses or is down is
        # re-pushed by the next resources poll, so this never rolls back the switch.
        try:
            ctx.settings.build_models_client().set_slots(
                model_slots.resolve_keys(ctx.conn, ctx.settings)
            )
        except Exception:  # noqa: BLE001
            logger.warning("model slots stored but not pushed; the next poll retries")
        return _settings_response(request, None)

    @app.post("/settings/models/download", response_class=HTMLResponse)
    def download_model(
        request: Request, slot: str = Form(...), key: str = Form(...)
    ) -> HTMLResponse:
        if not model_catalog.has(slot, key):
            raise HTTPException(status_code=400, detail=f"unknown model: {slot}/{key}")
        try:
            context().settings.build_models_client().download(slot, key)
        except Exception as exc:  # noqa: BLE001 - shown in the fragment, blocks nothing
            logger.warning("download of %s/%s could not be started: %s", slot, key, exc)
        return _settings_response(request, f"{slot}:{key}")

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

    # Model-derived stages a single photo can re-run from its own page. Thumbnails
    # and embeddings are static (the bytes never change), so they are not offered.
    _PHOTO_REPROCESS_STAGES = ("taxonomy", "caption")

    @app.post("/photo/{photo_id}/reprocess")
    def reprocess_photo(
        photo_id: int, stage: str = Form(...), back: str = Form(""),
    ) -> RedirectResponse:
        # Re-run one model stage for THIS photo only (re-tag / re-caption). `back`
        # carries the photo's collection query so the user lands back where they were.
        ctx = context()
        owned = ctx.conn.execute(
            "SELECT 1 FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, ctx.settings.owner_id),
        ).fetchone()
        if owned is None:
            raise HTTPException(status_code=404)
        if stage in _PHOTO_REPROCESS_STAGES:
            reprocess_one(ctx.conn, photo_id, stage)
        target = f"/photo/{photo_id}" + (f"?{back}" if back else "")
        return RedirectResponse(target, status_code=303)

    @app.get("/organize", response_class=HTMLResponse)
    def organize(
        request: Request, by: str | None = None, grain: str | None = None
    ) -> HTMLResponse:
        ctx = context()
        # Return to the organizer/grain last opened, so loading /organize from the
        # nav restores it instead of snapping back to the default date view
        # ("never lose the user's place"). The choice rides a cookie, rewritten
        # below to the resolved organizer so a stale value self-heals.
        if by is None:
            by = request.cookies.get("organize_by")
            if grain is None:
                grain = request.cookies.get("organize_grain")
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
        response = templates.TemplateResponse(
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
        # Remember this choice for the next bare /organize load (one year).
        response.set_cookie("organize_by", organizer.name, max_age=31_536_000,
                            samesite="lax", httponly=True)
        if is_date:
            response.set_cookie("organize_grain", active_grain, max_age=31_536_000,
                                samesite="lax", httponly=True)
        return response

    @app.post("/organize/memories/rebuild")
    def memories_rebuild() -> RedirectResponse:
        # One build at a time per process (§3.2). The build makes many model calls,
        # so it runs on a daemon thread; the thread gets its own DB connection via
        # the thread-local AppContext.conn.
        ctx = context()
        if not app.state.memories_building:
            client = app.state.inference_override or ctx.settings.build_inference_client()[0]
            model = planner_model()
            owner_id = ctx.settings.owner_id
            app.state.memories_building = True
            app.state.memories_progress = {"done": 0, "total": 0}

            def on_progress(done: int, total: int) -> None:
                app.state.memories_progress = {"done": done, "total": total}

            def run() -> None:
                # ctx.conn is thread-local (§5): build the memories on this daemon
                # thread's own connection. Model work goes through the `models`
                # service conveyor (plan 18) — no coordinator here.
                try:
                    build_memories(ctx.conn, client, model, owner_id,
                                   force=True, progress=on_progress,
                                   use_captions=ctx.settings.similar_use_captions)
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
            fused = candidates(ctx.conn, embedder, owner, Query(text=query, k=200))
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
        if ctx_param and ctx_param.startswith("chat-memory:"):
            # A memory shown IN chat, drilled into: the grid IS that memory, so it
            # pages within the memory's photos exactly like the Organize leaf does
            # — same resolution as `album:memories`, only the origin differs (close
            # returns to /chat, see _origin_url). §10, §13.1.
            key = ctx_param.split(":", 1)[1]
            for album in get_organizer(MemoriesOrganizer.name).organize(ctx.conn, owner):
                if album.key == key:
                    return {"ids": album.photo_ids, "title": album.title,
                            "description": album.description}
            # The memory is gone (rebuilt) — fall back to the library.
        if ctx_param and ctx_param.startswith("similar:"):
            # Drilled into a photo FROM another photo's "similar" strip: this layer
            # IS that origin photo's similar set, and it pages within it (§9, §13).
            origin_key, with_loose = _parse_similar_ctx(ctx_param)
            origin = _lookup_photo(ctx.conn, owner, origin_key)
            if origin is not None:
                rows = _similar_strip(ctx, origin["id"], loose=False)
                if with_loose:
                    rows = rows + _similar_strip(ctx, origin["id"], loose=True)
                ids = [r["id"] for r in rows]
                label = origin["ai_title"] or origin["caption"] or f"photo #{origin['id']}"
                return {
                    "ids": ids, "title": f"Similar to {label}", "description": None,
                    "origin_id": origin["id"],  # the photo this layer is similar to
                    # The base photo's own words, so the leaf can show what it is
                    # being compared TO right above the comparison table (§13).
                    "origin_title": origin["ai_title"],
                    "origin_description": origin["ai_description"],
                    "origin_caption": origin["caption"],
                }
            # The origin is gone — fall back to the library.
        if ctx_param == "chat":
            # A cited photo's grid IS the conversation (§13.1): page within the
            # photos this chat session actually cited, and "close" returns to /chat.
            return {
                "ids": _session_cited_ids(ctx.conn, owner),
                "title": "Chat",
                "description": None,
            }
        query = params.get("q", "").strip()
        return {
            "ids": _ordered_ids(params),
            "title": f"Search: {query}" if query else "Library",
            "description": None,
        }

    def _collection_for_photo(
        ctx_param: str | None, params: dict[str, str], photo_id: int
    ) -> tuple[dict, str | None, list[int]]:
        # Resolve the collection a photo pages within, falling back to the whole
        # library if the photo isn't actually a member (a stale link, a "similar"
        # jump). Shared by /photo and /photo/{id}/similar so both agree on exactly
        # which collection's members the similar strip must exclude (§13).
        collection = _photo_context(ctx_param, params)
        ids = collection["ids"]
        if photo_id not in ids:
            ids = _ordered_ids({})
            collection = {"title": "Library", "description": None, "ids": ids}
            ctx_param = None
        return collection, ctx_param, ids

    def _session_cited_ids(conn, owner_id: int) -> list[int]:
        # The photos the current chat session cited, first-appearance order — the
        # ordered "grid" a cited photo pages within (§13.1).
        session_id = current_session(conn, owner_id)
        seen: dict[int, None] = {}
        for row in conn.execute(
            "SELECT sources FROM chat_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ):
            for pid in json.loads(row["sources"] or "[]"):
                seen.setdefault(int(pid), None)
        return list(seen)

    def _lookup_photo(conn, owner_id: int, raw_id: str):
        try:
            return conn.execute(
                "SELECT id, ai_title, ai_description, caption FROM photos"
                " WHERE id = ? AND owner_id = ?",
                (int(raw_id), owner_id),
            ).fetchone()
        except (ValueError, TypeError):
            return None

    def _origin_url(ctx_param: str | None, params: dict[str, str]) -> str:
        # ONE level up from this leaf — where "close" goes (§13.1 rule 4).
        if ctx_param == "chat" or (ctx_param and ctx_param.startswith("chat-memory:")):
            # A memory shown in chat pages within the memory, but "close" goes back
            # UP to the conversation it was surfaced in, not to Organize (§13.1).
            return "/chat"
        if ctx_param and ctx_param.startswith("album:"):
            parts = ctx_param.split(":", 3)
            if len(parts) == 4:
                _, by, grain, _key = parts
                return f"/organize?by={by}" + (f"&grain={grain}" if grain else "")
        library_query = _query_string(params)
        library_url = "/library" + (f"?{library_query}" if library_query else "")
        if ctx_param and ctx_param.startswith("similar:"):
            # The origin photo is the ONE extra level above a similar leaf: close
            # returns to it (carrying the library's own state so ITS close still
            # lands on the filtered grid), and closing there goes up to the grid.
            origin_id, _ = _parse_similar_ctx(ctx_param)
            sep = "&" if library_query else ""
            return f"/photo/{origin_id}?ctx=library{sep}{library_query}"
        return library_url


    def _parse_similar_ctx(ctx_param: str) -> tuple[str, bool]:
        """`similar:<id>` -> (id, False); `similar:<id>:more` -> (id, True).

        The `:more` suffix is set on thumbnails revealed by "Show more", so prev/next
        pages the set the user can actually SEE and never walks into photos that were
        never on screen (§13.1 rule 6). Everything else about the layer is identical,
        so `close` still goes origin → grid.

        A COLON, not a `+`: in a query string `+` decodes to a space, so `similar:67+more`
        arrived as `similar:67 more`, the photo lookup failed, and the layer silently
        fell back to the whole library — losing the user's place, which §13.1 forbids.
        """
        raw = ctx_param.split(":", 1)[1]
        return (raw[: -len(":more")], True) if raw.endswith(":more") else (raw, False)

    def _similar_strip(ctx, photo_id: int, *, loose: bool) -> list[dict]:
        """The similar strip for one photo (§9).

        `loose=False` is the default strip: strict gates, so a result is there because
        something real matched. `loose=True` returns ONLY the extra photos "Show more"
        reveals — the loose pass minus everything the strict pass already showed, so
        the two never overlap and a weak match can never outrank a real one.
        """
        strict = similar_photos(
            ctx.conn, ctx.settings.owner_id, photo_id, k=SIMILAR_K,
            min_cosine=ctx.settings.similar_min_cosine,
            caption_min=ctx.settings.similar_caption_min,
            score_min=ctx.settings.similar_score_min,
            dimension_weights=vocab.dimension_weights,
            use_captions=ctx.settings.similar_use_captions,
        )
        if not loose:
            return strict
        seen = {s["id"] for s in strict}
        wider = similar_photos(
            ctx.conn, ctx.settings.owner_id, photo_id, k=SIMILAR_K + signals.LOOSE_LIMIT,
            dimension_weights=vocab.dimension_weights,
            use_captions=ctx.settings.similar_use_captions,
            loose=True,
        )
        return [s for s in wider if s["id"] not in seen][: signals.LOOSE_LIMIT]

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
        # Page within the collection this photo was opened from (§13) — never leak
        # into another memory/album. `ctx` names the collection; ids are its order.
        params = _params(request)
        ctx_param = params.get("ctx")
        collection, ctx_param, ids = _collection_for_photo(ctx_param, params, photo_id)
        index = ids.index(photo_id) if photo_id in ids else -1
        prev_id = ids[index - 1] if index > 0 else None
        next_id = ids[index + 1] if 0 <= index < len(ids) - 1 else None
        # A memory/album is a bounded set, so the leaf can show the WHOLE collection
        # as a collage (§13). The library/search/similar collections are unbounded
        # or already shown, so they get no collage.
        collection_grid = ids if _has_member_collage(ctx_param) else None

        # Opened from another photo's "similar" strip: explain the match, base vs
        # this photo, strongest facet first (§13).
        similarity = None
        if collection.get("origin_id") and collection["origin_id"] != photo_id:
            similarity = similarity_breakdown(
                ctx.conn, ctx.settings.owner_id, collection["origin_id"], photo_id,
                dimension_weights=vocab.dimension_weights,
                min_cosine=ctx.settings.similar_min_cosine,
                caption_min=ctx.settings.similar_caption_min,
                use_captions=ctx.settings.similar_use_captions,
            )

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
                "wasted_bytes": wasted,
                "similarity": similarity,
                "embedded": photo["embedding_model"] is not None,
                "tags": tags,
                "prev_id": prev_id, "next_id": next_id,
                "ctx_query": urlencode(keep),
                "origin_url": _origin_url(ctx_param, params),
                "collection": collection,
                "collection_grid": collection_grid,
                "position": index + 1 if index >= 0 else None,
                "total": len(ids),
            },
        )

    @app.get("/photo/{photo_id}/similar", response_class=HTMLResponse)
    def photo_similar(request: Request, photo_id: int) -> HTMLResponse:
        # The async counterpart to /photo's placeholder (§9.2, §13): runs the
        # expensive full-library similar scan off the critical path, resolving the
        # SAME collection (and its member-exclusion) as the main route so the two
        # never disagree.
        ctx = context()
        photo = ctx.conn.execute(
            "SELECT id FROM photos WHERE id = ? AND owner_id = ?",
            (photo_id, ctx.settings.owner_id),
        ).fetchone()
        if photo is None:
            raise HTTPException(status_code=404)
        params = _params(request)
        loose = params.get("loose") == "1"
        similar = _similar_strip(ctx, photo_id, loose=loose)
        ctx_param = params.get("ctx")
        _collection, ctx_param, ids = _collection_for_photo(ctx_param, params, photo_id)
        collection_grid = ids if _has_member_collage(ctx_param) else None
        if collection_grid:
            # Don't repeat the collection's own photos in the "similar" strip — the
            # collage already shows them; a member appearing twice is noise (§13).
            members = set(collection_grid)
            similar = [s for s in similar if s["id"] not in members]
        return templates.TemplateResponse(
            request, "_similar.html",
            {"photo": photo, "similar": similar, "loose": loose,
             # Offer "Show more" only when the strict pass did not fill the strip —
             # a full strip already has more than anyone scrolls.
             "can_widen": not loose and len(similar) < SIMILAR_K,
             "ctx_query": urlencode({k: v for k, v in params.items() if k != "loose"})},
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
        last = ctx.conn.execute(
            "SELECT root_label FROM uploads WHERE owner_id = ? AND files_sent > 0"
            " ORDER BY id DESC LIMIT 1",
            (ctx.settings.owner_id,),
        ).fetchone()
        photo_count = ctx.conn.execute(
            "SELECT COUNT(*) AS n FROM photos WHERE owner_id = ?", (ctx.settings.owner_id,)
        ).fetchone()["n"]
        return {
            "stages": [
                (stage, stage_counts(ctx.conn, stage), format_speed(stage_speed(ctx.conn, stage)))
                for stage in STAGES
            ],
            "failures": failures,
            # The folder currently in the library — persisted in `uploads`, so it
            # survives restarts (the file picker cannot remember a selection).
            "last_folder": last["root_label"] if last else None,
            "photo_count": photo_count,
            # The persistent folder list (§3.2c): every uploaded folder, deletable.
            "folders": list_folders(ctx.conn, ctx.settings.owner_id),
        }

    def _memory_card_html(question: str) -> str | None:
        # The Organize memory card(s) for a "show me a memory / all my memories"
        # question, rendered as HTML so chat shows the memory ITSELF (mosaic + title
        # + story), not just prose. A plural/all request renders EVERY memory's card
        # in order; a specific/singular one renders just its card. Reuses the shared
        # `_album_card.html` and links carry ctx=chat-memory:<key>, so drilling in
        # pages within the memory and "close" returns to the conversation (§10,
        # §13.1). Re-derived from the question (deterministic) — no extra stored
        # state, and history re-renders it.
        ctx = context()
        owner = ctx.settings.owner_id
        memories = memories_for_show(ctx.conn, owner, question)
        if not memories:
            return None
        albums = {a.key: a for a in get_organizer(MemoriesOrganizer.name).organize(ctx.conn, owner)}
        cards: list[str] = []
        for memory in memories:
            key = album_key_for_group(ctx.conn, owner, memory["id"])
            album = albums.get(key) if key else None
            if album is None:
                continue
            cards.append(templates.env.get_template("_album_card.html").render(
                album=album, ctx=f"ctx=chat-memory:{key}"
            ))
        return "\n".join(cards) if cards else None

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page(request: Request) -> HTMLResponse:
        # Render the current session's persisted turns as static history (§10). A
        # "show me a memory" turn also re-renders its memory card, so the memory
        # survives reload just like the answer text does.
        ctx = context()
        prefs = get_prefs(ctx.conn, ctx.settings.owner_id)
        session_id = current_session(ctx.conn, ctx.settings.owner_id)
        messages = session_messages(ctx.conn, session_id)
        for m in messages:
            # Memory cards are a direct-DB affordance (§10): re-render them only when
            # "Direct answers" is on, so history matches how the turn was answered.
            m["memory_html"] = _memory_card_html(m["question"]) if prefs.direct_answers else None
        return templates.TemplateResponse(
            request, "chat.html",
            {
                "messages": messages,
                "model": planner_model(),
                "prefs": prefs,
            },
        )

    @app.post("/chat/new")
    def chat_new() -> RedirectResponse:
        ctx = context()
        new_session(ctx.conn, ctx.settings.owner_id)
        return RedirectResponse("/chat", status_code=303)

    @app.post("/chat/prefs")
    def chat_prefs(
        guardrails: str | None = Form(None), direct_answers: str | None = Form(None)
    ) -> RedirectResponse:
        # Global per-owner chat toggles (§10). An unchecked box omits its field, so
        # presence == on. Persisted across every session; read on each stream turn.
        ctx = context()
        set_prefs(
            ctx.conn, ctx.settings.owner_id,
            guardrails=guardrails is not None, direct_answers=direct_answers is not None,
        )
        return RedirectResponse("/chat", status_code=303)

    @app.get("/chat/stream")
    def chat_stream(q: str = "") -> StreamingResponse:
        # Gate off-topic questions, else retrieve + ground + stream (§10).
        # Retrieval gives the model candidates to reason over, but the UI and the
        # stored turn show only the photos the answer actually CITES (§6) — never
        # the loosely-related candidate set, which used to dump 30 thumbnails for
        # a one-photo answer. The finished turn is persisted.
        ctx = context()
        owner_id = ctx.settings.owner_id
        client, _ = ctx.settings.build_inference_client()
        model = planner_model()
        session_id = current_session(ctx.conn, owner_id)

        # "Thinking" is the wait BEFORE the answer starts — from the request landing
        # here to the first token going out. It covers routing, retrieval and, on the
        # Jetson, a model swap (a ~9 s llama-server reload, §8.1), which is the part
        # the user actually waits through. Streaming is deliberately NOT counted:
        # once tokens are arriving the user is reading, not waiting, and a long
        # answer is not a slow one.
        turn_started = time.monotonic()
        thinking_ms: int | None = None

        def _turn_ms() -> int:
            return int((time.monotonic() - turn_started) * 1000)

        def _done(**stats) -> str:
            # The done event carries the model and, for a generated answer, its
            # measured decode speed (tokens/sec). The wait is reported earlier, by
            # the `thinking` event — by `done` the user has read the whole answer.
            return "event: done\ndata: " + json.dumps({"model": model, **stats}) + "\n\n"

        prefs = get_prefs(ctx.conn, owner_id)

        def events():
            # Any failure below (most often the models service / llama-server being
            # down — see §8.1) is streamed to the UI as a plain error turn, never a
            # silent "(no answer)" bubble, and the partial answer + note is persisted.
            parts: list[str] = []

            def _thinking() -> str:
                """Stop the wait clock and announce it — emitted ONCE, immediately
                before the first text of the turn, so the UI can print "thought for
                12.4 s" ABOVE the answer as it starts rather than after it ends."""
                nonlocal thinking_ms
                if thinking_ms is None:
                    thinking_ms = _turn_ms()
                return (
                    "event: thinking\ndata: "
                    + json.dumps({"elapsed_ms": thinking_ms})
                    + "\n\n"
                )

            try:
                # The direct-DB layer (§10) — ONLY when "Direct answers" is on (default).
                # Every question the DB can answer unambiguously — counts, memory
                # show/list, periods — is answered straight from SQLite with NO model,
                # replies instantly even while ingest captions (§8.1), and NEVER reaches
                # the weak planner. A "show me a memory" turn also streams the memory
                # card. When the toggle is OFF, this whole step is skipped and the
                # fully-agentic loop below handles those questions with real count tools.
                if prefs.direct_answers:
                    quick = direct_answer(ctx.conn, owner_id, q)
                    if quick is not None:
                        yield _thinking()
                        yield f"data: {json.dumps({'delta': quick})}\n\n"
                        add_message(ctx.conn, session_id, q, quick, [], thinking_ms)
                        card = _memory_card_html(q)
                        if card is not None:
                            yield "event: memory\ndata: " + json.dumps({"html": card}) + "\n\n"
                        yield _done()
                        return
                # NEVER wake SigLIP until something has established this question is
                # about the photos (§10). gemma (3800) and siglip (3400) cannot both
                # fit the 5000 MB jetson budget (§8.1), so loading one evicts the
                # other, and bringing gemma back respawns `llama-server` with ~3 GB
                # of weights — ~9 s, measured. gemma is also the model most likely to
                # be ALREADY resident, so a speculative SigLIP load is not free: it
                # costs an eviction now and a reload after, and "how many planets are
                # in the solar system" pays both for a search it never needed.
                #
                # Guardrails ON establishes it BY POLICY — an off-topic message is
                # refused, so anything answered is about the library. Only then is
                # retrieving up front the cheap order (siglip → gemma, ONE swap).
                #
                # Guardrails OFF means only the model can decide, so gemma goes
                # first: a general question then costs NO swap at all, and a photo
                # question pays one only when it genuinely needs photos.
                primed_ids: list[int] | None = None
                if prefs.guardrails:
                    embedder, _ = ctx.settings.build_embedder()
                    primed_ids = search_library(ctx.conn, embedder, owner_id, q, k=12)

                decision = None
                if prefs.guardrails or prefs.direct_answers:
                    decision = route(client, model, q)
                # Guardrails ON (§10): a message the router sends to `none` is off-topic —
                # refuse with a fixed redirect, no generation, no search. Opt-in; off by
                # default so chat stays a general assistant. App-specific questions (counts,
                # memories, albums, uploads, tags…) are NEVER off-topic — `is_app_topic`
                # overrides the weak router so app functionality is never turned away.
                if prefs.guardrails and decision["tool"] == "none" and not is_app_topic(q):
                    yield _thinking()
                    yield f"data: {json.dumps({'delta': GUARDRAIL_REFUSAL})}\n\n"
                    add_message(ctx.conn, session_id, q, GUARDRAIL_REFUSAL, [], thinking_ms)
                    yield _done()
                    return
                if prefs.direct_answers:
                    if decision["tool"] == "search_library":
                        if primed_ids is not None:
                            # Guardrails ON: retrieval already ran BEFORE gemma, so
                            # reuse it — embedding again here would evict gemma and
                            # pay a ~9 s reload to bring it straight back.
                            ids = primed_ids
                        else:
                            embedder, _ = ctx.settings.build_embedder()
                            ids = search_library(
                                ctx.conn, embedder, owner_id, decision["query"] or q, k=12
                            )
                        messages = chat_messages(q, build_chat_context(ctx.conn, ids))
                    elif decision["tool"] == "search_memories":
                        mems = search_memories(ctx.conn, owner_id, decision["query"] or q)
                        block = "memories: " + (
                            "; ".join(f"{m['name']} ({m['size']} photos)" for m in mems)
                            or "none"
                        )
                        messages = chat_messages(q, block)
                    else:  # general knowledge / chit-chat — answer directly, no search
                        messages = general_chat_messages(q)
                else:
                    # Fully-agentic RAG (§10): "Direct answers" off, so the model calls
                    # REAL tools (count_photos / list_memories / count_periods / search) to
                    # gather facts + candidate photos, then a grounded answer streams from
                    # those facts. Memory-show is prose here (no card). Nothing gathered ->
                    # answer from general knowledge.
                    embedder, _ = ctx.settings.build_embedder()
                    # `prime` only under guardrails — see the note above the router.
                    # With guardrails off, the loop's own STEP 1 decides whether this
                    # is even a photo question, and gemma (usually already resident)
                    # answers a general one without ever waking SigLIP.
                    # Prime when we KNOW the question is about photos, without a
                    # model: guardrails guarantees it by policy, and "find/show me
                    # photos of X" says it outright. Both license waking SigLIP
                    # first; a general question still never loads it.
                    block, grounded = agentic_gather(
                        ctx.conn, embedder, client, model, owner_id, q,
                        prime=prefs.guardrails or is_photo_show(q),
                        prime_as_evidence=is_photo_show(q),
                    )
                    messages = (
                        agentic_answer_messages(q, block) if grounded
                        else general_chat_messages(q)
                    )
                # A citation is a claim about the user's own library, so it may only
                # name a photo the model was actually GIVEN. The prompt says so in
                # capitals and gemma4-E2B still emitted `[photo:1]` off a
                # `count: 1 photo(s)` line — reading a quantity as an id. Enforced
                # here instead, over the stream, because the answer arrives a
                # character at a time and a bad citation must never render at all.
                citations = CitationFilter(_context_photo_ids(messages))
                started_at = None
                chunks = 0
                for delta in client.stream(model, messages, temperature=0):
                    if started_at is None:  # clock starts at the first token, not the wait before it
                        started_at = time.monotonic()
                    chunks += 1
                    clean = citations.feed(delta)
                    if clean:
                        parts.append(clean)
                        # The wait ends at the first text the user SEES — a citation
                        # filtered out of the leading chunks is not an answer yet.
                        if thinking_ms is None:
                            yield _thinking()
                        yield f"data: {json.dumps({'delta': clean})}\n\n"
                tail = citations.flush()
                if tail:
                    parts.append(tail)
                    if thinking_ms is None:  # the whole answer was held back until now
                        yield _thinking()
                    yield f"data: {json.dumps({'delta': tail})}\n\n"
                if citations.dropped:
                    logger.warning(
                        "dropped fabricated citation(s) %s for q=%r", citations.dropped, q
                    )
                # Chunk count is a close proxy for tokens with an OpenAI-style stream
                # (one token per chunk); good enough for a live decode-speed readout.
                elapsed = (time.monotonic() - started_at) if started_at else 0.0
                tok_per_sec = round(chunks / elapsed, 1) if elapsed > 0 else None
                answer = "".join(parts)
                add_message(ctx.conn, session_id, q, answer, cited_ids(answer), thinking_ms)
                yield _done(tok_per_sec=tok_per_sec, tokens=chunks)
            except Exception:
                logger.exception("chat stream failed for q=%r", q)
                note = (
                    "⚠️ The assistant is unavailable right now — the model service "
                    "could not be reached. Please try again in a moment."
                )
                # A break if partial text already streamed, so the note stays legible.
                delta = ("\n\n" + note) if parts else note
                if thinking_ms is None:
                    yield _thinking()
                yield f"data: {json.dumps({'delta': delta})}\n\n"
                add_message(ctx.conn, session_id, q, "".join(parts) + delta, [], thinking_ms)
                yield _done(error=True)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/upload", response_class=HTMLResponse)
    def upload_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "upload.html", progress_payload())

    @app.get("/upload/progress", response_class=HTMLResponse)
    def progress(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "_progress.html", progress_payload())

    @app.post("/upload/folder/delete")
    def delete_folder(root_label: str = Form(...)) -> RedirectResponse:
        # Remove a folder from the LIBRARY (never the source folder on disk, §3.2c):
        # enqueue the deletion, the worker cascades it away. Returns to /upload.
        ctx = context()
        enqueue_folder_deletion(ctx.conn, ctx.settings.owner_id, root_label)
        return RedirectResponse("/upload", status_code=303)

    return app


def app_factory() -> FastAPI:
    from config import get_settings

    return create_app(get_settings())




