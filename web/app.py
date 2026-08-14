from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from albums.registry import ORGANIZERS, get_organizer
from config import Settings
from ingest.embed import backfill_embeds, embed_handler
from ingest.jobs import STAGES, stage_counts
from ingest.thumbs import thumb_key
from ingest.worker import drain, thumbnail_handler
from search.facets import (
    SIDEBAR_GROUPS,
    SORTABLE,
    build_where,
    facet_counts,
    parse_filters,
)
from search.semantic import search_photos, similar_photos
from web.deps import AppContext, build_context
from web.upload_api import register as register_upload_api

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

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

    def context() -> AppContext:
        return app.state.context

    def drain_now() -> None:
        """Build thumbnails and embeddings for whatever has arrived.

        The `worker` container drains continuously in deployment. Doing it here
        too means a single-container run — and every test — still produces a
        usable, searchable grid without waiting on a poll.
        """
        ctx = context()
        embedder, model_name = ctx.settings.build_embedder()
        backfill_embeds(ctx.conn)
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
            },
        )

    register_upload_api(app, context, drain_now)

    def _semantic_page(ctx: AppContext, query: str, offset: int) -> list:
        embedder, _ = ctx.settings.build_embedder()
        ids = search_photos(ctx.conn, embedder, ctx.settings.owner_id, query, k=200)
        page_ids = ids[offset : offset + ctx.settings.page_size]
        if not page_ids:
            return []
        placeholders = ", ".join("?" for _ in page_ids)
        ranking = " ".join(f"WHEN {pid} THEN {rank}" for rank, pid in enumerate(page_ids))
        rows = ctx.conn.execute(
            SELECT_COLS
            + f" WHERE p.owner_id = ? AND p.id IN ({placeholders})"
            + f" ORDER BY CASE p.id {ranking} END",
            (ctx.settings.owner_id, *page_ids),
        )
        return list(rows)

    def fetch_page(offset: int, params: dict[str, str]) -> list:
        ctx = context()
        query = params.get("q", "").strip()
        if query:
            return _semantic_page(ctx, query, offset)
        where, where_params = build_where(parse_filters(params))
        if params.get("dupes"):
            where += DUPES_ONLY
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

    def _query_string(params: dict[str, str]) -> str:
        keep = {
            k: v for k, v in params.items()
            if k.startswith(("f_", "n_")) or k in ("sort", "q", "dupes")
        }
        return urlencode(keep)

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

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/library")

    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request) -> HTMLResponse:
        params = _params(request)
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

    @app.get("/organize", response_class=HTMLResponse)
    def organize(request: Request, by: str | None = None) -> HTMLResponse:
        ctx = context()
        organizer = get_organizer(by)
        albums = organizer.organize(ctx.conn, ctx.settings.owner_id)
        return templates.TemplateResponse(
            request,
            "organize.html",
            {
                "organizers": [(o.name, o.label) for o in ORGANIZERS.values()],
                "active": organizer.name,
                "albums": albums,
                "total_photos": sum(a.size for a in albums),
            },
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
        similar = similar_photos(ctx.conn, ctx.settings.owner_id, photo_id, k=12)
        return templates.TemplateResponse(
            request,
            "photo.html",
            {
                "photo": photo, "sources": sources, "facets": facets,
                "similar": similar, "wasted_bytes": wasted,
                "embedded": photo["embedding_model"] is not None,
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




