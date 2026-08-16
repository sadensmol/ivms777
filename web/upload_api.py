import logging
from collections.abc import Callable
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger("ivms777.upload")

from ingest.receive import (
    HashMismatchError,
    UnreadableImageError,
    known_hashes,
    link_existing,
    receive,
)
from web.deps import AppContext

# Refuse an upload that would leave the disk this close to full. Failing at the
# door with a clear message beats failing halfway through a 5,000-photo library.
FREE_SPACE_FLOOR = 512 * 1024 * 1024


class StartRequest(BaseModel):
    root_label: str = Field(default="photos", max_length=200)


class ProbeFile(BaseModel):
    hash: str = Field(min_length=64, max_length=64)
    rel_path: str = Field(min_length=1, max_length=1024)


class ProbeRequest(BaseModel):
    upload_id: int
    files: list[ProbeFile]


class FinishRequest(BaseModel):
    upload_id: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register(
    app: FastAPI,
    context: Callable[[], AppContext],
    drain_now: Callable[[], None],
) -> None:
    def _require_upload(ctx: AppContext, upload_id: int) -> None:
        row = ctx.conn.execute(
            "SELECT id FROM uploads WHERE id = ? AND owner_id = ?",
            (upload_id, ctx.settings.owner_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="unknown upload")

    @app.post("/api/upload/start")
    def start(payload: StartRequest) -> dict:
        ctx = context()
        free = ctx.originals.free_bytes()
        if free < FREE_SPACE_FLOOR:
            raise HTTPException(
                status_code=507,
                detail=f"only {free // (1024 * 1024)} MB free on the server",
            )
        cursor = ctx.conn.execute(
            "INSERT INTO uploads(owner_id, root_label, started_at) VALUES (?, ?, ?)",
            (ctx.settings.owner_id, payload.root_label, _now()),
        )
        return {"upload_id": int(cursor.lastrowid)}

    @app.post("/api/upload/probe")
    def probe(payload: ProbeRequest) -> dict:
        ctx = context()
        _require_upload(ctx, payload.upload_id)
        owner_id = ctx.settings.owner_id
        held = known_hashes(ctx.conn, owner_id, [item.hash for item in payload.files])

        needed: list[str] = []
        for item in payload.files:
            if item.hash in held:
                # Bytes we already have, at a path we may not. Record the path now;
                # the client will not send this file at all.
                link_existing(
                    ctx.conn,
                    owner_id=owner_id,
                    upload_id=payload.upload_id,
                    rel_path=item.rel_path,
                    content_hash=item.hash,
                )
            elif item.hash not in needed:
                needed.append(item.hash)

        ctx.conn.execute(
            "UPDATE uploads SET files_offered = files_offered + ? WHERE id = ?",
            (len(payload.files), payload.upload_id),
        )
        return {"needed": needed}

    @app.post("/api/upload/file")
    def upload_file(
        upload_id: int = Form(...),
        rel_path: str = Form(...),
        content_hash: str = Form(...),
        file: UploadFile = File(...),  # noqa: B008 - FastAPI's parameter-in-default idiom
    ) -> dict:
        ctx = context()
        _require_upload(ctx, upload_id)
        data = file.file.read()
        try:
            result = receive(
                ctx.conn,
                ctx.originals,
                owner_id=ctx.settings.owner_id,
                upload_id=upload_id,
                rel_path=rel_path,
                declared_hash=content_hash,
                data=data,
            )
        except HashMismatchError as error:
            ctx.conn.execute(
                "UPDATE uploads SET files_failed = files_failed + 1 WHERE id = ?",
                (upload_id,),
            )
            raise HTTPException(status_code=422, detail=str(error)) from error
        except UnreadableImageError as error:
            ctx.conn.execute(
                "UPDATE uploads SET files_failed = files_failed + 1 WHERE id = ?",
                (upload_id,),
            )
            raise HTTPException(status_code=415, detail=str(error)) from error

        ctx.conn.execute(
            "UPDATE uploads SET files_sent = files_sent + 1 WHERE id = ?", (upload_id,)
        )
        return {
            "status": "stored" if result.created else "linked",
            "photo_id": result.photo_id,
        }

    @app.post("/api/upload/finish")
    def finish(payload: FinishRequest) -> dict:
        ctx = context()
        _require_upload(ctx, payload.upload_id)
        ctx.conn.execute(
            "UPDATE uploads SET finished_at = ? WHERE id = ?", (_now(), payload.upload_id)
        )
        # The upload RECEIPT is complete once the bytes are stored and the jobs are
        # queued — processing is the `worker`'s job (§5, §8). Drain inline too, as a
        # convenience so a single-container / mac run produces a usable grid without
        # waiting on a poll — but BEST-EFFORT: a drain failure (e.g. the app
        # container cannot init CUDA on jetson) must never fail the upload, which
        # already succeeded. The jobs stay pending and visible in the UI, and the
        # worker container drains them regardless.
        try:
            drain_now()
        except Exception:  # a convenience drain must never fail the receipt
            logger.exception(
                "inline drain after upload %s failed; the worker will process it",
                payload.upload_id,
            )
        row = ctx.conn.execute(
            "SELECT files_offered, files_sent, files_failed FROM uploads WHERE id = ?",
            (payload.upload_id,),
        ).fetchone()
        return {
            "offered": row["files_offered"],
            "sent": row["files_sent"],
            "failed": row["files_failed"],
        }
