"""The face-group pages (design §13, §13.1): the leaf `/faces/{id}` under the People grid,
its actions, and the face crops.

Every action is a `fetch` POST that answers the group leaf it was taken on (faces.js swaps
its #face-group in place) or a JSON `{"detail": …}` error; none pushes history. Every
lookup is scoped to the owner: another owner's group or face is 404.
"""

import io
from collections.abc import Callable

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from ingest.people import (
    FaceGroup,
    PersonError,
    assign_group,
    create_person_for_group,
    group_detail,
    list_people,
    release_face,
    unassign_group,
)
from ingest.thumbs import thumb_key
from web.deps import AppContext
from web.people_api import refuse_person

GROUP_NOT_FOUND = "This face group does not exist."
FACE_NOT_FOUND = "This face does not exist."
PERSON_NOT_FOUND = "This person does not exist."
# The crop shows a little of the head around the detected box, a share of its side.
CROP_MARGIN = 0.2
CROP_QUALITY = 85


def _crop(detail: bytes, box: tuple[float, float, float, float]) -> bytes:
    """The face `box` (x, y, w, h in 0..1 of the image) cut from a JPEG, with a margin."""
    with Image.open(io.BytesIO(detail)) as image:
        width, height = image.size
        x, y, w, h = box
        left = max(0, round((x - w * CROP_MARGIN) * width))
        top = max(0, round((y - h * CROP_MARGIN) * height))
        right = min(width, max(left + 1, round((x + w * (1 + CROP_MARGIN)) * width)))
        bottom = min(height, max(top + 1, round((y + h * (1 + CROP_MARGIN)) * height)))
        out = io.BytesIO()
        image.crop((left, top, right, bottom)).convert("RGB").save(
            out, format="JPEG", quality=CROP_QUALITY
        )
    return out.getvalue()


def register(
    app: FastAPI,
    context: Callable[[], AppContext],
    templates: Jinja2Templates,
) -> None:
    def _leaf(request: Request, ctx: AppContext, group: FaceGroup) -> HTMLResponse:
        people = list_people(ctx.conn, ctx.settings.owner_id)
        return templates.TemplateResponse(
            request, "face_group.html", {"group": group, "people": people}
        )

    @app.get("/faces/crop/{face_id}")
    def face_crop(face_id: int) -> Response:
        ctx = context()
        row = ctx.conn.execute(
            "SELECT f.box_x, f.box_y, f.box_w, f.box_h, p.content_hash, p.thumb_key"
            " FROM faces f JOIN photos p ON p.id = f.photo_id"
            " WHERE f.id = ? AND p.owner_id = ?",
            (face_id, ctx.settings.owner_id),
        ).fetchone()
        if row is None or row["thumb_key"] is None:
            raise HTTPException(status_code=404, detail=FACE_NOT_FOUND)
        key = thumb_key(row["content_hash"], ctx.settings.thumb_detail_px)
        if not ctx.derived.exists(key):
            raise HTTPException(status_code=404, detail=FACE_NOT_FOUND)
        box = (row["box_x"], row["box_y"], row["box_w"], row["box_h"])
        try:
            crop = _crop(ctx.derived.read(key), box)
        except UnidentifiedImageError as error:
            raise HTTPException(status_code=404, detail=FACE_NOT_FOUND) from error
        return Response(crop, media_type="image/jpeg")

    @app.get("/faces/{group_id}", response_class=HTMLResponse)
    def face_group_page(request: Request, group_id: int) -> HTMLResponse:
        ctx = context()
        group = group_detail(ctx.conn, ctx.settings.owner_id, group_id)
        if group is None:
            raise HTTPException(status_code=404, detail=GROUP_NOT_FOUND)
        return _leaf(request, ctx, group)

    @app.post("/faces/{group_id}/assign", response_class=HTMLResponse)
    def face_group_assign(
        request: Request, group_id: int, person_id: int = Form(default=0)
    ) -> HTMLResponse:
        ctx = context()
        owner_id = ctx.settings.owner_id
        if group_detail(ctx.conn, owner_id, group_id) is None:
            raise HTTPException(status_code=404, detail=GROUP_NOT_FOUND)
        group = assign_group(ctx.conn, owner_id, group_id, person_id)
        if group is None:
            raise HTTPException(status_code=404, detail=PERSON_NOT_FOUND)
        return _leaf(request, ctx, group)

    @app.post("/faces/{group_id}/person", response_class=HTMLResponse)
    def face_group_new_person(
        request: Request, group_id: int, name: str = Form(default="")
    ) -> HTMLResponse:
        ctx = context()
        try:
            group = create_person_for_group(ctx.conn, ctx.settings.owner_id, group_id, name)
        except PersonError as error:
            raise refuse_person(error) from error
        if group is None:
            raise HTTPException(status_code=404, detail=GROUP_NOT_FOUND)
        return _leaf(request, ctx, group)

    @app.post("/faces/face/{face_id}/release", response_class=HTMLResponse)
    def face_release(request: Request, face_id: int) -> HTMLResponse:
        """Answers the group the face left, or its new group when the old one is gone."""
        ctx = context()
        owner_id = ctx.settings.owner_id
        before = ctx.conn.execute(
            "SELECT f.group_id FROM faces f JOIN photos p ON p.id = f.photo_id"
            " WHERE f.id = ? AND p.owner_id = ?",
            (face_id, owner_id),
        ).fetchone()
        if before is None:
            raise HTTPException(status_code=404, detail=FACE_NOT_FOUND)
        released = release_face(ctx.conn, owner_id, face_id)
        if released is None:
            raise HTTPException(status_code=404, detail=FACE_NOT_FOUND)
        left = (
            group_detail(ctx.conn, owner_id, before["group_id"])
            if before["group_id"] is not None
            else None
        )
        return _leaf(request, ctx, left if left is not None else released)

    @app.post("/faces/{group_id}/unassign", response_class=HTMLResponse)
    def face_group_unassign(request: Request, group_id: int) -> HTMLResponse:
        ctx = context()
        group = unassign_group(ctx.conn, ctx.settings.owner_id, group_id)
        if group is None:
            raise HTTPException(status_code=404, detail=GROUP_NOT_FOUND)
        return _leaf(request, ctx, group)
