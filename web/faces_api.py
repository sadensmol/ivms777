"""The face-group pages (design §13, §13.1): the leaf `/faces/{id}` under the Voices grid,
its actions, and the face crops.

Every action is a `fetch` that answers a fragment or a JSON `{"detail": …}` error; none
pushes history. Every lookup is scoped to the owner: another owner's group or face is 404.
"""

from collections.abc import Callable

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from ingest.people import (
    assign_group,
    create_person_for_group,
    group_detail,
    release_face,
    unassign_group,
)
from web.deps import AppContext

GROUP_NOT_FOUND = "This face group does not exist."
FACE_NOT_FOUND = "This face does not exist."


def register(
    app: FastAPI,
    context: Callable[[], AppContext],
    templates: Jinja2Templates,
) -> None:
    @app.get("/faces/crop/{face_id}")
    def face_crop(face_id: int) -> Response:
        raise HTTPException(status_code=404, detail=FACE_NOT_FOUND)

    @app.get("/faces/{group_id}", response_class=HTMLResponse)
    def face_group_page(request: Request, group_id: int) -> HTMLResponse:
        ctx = context()
        if group_detail(ctx.conn, ctx.settings.owner_id, group_id) is None:
            raise HTTPException(status_code=404, detail=GROUP_NOT_FOUND)
        return HTMLResponse("")

    @app.post("/faces/{group_id}/assign", response_class=HTMLResponse)
    def face_group_assign(group_id: int, person_id: int = Form(default=0)) -> HTMLResponse:
        ctx = context()
        if assign_group(ctx.conn, ctx.settings.owner_id, group_id, person_id) is None:
            raise HTTPException(status_code=404, detail=GROUP_NOT_FOUND)
        return HTMLResponse("")

    @app.post("/faces/{group_id}/person", response_class=HTMLResponse)
    def face_group_new_person(group_id: int, name: str = Form(default="")) -> HTMLResponse:
        ctx = context()
        if create_person_for_group(ctx.conn, ctx.settings.owner_id, group_id, name) is None:
            raise HTTPException(status_code=404, detail=GROUP_NOT_FOUND)
        return HTMLResponse("")

    @app.post("/faces/face/{face_id}/release", response_class=HTMLResponse)
    def face_release(face_id: int) -> HTMLResponse:
        ctx = context()
        if release_face(ctx.conn, ctx.settings.owner_id, face_id) is None:
            raise HTTPException(status_code=404, detail=FACE_NOT_FOUND)
        return HTMLResponse("")

    @app.post("/faces/{group_id}/unassign", response_class=HTMLResponse)
    def face_group_unassign(group_id: int) -> HTMLResponse:
        ctx = context()
        if unassign_group(ctx.conn, ctx.settings.owner_id, group_id) is None:
            raise HTTPException(status_code=404, detail=GROUP_NOT_FOUND)
        return HTMLResponse("")
