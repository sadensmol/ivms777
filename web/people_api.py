"""The People pages (design §6, §13, §13.1): the grid `/people` with "Faces to name", and
the leaf `/people/{id}`.

Every action is a `fetch` POST answered in place (people.js) or a JSON `{"detail": …}`
error; none pushes history. Every lookup is scoped to the owner: another owner's person is
404.
"""

import json
import sqlite3
from collections.abc import Callable

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from ingest.people import (
    PersonError,
    PersonHasVoices,
    PersonNameTaken,
    PersonNotFound,
    create_person,
    delete_person,
    get_person,
    groups_to_name,
    list_people,
    person_groups,
    person_photos,
    person_voices,
    rename_person,
)
from web.deps import AppContext

PERSON_NOT_FOUND = "This person does not exist."


def refuse_person(error: PersonError) -> HTTPException:
    """A person error as the HTTP answer that carries its reason: 404 missing, 409 taken
    or still owning a voice, 400 for a bad name."""
    if isinstance(error, PersonNotFound):
        return HTTPException(status_code=404, detail=str(error))
    status = 409 if isinstance(error, PersonNameTaken | PersonHasVoices) else 400
    return HTTPException(status_code=status, detail=str(error))


def newest_shot_first(conn: sqlite3.Connection, photo_ids: list[int]) -> list[int]:
    """The photos newest shot first; a photo with no shot time comes last."""
    return [
        row["id"]
        for row in conn.execute(
            "SELECT id FROM photos WHERE id IN (SELECT value FROM json_each(?))"
            " ORDER BY shot_at IS NULL, shot_at DESC, id DESC",
            (json.dumps(photo_ids),),
        )
    ]


def register(
    app: FastAPI,
    context: Callable[[], AppContext],
    templates: Jinja2Templates,
) -> None:
    def _leaf(request: Request, ctx: AppContext, person_id: int) -> HTMLResponse:
        owner_id = ctx.settings.owner_id
        person = get_person(ctx.conn, owner_id, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=PERSON_NOT_FOUND)
        return templates.TemplateResponse(
            request,
            "person.html",
            {
                "person": person,
                "voices": person_voices(ctx.conn, owner_id, person_id),
                "groups": person_groups(ctx.conn, owner_id, person_id),
                "photo_ids": newest_shot_first(
                    ctx.conn, person_photos(ctx.conn, owner_id, person_id)
                ),
            },
        )

    @app.get("/people", response_class=HTMLResponse)
    def people_page(request: Request) -> HTMLResponse:
        ctx = context()
        owner_id = ctx.settings.owner_id
        return templates.TemplateResponse(
            request,
            "people.html",
            {
                "people": list_people(ctx.conn, owner_id),
                "face_groups": groups_to_name(
                    ctx.conn, owner_id, min_faces=ctx.settings.face_group_min_faces
                ),
            },
        )

    @app.post("/people", response_class=HTMLResponse)
    def people_create(name: str = Form(default="")) -> HTMLResponse:
        ctx = context()
        owner_id = ctx.settings.owner_id
        try:
            create_person(ctx.conn, owner_id, name)
        except PersonError as error:
            raise refuse_person(error) from error
        return HTMLResponse(
            templates.get_template("_people_cards.html").render(
                {"people": list_people(ctx.conn, owner_id)}
            )
        )

    @app.get("/people/{person_id}", response_class=HTMLResponse)
    def person_page(request: Request, person_id: int) -> HTMLResponse:
        return _leaf(request, context(), person_id)

    @app.post("/people/{person_id}/rename", response_class=HTMLResponse)
    def person_rename(
        request: Request, person_id: int, name: str = Form(default="")
    ) -> HTMLResponse:
        ctx = context()
        if get_person(ctx.conn, ctx.settings.owner_id, person_id) is None:
            raise HTTPException(status_code=404, detail=PERSON_NOT_FOUND)
        try:
            rename_person(ctx.conn, ctx.settings.owner_id, person_id, name)
        except PersonError as error:
            raise refuse_person(error) from error
        return _leaf(request, ctx, person_id)

    @app.post("/people/{person_id}/delete")
    def person_delete(person_id: int) -> Response:
        ctx = context()
        try:
            delete_person(ctx.conn, ctx.settings.owner_id, person_id)
        except PersonError as error:
            raise refuse_person(error) from error
        return Response(status_code=204)
