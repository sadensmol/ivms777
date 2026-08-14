from albums.base import Organizer
from albums.by_camera import ByCameraOrganizer
from albums.by_date import ByDateOrganizer
from albums.by_place import ByPlaceOrganizer
from albums.memories import MemoriesOrganizer

# Order here is the dropdown order. `date` is the default. Memories (the agentic
# organizer that replaces the old "By similarity") sits second — see
# docs/design.md §11.
ORGANIZERS: dict[str, Organizer] = {
    o.name: o
    for o in (
        ByDateOrganizer(),
        MemoriesOrganizer(),
        ByCameraOrganizer(),
        ByPlaceOrganizer(),
    )
}

DEFAULT_ORGANIZER = "date"


def get_organizer(name: str | None) -> Organizer:
    return ORGANIZERS.get(name or DEFAULT_ORGANIZER, ORGANIZERS[DEFAULT_ORGANIZER])
