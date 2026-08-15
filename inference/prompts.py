"""Caption prompts, keyed per model, all satisfying one JSON schema (§4).

Adding a model is adding a system-prompt entry to `_SYSTEM_BY_MODEL`, not
touching the pipeline. Every model is asked for the same JSON shape.
"""

from inference.client import ChatMessage

# The one shape every caption response must satisfy.
CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "tags": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
    },
    "required": ["caption", "title", "description", "tags"],
}

_DEFAULT_SYSTEM = (
    "You caption personal photographs. Describe only what is visibly present in the "
    "image. Never invent people, places, brands, dates, or events. Reply with ONLY a "
    "JSON object — no prose, no markdown fences."
)

# Per-model system-prompt overrides; the default covers any vision model.
_SYSTEM_BY_MODEL: dict[str, str] = {}


def _user_text(dimensions: list[str]) -> str:
    dims = ", ".join(dimensions)
    return (
        "Return a JSON object describing this photo with these keys:\n"
        '- "caption": one plain sentence describing the photo.\n'
        '- "title": a short title, 3-6 words.\n'
        '- "description": one or two sentences of extra detail.\n'
        f'- "tags": an object mapping any of these dimensions to a list of labels '
        f"that clearly apply: {dims}. Omit a dimension if none apply.\n"
        "Reply with ONLY the JSON object."
    )


def caption_messages(
    model: str, image_data_uri: str, dimensions: list[str]
) -> list[ChatMessage]:
    system = _SYSTEM_BY_MODEL.get(model, _DEFAULT_SYSTEM)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _user_text(dimensions)},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]


_CHAT_SYSTEM = (
    "You answer questions about a personal photo library. Use ONLY the photos "
    "provided below — their captions, tags, and EXIF facts — plus any fact lines "
    "already given in the context (lines starting with 'count:', 'memories:', or "
    "'month(s)/year(s) with photos:'). Never invent photos, people, places, or "
    "dates. For totals or counts, use the count/memories/periods fact given in "
    "the context — never guess or infer a count from the number of photos shown "
    "below. State the number in a natural sentence; do NOT quote the raw fact "
    "line or say \"the fact provided\". Cite every photo you rely on inline as [photo:ID], using the exact ID "
    "from the context. If no photos are provided, or none are relevant, say you "
    "have no photos matching that and stop."
)


_INTENT_SYSTEM = (
    "The user is using an app that stores, organizes, and answers questions about "
    "their own personal photo library. Almost every message is about their "
    "photos or this library — finding or showing them (\"find a photo with a "
    "dog\", \"photos of cars\", \"show beach shots\"), asking what / when / where "
    "/ who / how about them, OR asking about the collection itself: totals and "
    "counts (\"how many photos do I have\", \"how many photos with dogs\"), "
    "memories, albums, or organizers (\"latest memory of Tbilisi\", \"how many "
    "memories\"), dates or periods (\"how many months\", \"what year\"), cameras, "
    "places, or a follow-up about a number the app already showed (\"why do I "
    "see over 800\"). Answer 'yes' for any such request. Answer 'no' ONLY when "
    "the message is clearly unrelated to the photo library — general life "
    "advice, math, coding, world trivia, or chit-chat with no connection to "
    "photos or this app. When unsure, answer 'yes'. Reply with a single word: "
    "yes or no."
)


def intent_messages(question: str) -> list[ChatMessage]:
    """A yes/no gate: is this question about the photo library at all? (§10)"""
    return [
        {"role": "system", "content": _INTENT_SYSTEM},
        {"role": "user", "content": question},
    ]


def chat_messages(question: str, context_block: str) -> list[ChatMessage]:
    """Grounded ask-your-library prompt (§10): answer only from `context_block`,
    cite photos as [photo:ID], say so when nothing relevant was retrieved.
    """
    user = f"Photos:\n{context_block}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "user", "content": user},
    ]
