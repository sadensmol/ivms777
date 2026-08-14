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
    "provided below — their captions, tags, and EXIF facts. Never invent photos, "
    "people, places, or dates. Cite every photo you rely on inline as [photo:ID], "
    "using the exact ID from the context. If no photos are provided, or none are "
    "relevant, say you have no photos matching that and stop."
)


_INTENT_SYSTEM = (
    "The user is using an app that searches and answers questions about their own "
    "photo library. Almost every message is about their photos — finding or "
    "showing them (\"find a photo with a dog\", \"photos of cars\", \"show beach "
    "shots\") or asking what / when / where / who / how about them. Answer 'yes' "
    "for any such request. Answer 'no' ONLY when the message is clearly unrelated "
    "to the photo library — general life advice, math, coding, trivia, or "
    "chit-chat. When unsure, answer 'yes'. Reply with a single word: yes or no."
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
