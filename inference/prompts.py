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
    },
    "required": ["caption", "title", "description"],
}

_DEFAULT_SYSTEM = (
    "You caption personal photographs. Describe only what is visibly present in the "
    "image. Never invent people, places, brands, dates, or events. Reply with ONLY a "
    "JSON object — no prose, no markdown fences."
)

# Per-model system-prompt overrides; the default covers any vision model.
_SYSTEM_BY_MODEL: dict[str, str] = {}


def _user_text() -> str:
    return (
        "Return a JSON object describing this photo with these keys:\n"
        '- "caption": one plain sentence describing the photo.\n'
        '- "title": a short title, 3-6 words.\n'
        '- "description": one or two sentences of extra detail.\n'
        "Reply with ONLY the JSON object."
    )


def caption_messages(model: str, image_data_uri: str) -> list[ChatMessage]:
    system = _SYSTEM_BY_MODEL.get(model, _DEFAULT_SYSTEM)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _user_text()},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]


_CHAT_SYSTEM = (
    "You answer questions about a personal photo library STRICTLY from the context "
    "below — photo captions, tags, and EXIF facts, plus any fact lines already given "
    "(lines starting with 'count:', 'memories:', or 'month(s)/year(s) with photos:'). "
    "A photo matches the question ONLY IF its caption or tags EXPLICITLY contain what "
    "is asked. NEVER infer, assume, or invent a subject, person, animal, place, object, "
    "or date that is not written in that photo's caption/tags — if the caption does not "
    "mention it, the photo does NOT match, so leave it out. Do not describe a photo "
    "beyond what its caption says. Cite EVERY matching photo, and ONLY matching photos, "
    "inline as [photo:ID] using the exact ID from the context. "
    "For totals or counts, use the count/memories/periods fact given in the context — "
    "never guess or infer a count from the number of photos shown. State the number in "
    "a natural sentence; do NOT quote the raw fact line or say \"the fact provided\". "
    "If no photos are provided, or none match, say you have no photos matching that and stop."
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


_GENERAL_SYSTEM = (
    "You are a helpful assistant inside a personal photo app. Answer the user's "
    "question directly and concisely from your own knowledge. This message was routed "
    "as NOT about the user's own photos or memories, so do not mention searching their "
    "library or say you have no matching photos — just answer the question."
)


def general_chat_messages(question: str) -> list[ChatMessage]:
    """Non-photo chat (plan 17): the message was routed as general knowledge / chit-chat,
    so answer it directly — chat is not limited to photo questions."""
    return [
        {"role": "system", "content": _GENERAL_SYSTEM},
        {"role": "user", "content": question},
    ]


# Guardrails ON (§10): a fixed, model-free refusal for a question the router
# classified as NOT about the user's photos/memories. Streamed verbatim — no lease,
# no generation, no search.
GUARDRAIL_REFUSAL = (
    "I can only answer questions about your photos and this app — finding photos, "
    "memories, counts, or how the library works. Ask me about those."
)


# Direct answers OFF (§10): the fully-agentic loop already gathered REAL facts
# (count/memories/periods lines) and candidate photos; this final call turns them
# into the streamed answer. Unlike `_CHAT_SYSTEM` it must state a count even when no
# photo is cited (a "how many" answer has a number, not a thumbnail).
_AGENTIC_ANSWER_SYSTEM = (
    "You answer a question about a personal photo library using ONLY the gathered "
    "data below. It has TWO kinds of lines and you MUST treat them differently.\n"
    "FACT lines — 'count: N ...', 'memories: ...', 'month(s)/year(s) with photos: N' "
    "— are REAL values already computed for you. State the number or the list in a "
    "natural sentence. A count is a QUANTITY, never a photo id: 'count: 1' means one "
    "photo EXISTS, it is NOT photo number 1.\n"
    "PHOTO blocks — each starts with its OWN '[photo:ID]' tag, then that photo's "
    "visual match / date / caption / tags — are the ONLY things you may cite. Copy a "
    "block's exact '[photo:ID]' tag.\n"
    "A 'visual match: rank N' line means an IMAGE model compared the question "
    "against the actual PIXELS and ranked these photos by how well they match, best "
    "first. It SAW the photo; you have not. The caption is one short sentence and "
    "routinely omits things that are plainly visible — clothing colour, small "
    "objects, background detail. So a caption that does not repeat the asked-for "
    "word is NOT evidence the photo lacks it. When the blocks carry visual-match "
    "ranks, present the top ones IN ORDER as the closest matches, and cite them. Do "
    "not silently drop rank 1 because its caption is worded differently.\n"
    "Describe a photo only from its caption; never invent detail. If the best match "
    "still looks unrelated, say these are the closest and may not be exact — do not "
    "claim there are none when photos were ranked for you.\n"
    "HARD RULE: a [photo:...] citation may come ONLY from a PHOTO block above. If the "
    "gathered data has NO photo blocks, your answer MUST NOT contain '[photo:' at "
    "all — a bare count line is not a photo.\n"
    "Never invent a subject, id, number, or date not written below. If nothing "
    "answers the question, say so plainly.\n"
    "Examples:\n"
    "Gathered: 'count: 1 photo(s) matching \"dog\"' → 'You have 1 photo of a dog.' "
    "(NO citation — there is no photo block)\n"
    "Gathered: 'count: 206 photo(s) in the library' → 'You have 206 photos.' (no "
    "citation)\n"
    "Gathered: '[photo:178] caption: A dog on a rooftop' → 'Here is your dog photo: "
    "[photo:178].'"
)


def agentic_answer_messages(question: str, gathered_block: str) -> list[ChatMessage]:
    """Final grounded answer for the fully-agentic direct-OFF path (§10): answer from
    the gathered fact lines + photo context, cite photos as [photo:ID]."""
    return [
        {"role": "system", "content": _AGENTIC_ANSWER_SYSTEM},
        {"role": "user", "content": f"Gathered:\n{gathered_block}\n\nQuestion: {question}"},
    ]
