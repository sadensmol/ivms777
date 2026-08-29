"""The story pass (design §11.1) — the ONE model call in the whole film pipeline.

The director owns craft, which is measurable. This owns story, which is taste: what
the film says, how it says it, and what it should sound like. It is one call, made
offline, and its output is text — so a failure costs the words, never the film.

The shot lines exist because the raw captions cannot carry a film: `photos.ai_title`
is object labelling ("Spray Bottle in Hand"), accurate and inert. The model is given
those labels plus the memory's own description and asked for something a person would
write under a photograph.
"""

import json
import logging
import re
from dataclasses import dataclass, replace

from feed.director import Edit
from feed.language import DEFAULT

logger = logging.getLogger(__name__)

# The narration carries the film; the on-screen text does not. An earlier version
# asked for a line per shot at <= 6 words and 3-5 sentences of narration — which
# specifies CAPTIONS, not a story, and read exactly that way ("Two people outdoors").
#
# The length SCALES WITH THE PHOTOGRAPHS, because the narration decides the runtime
# (§11.1) and the runtime is shared out among the shots. A fixed 200-400 words over a
# three-photo memory is 60 seconds of speech across three pictures — a 40-second hold
# on a single still, which is not a long take, it is a stall.
WORDS_PER_SHOT = (14, 22)
# ROOM PER NOTE, on top of the photographs' share. A note is not free: it is a fact the
# story has to carry, in its own place, with enough words around it to read as prose
# rather than as an inserted line. Without this the budget counted only photographs — a
# three-photograph memory was told to write 55-75 words TOTAL while two notes worth 32
# words of content were "not optional", so the model obeyed the cap and dropped one.
# That was the whole reason a viewer's sentence kept vanishing from their own film.
WORDS_PER_NOTE = 30
# Floor: below this there is no story, only a caption. Ceiling: past this a "short
# film" is a podcast, however many photographs are available.
NARRATION_BOUNDS = (55, 400)


def narration_words(shot_count: int, note_count: int = 0) -> tuple[int, int]:
    """The word band to ask for, given the photographs to fill AND the notes to carry.

    The CEILING is hard — a short film is not a podcast however many photographs are
    available — so a memory big enough to hit it gets a narrower band rather than one
    that overshoots. The notes' room is added ON TOP of the ceiling on purpose: the
    viewer's own words are the one thing that is never traded away for brevity.
    """
    per_low, per_high = WORDS_PER_SHOT
    floor, ceiling = NARRATION_BOUNDS
    high = max(floor + 20, min(ceiling, shot_count * per_high))
    low = max(floor, min(high - 20, shot_count * per_low))
    room = note_count * WORDS_PER_NOTE
    return low + room, high + room


def _note_count(notes: str) -> int:
    """How many notes the prompt carries. Counts OUR OWN position markers, written by
    `notes.notes_for_prompt` — not the viewer's words."""
    return sum(1 for line in notes.splitlines() if line.strip().startswith("[at "))


# How many shots get a title card. Sparse on purpose: a card every shot competes with
# the photograph, and 40 of them is a caption track, not a film.
MAX_TITLE_CARDS = 6

# What the model is allowed to WRITE. A 200-400 word narration plus cards does not fit
# in 900.
MAX_OUTPUT_TOKENS = 2000

# --- fitting the prompt to the server (§11.1) ---------------------------------------
# llama-server does NOT truncate a prompt it cannot fit: it rejects the whole call
# ("request (6306 tokens) exceeds the available context size (4096 tokens)"), the story
# comes back empty, and the film fails for want of narration. The photo listing is the
# only part that grows without limit — one brief per photograph, ~130 tokens each, so a
# 40-shot memory spent 5 258 of its 6 297 tokens on it — so that is what gets cut to fit
# whatever context the server actually serves.
#
# Tokens are ESTIMATED from characters: `app` is a thin client (§5.1) and has no
# tokenizer. Measured on this very prompt, 22 424 chars = 6 297 tokens (ru) and 22 729 =
# 6 388 (en), i.e. ~3.5 chars/token; 3.0 is used here so the estimate errs towards
# trimming a little too much rather than one token too few.
CHARS_PER_TOKEN = 3.0
# Room kept for the answer plus the chat template's own tokens — the server counts the
# request as prompt + what it is asked to generate.
RESERVED_TOKENS = MAX_OUTPUT_TOKENS + 256
# Below this a brief says nothing about its photograph, and a story written from
# nothing is worse than a shorter listing. A memory whose listing cannot fit even at
# this width is over the context however it is cut — that is a floor, not a promise,
# and it is logged.
MIN_BRIEF_CHARS = 80
# `"40. "` plus the clip marker and the newline — charged per line so the estimate is
# the WHOLE listing rather than just its briefs.
LINE_OVERHEAD_CHARS = 7
_MUSIC_TAIL = (
    "Atmospheric cinematic instrumental, slow, warm pads and soft piano, gentle, "
    "unobtrusive, no vocals, no drums."
)

STORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        # THE SPINE, and the film's FIRST CITIZEN. An ordered list of `{text, shot}`:
        # a passage of the STORY, and the photograph shown while it is spoken.
        #
        # The story leads and the photographs illustrate it — not the other way round.
        # So a passage may reuse a photograph already seen, and a photograph the story
        # has no use for is simply not shown. A list with one entry per photo is a
        # caption track; this is a film.
        #
        # Writing it this way is also what makes the film SYNCHRONISED by construction:
        # each passage is spoken as its own audio segment, so the shot begins exactly
        # when its words do. The alternative — prose, then guessing which sentence goes
        # with which picture — is where sync died (§11.1).
        "script": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "shot": {"type": "integer"},
                    # WHICH SENTENCE of the story this is — 1-based, in the order they
                    # were given. The words themselves are not the model's to write
                    # here: the story was written in its own pass and is not up for
                    # revision by a matching call (§11.1).
                    "sentence": {"type": "integer"},
                    # REQUIRED, but an EMPTY STRING means "the film's narrator" — the
                    # model must decide for every passage rather than omit the key,
                    # which is what an optional field gets you under a strict decoder.
                    "voice": {"type": "string"},
                },
                "required": ["shot", "sentence", "voice"],
            },
        },
        "narration": {"type": "string"},
        # The film's own narrator, chosen from the speakers offered in the prompt.
        "narrator": {"type": "string"},
        # `[{"shot": <1-based index>, "text": "..."}]` — only the beats worth titling,
        # NOT one per shot.
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"shot": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["shot", "text"],
            },
        },
        "music_prompt": {"type": "string"},
        # WHERE THE MUSIC CHANGES (§11.1). `[{"from": <1-based passage>, "brief": "…"}]`
        # — two to four moments, each naming what the score becomes there. The model
        # that wrote the passages is the one that knows where the film turns; a single
        # `music_prompt` can only describe the average of the whole memory, which is
        # what made every film sound like the same bed under different pictures.
        "score": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"from": {"type": "integer"}, "brief": {"type": "string"}},
                "required": ["from", "brief"],
            },
        },
    },
    # `narrator` IS REQUIRED. It was optional, and a strict-schema decoder emits only
    # what it must: gemma returned no narrator on every single film, so every film fell
    # back to the mood's default speaker and a viewer's "[суровый мужской голос]" was
    # read by a woman with nothing saying why (measured — `decisions.cast` showed
    # `narrator_asked=""` on a ru film whose notes asked for a voice).
    "required": ["script", "cards", "music_prompt", "narrator"],
}

# More changes than this is not a score, it is a playlist — and each one costs a whole
# generation (`feed/score.py`).
MAX_SCORE_SECTIONS = 4

_LANGUAGE_NAME = {"en": "English", "ru": "Russian"}
# A script that must be present for the output to count as written in that language.
# gemma4-E2B silently ignores "write in Russian" and answers in English — a Russian
# voice reading English text is worse than no narration, so the result is CHECKED and
# the call retried once with a blunter instruction.
_REQUIRED_SCRIPT = {"ru": ("\u0400", "\u04FF")}


def _is_in_language(text: str, language: str) -> bool:
    span = _REQUIRED_SCRIPT.get(language)
    if span is None or not text:
        return True
    low, high = span
    return any(low <= ch <= high for ch in text)


@dataclass(frozen=True)
class Story:
    narration: str
    cards: tuple[tuple[int, str], ...]   # (0-based shot index, text)
    music_prompt: str
    # The speaker the whole film is read by, and per passage an optional different one.
    # Validated against the catalog before use (`feed/voice.py`).
    narrator: str = ""
    # `(0-based shot index, what the voice says over it)`, IN RUNNING ORDER. This is
    # the film: the order the photographs appear in, and the words spoken over each.
    # Empty only when the model would not produce one, in which case the film falls
    # back to the old "prose plus a guess" path.
    script: tuple[tuple[int, str], ...] = ()

    # Per-passage speaker overrides, positionally matching `script`. Empty string means
    # "the film's narrator".
    voices: tuple[str, ...] = ()

    # `(0-based passage index, what the music becomes there)`, in order. The story's own
    # plan for the score (§11.1); empty when it did not make one, in which case
    # `feed/score.py` splits the film evenly instead.
    score: tuple[tuple[int, str], ...] = ()

    @property
    def chunks(self) -> list[str]:
        """The exact strings to hand to the voice, in order, so that segment `i` of the
        returned audio is the narration for shot `i` of the script."""
        return [text for _, text in self.script]

    def lines_for(self, shot_count: int) -> list[str]:
        """The per-shot text, empty everywhere except at a title card. The director
        takes a full-length list, so the sparseness is expressed here."""
        lines = [""] * shot_count
        for index, text in self.cards:
            if 0 <= index < shot_count:
                lines[index] = text
        return lines


def write_story(
    client, model: str, *, title: str, description: str, edit: Edit,
    shots: list[str] | None = None, language: str = DEFAULT, digest: str = "",
    notes: str = "", directions: str = "", speakers: str = "",
    context_tokens: int | None = None,
) -> Story:
    """The film's words, in TWO STAGES (design §11.1).

    1. **Tell the story** — prose, in the film's language, with a beginning, a middle
       and an end, carrying every note. Nothing else: no photographs to number, no
       cards, no cast. One job, which is the only way this model writes a story rather
       than a caption per picture.
    2. **Cut it to the pictures and dress the film** — a second call gives each SENTENCE
       its photograph and adds the title cards, the narrator, any per-passage voice and
       the score plan. It never rewrites a word: the story is already written.

    Between the two, the story is checked and repaired against what was asked for — the
    film's language, its length, and every one of the viewer's notes.

    NEVER raises: a film with no words still plays.
    """
    low, high = narration_words(len(edit.shots), _note_count(notes))
    prose = _tell(client, model, title, description, digest, notes, language, low, high)
    if prose and not _is_in_language(prose, language):
        # Composing in a second language is hard for a small model; TRANSLATING text it
        # can already see is much easier, so the fallback is a translation pass rather
        # than another attempt at the same hard task.
        logger.warning("the story came back in the wrong language; translating to %s",
                       language)
        prose = _translate_text(client, model, prose, language)
    if not prose:
        logger.error("no story could be written in %s; the film plays without a voice",
                     language)
        return Story(narration="", cards=(), music_prompt=_fallback_music(title))
    # A [bracket] is an INSTRUCTION and is never narrated (§11.3). The story pass is
    # told that and still copies one into its first sentence, which the voice then reads
    # aloud — "суровый мужской голос" spoken over the opening shot. Stripped here, where
    # it cannot be forgotten.
    told = [text for text in
            (_BRACKET.sub(" ", part).strip() for part in sentences(prose)) if text]
    told = _long_enough(client, model, told, title, digest, language, low)
    told = _with_uncovered(client, model, told, notes, language)
    told = _in_place(told, notes)
    return _dress(client, model, told, title, description, edit, shots, language,
                  digest, notes, directions, speakers, context_tokens)


def _dress(client, model: str, told: list[str], title: str, description: str, edit: Edit,
           shots: list[str] | None, language: str, digest: str, notes: str,
           directions: str, speakers: str, context_tokens: int | None) -> Story:
    """Stage 2: which photograph carries each sentence, plus the cards, the cast and the
    score plan. Asked ONCE, then repaired — and repaired hard, because the story itself
    is already safe: the worst a bad answer here can do is show the wrong picture."""
    story = _attempt(client, model, title, description, edit, shots, language,
                     digest, notes, directions, speakers, insist=False,
                     context_tokens=context_tokens, told=told)
    # A bracket asking for a voice that was never cast is silent: the film looks fine
    # and something the person asked for is simply gone. This does not read the bracket
    # — it notices one exists and was not answered (§11.3).
    if (_asks_for_a_voice(notes) and speakers
            and not any(v.strip() for v in story.voices)):
        logger.warning("the notes ask for a voice and none was cast; asking again")
        second = _attempt(
            client, model, title, description, edit, shots, language, digest,
            notes + "\n\nTHE NOTES CONTAIN [BRACKETS] AND YOU CAST NO VOICE FOR ANY "
            "PASSAGE. A bracket describing how something should SOUND is an "
            "instruction: work out which sentence it belongs to and give that sentence "
            'a "voice" from the list, in the wording they used, whatever language it '
            "is in. Leave every other sentence without one so the film keeps its "
            "narrator.",
            directions, speakers, insist=False, context_tokens=context_tokens, told=told,
        )
        if second.script and any(v.strip() for v in second.voices):
            story = second
    return story


# --- STAGE 1: THE STORY, and nothing else (§11.1) ------------------------------------
#
# The old single call asked for one JSON object per photograph, each carrying its own
# text. That makes the PHOTOGRAPH the unit of composition, and a 2B model answers
# exactly as asked: one sentence per picture. "На улице был ветер. Обе смотрели в
# объектив." is not a story, it is a caption track — and no amount of "write ONE
# continuous piece" inside the same prompt fixed it, because the output SHAPE was
# asking for captions.
#
# Measured, same model, same memory, minutes apart: freed of the shot bookkeeping it
# wrote "Помнишь тот вечер? … Нарастало странное напряжение. … И вот так всё
# закончилось." — an opening, a rise and an ending. So the story is written FIRST, as
# prose, with no photographs to number; the pictures are matched to it afterwards.
_TELL_PROMPT = {
    "ru": ("Ты пишешь короткий рассказ, который прозвучит закадровым голосом в фильме "
           "из фотографий одного дня. Это ЧУЖОЕ воспоминание — ты рассказываешь его "
           "человеку, который его прожил.\n\n"
           "ЧТО ЭТО ЗА ДЕНЬ: {title}. {description}\n\n"
           "ЧТО ИЗВЕСТНО ПРО ЭТОТ ДЕНЬ (это ФОН, а не список фактов для "
           "перечисления):\n{digest}\n\n"
           "{notes}"
           "Напиши {low}-{high} слов по-русски. ЭТО РАССКАЗ, А НЕ ПОДПИСИ К ФОТО: "
           "у него есть начало, середина и КОНЕЦ — последнее предложение закрывает "
           "историю. Не описывай кадры («девушка смотрит в камеру», «обе смотрели в "
           "объектив») — рассказывай, ЧТО ПРОИСХОДИЛО, что чувствовали и чем всё "
           "закончилось. НЕ ПЕРЕЧИСЛЯЙ МЕТКИ: «свет был тусклым», «ветер дул "
           "снаружи», «атмосфера была хаотичной» — это не рассказ, а опись. Пусть "
           "погода и свет чувствуются в том, что происходит с людьми. Короткие "
           "предложения, живой язык, это читают вслух. "
           'Верни JSON: {{"story": "..."}}'),
    "en": ("You are writing a short story to be read aloud over a film made of one "
           "day's photographs. It is SOMEONE ELSE'S memory — you are telling it back "
           "to the person who lived it.\n\n"
           "WHAT THE DAY WAS: {title}. {description}\n\n"
           "WHAT IS KNOWN ABOUT IT (this is BACKGROUND, not a list of facts to "
           "recite):\n{digest}\n\n"
           "{notes}"
           "Write {low}-{high} words in English. THIS IS A STORY, NOT PHOTO CAPTIONS: "
           "it has a beginning, a middle and an END — the last sentence closes it. "
           "Never describe the frames ('a woman looks at the camera', 'both looked "
           "into the lens'); tell WHAT HAPPENED, what it felt like, and how it ended. "
           "NEVER RECITE THE LABELS — 'the light was dim', 'the wind was outside', "
           "'the mood was chaotic' is an inventory, not a story. Let the weather and "
           "the light be felt in what happens to the people. "
           "Short sentences, plain words, it is read aloud. Return JSON: "
           '{{"story": "..."}}'),
}
_TELL_NOTES = {
    "ru": ("ЧТО РАССКАЗАЛ САМ ЧЕЛОВЕК. Это правда о том дне, важнее всех меток выше, "
           "и КАЖДОЕ должно прозвучать в рассказе — на своём месте, в том порядке, в "
           "каком записано. Текст в [квадратных скобках] — это указание, как это "
           "произнести; никогда не читай его вслух, просто учти.\n{notes}\n\n"),
    "en": ("WHAT THE PERSON THEMSELVES SAID. This is the truth about the day, it "
           "outranks every label above, and EVERY ONE of them must be in the story, in "
           "its own place, in the order written. Anything in [square brackets] is an "
           "instruction about how it should sound — never read it aloud, just obey "
           "it.\n{notes}\n\n"),
}


def _tell(client, model: str, title: str, description: str, digest: str, notes: str,
          language: str, low: int, high: int) -> str:
    """The story. One call, one job, no bookkeeping. "" when it could not be written."""
    template = _TELL_PROMPT.get(language, _TELL_PROMPT[DEFAULT])
    note_block = (_TELL_NOTES.get(language, _TELL_NOTES[DEFAULT]).format(notes=notes)
                  if notes.strip() else "")
    prompt = template.format(title=title, description=description,
                             digest=digest or "-", notes=note_block, low=low, high=high)
    try:
        payload = _loads(client.text_complete(
            model, [{"role": "user", "content": prompt}],
            json_schema=_STORY_SCHEMA_TEXT, temperature=0.85,
            max_tokens=MAX_OUTPUT_TOKENS,
        ))
    except Exception:
        logger.exception("the story pass failed; falling back to a wordless film")
        return ""
    return str(payload.get("story") or "").strip()


# Sentence boundaries for the cut. Deterministic on purpose: splitting prose is
# arithmetic, and a model asked to "return the sentences" returns them changed.
_SENTENCE = re.compile(r"[^.!?…]+[.!?…]*", re.DOTALL)
# A passage is a BEAT of the film — one photograph is on screen for exactly as long as
# it is spoken — so a four-word sentence is a cut every two seconds. Measured on a real
# film: 109 words came back as 15 sentences, which is a cut every seven words and reads
# as a stutter. Short sentences are joined until a passage is worth holding a picture
# for; the story keeps its own punctuation, only the CUTTING is coarser.
_MIN_SENTENCE_CHARS = 45


def sentences(text: str) -> list[str]:
    """The story, split into the passages a film can cut on."""
    parts = [part.strip() for part in _SENTENCE.findall(text) if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < _MIN_SENTENCE_CHARS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged[:MAX_PASSAGES]


# --- keeping the story honest, BEFORE any photograph is attached to it ---------------
#
# Both repairs work on the prose, not on the finished film: a sentence is a sentence
# until stage 2 gives it a picture, so length and note coverage are fixed while the
# story is still just words.

# Asking for more words does not get more words. Measured: the band was raised from
# 55-75 to 115-135 and gemma wrote 45, then 60 — a film that stopped talking twenty
# seconds in. A SHORT, CONCRETE follow-up does work: it is shown the story it already
# wrote and asked to carry it on and END it.
_EXPAND_PROMPT = {
    "ru": ("Вот рассказ, который читают за кадром в фильме про этот день "
           "({title}):\n\n{story}\n\nЧТО ЕЩЁ ИЗВЕСТНО ПРО ЭТОТ ДЕНЬ:\n{digest}\n\n"
           "Рассказ обрывается слишком рано. Допиши ещё {count} предложения, которые "
           "ПРОДОЛЖАЮТ и ЗАКАНЧИВАЮТ эту историю — что было дальше и чем всё "
           "кончилось. Не повторяй уже сказанное и не описывай фотографии. Пиши "
           'по-русски, живым языком. Верни JSON: {{"story": "..."}}'),
    "en": ("Here is the narration of a film about this day ({title}):\n\n{story}\n\n"
           "WHAT ELSE IS KNOWN ABOUT THE DAY:\n{digest}\n\n"
           "The story stops too early. Write {count} more sentences that CARRY IT ON "
           "and END it — what happened next and how it finished. Do not repeat what is "
           "already said and do not describe photographs. Write in English. Return "
           'JSON: {{"story": "..."}}'),
}
_STORY_SCHEMA_TEXT: dict = {
    "type": "object", "properties": {"story": {"type": "string"}}, "required": ["story"],
}
# Below this share of the band's floor the film is not a short story, it is a trailer —
# the point at which the follow-up is worth its ten seconds.
SHORT_ENOUGH_TO_EXTEND = 0.8


def _long_enough(client, model: str, told: list[str], title: str, digest: str,
                 language: str, low: int) -> list[str]:
    """A film that stops talking twenty seconds in is a clip, not a short film.

    The narration decides the runtime (§11.1), so a story far short of its own word band
    IS a short film — and the band cannot be enforced by asking for it.
    """
    words = sum(len(text.split()) for text in told)
    if not told or words >= low * SHORT_ENOUGH_TO_EXTEND:
        return told
    logger.info("the story is %d words against a floor of %d; asking for more",
                words, low)
    return _expand(client, model, told, title, digest, language, low)


def _expand(client, model: str, told: list[str], title: str, digest: str,
            language: str, wanted: int) -> list[str]:
    """One follow-up call for more story. Returns what it had if it cannot be had."""
    template = _EXPAND_PROMPT.get(language, _EXPAND_PROMPT[DEFAULT])
    words = sum(len(text.split()) for text in told)
    prompt = template.format(story=" ".join(told), title=title, digest=digest or "-",
                             count=max(2, min(6, (wanted - words) // 20)))
    try:
        payload = _loads(client.text_complete(
            model, [{"role": "user", "content": prompt}],
            json_schema=_STORY_SCHEMA_TEXT, temperature=0.85, max_tokens=800,
        ))
    except Exception:
        logger.exception("extending the story failed")
        return told
    more = str(payload.get("story") or "").strip()
    if not more or not _is_in_language(more, language):
        logger.warning("the story pass would not extend a %d-word film", words)
        return told
    # "Do not repeat what is already said" is an instruction, not a guarantee — it
    # handed back sentences of its own story word for word. So repeats are DROPPED.
    # Containment, not equality: sentences are MERGED into beats, so a repeat can arrive
    # glued to a new sentence ("Мы приехали. Вечер закончился тихо.") and an equality
    # check would let it through.
    said = [_normalised(text) for text in told]
    extra = []
    for text in sentences(more):
        key = _normalised(text)
        if any(key in seen or seen in key for seen in said):
            logger.info("dropped a repeated sentence from the extension: %r", text[:60])
            continue
        said.append(key)
        extra.append(text)
    if not extra:
        return told
    logger.info("extended the story from %d to %d sentences", len(told),
                len(told) + len(extra))
    return (told + extra)[:MAX_PASSAGES]


def _normalised(text: str) -> str:
    """A sentence's identity for repeat-detection: words only, case and punctuation
    dropped, so "Обе смотрели в объектив." and "обе смотрели в объектив" are one."""
    return " ".join(re.findall(r"\w+", text.lower()))


# Our OWN position marker, written by `notes.notes_for_prompt` — not the viewer's words.
# Reading it back is not the keyword-matching this module refuses to do (§11.3): it is
# parsing something we generated, to recover where the viewer was standing.
_SHARE_MARKER = re.compile(r"about (\d+)% of the way through")
_BRACKET = re.compile(r"\[[^\]]*\]")

# The ONE-SENTENCE rewrite, in the film's own language. Composing a whole story around a
# fact is what the big prompt failed at; rewriting a single sentence to carry that fact
# is a task a 2B model does well.
_WEAVE_PROMPT = {
    "ru": ("Ты переписываешь ОДНО предложение закадрового текста.\n\n"
           "ПРЕДЛОЖЕНИЕ: {passage}\n\n"
           "ФАКТ, который человек рассказал про этот момент: {note}\n\n"
           "Перепиши это предложение так, чтобы оно рассказывало этот факт. Пиши "
           "по-русски, 1-2 коротких предложения, живым языком, от лица человека, "
           'который это помнит. Верни JSON: {{"text": "..."}}'),
    "en": ("You are rewriting ONE sentence of a film's narration.\n\n"
           "SENTENCE: {passage}\n\n"
           "WHAT THE PERSON SAID ABOUT THIS MOMENT: {note}\n\n"
           "Rewrite the sentence so that it tells this. Write in English, one or two "
           "short sentences, in the voice of someone remembering it. Return JSON: "
           '{{"text": "..."}}'),
}
_WEAVE_SCHEMA: dict = {
    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
}


def _weave(client, model: str, passage: str, note: str, language: str) -> str:
    """Rewrite one sentence so it carries one note. "" when it could not be done."""
    template = _WEAVE_PROMPT.get(language, _WEAVE_PROMPT[DEFAULT])
    try:
        payload = _loads(client.text_complete(
            model,
            [{"role": "user", "content": template.format(passage=passage, note=note)}],
            json_schema=_WEAVE_SCHEMA, temperature=0.7, max_tokens=300,
        ))
    except Exception:
        logger.exception("weaving a note into its sentence failed")
        return ""
    text = str(payload.get("text") or "").strip()
    return text if text and _is_in_language(text, language) else ""


def _with_uncovered(client, model: str, told: list[str], notes: str,
                    language: str) -> list[str]:
    """THE VIEWER'S WORDS GO IN THE FILM, whether or not the model would write them.

    Asking twice is not a guarantee, and it was measured not to be one: gemma dropped
    the same note through a raised word budget, an explicit complaint on retry, and a
    schema field that forced one entry per note. So anything still uncovered is either
    WOVEN into the sentence at its position by one small call, or — if even that fails —
    inserted there verbatim. A sentence that reads a little flat is worth more than one
    that is missing.

    A `[bracket]` is an instruction and is never narrated (§11.3), so it is stripped;
    and a note not written in the film's language is left alone rather than fed to a
    voice that would read it phonetically.
    """
    missing = uncovered_notes(" ".join(told), notes)
    if not missing or not told:
        return told
    for line in missing:
        marker, _, text = line.partition("] ")
        text = _BRACKET.sub(" ", text).strip()
        if not text or not _is_in_language(text, language):
            logger.warning("cannot carry a note the film's language cannot speak: %r",
                           line[:80])
            continue
        share = _SHARE_MARKER.search(marker)
        at = (min(max(int(share.group(1)), 0), 100) / 100.0) if share else 1.0
        index = min(round(at * len(told)), len(told))
        target = min(index, len(told) - 1)
        woven = _weave(client, model, told[target], text, language)
        if woven and not uncovered_notes(woven, line):
            told = [*told[:target], woven, *told[target + 1:]]
            logger.info("wove a note into sentence %d: %r", target, woven[:60])
            continue
        # A note is typed without punctuation as often as not, and a passage that does
        # not close runs into the next one — both in the split and in the voice, which
        # then reads two thoughts as one breath.
        if text[-1] not in ".!?…":
            text += "."
        told = [*told[:index], text, *told[index:]]
        logger.info("inserted a note verbatim at sentence %d: %r", index, text[:60])
    return told[:MAX_PASSAGES]


def _in_place(told: list[str], notes: str) -> list[str]:
    """MOVE a sentence that carries a note to where the note was written.

    Coverage and placement are different failures and the second is easy to miss: every
    note is in the story, so nothing looks wrong, and the person hears their remark
    about the middle of the evening read out over the closing shot.

    The old pipeline noticed this and asked the model to try again. It does not need
    asking: the note says where it belongs and the sentence carrying it is known, so the
    sentence is simply moved. Measured in SHARES of the film, because the new cut is a
    different length from the one they were watching.
    """
    if not told or not notes:
        return told
    for line in notes.splitlines():
        marker, _, prose = line.partition("] ")
        share = _share(marker)
        if share is None or not prose.strip():
            continue
        words = re.findall(r"\w{5,}", prose.lower())
        found = [i for i, text in enumerate(told)
                 if any(word[:6] in text.lower() for word in words)]
        if not found:
            continue                     # not there at all — coverage catches that
        at = found[0]
        landed = (at + 0.5) / len(told)
        if abs(landed - share) <= _PLACEMENT_TOLERANCE:
            continue
        wanted = min(round(share * len(told)), len(told) - 1)
        sentence = told.pop(at)
        told.insert(wanted, sentence)
        logger.info("moved a note's sentence from %d%% to %d%% of the story",
                    round(landed * 100), round(share * 100))
    return told


def _translate_text(client, model: str, text: str, language: str) -> str:
    """The story, in the film's language. "" when it could not be translated."""
    name = _LANGUAGE_NAME.get(language, _LANGUAGE_NAME[DEFAULT])
    try:
        payload = _loads(client.text_complete(
            model,
            [{"role": "user", "content":
              f"Translate into {name}. Translate faithfully and naturally — this is "
              f'read aloud. Return JSON: {{"story": "..."}}\n\n{text}'}],
            json_schema=_STORY_SCHEMA_TEXT, temperature=0.2, max_tokens=MAX_OUTPUT_TOKENS,
        ))
    except Exception:
        logger.exception("translating the story failed")
        return ""
    out = str(payload.get("story") or "").strip()
    return out if out and _is_in_language(out, language) else ""


def uncovered_notes(narration: str, notes: str) -> list[str]:
    """The note lines whose content does not appear in the narration AT ALL.

    Telling a model that every note must be used is not the same as it using them, and
    the failure is silent: the film is fine, one thing the person said is simply gone.
    So the output is checked.

    Deliberately crude — any distinctive word of a note turning up anywhere counts as
    covered. The model paraphrases, reorders and translates, so anything stricter would
    fire constantly; this catches the case that actually happens, which is a note left
    out completely.
    """
    if not narration or not notes:
        return []
    body = narration.lower()
    missing = []
    for line in notes.splitlines():
        prose = line.split("] ", 1)[-1].strip()
        words = re.findall(r"\w{5,}", prose.lower())
        if words and not any(word[:6] in body for word in words):
            missing.append(line.strip())
    return missing


# How far a note's words may sit from where the note was written, as a share of the
# film, before it counts as misplaced. A quarter of the film is generous — the story
# reorganises — but it catches the failure that happens, which is a note appended to
# the end because the retry only asked for it to be INCLUDED.
_PLACEMENT_TOLERANCE = 0.22


def misplaced_notes(story: "Story", notes: str) -> list[str]:
    """Note lines whose words landed far from where the note was written.

    Coverage and placement are different failures and the second is easy to miss:
    every note is in the story, so nothing looks wrong, and the person hears their
    remark about the middle of the evening read out over the closing shot.

    Measured in SHARES of the film rather than seconds, because the new cut is a
    different length from the one they were watching.
    """
    if not story.script or not notes:
        return []
    passages = [text.lower() for _index, text in story.script]
    total = len(passages)
    late = []
    for line in notes.splitlines():
        marker, _, prose = line.partition("] ")
        share = _share(marker)
        if share is None or not prose.strip():
            continue
        words = re.findall(r"\w{5,}", prose.lower())
        found = [i for i, text in enumerate(passages)
                 if any(word[:6] in text for word in words)]
        if not found:
            continue                     # not there at all — coverage catches that
        landed = (min(found) + 0.5) / total
        if abs(landed - share) > _PLACEMENT_TOLERANCE:
            late.append(f"{line.strip()}\n     → you put this about "
                        f"{round(landed * 100)}% in; it belongs about "
                        f"{round(share * 100)}% in")
    return late


def _share(marker: str) -> float | None:
    """The `about NN% of the way through` out of a note's position marker."""
    match = re.search(r"about (\d+)% of the way through", marker)
    return int(match.group(1)) / 100 if match else None


def _asks_for_a_voice(notes: str) -> bool:
    """Whether any note carries a bracket at all.

    A bracket is an instruction (§11.3) and this is the whole of what code decides
    about one: that it is there. What it MEANS — a hoarse man, a bright chord, a
    glitch — is read by the model, in whatever wording and language it was typed.

    Only the note's own text counts. Each line opens with a POSITION marker that is
    itself in brackets — `[at 0:12, about 47% …]` — so a bare `"[" in notes` was true
    of every note ever written.
    """
    return any("[" in line.split("] ", 1)[-1] for line in notes.splitlines())


def _answered(story: "Story", notes: str, speakers: str) -> int:
    """How many of the things asked for this telling actually answered.

    Used only to compare a retry against its predecessor, so a second attempt that came
    back worse is discarded.
    """
    score = -len(uncovered_notes(story.narration, notes)) - len(misplaced_notes(story, notes))
    if _asks_for_a_voice(notes) and speakers and any(v.strip() for v in story.voices):
        score += 1
    return score


def _translate(client, model: str, story: Story, language: str) -> Story | None:
    """Translate a finished story into `language`. None when it cannot be done."""
    name = _LANGUAGE_NAME.get(language, _LANGUAGE_NAME[DEFAULT])
    cards = [text for _, text in story.cards]
    prompt = (
        f"Translate into {name}. Return JSON with two keys: \"narration\" (the "
        f"translation of the text below) and \"cards\" (a list of {len(cards)} "
        f"strings, the translations of the card texts, in order).\n"
        f"Translate faithfully and naturally — this is read aloud.\n\n"
        f"TEXT:\n{story.narration}\n\nCARDS:\n"
        + "\n".join(f"{i + 1}. {text}" for i, text in enumerate(cards))
    )
    schema = {
        "type": "object",
        "properties": {
            "narration": {"type": "string"},
            "cards": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["narration", "cards"],
    }
    try:
        payload = _loads(
            client.text_complete(model, [{"role": "user", "content": prompt}],
                                 json_schema=schema, temperature=0.2, max_tokens=2000)
        )
    except Exception:
        logger.exception("translation pass failed")
        return None
    narration = str(payload.get("narration") or "")
    if not narration or not _is_in_language(narration, language):
        return None
    texts = payload.get("cards")
    if isinstance(texts, list) and len(texts) == len(story.cards):
        cards = tuple((index, str(text))
                      for (index, _), text in zip(story.cards, texts))
    else:
        cards = story.cards
    return replace(story, narration=narration, cards=cards)


def _attempt(client, model, title, description, edit, shots, language, digest="",
             notes="", directions="", speakers="", *, insist: bool,
             context_tokens: int | None = None,
             told: list[str] | None = None) -> Story:
    prompt = _prompt(title, description, edit, shots, language, digest, notes,
                     directions, speakers, insist=insist,
                     context_tokens=context_tokens, told=told)
    try:
        raw = client.text_complete(
            model,
            [{"role": "user", "content": prompt}],
            json_schema=STORY_SCHEMA,
            temperature=0.8,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        payload = _loads(raw)
    except Exception:  # a wordless film beats no film
        logger.exception("story pass failed; falling back to a wordless film")
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    script = _script(payload.get("script"), len(edit.shots), told)
    # `narration` is what the story SAYS; it is the script joined, so the two can never
    # disagree — and with a story already written, the words are ours, not the model's.
    narration = (" ".join(text for _, text in script) if script
                 else " ".join(told or []) or str(payload.get("narration") or ""))
    return Story(
        narration=narration,
        cards=_cards(payload.get("cards"), len(edit.shots)),
        music_prompt=str(payload.get("music_prompt") or "") or _fallback_music(title),
        script=script,
        narrator=str(payload.get("narrator") or "").strip(),
        voices=_voices(payload.get("script"), len(script)),
        score=_score(payload.get("score"), len(script)),
    )


# A story told in fewer passages than this is not a story; more than this and the
# picture is changing faster than anyone can follow.
MAX_PASSAGES = 24


def _voices(raw, length: int) -> tuple[str, ...]:
    """The per-passage speaker names, positionally matching the script.

    NOT validated here — `feed/build.py` runs every one through `voice.resolve` against
    the catalog, because this module has no business knowing which speakers exist.
    """
    if not isinstance(raw, list):
        return ()
    names = []
    for item in raw[:length]:
        names.append(str(item.get("voice") or "").strip()
                     if isinstance(item, dict) else "")
    return tuple(names + [""] * (length - len(names)))


def _script(raw, shot_count: int, told: list[str] | None = None) -> tuple[tuple[int, str], ...]:
    """Parse the story into `(photo index, what is said over it)`, in TELLING ORDER.

    The story is the first citizen, so this deliberately does NOT require every
    photograph to be used, nor forbid one from being used twice: a passage that returns
    to a face already seen is a thing films do, and a photograph the story has no use
    for is better left out than forced in. Only the words are mandatory.

    A passage naming a photograph that does not exist keeps its text and falls back to
    the previous picture — losing a line of the story to a bad index would be much
    worse than holding a shot a moment longer.
    """
    if not isinstance(raw, list):
        return _even(told, shot_count) if told else ()
    passages: list[tuple[int, str]] = []
    seen: set[int] = set()
    for item in raw[:MAX_PASSAGES]:
        if not isinstance(item, dict):
            continue
        if told is not None:
            # The words come from the story pass; this call only says which photograph
            # goes under which sentence.
            try:
                which = int(item["sentence"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= which < len(told) or which in seen:
                continue
            seen.add(which)
            text = told[which]
        else:
            text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            index = int(item["shot"]) - 1
        except (KeyError, TypeError, ValueError):
            index = -1
        if not 0 <= index < shot_count:
            index = passages[-1][0] if passages else 0
        passages.append((index, text))
    if told is not None:
        # EVERY SENTENCE IS IN THE FILM. One the matching call forgot is not dropped —
        # the story was already written and approved; it just needs a picture.
        missing = [i for i in range(len(told)) if i not in seen]
        for which in missing:
            at = min(which, len(passages))
            shot = passages[at - 1][0] if at else (passages[0][0] if passages else 0)
            passages.insert(at, (shot, told[which]))
        if not passages:
            return _even(told, shot_count)
    return tuple(passages)


def _even(told: list[str], shot_count: int) -> tuple[tuple[int, str], ...]:
    """Sentences laid over the photographs in order — the fallback when the matching
    call gives nothing usable. Every sentence still gets a picture."""
    if not told:
        return ()
    return tuple((min(int(i * shot_count / len(told)), max(shot_count - 1, 0)), text)
                 for i, text in enumerate(told))


def _cards(raw, shot_count: int) -> tuple[tuple[int, str], ...]:
    """Parse the title cards, dropping anything malformed rather than failing.

    The model is asked for 1-based shot numbers because that is how the prompt lists
    them; they are converted here so nothing downstream has to remember."""
    if not isinstance(raw, list):
        return ()
    cards: list[tuple[int, str]] = []
    for item in raw[:MAX_TITLE_CARDS]:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["shot"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if text and 0 <= index < shot_count:
            cards.append((index, text))
    return tuple(cards)


def _score(raw, passage_count: int) -> tuple[tuple[int, str], ...]:
    """Parse the score plan, dropping anything malformed rather than failing.

    1-based in (that is how the prompt numbers passages), 0-based out. A change pointing
    past the end of the script is dropped rather than clamped: the model meant a passage
    that does not exist, so where it wanted the music to turn is not known.
    """
    if not isinstance(raw, list):
        return ()
    changes: dict[int, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["from"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        brief = str(item.get("brief") or "").strip()
        if brief and 0 <= index < passage_count:
            changes.setdefault(index, brief)
    return tuple(sorted(changes.items())[:MAX_SCORE_SECTIONS])


def _loads(raw: str) -> dict:
    """Parse the model's JSON, tolerating what small models actually emit.

    gemma wraps its answer in a ```json fence often enough that a bare `json.loads`
    throws, and the whole story is then discarded — which shows up as a film with no
    narration and therefore no voice, with nothing to indicate why. Strip the fence,
    and failing that take the outermost {...} block.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    return payload if isinstance(payload, dict) else {}


def _fallback_music(title: str) -> str:
    """Kept for callers with no signals. `feed/build.py` uses `mood.music_prompt`,
    which matches the DAY — one generic sentence over a birthday is the funeral-music
    bug (§11.1)."""
    return f"{title}. {_MUSIC_TAIL}".strip()


def brief_chars(shot_count: int, fixed_chars: int, context_tokens: int | None) -> int | None:
    """How many characters each photograph's brief may spend, or None for no limit.

    Everything else in the prompt is already written by the time this is asked, so the
    listing gets what is left of the server's context after the answer is reserved —
    shared equally, because no photograph is more worth describing than another.
    """
    if not context_tokens:
        return None
    room = ((context_tokens - RESERVED_TOKENS) * CHARS_PER_TOKEN - fixed_chars
            - shot_count * LINE_OVERHEAD_CHARS)
    return max(MIN_BRIEF_CHARS, int(room // max(1, shot_count)))


def _listing(texts: list[str], shot_count: int, cap: int | None) -> str:
    """The numbered photo lines. NUMBERING IS THE CONTRACT — the story answers with
    `shot` numbers — so a brief that does not fit is clipped, never dropped."""
    lines = []
    for index in range(shot_count):
        text = texts[index] if index < len(texts) and texts[index] else ""
        if cap is not None and len(text) > cap:
            text = text[:cap].rstrip() + "…"
        lines.append(f"{index + 1}. {text or f'(shot {index + 1})'}")
    return "\n".join(lines)


def _prompt(
    title: str, description: str, edit: Edit, shots: list[str] | None, language: str,
    digest: str = "", notes: str = "", directions: str = "", speakers: str = "",
    *, insist: bool = False, context_tokens: int | None = None,
    told: list[str] | None = None,
) -> str:
    """The story prompt, CUT TO FIT the server's context (§11.1).

    Assembled twice on purpose: once with no listing at all, to measure what the fixed
    instructions, the notes and the digest already cost, and again with the listing
    given whatever room is left. Guessing that share in advance would either waste
    context or overrun it, and an overrun costs the whole call.
    """
    texts = shots or [shot.line for shot in edit.shots]
    fixed = _assemble(title, description, edit, language, digest, notes, directions,
                      speakers, listing="", insist=insist, told=told)
    cap = brief_chars(len(edit.shots), len(fixed), context_tokens)
    listing = _listing(texts, len(edit.shots), cap)
    prompt = _assemble(title, description, edit, language, digest, notes, directions,
                       speakers, listing=listing, insist=insist, told=told)
    if context_tokens and _estimated_tokens(prompt) + RESERVED_TOKENS > context_tokens:
        # Nothing left to cut: the instructions, the notes and one floor-width line per
        # photograph already fill the context. Say so — if the server does reject this,
        # the film fails for want of narration, and that reads as a broken feature
        # unless the log says otherwise. The estimate is deliberately pessimistic, so
        # this is a warning about a risk, not a prediction of a failure.
        logger.warning(
            "the story prompt is ~%d estimated tokens against a %d-token context and "
            "may be rejected: this memory has too many photographs for this context",
            _estimated_tokens(prompt), context_tokens,
        )
    return prompt


def _estimated_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _assemble(
    title: str, description: str, edit: Edit, language: str, digest: str, notes: str,
    directions: str, speakers: str, *, listing: str, insist: bool,
    told: list[str] | None = None,
) -> str:
    name = _LANGUAGE_NAME.get(language, _LANGUAGE_NAME[DEFAULT])
    told = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(told or []))
    minutes = max(1, round(edit.seconds / 60))
    # The language demand goes FIRST and is repeated at the end: buried in the middle
    # of a long prompt, a small model drops it and answers in English.
    header = f"Write every word of your answer in {name}.\n\n"
    if insist:
        header = (
            f"CRITICAL: your previous answer was NOT in {name}. Write EVERY word — the "
            f"narration and every card — in {name} and nothing else. Do not translate "
            f"at the end; think and write in {name} from the first word.\n\n"
        )
    return (
        header
        + "You are writing the narration for a short film made from one person's own "
        "photographs. It is their memory; you are telling it back to them.\n\n"
        f"Memory: {title}\n"
        f"What they said about it: {description}\n"
        f"Runtime: about {minutes} minute(s), {len(edit.shots)} shots.\n\n"
        # FIRST, above every machine label: the viewer's own words, typed while
        # watching an earlier cut at the second of the film they belong to (§11.3).
        # Position matters as much as content — a note is a person pointing at a
        # moment, and answering it somewhere else answers nothing.
        + (f"### WHAT THE PERSON WHOSE MEMORY THIS IS TOLD YOU — READ THIS FIRST\n"
           f"They wrote these while watching an earlier cut of this film. This is "
           f"TESTIMONY: every label further down is a machine's guess, and this is the "
           f"only thing here that comes from someone who was actually there. It "
           f"outranks all of it, including the memory's own title and description.\n"
           f"**EACH NOTE IS A FACT ABOUT THE WHOLE MEMORY, NOT A LINE TO INSERT.**\n"
           f"This is the part that is usually got wrong. If they wrote 'this is "
           f"Julia' over photograph 3, then the woman in photograph 3 IS Julia — for "
           f"the entire film. Name her in the passage about photograph 3, name her "
           f"again wherever that photograph comes back, name her in any OTHER "
           f"photograph that is plainly the same person, and name her in passages "
           f"BEFORE that point too, because you now know who she is while you are "
           f"writing them. Rewriting one sentence around the timestamp and leaving her "
           f"'a woman' everywhere else is exactly the failure this instruction "
           f"exists to prevent.\n"
           f"The same holds for anything else they tell you: a place, a relationship, "
           f"a date, what was actually happening. Learn it, then write the whole story "
           f"as someone who knew it all along.\n\n"
           f"RULES, and they are not optional:\n"
           f"- EVERY note must be reflected in the narration. Not summarised at the "
           f"end, not implied — actually said.\n"
           f"- THE WORD COUNT BELOW ALREADY PAYS FOR THEM. It was raised by 30 words "
           f"for every note you were given, so writing them in fully is what the "
           f"length is FOR — never leave one out to stay short.\n"
           f"- ANYTHING IN [SQUARE BRACKETS] IS AN INSTRUCTION, NOT A SENTENCE. Never "
           f"narrate it — obey it. It may be in either language, in any wording: "
           f"'[страшным хриплым голосом]' asks for a frightening, hoarse, male "
           f"delivery, so cast a speaker like that for THAT passage. A bracket about "
           f"the music belongs in `music_prompt`; one about the picture or its length "
           f"is handled elsewhere and you may ignore it.\n"
           f"- Each is marked with where in the film it was written and WHICH "
           f"PHOTOGRAPH was on screen at the time. That photograph is what the note is "
           f"ABOUT. Its position is where the fact first matters most, not the only "
           f"place it may appear.\n"
           f"- **THEY ARE LISTED IN ORDER, AND THAT ORDER IS THE ORDER OF THE DAY.** "
           f"Whatever a note at 0:07 describes happened BEFORE what a note at 0:11 "
           f"describes. Your passages must follow the same sequence: cover the first "
           f"note, then the second, then the third. Telling the fourth and fifth and "
           f"then doubling back to the third makes the story contradict itself, and it "
           f"is the single most obvious way to get this wrong.\n"
           f"- Where a note CONTRADICTS a label or a caption below, the note is right "
           f"and the label is wrong. Say what they said, and stop using the label's "
           f"word for it anywhere in the film.\n"
           f"- They may be in any language, often not the film's. Translate what they "
           f"MEAN into {name} and write it in your own prose — never quoted raw, never "
           f"listed, never in brackets.\n"
           f"{notes}\n\n"
           if notes else "")
        # The whole memory before the individual pictures. A writer shown only three
        # captions writes three captions; it has to know these are one snowy holiday
        # evening in Tbilisi before it can write about the EVENING (§11.1).
        + (f"WHAT THIS MEMORY IS, from the library's own labels:\n{digest}\n\n"
           if digest else "")
        + f"EACH PHOTOGRAPH — what is in it, then its labels, where and when it was "
        f"taken, and how it was shot:\n{listing}\n\n"

        # `[bracketed]` directions about HOW it should be told, as opposed to what
        # happened. Separated from the prose so neither is mistaken for the other.
        + (f"### HOW THEY ASKED FOR IT TO BE TOLD\n"
           f"These are INSTRUCTIONS, not events. Never narrate them; obey them.\n"
           f"A delivery named here is already being applied to the VOICE that reads "
           f"this — a narrator has been chosen to match it. Your job is to make the "
           f"WORDS fit that delivery too: a story to be read in a low, frightening "
           f"voice is written in shorter, darker sentences than the same story read "
           f"brightly, and handing that narrator cheerful prose wastes the choice.\n"
           f"{directions}\n\n" if directions else "")
        + (f"### THE STORY, ALREADY WRITTEN. These are its sentences, numbered. They "
           f"are FINAL — you are not rewriting them, you are giving each one a "
           f"photograph.\n{told}\n\n" if told else "")
        + f"Write ALL of your output in {name}.\n\n"
        "Return JSON with three keys:\n"
        '- "script": THE FILM. An ordered list of '
        '{"sentence": <1-based sentence number>, "shot": <1-based photograph number>} '
        "objects — EVERY sentence exactly once, in the order they are written.\n"
        "  `shot` is whichever photograph best ILLUSTRATES that sentence — it is on "
        "screen exactly while those words are spoken, so if the sentence mentions the "
        "cat, give it the cat's photograph.\n"
        "  The photographs serve the story, not the reverse. You may show the same "
        "photograph again when the story returns to it, and you may leave a photograph "
        "out entirely if it adds nothing.\n"
        '- "narration": leave it empty. The story above IS the narration.\n'
        + (f'- "narrator": WHO READS THIS FILM. One name from the list below, chosen '
           f"for the memory — the mood, the occasion, who is in it. This voice reads "
           f"every passage.\n"
           f'  A passage may ALSO carry its own "voice" (same list) — but only where '
           f"the viewer asked for one in their notes above. That passage alone is read "
           f"by it and the film returns to the narrator afterwards; the parts are "
           f"spoken separately and merged. Never restyle the whole film because one "
           f"note asked for one moment to sound different.\n"
           f"  THE VOICES YOU MAY CAST, and what each sounds like:\n{speakers}\n"
           f"  Use these names EXACTLY. Anything not on this list is discarded.\n"
           if speakers else "")
        + f'- "cards": at most {MAX_TITLE_CARDS} title cards for the WHOLE film, as '
        'objects {"shot": <1-based shot number>, "text": "..."}. These are chapter '
        "markers, not captions — put them where the film turns (a new place, a new "
        "part of the day), and leave every other shot with no text at all. A few "
        "words each.\n"
        + ('- "score": WHERE THE MUSIC CHANGES. Two to four objects '
           '{"from": <1-based passage number>, "brief": "..."}, in order, the first '
           'one starting at passage 1. Each `brief` is one English sentence saying '
           'what the music BECOMES at that passage — the instrument, the tempo, the '
           'feeling — and they must differ from each other: an opening, a lift or a '
           'turn where the day changes, and an ending. This is what stops the film '
           'sounding like one loop under every photograph, so choose the passages '
           'where the story itself turns. English only, whatever the film\'s '
           'language, because the audio model reads nothing else.\n')
        + '- "music_prompt": one English sentence describing an instrumental score for '
        "this memory, for a text-to-audio model. **MATCH THE OCCASION AND THE MOOD "
        "GIVEN ABOVE.** A birthday is not scored like a funeral: name the tempo, the "
        "instruments and the feeling that actually fit — bright, fast and playful for "
        "a celebration; slow and spare for a quiet or sad one. Defaulting to 'warm "
        "ambient pads' over a party is the single worst thing you can do here. Always "
        "instrumental, no vocals. This key is the ONE exception — it is always "
        "English, because the audio model only understands English.\n\n"
        f"Reminder: every line of \"script\" and every card must be written in {name}."
    )
