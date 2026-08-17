from inference.prompts import (
    CAPTION_SCHEMA,
    GUARDRAIL_REFUSAL,
    agentic_answer_messages,
    caption_messages,
    chat_messages,
    intent_messages,
)


def test_guardrail_refusal_is_an_on_topic_redirect():
    # Guardrails ON refuses off-topic questions with this fixed line (§10).
    assert "only answer questions about your photos" in GUARDRAIL_REFUSAL.lower()


def test_agentic_answer_prompt_carries_facts_and_question():
    msgs = agentic_answer_messages("how many dogs?", 'count: 4 photo(s) matching "dogs"')
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "count: 4" in msgs[1]["content"]
    assert "how many dogs?" in msgs[1]["content"]


def test_agentic_answer_prompt_separates_counts_from_citations():
    # The "[photo:1] for a count of 1" bug: the prompt must tell the model a count
    # is a quantity, not a photo id, and that a citation may come ONLY from a photo
    # block — so a bare count line yields no [photo:...] at all.
    system = agentic_answer_messages("q", "count: 1")[0]["content"].lower()
    assert "quantity" in system and "not photo number 1" in system
    assert "only" in system and "photo block" in system
    assert "must not contain '[photo:'" in system


def test_caption_schema_requires_the_three_fields():
    # The caption model returns ONLY the caption sentence — no tags (design §7).
    props = CAPTION_SCHEMA["properties"]
    assert set(props) == {"caption", "title", "description"}
    assert set(CAPTION_SCHEMA["required"]) == {"caption", "title", "description"}
    assert "tags" not in props


def test_caption_messages_carry_the_image_and_ask_for_json():
    messages = caption_messages("qwen2.5vl:7b", "data:image/jpeg;base64,AAA")
    assert messages[0]["role"] == "system"
    user = messages[-1]
    assert user["role"] == "user"
    blob = str(user["content"])
    assert "AAA" in blob                       # the image data uri is included
    assert "caption" in blob.lower()           # asks for the caption sentence
    assert "json" in blob.lower()
    assert "tags" not in blob.lower()          # never asks the model for tags


def test_unknown_model_falls_back_to_a_default_template():
    messages = caption_messages("some-unknown-model", "data:image/jpeg;base64,AAA")
    assert messages and messages[0]["role"] == "system"


def test_chat_messages_ground_and_require_citation():
    msgs = chat_messages("what lens did I use?", "[photo:7]\ncaption: a cat")
    system = msgs[0]["content"]
    assert "[photo:" in system  # instructs the citation format
    assert "only" in system.lower()  # grounded only in the provided photos
    user = msgs[1]["content"]
    assert "[photo:7]" in user
    assert "what lens did I use?" in user


def test_chat_messages_handle_no_matches():
    msgs = chat_messages("anything", "No photos matched.")
    assert "No photos matched." in msgs[1]["content"]


def test_chat_messages_instructs_using_count_facts_not_guessing():
    # §10: totals/counts come from the count/memories/periods tool facts already
    # threaded into the context — never inferred from the photos shown.
    msgs = chat_messages("how many photos do I have?", "count: 897 photo(s) in total.")
    system = msgs[0]["content"].lower()
    assert "count" in system and "memories" in system and "periods" in system
    assert "never" in system


def test_intent_messages_covers_counts_totals_and_meta_questions():
    # §10: the gate must treat counts/totals/memories/dates/organization questions
    # as on-topic — only a clearly unrelated general question is refused.
    system = intent_messages("how many photos do I have?")[0]["content"].lower()
    for phrase in ("how many", "total", "memor", "month", "year", "camera", "place"):
        assert phrase in system
