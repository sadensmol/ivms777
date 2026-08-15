from inference.prompts import CAPTION_SCHEMA, caption_messages, chat_messages, intent_messages


def test_caption_schema_requires_the_four_fields():
    props = CAPTION_SCHEMA["properties"]
    assert {"caption", "title", "description", "tags"} <= set(props)
    assert set(CAPTION_SCHEMA["required"]) == {"caption", "title", "description", "tags"}


def test_caption_messages_carry_the_image_and_ask_for_json():
    messages = caption_messages("qwen2.5vl:7b", "data:image/jpeg;base64,AAA", ["subject", "vibe"])
    assert messages[0]["role"] == "system"
    user = messages[-1]
    assert user["role"] == "user"
    blob = str(user["content"])
    assert "AAA" in blob                       # the image data uri is included
    assert "subject" in blob and "vibe" in blob  # the vocabulary dimensions
    assert "json" in blob.lower()


def test_unknown_model_falls_back_to_a_default_template():
    messages = caption_messages("some-unknown-model", "data:image/jpeg;base64,AAA", ["subject"])
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
