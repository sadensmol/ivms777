from inference.prompts import CAPTION_SCHEMA, caption_messages


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
