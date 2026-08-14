from inference.fakes import FakeInferenceClient
from scripts.bakeoff import format_table, run_bakeoff


def fake_clock(values):
    ticks = iter(values)
    return lambda: next(ticks)


def test_runs_every_model_against_every_image():
    client = FakeInferenceClient(["cap-a1", "cap-a2", "cap-b1", "cap-b2"])
    clock = fake_clock([0.0, 1.0, 1.0, 3.0, 3.0, 3.5, 3.5, 4.5])

    rows = run_bakeoff(
        client,
        models=["model-a", "model-b"],
        images=[("one.jpg", b"x"), ("two.jpg", b"y")],
        clock=clock,
    )

    assert len(rows) == 4
    assert [row.model for row in rows] == ["model-a", "model-a", "model-b", "model-b"]
    assert rows[0].caption == "cap-a1"
    assert rows[0].seconds == 1.0
    assert rows[1].seconds == 2.0


def test_image_is_sent_as_a_data_uri():
    client = FakeInferenceClient(["cap"])
    run_bakeoff(client, ["m"], [("one.jpg", b"x")], clock=fake_clock([0.0, 1.0]))

    _model, messages = client.calls[0]
    parts = messages[0]["content"]
    assert any(part["type"] == "image_url" for part in parts)
    image_part = next(part for part in parts if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_format_table_reports_mean_seconds_per_model():
    client = FakeInferenceClient(["a", "b"])
    rows = run_bakeoff(
        client, ["m"], [("1.jpg", b"x"), ("2.jpg", b"y")], clock=fake_clock([0.0, 2.0, 2.0, 6.0])
    )

    table = format_table(rows)
    assert "m" in table
    assert "3.00" in table  # mean of 2.0 and 4.0
