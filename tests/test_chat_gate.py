from chat.retrieve import is_photo_question
from inference.fakes import FakeInferenceClient


def test_clear_no_is_refused():
    assert is_photo_question(FakeInferenceClient(responses=["no"]), "m", "walk or drive?") is False


def test_yes_is_a_photo_question():
    assert is_photo_question(FakeInferenceClient(responses=["yes"]), "m", "find a dog") is True


def test_anything_but_no_is_treated_as_a_photo_question():
    # A chatty or unexpected classifier reply must not block a real search.
    assert is_photo_question(FakeInferenceClient(responses=["Sure, photos."]), "m", "cars") is True


def test_classifier_error_fails_open():
    assert is_photo_question(FakeInferenceClient(responses=[]), "m", "find a dog") is True
