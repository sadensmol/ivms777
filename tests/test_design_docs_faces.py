"""The docs describe face recognition as built (CLAUDE.md: `docs/design.md` is the source
of truth): the declared face model values and the ONNX converter the docs name are the ones
the code offers and installs."""

import re
import tomllib
from pathlib import Path

import pytest

from models import catalog

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = ("docs/design.md", "docs/models.md")


def _requirement_names(requirements: list[str]) -> set[str]:
    return {re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].lower()
            for requirement in requirements}


@pytest.mark.parametrize("doc", DOCS)
@pytest.mark.parametrize("profile", ["mac", "jetson", "cloud"])
def test_the_docs_declare_the_face_model_values_the_profile_offers(doc, profile):
    entry = catalog.face_model_for(profile)
    assert entry is not None, profile
    text = (REPO_ROOT / doc).read_text(encoding="utf-8")
    for declared in (f"`score_floor` {entry.score_floor}",
                     f"`match_threshold` {entry.match_threshold}",
                     f"`input_edge` {entry.input_edge}",
                     f"`min_face_px` {entry.min_face_px}"):
        assert declared in text, (doc, declared)


def test_the_models_extra_installs_the_onnx_converter_the_docs_name():
    for doc in DOCS:
        assert "onnx2torch" in (REPO_ROOT / doc).read_text(encoding="utf-8"), doc
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    models_extra = _requirement_names(project["project"]["optional-dependencies"]["models"])
    assert {"onnx", "onnx2torch"} <= models_extra
    assert not {"onnx", "onnx2torch"} & _requirement_names(project["project"]["dependencies"])


def _section(text: str, heading: str, next_heading: str) -> str:
    return text[text.index(heading):text.index(next_heading)]


def test_the_docs_describe_people_as_their_own_record():
    plan = REPO_ROOT / "docs/plans/28-people.md"
    assert plan.exists()
    design = (REPO_ROOT / "docs/design.md").read_text(encoding="utf-8")
    data_model = (REPO_ROOT / "docs/data-model.md").read_text(encoding="utf-8")
    ui = (REPO_ROOT / "docs/ui.md").read_text(encoding="utf-8")
    data_section = _section(design, "## 6. Data model", "## 6.1 ")
    ui_section = _section(design, "## 13. UI", "### 13.1 ")
    navigation = design[design.index("### 13.1 "):]

    assert "A person IS a `saved_voices` row" not in design
    assert "**`people`**" in data_section
    assert "/people" in ui_section and "/people/{id}" in ui_section
    assert "/people" in navigation
    assert "people pages" in design[:design.index("## 6. Data model")]
    assert "REFERENCES people(id)" in data_model
    assert "/people/{id}" in ui
    assert "people" in plan.read_text(encoding="utf-8")
