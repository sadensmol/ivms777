from pathlib import Path

import pytest

from config import Settings
from db.connection import connect, migrate


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Fake embedder keeps the suite offline and instant; the real SigLIP (torch)
    # is exercised only by the slow-marked test and by hand. `text_embed_model="fake"`
    # so the §9 caption embedder adds no task prefix (see embedding.caption_text) and
    # the deterministic fake vectors match between caption docs and queries.
    return Settings(
        data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True,
        text_embed_model="fake",
    )


@pytest.fixture
def conn(settings: Settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    c = connect(settings.db_path)
    migrate(c)
    yield c
    c.close()
