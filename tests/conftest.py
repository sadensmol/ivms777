from pathlib import Path

import pytest

from config import Settings
from db.connection import connect, migrate


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Fake embedder keeps the suite offline and instant; the real SigLIP (torch)
    # is exercised only by the slow-marked test and by hand. No `text_embed_model`
    # override: the §9 caption embedder is named by the `text_embed` SLOT (§4.1), and
    # an override that is not a catalog key is ignored by the resolver anyway — a test
    # that cares about the name stores one (`model_slot.text_embed`).
    # `models_base_url` is pinned to an unresolvable host on purpose. The mac
    # profile now defaults to localhost:9000 (everything is host-native, §3.1),
    # so a developer running `make up` would otherwise have the suite talk to
    # their LIVE models service. The suite is offline and hermetic.
    return Settings(
        data_dir=tmp_path, use_fake_embedder=True, use_fake_inference=True,
        models_base_url="http://models.invalid:9000",
    )


@pytest.fixture
def conn(settings: Settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    c = connect(settings.db_path)
    migrate(c)
    yield c
    c.close()
