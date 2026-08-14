from chat.retrieve import retrieve
from embedding.fakes import FakeEmbedder
from embedding.store import write_vector
from tests.factories import add_photo


def test_retrieve_fuses_semantic_and_keyword(conn):
    fake = FakeEmbedder()
    for pid, word in ((1, "beach"), (2, "keyboard")):
        add_photo(conn, photo_id=pid, content_hash=word, thumb_key=f"{word}.jpg")
        write_vector(conn, pid, fake.embed_texts([word])[0])
    ids = retrieve(conn, fake, owner_id=1, question="beach", k=30)
    assert ids and ids[0] == 1


def test_retrieve_empty_question_returns_nothing(conn):
    assert retrieve(conn, FakeEmbedder(), owner_id=1, question="   ", k=30) == []


def test_retrieve_caps_at_k(conn):
    fake = FakeEmbedder()
    for pid in range(1, 6):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}" * 8, thumb_key=f"{pid}.jpg")
        write_vector(conn, pid, fake.embed_texts([f"h{pid}"])[0])
    assert len(retrieve(conn, fake, owner_id=1, question="anything", k=2)) <= 2
