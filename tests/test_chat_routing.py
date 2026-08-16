"""Routing matrix (plan 15): which layer answers each phrasing.

`direct_answer` returns a string => the direct-DB layer answered it with NO model
(the phrasing never reaches the weak planner). It returns None => it DECLINED and
the question falls through to the agent loop.

This matrix is the regression net the "all" bug slipped through: every class ×
many phrasings, including the adversarial quantifiers, a typo, and the negatives
that must NOT be caught. A matcher may only fire when confident, else decline —
so a wrong phrasing degrades to the agent, never to a confidently-wrong answer.
"""

import pytest

from albums.memory_store import Memory, replace_memories
from chat.agent import direct_answer
from tests.factories import add_photo


@pytest.fixture
def library(conn):
    # 5 photos spanning two years, and two memories (Borjomi=2, Tbilisi=3 photos).
    for pid, when in enumerate(
        ("2023-06-01T00:00:00", "2023-07-01T00:00:00", "2024-01-01T00:00:00",
         "2024-02-01T00:00:00", "2024-03-01T00:00:00"), start=1):
        add_photo(conn, photo_id=pid, content_hash=f"h{pid}", suffix=".jpg", shot_at=when)
    replace_memories(conn, 1, [
        Memory("A day at Borjomi", "Hiking in Borjomi park.", [1, 2], "sig"),
        Memory("Tbilisi evening", "Old town at night.", [3, 4, 5], "sig"),
    ])
    return conn


DIRECT = [
    # total count
    "how many photos do I have?",
    "how many images in my libray?",              # typo
    "total number of pics",
    "how many photos do I have in total?",
    "how many photos altogether",
    # subject count
    "how many photos with dogs",
    "number of beach photos",
    "how many photos of my car",
    # memory count
    "how many memories do I have?",
    "number of memories",
    # memory show / list — the "all" battleground
    "show me all my memories",
    "show me my memories",
    "list my memories",
    "list all memories",
    "show me every memory",
    "show me a memory",
    "show me my borjomi memory",
    "open my antarctica memory",                  # miss, still answered by the DB
    # periods
    "how many months have photos",
    "how many years",
]

AGENT = [
    "photos of dogs on a beach",
    "show me sunset pictures",                    # "show me X", X != memories
    "what lens did I use most in Italy?",
    "what years do my photos span?",              # no count-intent → decline (conservative)
    "how many photos similar to this dog?",       # relational count → decline
    "how many photos are in my Borjomi memory?",  # photos-in-a-memory != memory count
    "do I have any photos of cats?",
]


@pytest.mark.parametrize("question", DIRECT)
def test_direct_db_answers_without_the_planner(library, question):
    assert direct_answer(library, owner_id=1, question=question) is not None


@pytest.mark.parametrize("question", AGENT)
def test_semantic_questions_decline_to_the_agent(library, question):
    assert direct_answer(library, owner_id=1, question=question) is None


def test_show_all_memories_reports_the_count(library):
    answer = direct_answer(library, owner_id=1, question="show me all my memories")
    assert "2" in answer                                   # both memories, not "no match"


def test_specific_memory_miss_is_honest_not_a_planner_trip(library):
    answer = direct_answer(library, owner_id=1, question="open my antarctica memory")
    assert answer is not None and "antarctica" in answer.lower()
