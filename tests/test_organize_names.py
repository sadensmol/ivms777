from organize.names import folder_name, unique_folder_names


def test_reserved_and_separator_characters_become_dashes():
    assert folder_name('Trip: Kyiv/Rome <2024>', fallback="k") == "Trip- Kyiv-Rome -2024-"


def test_whitespace_is_collapsed_and_trimmed():
    assert folder_name("  June   2024 \n", fallback="k") == "June 2024"


def test_trailing_dots_are_stripped_windows_cannot_hold_them():
    assert folder_name("June 2024...", fallback="k") == "June 2024"


def test_a_title_that_sanitizes_to_nothing_falls_back_to_the_album_key():
    assert folder_name("///", fallback="memory-3") == "memory-3"


def test_long_titles_are_truncated_to_120_characters():
    assert len(folder_name("x" * 400, fallback="k")) == 120


def test_colliding_folder_names_get_a_deterministic_suffix():
    names = unique_folder_names(
        [("Waterfalls", "a"), ("Waterfalls", "b"), ("waterfalls", "c")]
    )
    assert names == ["Waterfalls", "Waterfalls_2", "waterfalls_3"]
