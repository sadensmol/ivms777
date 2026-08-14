from ingest.geocode import reverse, reverse_many


def test_kyiv_resolves_to_a_city_and_country():
    place = reverse(50.4501, 30.5234)
    assert place is not None
    assert place.city in ("Kyiv", "Kiev")
    assert place.country == "Ukraine"
    assert place.label == f"{place.city}, Ukraine"


def test_rome_resolves():
    place = reverse(41.9028, 12.4964)
    assert place.city == "Rome"
    assert place.country == "Italy"


def test_no_coordinates_leak_into_the_label():
    place = reverse(43.6532, -79.3832)  # Toronto
    assert "°" not in place.label
    assert place.country == "Canada"


def test_reverse_many_returns_one_place_per_point():
    places = reverse_many([(41.9028, 12.4964), (43.6532, -79.3832)])
    assert [p.country for p in places] == ["Italy", "Canada"]
