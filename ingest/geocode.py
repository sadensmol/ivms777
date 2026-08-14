"""Offline reverse geocoding: GPS -> place name.

Uses a bundled GeoNames dataset (via `reverse_geocoder`), so a coordinate is
turned into a city/country entirely on the box — no network, nothing leaves the
machine, works on the Jetson offline. Country codes become full names via
`pycountry`.
"""

from dataclasses import dataclass

import pycountry
import reverse_geocoder


@dataclass(frozen=True)
class Place:
    city: str | None
    region: str | None
    country: str | None

    @property
    def label(self) -> str:
        # "City, Country" when both are known; otherwise the coarsest part we have.
        if self.city and self.country:
            return f"{self.city}, {self.country}"
        return self.city or self.country or self.region or "Unknown location"


def _country_name(code: str | None) -> str | None:
    if not code:
        return None
    match = pycountry.countries.get(alpha_2=code)
    return match.name if match else code


def _to_place(record: dict) -> Place:
    return Place(
        city=record.get("name") or None,
        region=record.get("admin1") or None,
        country=_country_name(record.get("cc")),
    )


def reverse_many(coords: list[tuple[float, float]]) -> list[Place]:
    """Resolve many points in one pass (a single KDTree query)."""
    if not coords:
        return []
    results = reverse_geocoder.search(coords, mode=1)
    return [_to_place(record) for record in results]


def reverse(lat: float, lon: float) -> Place | None:
    places = reverse_many([(lat, lon)])
    return places[0] if places else None
