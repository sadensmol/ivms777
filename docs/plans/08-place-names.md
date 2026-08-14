# Photo Library Organizer — Plan 08: Place names (offline reverse geocoding)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn GPS into **real place names**. The "By place" organizer stops showing "Location 1 / Location 2" (the coordinate-free placeholder from §11) and instead groups and titles albums by recognizable names — "Kyiv", "Rome", "Lake Ontario". Raw coordinates never appear in Organize; they stay a technical detail on `/photo` (§13). Place names also become a queryable facet in the `/library` sidebar. All offline — no geocoding API, nothing leaves the box.

**Architecture:** A small **offline reverse geocoder** maps a lat/long to the nearest known place (city / region / country) from a bundled dataset, with no network. The EXIF **facets** stage derives `place_city` and `place_country` facets from a photo's GPS (§6.2), so place is filterable like any other facet and the organizer just groups by it. The organizer groups photos by place label (falling back to the ordinal "Location N" only when a point can't be resolved) and titles each album with the name.

**Tech Stack:** Python 3.12, an offline reverse-geocoding dataset (GeoNames `cities1000`, ~2 MB, bundled — via the `reverse_geocoder` package or a small in-repo KDTree over the same data), SQLite, FastAPI, Jinja2. No new runtime services.

**Spec:** `docs/design.md` — §11 (By place → real names, no coordinates in Organize), §6.2 (place facets), §13 (`/photo` is the only place coordinates appear; `/library` place filter), §18 (this is the reverse-geocoding item).

**Builds on:** the facet pipeline (plan 02) and the `albums/by_place.py` organizer (already live, currently ordinal-labelled). Only photos with GPS are affected; everything else is untouched.

**Covers:** offline reverse geocoding, place facets, the named "By place" organizer, and a place filter in the library sidebar. **Deferred:** on-device map view; user-editable place labels; sub-city neighbourhoods.

## Global Constraints

- Python 3.12. Dependencies via `uv` with a committed `uv.lock`.
- **Never run `git commit`/`git add`.** The user commits. Every task ends at a checkpoint.
- Every user-scoped query filters on `owner_id`.
- **Offline only** — the geocoder must never hit the network; its data is bundled and loaded from disk. Tests assert no network use (known coordinates resolve to known names).
- **No coordinates in Organize** (§11). A raw lat/long may appear only on `/photo`.
- The full fast suite passes at the end of every task: `uv run pytest -q`; `uv run ruff check .` clean.

---

### Task 1: The offline reverse geocoder

**Files:**
- Create: `ingest/geocode.py`
- Create: `tests/test_geocode.py`
- Modify: `pyproject.toml` (add the offline dataset dependency)

**Interfaces:**
- Produces:
  - `ingest.geocode.Place` — `(city: str | None, region: str | None, country: str | None)` with a `label` property ("Kyiv, Ukraine", or the coarsest part available).
  - `ingest.geocode.reverse(lat: float, lon: float) -> Place | None` — nearest place, `None` for an unresolvable point (e.g. mid-ocean beyond a distance cutoff). The dataset loads once and is reused (module-level, lazy).

- [ ] **Step 1: Write the failing test** — known coordinates resolve to known names, offline:

```python
from ingest.geocode import reverse


def test_kyiv_coordinates_resolve_to_kyiv():
    place = reverse(50.4501, 30.5234)
    assert place is not None
    assert "Kyiv" in place.label or "Kiev" in place.label
    assert "Ukraine" in place.label


def test_label_is_city_and_country():
    place = reverse(41.9028, 12.4964)  # Rome
    assert place.city and place.country
    assert place.label == f"{place.city}, {place.country}"


def test_two_nearby_points_share_a_place():
    a = reverse(50.4501, 30.5234)
    b = reverse(50.4600, 30.5300)  # ~1.5 km away, same city
    assert a.city == b.city
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: ingest.geocode`.

- [ ] **Step 3: Add the dataset dep** — `uv add reverse_geocoder` (bundles GeoNames `cities1000`; fully offline). If its transitive `scipy`/`numpy` footprint is unwanted, the alternative is to vendor `cities1000.txt` and build a `scipy.spatial.cKDTree` in-repo — same data, no extra package. Pick one and note the choice.

- [ ] **Step 4: Write `ingest/geocode.py`** — wrap the offline lookup; build `Place` from the nearest record's city/admin1/country; `label` joins the non-empty parts, coarsest fallback when city is absent. Guard the lazy load so the dataset is read once per process.

- [ ] **Step 5–7:** run the geocode tests (PASS), run the whole suite (PASS), checkpoint — report the dataset choice, its size, and that it loads once and never touches the network.

---

### Task 2: Place facets from GPS

Derive `place_city` and `place_country` in the facets stage (§6.2) so place is filterable and the organizer reads it instead of re-geocoding.

**Files:**
- Modify: `ingest/facets.py` (add the two facets when GPS is present)
- Create/extend: `tests/test_facets.py` (a GPS photo gets `place_city` / `place_country`)
- Add a `backfill`/reprocess note: `POST /reprocess?from=facets` re-derives facets for the existing library (the reprocess machinery from §8 already exists).

**Interfaces:**
- Produces: `photo_facets` rows `place_city` and `place_country` (categorical, `value_text`) for every photo with GPS.

- [ ] **Step 1: Failing test** — a photo with Kyiv GPS yields `place_city='Kyiv'` (or 'Kiev') and `place_country='Ukraine'` facets.
- [ ] **Step 2–4:** derive both facets in `ingest/facets.py` via `ingest.geocode.reverse(lat, lon)` when `gps_lat`/`gps_lon` are set; skip cleanly when absent or unresolvable. Register `place_city`/`place_country` in `FACET_KEYS`.
- [ ] **Step 5:** run tests + suite (PASS). Checkpoint — note that `from=facets` reprocess names the existing library.

---

### Task 3: The named "By place" organizer

**Files:**
- Modify: `albums/by_place.py`
- Modify: `tests/test_albums.py`

**Interfaces:**
- Produces: albums grouped by **place name**, titled with it ("Rome"), described "N photos in Rome." Photos whose GPS can't be resolved fall into a single "Location (unknown)" album — never a coordinate string.

- [ ] **Step 1: Failing test** — two photos with Rome GPS and one with Kyiv GPS produce a "Rome" album (size 2) and a "Kyiv" album (size 1); no album title contains a digit-degree/`°` coordinate.
- [ ] **Step 2–3:** group by the `place_city` facet (falling back to `place_country`, then an "unknown" bucket). Title = the place name; description = count + name. Keep the `~1 km` cell only as an internal tiebreak if two same-named clusters must stay separate — but prefer one album per named place, which is the whole point.
- [ ] **Step 4:** run tests + suite (PASS). Checkpoint.

---

### Task 4: Place filter in the library sidebar

Let people filter the grid by place, like any other facet.

**Files:**
- Modify: `search/facets.py` (`SIDEBAR_GROUPS` gains a "Place" group with `place_city`, `place_country`)
- Modify: `tests/test_web_facet_filters.py` (filtering `f_place_city=Rome` narrows the grid)

- [ ] **Step 1: Failing test** — `GET /library?f_place_city=Rome` returns only the Rome photos; the sidebar lists place values with counts.
- [ ] **Step 2–3:** add the Place group to `SIDEBAR_GROUPS`; the existing facet machinery (`build_where`, `facet_counts`, the auto-applying checkboxes) handles the rest with no new code.
- [ ] **Step 4:** run tests + suite (PASS); `ruff` clean. Checkpoint.

---

### Task 5: Verify against a real library and update the docs

- [ ] **Step 1:** on a library with GPS photos, run `POST /reprocess?from=facets`, then open **/organize?by=place** — albums are named places, no coordinates; and **/library** filters by place. Spot-check a few names against where the photos were actually taken. Confirm `/photo` still shows the raw coordinates in its EXIF panel (the one place they belong).
- [ ] **Step 2:** update `README.md` (place names are offline, from a bundled dataset) and confirm `docs/design.md` §11/§6.2/§13/§18 match the implementation.
- [ ] **Step 3:** checkpoint — report accuracy spot-checks and the dataset size. Stop. Do not commit.

---

## What plan 08 delivers

"By place" becomes a real thing: open Organize and see **Rome**, **Kyiv**, **Lake Ontario** — not "Location 1" and never a lat/long. Place is also a sidebar filter, so you can narrow the whole library to one city. It's all offline — a small bundled GeoNames dataset resolves coordinates on the box, nothing is sent anywhere — and coordinates keep their one honest home on the photo detail page, as the technical detail they are.
