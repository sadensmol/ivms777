"""`read_temps` — the board temperatures the resource bar shows (design §13).

Driven against a fake sysfs tree, so the suite stays offline and passes on a mac
where no thermal zone exists.
"""

from models import thermal
from models.thermal import read_temps


def _zones(tmp_path, zones):
    """Build a fake /sys/devices/virtual/thermal tree; return its glob."""
    for index, (kind, temp) in enumerate(zones):
        zone = tmp_path / f"thermal_zone{index}"
        zone.mkdir()
        (zone / "type").write_text(f"{kind}\n")
        (zone / "temp").write_text(temp)
    return str(tmp_path / "thermal_zone*")


def test_reads_cpu_gpu_and_junction_in_celsius(tmp_path, monkeypatch):
    # Millidegrees on the wire, °C in the payload.
    monkeypatch.setattr(
        thermal,
        "_ZONE_GLOB",
        _zones(tmp_path, [("cpu-thermal", "51406\n"), ("gpu-thermal", "52625\n"),
                          ("tj-thermal", "53000\n")]),
    )
    assert read_temps() == {"cpu_c": 51.406, "gpu_c": 52.625, "tj_c": 53.0}


def test_unreadable_zones_are_skipped_never_reported_as_zero(tmp_path, monkeypatch):
    # The Orin's cv0/cv1/cv2 zones read EMPTY. Reporting them as 0 °C would put a
    # bogus number in the bar, which is worse than omitting the reading.
    monkeypatch.setattr(
        thermal,
        "_ZONE_GLOB",
        _zones(tmp_path, [("cpu-thermal", "51406\n"), ("cv0-thermal", ""),
                          ("gpu-thermal", "")]),
    )
    assert read_temps() == {"cpu_c": 51.406}


def test_soc_and_cv_zones_are_not_surfaced(tmp_path, monkeypatch):
    # Two numbers the user can act on, not nine.
    monkeypatch.setattr(
        thermal,
        "_ZONE_GLOB",
        _zones(tmp_path, [("soc0-thermal", "51968\n"), ("soc1-thermal", "52687\n"),
                          ("cpu-thermal", "51406\n")]),
    )
    assert read_temps() == {"cpu_c": 51.406}


def test_no_thermal_zones_off_tegra_returns_empty(tmp_path, monkeypatch):
    # mac/cloud have no such tree — the bar must simply omit the field.
    monkeypatch.setattr(thermal, "_ZONE_GLOB", str(tmp_path / "nothing_here*"))
    assert read_temps() == {}
