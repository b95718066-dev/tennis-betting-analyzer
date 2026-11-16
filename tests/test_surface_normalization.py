import pytest

from app.data.normalization import normalize_surface


@pytest.mark.parametrize(
    ("source_value", "expected"),
    [
        ("hard", "hard"),
        ("dur", "hard"),
        ("dur intérieur", "hard"),
        ("clay", "clay"),
        ("terre battue", "clay"),
        ("grass", "grass"),
        ("gazon", "grass"),
        ("carpet", "carpet"),
        ("moquette intérieur", "carpet"),
    ],
)
def test_normalizes_canonical_and_valuebetennis_surface_labels(source_value, expected):
    assert normalize_surface(source_value) == expected


def test_unknown_surface_is_not_guessed():
    assert normalize_surface("synthetic unknown court") is None
    assert normalize_surface(None) is None
