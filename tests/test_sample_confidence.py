import pytest

from app.analytics.sample_confidence import assess_serve_sample, wilson_interval


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(24, 35)
    assert low < 24 / 35 < high
    assert 0 <= low < high <= 1


def test_sample_confidence_flags_small_live_sample():
    result = assess_serve_sample(12, 18)
    assert result["sample_quality"] == "low"
    assert result["sample_size"] == 18
    assert result["warning"]
    assert result["wilson_95_interval"][0] < result["observed_rate"]
    assert result["wilson_95_interval"][1] > result["observed_rate"]


def test_sample_confidence_increases_with_more_points():
    small = assess_serve_sample(20, 30)
    large = assess_serve_sample(40, 60)
    small_width = small["wilson_95_interval"][1] - small["wilson_95_interval"][0]
    large_width = large["wilson_95_interval"][1] - large["wilson_95_interval"][0]
    assert large_width < small_width
    assert large["sample_quality"] == "stronger"


@pytest.mark.parametrize("won,played", [(-1, 10), (11, 10), (0, 0)])
def test_invalid_sample_counts_are_rejected(won, played):
    with pytest.raises(ValueError):
        assess_serve_sample(won, played)
