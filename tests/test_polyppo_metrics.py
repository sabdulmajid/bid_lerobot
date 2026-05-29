import math

import pytest

from lerobot.common.polyppo.metrics import compute_pass_at_k, mean_confidence_interval


def test_compute_pass_at_k_uses_ordered_attempt_prefixes():
    successes = [
        [False, True, False, False],
        [False, False, False, True],
        [False, False, False, False],
    ]

    out = compute_pass_at_k(successes, ks=(1, 2, 4, 8))

    assert out["pass@1"] == 0.0
    assert out["pass@2"] == pytest.approx(1 / 3)
    assert out["pass@4"] == pytest.approx(2 / 3)
    assert out["pass@8"] == pytest.approx(2 / 3)


def test_mean_confidence_interval_ignores_non_finite_values():
    out = mean_confidence_interval([1.0, 2.0, float("nan"), float("inf"), 3.0])

    assert out["n"] == 3
    assert out["mean"] == pytest.approx(2.0)
    assert math.isfinite(out["ci95_low"])
    assert math.isfinite(out["ci95_high"])
