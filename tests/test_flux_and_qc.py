"""Calibration, energy-balance closure to ET, block QC and the sign conventions."""
import numpy as np
import pytest

from srflux import (calibrate_alpha, daily_et, fit_and_score, latent_heat_residual,
                    prepare_block, ramp_flux, sensible_heat, sign_from_skewness,
                    sign_from_stability)
from srflux.preprocess import TEMP_RANGE, increment_skewness, moving_average
from srflux.synthetic import ramp_series


def test_count_and_period_forms_agree_when_consistent():
    """N/block and 1/tau are the same rate, so the two forms must coincide."""
    f_count = ramp_flux(0.3, count=30, z=2.0, block_s=1800.0)
    f_period = ramp_flux(0.3, period=60.0, z=2.0)
    assert f_count == pytest.approx(f_period)


def test_ramp_flux_requires_exactly_one_rate():
    with pytest.raises(ValueError):
        ramp_flux(0.3, z=2.0)
    with pytest.raises(ValueError):
        ramp_flux(0.3, count=30, period=60.0, z=2.0)


def test_flux_is_linear_in_amplitude_and_height():
    base = ramp_flux(0.2, count=30, z=2.0)
    assert ramp_flux(0.4, count=30, z=2.0) == pytest.approx(2 * base)
    assert ramp_flux(0.2, count=30, z=4.0) == pytest.approx(2 * base)


def test_calibrate_alpha_recovers_a_known_scaling():
    rng = np.random.default_rng(0)
    F = rng.uniform(5, 60, 500)
    assert calibrate_alpha(F, 3.4 * F) == pytest.approx(3.4)


def test_calibrate_alpha_is_unbiased_under_symmetric_noise():
    rng = np.random.default_rng(1)
    F = rng.uniform(5, 60, 20000)
    H = 2.0 * F + rng.normal(0, 10, F.size)
    assert calibrate_alpha(F, H) == pytest.approx(2.0, rel=0.02)


def test_calibrate_alpha_ignores_missing_pairs():
    F = np.array([1.0, 2.0, np.nan, 4.0])
    H = np.array([2.0, 4.0, 100.0, np.nan])
    assert calibrate_alpha(F, H) == pytest.approx(2.0)


def test_calibrate_alpha_degenerate_inputs():
    assert np.isnan(calibrate_alpha([1.0], [2.0]))
    assert np.isnan(calibrate_alpha([0.0, 0.0], [1.0, 2.0]))


def test_alpha_is_the_minimum_rmse_scaling():
    """Through-origin is a least-squares fit: no other alpha can beat it on RMSE."""
    rng = np.random.default_rng(2)
    F = rng.uniform(5, 60, 2000)
    H = 2.0 * F + rng.normal(0, 15, F.size)
    fit = fit_and_score(F, H)
    for other in (fit.alpha * 0.9, fit.alpha * 1.1):
        assert np.sqrt(np.mean((other * F - H) ** 2)) > fit.rmse


def test_fit_and_score_reports_scale_invariant_r():
    """r does not depend on alpha, so rescaling F must not change it."""
    rng = np.random.default_rng(3)
    F = rng.uniform(5, 60, 1000)
    H = 2.0 * F + rng.normal(0, 12, F.size)
    assert fit_and_score(F, H).r == pytest.approx(fit_and_score(10 * F, H).r)


def test_sensible_heat_applies_direction():
    assert sensible_heat(100.0, 2.0, sign=-1.0) == pytest.approx(-200.0)


def test_daily_et_of_a_known_energy_input():
    """48 blocks of pure LE at 100 W m-2 for 24 h -> 4.32 MJ/m2 -> ~1.76 mm."""
    le = np.full(48, 100.0)
    assert daily_et(le) == pytest.approx(48 * 100.0 * 1800.0 / 2.45e6, rel=1e-9)


def test_residual_closes_the_energy_balance():
    rn, g, h = 500.0, 60.0, 150.0
    assert latent_heat_residual(rn, g, h) == pytest.approx(290.0)


def test_et_error_from_an_alpha_error_is_bounded_by_the_h_term():
    """A wrong alpha can only move ET through H, so the ET error is exactly the H error."""
    F = np.full(20, 120.0)
    rn, g = np.full(20, 600.0), np.full(20, 70.0)
    et_true = daily_et(latent_heat_residual(rn, g, sensible_heat(F, 2.0)))
    et_wrong = daily_et(latent_heat_residual(rn, g, sensible_heat(F, 3.0)))
    expected = daily_et(np.full(20, (3.0 - 2.0) * 120.0))
    assert et_true - et_wrong == pytest.approx(expected)


def test_prepare_block_clips_out_of_range_sentinels():
    v = ramp_series(n=1800, period_s=60, amplitude=1.0) + 20.0
    v[100:110] = -8190.0
    block = prepare_block(v)
    assert block.ok and block.n_clipped == 10
    assert block.values.min() > TEMP_RANGE[0] and block.values.max() < TEMP_RANGE[1]


def test_prepare_block_rejects_a_mostly_missing_block():
    v = np.full(1800, np.nan)
    v[:400] = 20.0
    assert not prepare_block(v).ok


def test_prepare_block_rejects_a_long_outage():
    v = ramp_series(n=1800, period_s=60, amplitude=1.0) + 20.0
    v[200:1000] = np.nan
    assert not prepare_block(v).ok


def test_prepare_block_fills_short_gaps():
    v = ramp_series(n=1800, period_s=60, amplitude=1.0) + 20.0
    v[300:310] = np.nan
    block = prepare_block(v)
    assert block.ok and np.isfinite(block.values).all() and block.n_valid == 1790


def test_moving_average_keeps_the_edges():
    v = np.arange(100, dtype=float)
    assert np.isfinite(moving_average(v, 21)).all()


def test_skewness_separates_warm_and_cold_ramps():
    warm = ramp_series(n=3600, period_s=60, amplitude=1.0, warm=True)
    cold = ramp_series(n=3600, period_s=60, amplitude=1.0, warm=False)
    assert increment_skewness(warm) < 0 < increment_skewness(cold)
    assert sign_from_skewness(warm) == 1.0
    assert sign_from_skewness(cold) == -1.0


def test_sign_from_stability_follows_zeta():
    assert list(sign_from_stability([-0.5, 0.5, 0.0])) == [1.0, -1.0, 1.0]


def test_sign_from_skewness_needs_enough_samples():
    assert np.isnan(sign_from_skewness(np.zeros(10)))


def test_nse_is_negative_for_a_near_zero_prediction():
    """The failure r cannot see: a prediction of ~0 whose sign follows the truth."""
    from srflux.flux import scores
    rng = np.random.default_rng(21)
    H = np.r_[rng.uniform(50, 200, 40), rng.uniform(-80, -20, 40)]
    F = np.sign(H) * 1e-4
    sc = scores(F, H, 1.0)
    assert sc["r"] > 0.8
    assert sc["rmse"] == pytest.approx(np.sqrt(np.mean(H ** 2)), rel=0.01)
    assert sc["nse"] < 0


def test_nse_is_high_for_a_good_fit():
    rng = np.random.default_rng(22)
    F = rng.uniform(5, 60, 500)
    H = 2.0 * F + rng.normal(0, 5, F.size)
    fit = fit_and_score(F, H)
    assert fit.nse > 0.9
    assert fit.nse <= 1.0


def test_nse_of_the_mean_prediction_is_zero():
    rng = np.random.default_rng(23)
    H = rng.normal(100, 40, 300)
    F = np.full(H.size, H.mean())
    from srflux.flux import scores
    assert scores(F, H, 1.0)["nse"] == pytest.approx(0.0, abs=1e-9)


def test_skewness_sign_needs_a_lag_matched_to_the_ramp():
    """A slow ramp buried in fast noise: lag 1 samples the noise, a longer lag the ramp.

    Scored over many realisations rather than one, since at lag 1 the sign is close to a coin
    toss and any single draw can come out either way -- which is the point.
    """
    from srflux import sign_from_skewness
    rng = np.random.default_rng(31)
    hits = {1.0: 0, 20.0: 0}
    n = 40
    for _ in range(n):
        v = ramp_series(n=3600, period_s=300, amplitude=1.0, rise_fraction=0.9, warm=True,
                        noise=0.25, seed=int(rng.integers(1e6)))
        for lag in hits:
            hits[lag] += sign_from_skewness(v, lag_s=lag) == 1.0
    assert hits[20.0] >= 0.9 * n
    assert hits[1.0] <= 0.7 * n


def test_longer_lag_raises_the_ramp_signal_above_the_noise():
    """The mechanism behind the test above: |skewness| grows with lag on a noisy slow ramp."""
    from srflux.preprocess import increment_skewness
    v = ramp_series(n=3600, period_s=300, amplitude=1.0, rise_fraction=0.9, noise=0.25, seed=7)
    assert abs(increment_skewness(v, lag=20)) > 3 * abs(increment_skewness(v, lag=1))


def test_fit_skewness_sign_recovers_the_useful_lag():
    from srflux import fit_skewness_sign
    rng = np.random.default_rng(32)
    blocks, ref = [], []
    for i in range(40):
        warm = i % 2 == 0
        v = ramp_series(n=1800, period_s=300, amplitude=1.0, warm=warm,
                        seed=int(rng.integers(1e6)))
        blocks.append(v + rng.normal(0, 0.3, v.size))
        ref.append(120.0 if warm else -60.0)
    lag_s, tau, acc = fit_skewness_sign(blocks, ref, lags_s=(1, 5, 20))
    assert acc > 0.9
    assert lag_s > 1.0
    assert np.isfinite(tau)
