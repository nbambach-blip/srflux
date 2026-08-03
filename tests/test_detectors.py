"""Detector behaviour on signals whose answer is known by construction."""
import numpy as np
import pytest

from srflux import HaarDetector, VanAttaDetector, chen_lag, haar_kernel, prepare_block
from srflux.detectors.vanatta import solve_cubic
from srflux.synthetic import ramp_series, step_series


# ---------------------------------------------------------------- Haar / SR-WL
def test_haar_kernel_normalised():
    """Convolving a unit step returns a coefficient of unit magnitude."""
    k = haar_kernel(32, fs=1.0)
    assert len(k) == 32
    assert k.sum() == pytest.approx(0.0, abs=1e-12)
    v = np.r_[np.zeros(200), np.ones(200)]
    assert np.abs(np.convolve(v, k, mode="same")).max() == pytest.approx(1.0, rel=0.02)


def test_haar_recovers_step_height():
    """A square wave of known height is the cleanest amplitude test."""
    v = step_series(n=1800, step_s=100, amplitude=2.0)
    res = HaarDetector(fs=1.0).detect(v)
    assert res.valid
    assert res.amplitude == pytest.approx(2.0, rel=0.15)
    assert res.count == pytest.approx(1800 / 100, abs=2)     # one front per half-period


def test_haar_count_tracks_ramp_rate():
    """Halving the ramp period should roughly double the number of fronts."""
    slow = HaarDetector(fs=1.0).detect(ramp_series(period_s=120, amplitude=1.0))
    fast = HaarDetector(fs=1.0).detect(ramp_series(period_s=60, amplitude=1.0))
    assert fast.count > slow.count


def test_haar_amplitude_scales_linearly():
    a1 = HaarDetector(fs=1.0).detect(ramp_series(period_s=60, amplitude=1.0)).amplitude
    a3 = HaarDetector(fs=1.0).detect(ramp_series(period_s=60, amplitude=3.0)).amplitude
    assert a3 / a1 == pytest.approx(3.0, rel=0.05)


def test_haar_is_scale_free_to_threshold():
    """The sigma-relative threshold means a rescaled signal gives the same COUNT."""
    v = ramp_series(period_s=60, amplitude=1.0, noise=0.05, seed=3)
    r1 = HaarDetector(fs=1.0).detect(v)
    r2 = HaarDetector(fs=1.0).detect(10 * v)
    assert r1.count == r2.count
    assert r2.amplitude / r1.amplitude == pytest.approx(10.0, rel=1e-6)


def test_haar_rejects_flat_signal():
    assert not HaarDetector(fs=1.0).detect(np.zeros(1800)).valid


def test_haar_survives_a_trend():
    """The 300 s detrend should stop a linear drift from swamping the ramps."""
    clean = HaarDetector(fs=1.0).detect(ramp_series(period_s=60, amplitude=1.0, seed=0))
    drift = HaarDetector(fs=1.0).detect(
        ramp_series(period_s=60, amplitude=1.0, trend=20.0, seed=0))
    assert drift.amplitude == pytest.approx(clean.amplitude, rel=0.1)


def test_haar_handles_higher_sampling_rate():
    """Scale is specified in seconds, so 4 Hz data must give the same answer as 1 Hz."""
    r1 = HaarDetector(fs=1.0).detect(ramp_series(n=1800, fs=1.0, period_s=60, amplitude=1.0))
    r4 = HaarDetector(fs=4.0).detect(ramp_series(n=7200, fs=4.0, period_s=60, amplitude=1.0))
    assert r4.amplitude == pytest.approx(r1.amplitude, rel=0.15)
    assert r4.count == pytest.approx(r1.count, rel=0.2)


# ---------------------------------------------------------------- Van Atta / SR-VA
def test_vanatta_recovers_synthetic_amplitude():
    v = ramp_series(n=3600, period_s=60, amplitude=1.0, rise_fraction=0.9)
    res = VanAttaDetector(fs=1.0).detect(v)
    assert res.valid
    assert res.amplitude == pytest.approx(1.0, rel=0.5)      # cubic on an ideal sawtooth


def test_vanatta_amplitude_scales_linearly():
    a1 = VanAttaDetector(fs=1.0).detect(ramp_series(n=3600, period_s=60, amplitude=1.0))
    a3 = VanAttaDetector(fs=1.0).detect(ramp_series(n=3600, period_s=60, amplitude=3.0))
    assert a3.amplitude / a1.amplitude == pytest.approx(3.0, rel=0.05)


def test_vanatta_sign_follows_ramp_direction():
    """The signed root, not |A|, carries the flux direction."""
    warm = VanAttaDetector(fs=1.0).detect(
        ramp_series(n=3600, period_s=60, amplitude=1.0, warm=True))
    cold = VanAttaDetector(fs=1.0).detect(
        ramp_series(n=3600, period_s=60, amplitude=1.0, warm=False))
    assert warm.extra["signed_amplitude"] * cold.extra["signed_amplitude"] < 0


def test_chen_lag_grows_with_ramp_period():
    """The adaptive lag is the point of the Chen criterion: slow ramps need a longer lag."""
    fast = chen_lag(ramp_series(n=3600, period_s=20, amplitude=1.0), fs=1.0)
    slow = chen_lag(ramp_series(n=3600, period_s=200, amplitude=1.0), fs=1.0)
    assert slow > fast


def test_fixed_short_lag_collapses_on_a_smooth_ramp():
    """Documented failure mode: a 1 s lag has almost no signal on a slow, smooth ramp."""
    v = ramp_series(n=3600, period_s=300, amplitude=1.0, rise_fraction=0.7)
    fixed = VanAttaDetector(fs=1.0, lag=1.0).detect(v)
    chen = VanAttaDetector(fs=1.0, lag="chen").detect(v)
    assert chen.amplitude > 3 * fixed.amplitude


def test_solve_cubic_rejects_impossible_lag():
    x = ramp_series(n=400, period_s=60)
    assert np.isnan(solve_cubic(x, 0)[0])
    assert np.isnan(solve_cubic(x, len(x))[0])


# ---------------------------------------------------------------- both
@pytest.mark.parametrize("detector", [HaarDetector(fs=1.0), VanAttaDetector(fs=1.0)])
def test_detectors_share_the_result_contract(detector):
    res = detector.detect(ramp_series(n=1800, period_s=60, amplitude=1.0, seed=5))
    assert res.valid and res.count > 0 and res.amplitude > 0 and res.period > 0
    assert res.detector in {"haar", "van_atta"}


@pytest.mark.parametrize("detector", [HaarDetector(fs=1.0), VanAttaDetector(fs=1.0)])
def test_detectors_accept_a_prepared_block(detector):
    raw = ramp_series(n=1800, period_s=60, amplitude=1.0, noise=0.02, seed=2)
    raw[500:520] = np.nan                       # a short gap the QC should interpolate
    block = prepare_block(raw)
    assert block.ok
    assert detector.detect(block.values).valid


# ---------------------------------------------------------------- unit-period Van Atta
def test_unit_period_reports_amplitude_only():
    """period_mode='unit' discards tau: the period is 1 by definition, A is unchanged."""
    v = ramp_series(n=3600, period_s=60, amplitude=1.0)
    fitted = VanAttaDetector(fs=1.0, lag=4.0).detect(v)
    unit = VanAttaDetector(fs=1.0, lag=4.0, period_mode="unit").detect(v)
    assert unit.period == 1.0
    assert unit.amplitude == pytest.approx(fitted.amplitude)
    assert unit.valid
    assert unit.extra["period_mode"] == "unit"


def test_unit_period_keeps_the_fitted_tau_for_diagnostics():
    v = ramp_series(n=3600, period_s=60, amplitude=1.0)
    unit = VanAttaDetector(fs=1.0, lag=4.0, period_mode="unit").detect(v)
    assert np.isfinite(unit.extra["tau_fitted"])
    assert unit.extra["tau_fitted"] != 1.0


def test_unit_period_survives_an_unusable_tau():
    """The point of the mode: a block whose tau is garbage still yields an amplitude."""
    rng = np.random.default_rng(11)
    v = rng.normal(0, 0.05, 1800)                 # noise, no ramp structure
    fitted = VanAttaDetector(fs=1.0, lag=4.0).detect(v)
    unit = VanAttaDetector(fs=1.0, lag=4.0, period_mode="unit").detect(v)
    assert unit.valid or not np.isfinite(unit.amplitude)
    if not fitted.valid:
        assert unit.valid                          # unit mode recovers what fitted drops


def test_unit_period_flux_is_amplitude_scaled():
    """F = rho cp z A when tau = 1, so it must be linear in the amplitude."""
    from srflux import ramp_flux
    v1 = ramp_series(n=3600, period_s=60, amplitude=1.0)
    v3 = ramp_series(n=3600, period_s=60, amplitude=3.0)
    det = VanAttaDetector(fs=1.0, lag=4.0, period_mode="unit")
    f1 = ramp_flux(det.detect(v1).amplitude, period=det.detect(v1).period, z=2.0)
    f3 = ramp_flux(det.detect(v3).amplitude, period=det.detect(v3).period, z=2.0)
    assert f3 / f1 == pytest.approx(3.0, rel=0.05)


def test_period_mode_is_validated():
    with pytest.raises(ValueError):
        VanAttaDetector(fs=1.0, period_mode="nonsense")
