# srflux

Surface-renewal sensible heat flux from a high-rate scalar series, with two ramp detectors:

| | method | what it measures |
|---|---|---|
| **SR-WL** | Haar wavelet front picker | counts microfronts and measures their amplitude directly |
| **SR-VA** | Van Atta structure-function cubic | fits an idealised ramp to S2/S3/S5 |

Surface renewal estimates the sensible heat flux from the temperature ramps that coherent
structures leave in a scalar time series, needing only a fast thermometer or radiometer
rather than a sonic anemometer. Both detectors produce an *uncalibrated* flux that is closed
by one dimensionless coefficient `alpha`.

```python
from srflux import HaarDetector, VanAttaDetector, prepare_block, ramp_flux, fit_and_score

block = prepare_block(temperature_1hz, fs=1.0)                  # clip, gap-fill, validate
wl = HaarDetector(fs=1.0).detect(block.values)                  # SR-WL
va = VanAttaDetector(fs=1.0, lag="chen").detect(block.values)   # SR-VA

F_wl = ramp_flux(wl.amplitude, count=wl.count, z=2.0)           # uncalibrated, W m-2
F_va = ramp_flux(va.amplitude, period=va.period, z=2.0)

fit = fit_and_score(F_wl_all_blocks, H_reference)               # alpha, r, NSE, RMSE, bias
H_sr = fit.apply(F_wl_all_blocks)
```

Two runnable examples:

- **`examples/quickstart.py`** — the whole chain on synthetic data, no field data needed.
- **`examples/ola_one_day.ipynb`** — one real day from an almond orchard: 1 Hz fine-wire and
  canopy-skin temperature plus the tower's Rn/G/H/LE, with α and the flux-direction
  convention loaded from a calibration fitted on other days. Ramp anatomy, the calibrated
  flux day and night, and the energy-balance route to ET. Data ships in `examples/data/`.

## Install

```bash
pip install -e .            # from a clone
pip install -e ".[test]"    # with pytest
pytest
```

Requires Python ≥ 3.9, numpy and pandas.

## The two detectors

**SR-WL — Haar (`srflux.detectors.haar`).** The microfront at the trailing edge of a ramp is
a step, so ramp detection is edge detection. The block is high-passed (300 s moving mean),
convolved with a normalised Haar step kernel, and local extrema of |coefficient| above a
threshold are taken as fronts. The kernel is normalised so the coefficient *is* the
amplitude in the scalar's own units.

The threshold is **sigma-relative** (`k · std(coef)`, default `k = 0.75`), not absolute — a
canopy skin temperature has ramps an order of magnitude smaller than air temperature, and a
fixed threshold does not travel between sites or scalars. The 32 s kernel sits in a broad
flat optimum (20–48 s); neither default is worth tuning per site.

**SR-VA — Van Atta (`srflux.detectors.vanatta`).** Solves `A³ + pA + q = 0` with
`p = 10·S2 − S5/S3`, `q = 10·S3` for the ramp amplitude and `tau = A²r/S2` for the period.
The root is chosen to match `sign(S3)`.

Lag selection dominates the result. A fixed 1 s lag suits a 20 Hz sonic but collapses on a
radiometric surface temperature, whose ramps carry almost no 1 s signal. `lag="chen"` adapts
per block via the first global maximum of `|S3(r)|/r`.

On a low-rate skin temperature also **drop the period**: `period_mode="unit"` sets τ = 1 and
uses the amplitude alone, `F = ρ·cp·z·A`, folding the ramp duration into α. The fitted τ on a
1 Hz IRT can swing by an order of magnitude between blocks (15–324 s observed), so dividing
by it adds noise rather than information. With τ discarded, choose the lag on a calibration
window rather than with the Chen criterion, which targets the period you have just dropped.

## Calibration

```
SR-WL:  F = rho·cp·z · N·A / block        SR-VA:  F = rho·cp·z · A / tau
        H = alpha · F
```

`alpha` is fitted through the origin, `alpha = Σ(F·H)/Σ(F²)`, against a reference flux —
ideally energy-balance-closure-corrected eddy covariance — **per regime**, so that a
coefficient fitted on daytime convection is not applied to the weak nocturnal flux and the
ramp direction never enters the fit. `examples/data/make_ola_calibration.py` shows the
pattern: fit on a window of days, publish only the coefficients, apply them elsewhere.

Five things that matter in practice:

- **`alpha` and `z` are not separately identifiable** — only their product enters the flux.
  For an air temperature `z` is the measurement height; for a surface temperature the renewed
  volume is the canopy layer, so canopy height is the physical choice. Two studies that pick
  differently cannot compare alphas; compare `alpha·z`.
- **Budget 10–21 days** spread across conditions, not one continuous campaign: a held-out
  test needed that much before the daily-ET error settled, and a continuous window converges
  three to five times more slowly because days inside it see the same weather.
- **Screen the calibration set per day before pooling.** A through-origin fit weights by F²,
  so one day on which the sensor is mis-scaled carries enormous leverage and can drag the
  pooled coefficient to nearly zero.
- **Bound the window by sensor drift, not the calendar.** Check the median ramp amplitude per
  day; a drifting sensor moves α with no change in the flux.
- **Score with NSE** (`srflux.scores`), not r. r is blind to scale, so an estimate with the
  right shape but the wrong magnitude still scores well.

## Flux direction

A detector measures how big the ramp is, not which way the heat is going.

`sign_from_skewness` is the temperature-only option: a warm ramp rises gradually and
collapses sharply, so the increment skewness is negative. **Fit its two parameters together
on a calibration window** with `fit_skewness_sign`:

- the **increment lag** must match the timescale of the fronts. The 1 s default suits a fine
  wire; a slower skin temperature needs several seconds, and a skin channel that appears to
  carry no direction information usually just needs a longer lag.
- the **flip threshold** is not zero — near the daytime H → 0 transition the raw sign flips
  early, and a small positive τ corrects it.

`sign_from_stability` uses `−sign(zeta)` where an eddy-covariance system is present.

## Layout

```
src/srflux/
  preprocess.py        block QC, detrending, increment skewness
  detectors/haar.py    SR-WL front picker
  detectors/vanatta.py SR-VA cubic + Chen lag selection
  flux.py              uncalibrated flux, through-origin alpha, skill scores, LE and ET
  sign.py              flux-direction conventions
  synthetic.py         ramp generator for tests and sanity checks
tests/                 tests on signals whose answer is known by construction
examples/              synthetic quickstart and a real-data notebook
```

## References

- Van Atta (1977) *Arch. Mech.* **29**, 161–171 — structure-function ramp model.
- Paw U, Qiu, Su, Watanabe & Brunet (1995) *Agric. For. Meteorol.* **74**, 119–137 — surface
  renewal for scalar fluxes.
- Chen, Novak, Black & Yang (1997a) *Boundary-Layer Meteorol.* **84**, 99–124 — lag criterion.
- Collineau & Brunet (1993) *Boundary-Layer Meteorol.* **65**, 357–379 — wavelet detection of
  coherent structures.
- Snyder, Spano & Paw U (1996) *Theor. Appl. Climatol.* **53**, 231–240 — surface renewal for
  sensible heat.
- Castellvi (2004) *Agric. For. Meteorol.* **122**, 121–135; Shapland, McElrone, Snyder &
  Paw U (2012) *Boundary-Layer Meteorol.* **145**, 5–25 — structure-function surface renewal
  in practice.

## License

MIT — see [LICENSE](LICENSE).
