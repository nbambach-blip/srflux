# srflux

Surface-renewal sensible heat flux from a high-rate scalar series, with the two ramp
detectors side by side:

| | method | what it measures | when to use it |
|---|---|---|---|
| **SR-WL** | Haar wavelet front picker | counts microfronts and measures their amplitude directly | default; robust across sensors and scalars |
| **SR-VA** | Van Atta structure-function cubic | fits an idealised ramp to S2/S3/S5 | when you want the classical formulation, or a per-block ramp period |

Surface renewal estimates the sensible heat flux from the temperature ramps that coherent
structures leave in a scalar time series, needing only a fast thermometer or radiometer
rather than a sonic anemometer. Both detectors here produce an *uncalibrated* flux that is
closed by one dimensionless coefficient `alpha`.

```python
from srflux import HaarDetector, VanAttaDetector, prepare_block, ramp_flux, fit_and_score

block = prepare_block(temperature_1hz, fs=1.0)          # clip, gap-fill, validate
wl = HaarDetector(fs=1.0).detect(block.values)           # SR-WL
va = VanAttaDetector(fs=1.0, lag="chen").detect(block.values)   # SR-VA

F_wl = ramp_flux(wl.amplitude, count=wl.count, z=2.0)    # uncalibrated, W m-2
F_va = ramp_flux(va.amplitude, period=va.period, z=2.0)

fit = fit_and_score(F_wl_all_blocks, H_eddy_covariance)  # alpha, r, RMSE, bias
H_sr = fit.apply(F_wl_all_blocks)
```

`python examples/quickstart.py` runs the whole chain — ramps, flux, calibration, daily ET —
on synthetic data, with no field data required.

## Install

```bash
pip install -e .            # from a clone
pip install -e ".[test]"    # with pytest
pytest                      # 42 tests, all synthetic
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
fixed threshold does not travel between sites or scalars. A scale sweep gave a broad flat
optimum at 20–48 s for reproducing eddy-covariance H, with 32 s at the peak; the wavelet
shape matters even less (Haar ties with a first-derivative-of-Gaussian, and both beat
symmetric or oscillatory kernels, which find ramp centres rather than fronts). Neither
default is worth tuning per site.

**SR-VA — Van Atta (`srflux.detectors.vanatta`).** Solves `A³ + pA + q = 0` with
`p = 10·S2 − S5/S3`, `q = 10·S3` for the ramp amplitude, and `tau = A²r/S2` for the period.
The root is chosen to match `sign(S3)`, which ties the amplitude to the flux direction.

Lag selection is the whole game. A fixed 1 s lag is the textbook choice for a 20 Hz sonic
but **collapses on a radiometric surface temperature** — a smooth ~90 s skin-temperature
ramp has almost no 1 s signal, and both amplitude and period go to noise. `lag="chen"`
(default) adapts per block via the first global maximum of `|S3(r)|/r`, giving r* ≈ 2 s on
air and ≈ 65 s on surface series. Use it unless you are reproducing a fixed-lag analysis.

## Calibration: what `alpha` does and does not transfer across

```
SR-WL:  F = rho·cp·z · N·A / block        SR-VA:  F = rho·cp·z · A / tau
        H = alpha · F
```

`alpha` is fitted through the origin, `alpha = Σ(F·H)/Σ(F²)`, against a reference flux —
ideally energy-balance-closure-corrected eddy covariance — on a **single well-defined
regime** (convective daytime blocks of one sign), so the ramp direction never enters the
calibration.

Findings from a six-site orchard and vineyard study that used this code (three almond, three
vineyard; ~46,000 blocks; daily ET as the energy-balance residual):

- **Year to year, `alpha` is stable.** Reusing another year's value costs a median
  0.44 mm d⁻¹ MAE in almonds and 0.53 in vineyards, against a 0.43 mm d⁻¹ floor when each
  site uses its own. Calibrate once, reuse — but track the median ramp amplitude annually:
  at one site `alpha` drifted 3.4 → 5.3 over four years while the amplitude fell 33 %, a
  drifting radiometer rather than a changing flux.
- **Site to site, it does not transfer**, even within a crop: 0.63 (almond) and
  0.82 (vineyard) mm d⁻¹ MAE, roughly double the floor; across crops the worst pair exceeded
  6 mm d⁻¹.
- **`alpha` and `z` are not separately identifiable** — only the product enters the flux. For
  an air temperature `z` is the measurement height; for a surface temperature the renewed
  volume is the canopy layer, so canopy height is the physical choice. Two studies that pick
  differently cannot compare alphas; compare `alpha·z`. In that form a 4.6× cross-site spread
  collapsed to 1.74×.
- **`alpha` scales inversely with the ramp amplitude the sensor resolves.** Five co-located
  radiometers on one mast, viewing the same canopy and the same flux and differing only in
  field of view, ranked perfectly inversely (Spearman −1.0) from A = 0.375 K, alpha = 4.4 to
  A = 0.098 K, alpha = 14.2. A view admitting more exposed soil returns a damped ramp and
  needs a larger alpha.
- **How much reference data?** 10–21 days spread through the record bring the added daily-ET
  error below 0.10 mm d⁻¹; 21–90 days bring `alpha` within ±10 %. The days must sample
  different conditions — a single continuous campaign converges 3–5× more slowly and plateaus
  at 30–50 % error until the window spans a season.

## Flux direction

Neither detector's amplitude is signed usefully on its own. `srflux.sign` offers, in
increasing reliability:

1. `sign_from_skewness` — pure surface renewal. A warm ramp rises gradually and collapses
   sharply, so the increment skewness is negative. Temperature-only, no other instrument.
   ~70–80 % correct through the daytime H → 0 transition; the flip threshold is not exactly
   zero and benefits from site calibration.
2. `sign_from_stability` — `−sign(zeta)`. Needs a sonic, ~98 % correct.

On a **surface** temperature the skewness convention is unreliable (51–88 % correct versus
92–99 % for air), because skin temperature decouples from the flux direction. If an air
series exists, take the sign from the air and apply it to the surface channel.

## Layout

```
src/srflux/
  preprocess.py        block QC (range clip, gap fill, validation), detrending, skewness
  detectors/haar.py    SR-WL front picker
  detectors/vanatta.py SR-VA cubic + Chen lag selection
  flux.py              uncalibrated flux, through-origin alpha, H, LE residual, daily ET
  sign.py              ramp-direction conventions
  synthetic.py         sawtooth ramp generator for tests and sanity checks
tests/                 42 tests on signals whose answer is known by construction
examples/quickstart.py end-to-end demo
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
