"""Common result type for the ramp detectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RampStats:
    """What a detector extracts from one block.

    Attributes
    ----------
    count : int
        Number of ramps (microfronts) found in the block. The Van Atta method does not
        count individual ramps, so it reports ``count = block_duration / period``.
    amplitude : float
        Ramp amplitude [same units as the input scalar], the median over the block for the
        counting detectors and the cubic root for Van Atta.
    period : float or nan
        Mean time between ramps [s]. Van Atta solves for it; the Haar detector infers it
        from the count as ``block_duration / count``.
    detector : str
        Name of the detector that produced the result.
    extra : dict
        Detector-specific diagnostics (threshold used, optimal lag, structure functions).
    """

    count: int
    amplitude: float
    period: float
    detector: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """True when the block yielded a usable amplitude."""
        return self.count > 0 and self.amplitude == self.amplitude and self.amplitude > 0
