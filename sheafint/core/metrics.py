# Container types for sheaf cohomology metrics.

from dataclasses import dataclass


@dataclass
class SheafMetrics:
    # Aggregated metrics returned by ScalableSheaf.evaluate.
    H0_dim: int
    H1_dim: int
    consistency_energy: float
    cycle_energy: float
    exact_fraction: float
    harmonic_fraction: float
    coexact_fraction: float
    exactness_error: float
