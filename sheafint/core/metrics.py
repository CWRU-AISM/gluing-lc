"""
Container types for sheaf cohomology metrics.
"""

from dataclasses import dataclass


@dataclass
class SheafMetrics:
    """
    Aggregated metrics returned by :meth:`ScalableSheaf.evaluate`.

    Cohomology dimensions, consistency / cycle energies, the exact /
    harmonic / coexact split of a 1-cochain, and the chain-complex
    exactness residual ``||delta_1 . delta_0||``.
    """

    H0_dim: int
    H1_dim: int
    consistency_energy: float
    cycle_energy: float
    exact_fraction: float
    harmonic_fraction: float
    coexact_fraction: float
    exactness_error: float
