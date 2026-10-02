"""
Top-level :class:`ScalableSheaf` class.

Wires together projections, coboundaries, and cohomology so user code can
go from raw activations to (H^0, H^1) dimensions and Hodge mass with a
single ``fit`` call.
"""

from typing import Dict, List, Optional, Tuple
import warnings
import torch
from .coboundary import build_delta0, build_delta1, exactness_error
from .cohomology import compute_cohomology_dims
from .projections import learn_joint_projections
from .spectrum import compute_laplacian_spectrum, project_signal_onto_h0


class ScalableSheaf:
    """Sheaf with scalable cohomology computation and chain-complex guarantees.

    The sheaf learns:
      1. Node projections P_i mapping hidden activations to a common edge space.
      2. Edge transports Q_ij encoding restriction maps.
      3. Coboundary operators delta0, delta1 that satisfy delta1 @ delta0 = 0 by construction.

    Randomized SVD keeps the cost at O(n^2 k) for high-dimensional LLM activations.
    """

    def __init__(
        self,
        edge_dim: int = 64,
        face_dim: int = 32,
        svd_rank: int = 100,
        use_randomized_svd: bool = True,
        tol: float = 1e-6,
        device: str = 'cuda',
    ):
        self.edge_dim = edge_dim
        self.face_dim = face_dim
        self.svd_rank = svd_rank
        self.use_randomized_svd = use_randomized_svd
        self.tol = tol
        self.device = device

        self.node_projections: Dict[str, torch.Tensor] = {}
        self.edge_transports: Dict[Tuple[str, str], torch.Tensor] = {}

        self.delta0: Optional[torch.Tensor] = None
        self.delta1: Optional[torch.Tensor] = None

        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self.faces: List[Tuple[str, str, str]] = []
        self.node_to_idx: Dict[str, int] = {}

        self.fitted = False

        self._h0_basis: Optional[torch.Tensor] = None

    def fit(
        self,
        features: Dict[str, torch.Tensor],
        pair_overlaps: Dict[Tuple[str, str], torch.Tensor],
        triple_overlaps: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        method: str = 'joint_pca',
    ):
        """Learn restriction maps and coboundary operators for the cover."""
        self.nodes = list(features.keys())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        features = {k: v.to(self.device) for k, v in features.items()}

        if method == 'joint_pca':
            projections, transports, actual_edge_dim = learn_joint_projections(
                features, pair_overlaps, self.nodes, self.edge_dim, self.device,
                use_randomized_svd=self.use_randomized_svd,
            )
            self.edge_dim = actual_edge_dim
        else:
            raise ValueError(f"Unknown method: {method}")

        self.node_projections = projections
        self.edge_transports = transports
        self.edges = list(pair_overlaps.keys())

        if triple_overlaps:
            self.faces = list(triple_overlaps.keys())

        self.delta0 = build_delta0(
            self.nodes, self.edges, self.edge_transports, self.edge_dim, self.device
        )
        self.delta1 = build_delta1(
            self.edges, self.faces, self.edge_dim, self.face_dim, self.device,
        )

        residual = exactness_error(self.delta0, self.delta1)
        if residual > self.tol * 10:
            warnings.warn(f"Chain complex not exact: ||delta1 @ delta0|| = {residual:.6f}")

        self.fitted = True

    def compute_cohomology(self) -> Dict[str, int]:
        """Cohomology dimensions of the sheaf chain complex."""
        if not self.fitted:
            raise ValueError("Sheaf must be fitted first")

        return compute_cohomology_dims(
            self.delta0, self.delta1,
            self.svd_rank, self.tol, self.use_randomized_svd, self.device,
        )

    def _assemble_node_signal(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack mean-projected node sections into a single C^0 vector."""
        signal = torch.zeros(len(self.nodes) * self.edge_dim, device=self.device)
        for node in self.nodes:
            if node in features and node in self.node_projections:
                pooled = (features[node] @ self.node_projections[node]).mean(dim=0)
                idx = self.node_to_idx[node]
                signal[idx * self.edge_dim:(idx + 1) * self.edge_dim] = pooled
        return signal

    def compute_laplacian_spectrum(self, k: int = 50) -> Dict:
        """Eigendecomposition of the node Laplacian. Caches the H0 basis."""
        if not self.fitted:
            raise ValueError("Sheaf must be fitted first")

        spectrum = compute_laplacian_spectrum(self.delta0, self.tol, self.device, k=k)
        eigenvectors = spectrum.pop('eigenvectors')

        if eigenvectors is not None:
            self._h0_basis = eigenvectors[:, :spectrum['h0_dim']] if spectrum['h0_dim'] > 0 else None
        else:
            self._h0_basis = None

        return spectrum

    def project_onto_cohomology(self, features: Dict[str, torch.Tensor]) -> Dict:
        """Project node features onto H0 (the consistent subspace) and report energy fractions."""
        if not self.fitted:
            raise ValueError("Sheaf must be fitted first")

        if self._h0_basis is None:
            self.compute_laplacian_spectrum()

        features = {k: v.to(self.device) for k, v in features.items()}
        signal = self._assemble_node_signal(features)
        return project_signal_onto_h0(
            signal, self._h0_basis, self.delta0,
            self.edge_dim, self.nodes, self.node_to_idx, self.tol,
        )
