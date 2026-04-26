# Top-level ScalableSheaf class wiring together projections, coboundaries, and cohomology.

from typing import Dict, List, Optional, Tuple
import warnings
import torch
from .coboundary import build_delta0, build_delta1, exactness_error
from .cohomology import compute_cohomology_dims, hodge_decomposition
from .metrics import SheafMetrics
from .projections import learn_joint_projections, learn_procrustes_projections
from .spectrum import compute_laplacian_spectrum, project_signal_onto_h0


class ScalableSheaf:
    # Sheaf with scalable cohomology computation and chain-complex guarantees.
    #
    # The sheaf learns:
    #   1. Node projections P_i mapping hidden activations to a common edge space.
    #   2. Edge transports Q_ij encoding restriction maps.
    #   3. Coboundary operators delta0, delta1 that satisfy delta1 @ delta0 = 0 by construction.
    #
    # Randomized SVD keeps the cost at O(n^2 k) for high-dimensional LLM activations.

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
        self.face_projections: Dict[Tuple[str, str, str], torch.Tensor] = {}

        self.delta0: Optional[torch.Tensor] = None
        self.delta1: Optional[torch.Tensor] = None

        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self.faces: List[Tuple[str, str, str]] = []
        self.node_to_idx: Dict[str, int] = {}

        self.fitted = False

        self._h0_basis: Optional[torch.Tensor] = None
        self._laplacian_eigenvectors: Optional[torch.Tensor] = None
        self._laplacian_eigenvalues: Optional[torch.Tensor] = None

    def fit(
        self,
        features: Dict[str, torch.Tensor],
        pair_overlaps: Dict[Tuple[str, str], torch.Tensor],
        triple_overlaps: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        method: str = 'joint_pca',
    ):
        # Learn restriction maps and coboundary operators for the cover.
        self.nodes = list(features.keys())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        features = {k: v.to(self.device) for k, v in features.items()}

        if method == 'joint_pca':
            projections, transports, actual_edge_dim = learn_joint_projections(
                features, pair_overlaps, self.nodes, self.edge_dim, self.device,
                use_randomized_svd=self.use_randomized_svd,
            )
            self.edge_dim = actual_edge_dim
        elif method == 'procrustes':
            projections, transports = learn_procrustes_projections(
                features, pair_overlaps, self.nodes, self.edge_dim, self.device,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.node_projections = projections
        self.edge_transports = transports
        self.edges = list(pair_overlaps.keys())

        if triple_overlaps:
            self.faces = list(triple_overlaps.keys())
            self._learn_face_projections(triple_overlaps)

        self.delta0 = build_delta0(
            self.nodes, self.edges, self.edge_transports, self.edge_dim, self.device
        )
        self.delta1 = build_delta1(
            self.edges, self.faces, self.face_projections,
            self.edge_dim, self.face_dim, self.device,
        )

        residual = exactness_error(self.delta0, self.delta1)
        if residual > self.tol * 10:
            warnings.warn(f"Chain complex not exact: ||delta1 @ delta0|| = {residual:.6f}")

        self.fitted = True

    def _learn_face_projections(self, triple_overlaps):
        # Identity-truncation face projections preserve exactness of the chain complex.
        default = torch.eye(self.face_dim, self.edge_dim, device=self.device)
        for (i, j, k), indices in triple_overlaps.items():
            if len(indices) < 2:
                continue
            edges = [(i, j), (j, k), (i, k)]
            if not all(e in self.edge_transports for e in edges):
                continue
            self.face_projections[(i, j, k)] = default.clone()

    def compute_cohomology(self) -> Dict[str, int]:
        # Cohomology dimensions of the sheaf chain complex.
        if not self.fitted:
            raise ValueError("Sheaf must be fitted first")

        return compute_cohomology_dims(
            self.delta0, self.delta1,
            self.svd_rank, self.tol, self.use_randomized_svd, self.device,
        )

    @torch.no_grad()
    def evaluate(
        self,
        features: Dict[str, torch.Tensor],
        pair_overlaps: Dict[Tuple[str, str], torch.Tensor],
    ) -> SheafMetrics:
        # Aggregate metrics: per-pair consistency energy plus Hodge fractions.
        if not self.fitted:
            raise ValueError("Sheaf must be fitted first")

        features = {k: v.to(self.device) for k, v in features.items()}
        consistency_energy = self._consistency_energy(features, pair_overlaps)

        node_section_signal = self._assemble_node_signal(features)
        edge_section = self.delta0 @ node_section_signal

        if self.delta1 is not None:
            cycle_energy = torch.norm(self.delta1 @ edge_section).item() ** 2
        else:
            cycle_energy = 0.0

        hodge = hodge_decomposition(edge_section, self.delta0, self.delta1)
        cohom = self.compute_cohomology()

        return SheafMetrics(
            H0_dim=cohom['H0_dim'],
            H1_dim=cohom['H1_dim'],
            consistency_energy=consistency_energy,
            cycle_energy=cycle_energy,
            exact_fraction=hodge['exact_fraction'],
            harmonic_fraction=hodge['harmonic_fraction'],
            coexact_fraction=hodge['coexact_fraction'],
            exactness_error=exactness_error(self.delta0, self.delta1),
        )

    def _consistency_energy(
        self,
        features: Dict[str, torch.Tensor],
        pair_overlaps: Dict[Tuple[str, str], torch.Tensor],
    ) -> float:
        # Mean ||s_j - Q_ij s_i||^2 across all edge overlaps.
        total_error = 0.0
        total_pairs = 0
        identity = torch.eye(self.edge_dim, device=self.device)

        for (node_i, node_j), indices in pair_overlaps.items():
            if node_i not in features or node_j not in features:
                continue
            if node_i not in self.node_projections or node_j not in self.node_projections:
                continue

            Pi = self.node_projections[node_i]
            Pj = self.node_projections[node_j]
            Q = self.edge_transports.get((node_i, node_j), identity)

            si = features[node_i][indices] @ Pi
            sj = features[node_j][indices] @ Pj
            transported = si @ Q.T

            errors = torch.sum((sj - transported) ** 2, dim=1)
            total_error += errors.sum().item()
            total_pairs += len(indices)

        return total_error / total_pairs if total_pairs > 0 else 0.0

    def _assemble_node_signal(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Stack mean-projected node sections into a single C^0 vector.
        signal = torch.zeros(len(self.nodes) * self.edge_dim, device=self.device)
        for node in self.nodes:
            if node in features and node in self.node_projections:
                pooled = (features[node] @ self.node_projections[node]).mean(dim=0)
                idx = self.node_to_idx[node]
                signal[idx * self.edge_dim:(idx + 1) * self.edge_dim] = pooled
        return signal

    def add_regularization(
        self,
        loss: torch.Tensor,
        features: Dict[str, torch.Tensor],
        lambda_cons: float = 0.01,
        lambda_cycle: float = 0.001,
    ) -> torch.Tensor:
        # Augment a downstream loss with sheaf consistency and cycle penalties.
        if not self.fitted:
            raise ValueError("Sheaf must be fitted first")

        identity = torch.eye(self.edge_dim, device=self.device)
        node_sections = {
            node: features[node] @ self.node_projections[node]
            for node in self.nodes
            if node in features and node in self.node_projections
        }

        cons_loss = 0.0
        n_edges = 0
        for (i, j) in self.edges:
            if i in node_sections and j in node_sections:
                Q = self.edge_transports.get((i, j), identity)
                cons_loss = cons_loss + torch.norm(node_sections[j] - node_sections[i] @ Q.T, dim=-1).mean()
                n_edges += 1
        if n_edges > 0:
            loss = loss + lambda_cons * cons_loss / n_edges

        if lambda_cycle > 0 and self.faces:
            cycle_loss = 0.0
            n_faces = 0
            for (i, j, k) in self.faces:
                if all(n in node_sections for n in [i, j, k]):
                    s_i = node_sections[i]
                    Q_ij = self.edge_transports.get((i, j), identity)
                    Q_jk = self.edge_transports.get((j, k), identity)
                    Q_ik = self.edge_transports.get((i, k), identity)
                    cycle = (s_i @ Q_ik.T) - (s_i @ Q_ij.T @ Q_jk.T)
                    cycle_loss = cycle_loss + torch.norm(cycle, dim=-1).mean()
                    n_faces += 1
            if n_faces > 0:
                loss = loss + lambda_cycle * cycle_loss / n_faces

        return loss

    def compute_laplacian_spectrum(self, k: int = 50) -> Dict:
        # Eigendecomposition of the node Laplacian. Caches eigenvectors and H0 basis.
        if not self.fitted:
            raise ValueError("Sheaf must be fitted first")

        spectrum = compute_laplacian_spectrum(self.delta0, self.tol, self.device, k=k)
        eigenvectors = spectrum.pop('eigenvectors')

        if eigenvectors is not None:
            self._laplacian_eigenvectors = eigenvectors
            self._h0_basis = eigenvectors[:, :spectrum['h0_dim']] if spectrum['h0_dim'] > 0 else None
        else:
            self._laplacian_eigenvectors = None
            self._h0_basis = None

        self._laplacian_eigenvalues = torch.tensor(spectrum['eigenvalues'], device=self.device)
        return spectrum

    def project_onto_cohomology(
        self,
        features: Dict[str, torch.Tensor],
        compute_spectrum_first: bool = True,
    ) -> Dict:
        # Project node features onto H0 (the consistent subspace) and report energy fractions.
        if not self.fitted:
            raise ValueError("Sheaf must be fitted first")

        if compute_spectrum_first or self._h0_basis is None:
            self.compute_laplacian_spectrum()

        features = {k: v.to(self.device) for k, v in features.items()}
        signal = self._assemble_node_signal(features)
        return project_signal_onto_h0(
            signal, self._h0_basis, self.delta0,
            self.edge_dim, self.nodes, self.node_to_idx, self.tol,
        )
