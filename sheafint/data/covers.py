# Build a sheaf cover from one or more BaseContext sources.

from typing import Dict, List, Tuple
import torch
from .base import BaseContext
from .paraphrase import MRPCContext, ParaphraseContext, QQPContext
from .prompts import PromptVariantContext


class ContextCover:
    # A cover is a list of nodes (per-context "a" / "b" sides) plus pair overlaps.

    def __init__(self):
        self.contexts: Dict[str, BaseContext] = {}
        self.node_names: List[str] = []
        self.edges: List[Tuple[int, int]] = []
        self.faces: List[Tuple[int, int, int]] = []

    def add_context(self, name: str, context: BaseContext):
        self.contexts[name] = context

    def build_cover(self) -> Dict:
        # Construct nodes/edges/overlaps used by ScalableSheaf.fit.
        all_nodes = []
        sample_counts = {}

        for ctx_name, ctx in self.contexts.items():
            node_a = f"{ctx_name}_a"
            node_b = f"{ctx_name}_b"
            all_nodes.extend([node_a, node_b])
            sample_counts[node_a] = len(ctx)
            sample_counts[node_b] = len(ctx)

        self.node_names = all_nodes
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}

        pair_overlaps = {}
        for ctx_name, ctx in self.contexts.items():
            node_a = f"{ctx_name}_a"
            node_b = f"{ctx_name}_b"
            i = node_to_idx[node_a]
            j = node_to_idx[node_b]
            if i > j:
                i, j = j, i
                node_a, node_b = node_b, node_a
            self.edges.append((i, j))
            pair_overlaps[(node_a, node_b)] = torch.arange(len(ctx))

        return {
            'nodes': self.node_names,
            'node_to_idx': node_to_idx,
            'edges': self.edges,
            'faces': self.faces,
            'pair_overlaps': pair_overlaps,
            'triple_overlaps': {},
            'sample_counts': sample_counts,
        }

    def get_texts_for_node(self, node_name: str) -> List[str]:
        # Retrieve raw texts associated with a node (either text_a or text_b side).
        ctx_name, side = node_name.rsplit('_', 1)
        if ctx_name not in self.contexts:
            raise ValueError(f"Unknown context: {ctx_name}")
        ctx = self.contexts[ctx_name]
        return [p.text_a for p in ctx.pairs] if side == 'a' else [p.text_b for p in ctx.pairs]


def create_standard_cover() -> ContextCover:
    # Hand-crafted paraphrases plus prompt variants.
    cover = ContextCover()
    cover.add_context('paraphrase', ParaphraseContext())
    cover.add_context('prompt', PromptVariantContext())
    return cover


def create_mrpc_cover(max_pairs: int = 200) -> ContextCover:
    # Cover backed by MRPC paraphrase pairs.
    cover = ContextCover()
    cover.add_context('mrpc', MRPCContext(split='train', max_pairs=max_pairs))
    return cover


def create_full_cover(mrpc_pairs: int = 100, qqp_pairs: int = 100) -> ContextCover:
    # MRPC + QQP + hand-crafted + prompt variants combined into one cover.
    cover = ContextCover()
    cover.add_context('paraphrase', ParaphraseContext())
    cover.add_context('mrpc', MRPCContext(split='train', max_pairs=mrpc_pairs))
    cover.add_context('qqp', QQPContext(split='train', max_pairs=qqp_pairs))
    cover.add_context('prompt', PromptVariantContext())
    return cover
