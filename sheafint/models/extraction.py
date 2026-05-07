"""
High-level helpers for batched activation extraction.

Wraps :class:`HookManager` so callers can register layer hooks, run a
batch of contexts through the model, and recover per-context, per-layer
activations without manually wiring forward passes.
"""

from typing import Dict, List, Union
import torch
import torch.nn as nn
from .hooks import HookManager
from .layer_names import gpt2_layer_names, llama_layer_names


def detect_model_type(model: nn.Module) -> str:
    # Heuristic detection of transformer family based on attribute layout.
    if hasattr(model, 'transformer'):
        return 'gpt2'
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return 'llama'
    raise ValueError(f"Could not auto-detect model type for {type(model).__name__}")


def create_extraction_hooks(
    model: nn.Module,
    model_type: str = 'auto',
    layers: Union[str, List[int]] = 'all',
    components: List[str] = ('residual',),
) -> HookManager:
    # Convenience wrapper that resolves model layout and registers hooks for selected components.
    if model_type == 'auto':
        model_type = detect_model_type(model)

    if model_type == 'gpt2':
        n_layers = len(model.transformer.h) if hasattr(model, 'transformer') else 12
        layer_groups = gpt2_layer_names(n_layers)
    elif model_type == 'llama':
        n_layers = len(model.model.layers) if hasattr(model, 'model') else 32
        layer_groups = llama_layer_names(n_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if layers == 'all':
        layer_indices = list(range(n_layers))
    elif layers == 'first_last':
        layer_indices = [0, n_layers - 1]
    elif isinstance(layers, list):
        layer_indices = layers
    else:
        raise ValueError(f"Invalid layers specification: {layers}")

    hook_names = []
    for component in components:
        if component in layer_groups:
            names = layer_groups[component]
            for idx in layer_indices:
                if idx < len(names):
                    hook_names.append(names[idx])

    manager = HookManager(model)
    manager.register_hooks(hook_names)
    return manager


def _pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    # Pool a (batch, seq, hidden) tensor according to the requested strategy.
    if mode == 'last':
        seq_lens = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), seq_lens]
    if mode == 'mean':
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    if mode == 'first':
        return hidden[:, 0]
    if mode == 'max':
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask - 1e9 * (1 - mask)).max(dim=1)[0]
    raise ValueError(f"Unknown pooling: {mode}")


@torch.no_grad()
def extract_features(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    layer_names: List[str],
    batch_size: int = 8,
    max_length: int = 128,
    pooling: str = 'last',
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    # Run the model on each batch of texts and return pooled per-layer activations.
    model.to(device)
    was_training = model.training
    model.eval()

    manager = HookManager(model)
    manager.register_hooks(layer_names)
    buffers = {name: [] for name in layer_names}

    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with manager.record():
                _ = model(**inputs)

            for name in layer_names:
                if name in manager.cache:
                    pooled = _pool_hidden(manager.cache[name], inputs['attention_mask'], pooling)
                    buffers[name].append(pooled.cpu())
    finally:
        manager.clear_hooks()
        if was_training:
            model.train()

    return {
        name: torch.cat(feats, dim=0) if feats else torch.tensor([])
        for name, feats in buffers.items()
    }


def extract_features_for_contexts(
    model: nn.Module,
    tokenizer,
    context_cover,
    layer_names: List[str],
    **kwargs,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # Extract features for every node in a ContextCover.
    cover_data = context_cover.build_cover()
    return {
        node_name: extract_features(
            model, tokenizer,
            context_cover.get_texts_for_node(node_name),
            layer_names,
            **kwargs,
        )
        for node_name in cover_data['nodes']
    }
