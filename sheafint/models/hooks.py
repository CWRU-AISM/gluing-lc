"""
Forward and backward hook utilities for transformer activation extraction.

Provides :class:`HookManager` (lifecycle / cleanup) and
:class:`ActivationCache` (layer-keyed storage) used by the higher-level
``extract_features`` helpers.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import torch
import torch.nn as nn


@dataclass
class ActivationCache:
    # Stores per-layer activations and gradients during a forward / backward pass.
    activations: Dict[str, torch.Tensor] = field(default_factory=dict)
    gradients: Dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        self.activations.clear()
        self.gradients.clear()

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.activations.get(name)

    def keys(self) -> List[str]:
        return list(self.activations.keys())

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self.activations)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.activations[name]

    def __contains__(self, name: str) -> bool:
        return name in self.activations


class HookManager:
    # Registers forward (and optional backward) hooks on a model and stores results.

    def __init__(self, model: nn.Module):
        self.model = model
        self.cache = ActivationCache()
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_names: List[str] = []

    def _resolve_module(self, name: str) -> nn.Module:
        # Resolve a dot-separated name like 'transformer.h.5' to a module.
        module = self.model
        for part in name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _make_forward_hook(self, name: str) -> Callable:
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                self.cache.activations[name] = output[0].detach()
            elif isinstance(output, torch.Tensor):
                self.cache.activations[name] = output.detach()
            else:
                self.cache.activations[name] = output
        return hook

    def _make_backward_hook(self, name: str) -> Callable:
        def hook(_module, _grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.cache.gradients[name] = grad_output[0].detach()
            elif isinstance(grad_output, torch.Tensor):
                self.cache.gradients[name] = grad_output.detach()
        return hook

    def register_hooks(self, layer_names: List[str], include_gradients: bool = False):
        # Register forward hooks (and optionally backward hooks) on the named modules.
        self.clear_hooks()
        self.layer_names = layer_names
        for name in layer_names:
            try:
                module = self._resolve_module(name)
                self.hooks.append(module.register_forward_hook(self._make_forward_hook(name)))
                if include_gradients:
                    self.hooks.append(module.register_full_backward_hook(self._make_backward_hook(name)))
            except (AttributeError, IndexError):
                continue

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_cache(self):
        self.cache.clear()

    @contextmanager
    def record(self):
        # Reset the cache before yielding control to a forward pass.
        self.clear_cache()
        try:
            yield self.cache
        finally:
            pass

    def __del__(self):
        self.clear_hooks()
