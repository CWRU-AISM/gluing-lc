"""
Layer-name lookup tables for common transformer architectures.

Maps friendly names ("transformer block 5") to the canonical
``transformer.h.5`` / ``model.layers.5`` paths used to register hooks.
"""

from typing import Dict, List


def gpt2_layer_names(n_layers: int = 12, use_lm_head: bool = False) -> Dict[str, List[str]]:
    # Returns layer-name groups for HuggingFace GPT-2 variants.
    prefix = 'transformer.' if use_lm_head else ''
    return {
        'residual': [f'{prefix}h.{i}' for i in range(n_layers)],
        'mlp': [f'{prefix}h.{i}.mlp' for i in range(n_layers)],
        'attn': [f'{prefix}h.{i}.attn' for i in range(n_layers)],
        'ln1': [f'{prefix}h.{i}.ln_1' for i in range(n_layers)],
        'ln2': [f'{prefix}h.{i}.ln_2' for i in range(n_layers)],
        'embed': [f'{prefix}wte', f'{prefix}wpe'],
        'final_ln': [f'{prefix}ln_f'],
    }


def llama_layer_names(n_layers: int = 32) -> Dict[str, List[str]]:
    # Returns layer-name groups for Llama / Mistral / Gemma-style models.
    return {
        'residual': [f'model.layers.{i}' for i in range(n_layers)],
        'mlp': [f'model.layers.{i}.mlp' for i in range(n_layers)],
        'attn': [f'model.layers.{i}.self_attn' for i in range(n_layers)],
        'input_ln': [f'model.layers.{i}.input_layernorm' for i in range(n_layers)],
        'post_attn_ln': [f'model.layers.{i}.post_attention_layernorm' for i in range(n_layers)],
        'embed': ['model.embed_tokens'],
        'final_ln': ['model.norm'],
    }
