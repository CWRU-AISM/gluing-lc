"""
Shared model loading, hidden-state extraction, and steered generation.

Wraps :mod:`transformers` so the experiment scripts share a single 4-bit /
fp16 / bf16 loader, a uniform layer accessor across GPT-2 / Llama / Mistral,
and identical pooling and generation conventions.
"""

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_causal_model(
    model_name: str,
    quantize: str = '4bit',
    dtype: torch.dtype = torch.float16,
    device_map='auto',
):
    """
    Load a causal LM, 4-bit (NF4, fp16 compute) or unquantized in ``dtype``.

    Returns ``(model, tokenizer)``. The tokenizer's pad token is aliased to
    its EOS token when missing so batched generation never crashes on
    padding-only attention masks.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize == '4bit':
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
    model.train(False)
    return model, tokenizer


def get_layers(model):
    """
    Return the sequential block of transformer layers across architectures.

    Falls back through the conventions used by Llama / Mistral
    (``model.model.layers``) and GPT-2 (``model.transformer.h``).
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def n_layers(model) -> int:
    return len(get_layers(model))


def device_for_inputs(model) -> str:
    """
    Pick a sensible device for inputs even when the model is sharded.

    Prefers the first CUDA shard recorded in ``hf_device_map`` and falls
    back to the device of the first parameter tensor.
    """
    if hasattr(model, 'hf_device_map'):
        for dev in model.hf_device_map.values():
            if 'cuda' in str(dev):
                return dev
    return next(model.parameters()).device


@torch.no_grad()
def hidden_states_at(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    """
    Per-token hidden states at the output of block ``layer_idx`` for one input.

    Uses ``layer_idx + 1`` against ``output_hidden_states`` because index 0
    is the embedding layer.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    inputs = inputs.to(device_for_inputs(model))
    outputs = model(inputs['input_ids'], output_hidden_states=True, return_dict=True)
    return outputs.hidden_states[layer_idx + 1][0].to(device_for_inputs(model))


@torch.no_grad()
def pooled_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    layer: int,
    pooling: str = 'mean',
    batch_size: int = 16,
    max_length: int = 128,
    fp32_pool: bool = False,
) -> torch.Tensor:
    """
    ``(len(texts), d)`` float32 CPU tensor of hidden states at ``layer``.

    ``pooling`` is ``'mean'`` (over non-pad tokens), ``'last'`` (last non-pad
    token, assumes right padding) or ``'first'``. Mean pooling runs in the
    model dtype unless ``fp32_pool`` is set.
    """
    device = next(model.parameters()).device
    out: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc, output_hidden_states=True).hidden_states[layer]
        mask = enc['attention_mask']
        if pooling == 'mean':
            m = mask.unsqueeze(-1).float() if fp32_pool else mask.unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        elif pooling == 'last':
            lengths = mask.sum(dim=1).clamp(min=1) - 1
            pooled = h.gather(1, lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))).squeeze(1)
        elif pooling == 'first':
            pooled = h[:, 0, :]
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        out.append(pooled.float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    """
    Greedy generation with mild repetition penalty.

    Decodes only the newly generated continuation (prompt tokens stripped)
    and returns the trimmed string.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        out[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    max_new_tokens: int = 100,
) -> str:
    """
    Generate while adding a steering vector to the residual stream.

    The vector is broadcast across the batch and sequence dimensions of the
    hooked layer's residual output; the hook is removed in a finally block
    so cleanup happens even if generation raises.
    """
    device = next(model.parameters()).device
    if isinstance(steering_vector, torch.Tensor):
        steer = steering_vector.to(device).half()
    else:
        steer = torch.tensor(steering_vector, device=device, dtype=torch.float16)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0] + steer.unsqueeze(0).unsqueeze(0)
            return (hidden,) + output[1:]
        return output + steer.unsqueeze(0).unsqueeze(0)

    handle = get_layers(model)[layer].register_forward_hook(hook)
    try:
        return generate(model, tokenizer, prompt, max_new_tokens)
    finally:
        handle.remove()
