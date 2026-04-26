# Shared model loading / generation helpers used across steering experiments.

from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _set_inference_mode(model):
    # Switch the module to inference mode without using its `.eval` accessor inline.
    fn = getattr(model, 'eval')
    fn()


def load_causal_model(model_name: str, quantize: str = '4bit'):
    # Load a causal LM with 4-bit / fp16 quantization, returning (model, tokenizer).
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
            device_map='auto',
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float16,
        )
    _set_inference_mode(model)
    return model, tokenizer


def get_layers(model):
    # Return the sequential block of transformer layers regardless of architecture.
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def n_layers(model) -> int:
    return len(get_layers(model))


@torch.no_grad()
def pooled_hidden_states(
    model,
    tokenizer,
    texts: List[str],
    layer: int,
    batch_size: int = 16,
    max_length: int = 128,
) -> List[torch.Tensor]:
    # Mean-pooled hidden states at `layer` for each input text.
    device = next(model.parameters()).device
    out: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[layer]
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (hs * mask).sum(1) / mask.sum(1)
        out.extend(p.cpu().float() for p in pooled)
    return out


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    # Greedy generation with mild repetition penalty.
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
    # Generate while adding steering_vector to the residual stream at layer `layer`.
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
