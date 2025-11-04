# scripts/inference_demo.py
# Minimal offline risk visualizer for a trained hallucination probe.
# Works with: OpenMeditron/Meditron3-8B + your saved probe (local or HF).
#
# It:
#  1) loads the base model + (optionally) LoRA adapters,
#  2) loads the linear value head from your probe folder,
#  3) generates text,
#  4) re-runs a forward pass with output_hidden_states=True and computes
#     token-wise risk = sigmoid(W·h_t + b) at the tapped layer,
#  5) prints tokens colored by risk (red if risk >= threshold).

import argparse
import os
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

try:
    from termcolor import colored
except Exception:
    def colored(s, *_args, **_kwargs):  # fallback if termcolor isn't installed
        return s

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_MODEL = "OpenMeditron/Meditron3-8B"

def find_layer_module(model, layer_index: int):
    """
    Return the Module corresponding to the decoder block at `layer_index`.
    Supports LLaMA/Med-LLaMA style (model.model.layers) and GPT-style (transformer.h)
    """
    # LLaMA-like
    layers = getattr(getattr(model, "model", model), "layers", None)
    if isinstance(layers, torch.nn.ModuleList) and 0 <= layer_index < len(layers):
        return layers[layer_index]
    # GPT-like
    h = getattr(getattr(model, "transformer", model), "h", None)
    if isinstance(h, torch.nn.ModuleList) and 0 <= layer_index < len(h):
        return h[layer_index]
    raise ValueError("Could not locate decoder block list on this model.")

def load_value_head(head_path: Path, hidden_size: int) -> nn.Module:
    """
    Load a simple Linear( hidden_size -> 1 ) from various common filenames.
    Expected to find something like:
      - lora_probe/probe_head.bin  (torch.save of {'weight','bias'} or state_dict)
      - probe_head.bin / probe_head.pt
    """
    candidates = [
        head_path / "lora_probe" / "probe_head.bin",
        head_path / "probe_head.bin",
        head_path / "probe_head.pt",
    ]
    state = None
    for p in candidates:
        if p.exists():
            try:
                state = torch.load(p, map_location="cpu")
                head_file = p
                break
            except Exception:
                pass
    if state is None:
        raise FileNotFoundError(f"Could not find probe head in {head_path} "
                                f"(tried: {', '.join(str(c) for c in candidates)})")
    head = nn.Linear(hidden_size, 1, bias=True)
    # Flexible loading
    if isinstance(state, dict) and "weight" in state and "bias" in state:
        with torch.no_grad():
            head.weight.copy_(state["weight"])
            head.bias.copy_(state["bias"])
    elif isinstance(state, dict):
        # maybe a nested state_dict, try common keys
        for key in ["module", "state_dict", "value_head", "head", "model"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        missing = head.load_state_dict(state, strict=False)
        if missing.missing_keys and missing.unexpected_keys:
            print("[warn] non-strict load; missing:", missing.missing_keys,
                  "unexpected:", missing.unexpected_keys)
    else:
        try:
            head.load_state_dict(state)
        except Exception:
            print("[warn] unknown head format; using random init")
    return head

@torch.no_grad()
def compute_risks(model, value_head, input_ids, attention_mask, layer_index: int, device: str):
    """
    Run a forward pass with output_hidden_states=True and compute σ(W·h + b)
    from the tapped decoder layer (post-block hidden states).
    Returns tensor of shape [seq_len] with risks in [0,1].
    """
    out = model(input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                use_cache=False)
    # For LLaMA-like models, hidden_states is a tuple: (emb, layer1, layer2, ..., final)
    # We'll pick the layer at index layer_index+1 (since [0] is embeddings).
    hidden_states = out.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states=True outputs.")
    idx = layer_index + 1
    if idx >= len(hidden_states):
        raise IndexError(f"Requested layer index {layer_index} is out of range "
                         f"for hidden_states of length {len(hidden_states)}")

    h = hidden_states[idx][0]  # [1, seq, hidden] -> [seq, hidden]
    logits = value_head(
        h.to(device=value_head.weight.device, dtype=value_head.weight.dtype)
    ).squeeze(-1)  # [seq]
    probs = torch.sigmoid(logits).float().cpu()
    return probs  # [seq_len]

def color_token(tok: str, risk: float, thr: float) -> str:
    if risk >= thr:
        # red for high risk; add risk value for clarity
        return colored(tok, "red") + colored(f" ({risk:.2f})", "red")
    # light grey for safe(ish)
    return tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="Base model (HF id).")
    ap.add_argument("--probe_source", type=str, choices=["local", "hf"], default="local",
                    help="Where to load the probe from.")
    ap.add_argument("--probe_dir", type=str, default="value_head_probes/meditron3_8b_lora_probe",
                    help="Local path to the saved probe (if probe_source=local).")
    ap.add_argument("--hf_repo", type=str, default="FayElhassan/hallucination-probes",
                    help="HF repo with uploaded probes (if probe_source=hf).")
    ap.add_argument("--hf_subpath", type=str, default="meditron3_8b_lora_probe",
                    help="Subfolder inside the HF repo where the probe is stored.")
    ap.add_argument("--layer", type=int, default=30, help="Tapped decoder layer index.")
    ap.add_argument("--threshold", type=float, default=0.46, help="Risk threshold for coloring.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--prompt", type=str, default="Explain how insulin works in the human body.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] base: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # Where is the probe?
    if args.probe_source == "local":
        probe_path = Path(args.probe_dir)
    else:
        # cache the HF subfolder locally for both LoRA and value head
        from huggingface_hub import snapshot_download
        probe_path = Path(
            snapshot_download(repo_id=args.hf_repo, allow_patterns=[f"{args.hf_subpath}/*"])
        ) / args.hf_subpath

    # Try to load LoRA adapters
    try:
        print(f"[load] applying LoRA from: {probe_path}")
        model = PeftModel.from_pretrained(model, str(probe_path))
        model = model.to(device)
    except Exception as e:
        print(f"[warn] Could not load LoRA adapters from {probe_path}: {e}\n"
              f"       Continuing with base model (risks may be slightly different).")

    # Load the value head (linear probe)
    hidden_size = model.config.hidden_size
    value_head = load_value_head(probe_path, hidden_size).to(device)
    value_head = value_head.to(dtype=getattr(model, 'dtype', torch.float32))
    value_head.eval()

    # 1) tokenize + generate
    inputs = tok(args.prompt, return_tensors="pt").to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature is not None and args.temperature > 0),
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    full_text = tok.decode(gen_ids[0], skip_special_tokens=True)
    print("\n[output]\n" + full_text + "\n")

    # 2) recompute hidden states on the *full* sequence, then score risks
    # NOTE: This computes risks offline for readability; online per-step would be similar.
    attn = torch.ones_like(gen_ids).to(device)
    risks = compute_risks(model, value_head, gen_ids, attn, args.layer, device)
    risks = risks.tolist()

    # 3) pretty-print a token heatmap (only for the generated tail for clarity)
    #    Find split between prompt and generation
    prompt_len = inputs["input_ids"].shape[1]
    toks = tok.convert_ids_to_tokens(gen_ids[0].tolist(), skip_special_tokens=False)

    print("[risk heatmap] tokens with risk >= threshold are shown in red with (risk):\n")
    shown = []
    for i, (t, r) in enumerate(zip(toks, risks)):
        # skip specials in display
        if t in (tok.pad_token, tok.eos_token) or t.startswith("<") and t.endswith(">"):
            continue
        disp = t.replace("▁", " ")  # sentencepiece whitespace
        mark = color_token(disp, r, args.threshold) if i >= prompt_len else disp
        shown.append(mark)

    # Wrap lines to avoid overly long single lines
    WIDTH = 96
    line = ""
    for piece in shown:
        if len(line) + len(piece) + 1 > WIDTH:
            print(line)
            line = piece
        else:
            line = (line + " " + piece) if line else piece
    if line:
        print(line)

    print(f"\n[done] threshold={args.threshold:.2f}  |  layer={args.layer}  |  "
          f"source={'HF:'+args.hf_repo+'/'+args.hf_subpath if args.probe_source=='hf' else str(probe_path)}")

if __name__ == "__main__":
    main()


