#!/usr/bin/env python
"""Merge a Fast-LoCo LoRA adapter into its base model and save full weights.

The base model is loaded in bf16/fp16 (never 4-bit: merging into quantized
weights would bake in quantization error), the adapter is applied with
``PeftModel`` and folded in via ``merge_and_unload()``. The result is a plain
``transformers`` checkpoint (sharded safetensors) usable by vLLM, TGI or
llama.cpp.

Usage::

    python merge_lora.py --adapter out/loco-lora --output-dir out/loco-merged
    # optional: --base deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --device-map auto
    #           --push-to-hub your-user/LoCo-lite-7B

Merging a 7B model on CPU needs ~16 GB of system RAM in bf16 (default
``--device-map cpu``); ``--device-map auto`` uses the GPU if it fits.

Serving with vLLM (OpenAI-compatible, port 8002 as in CONTRACT.md)::

    vllm serve out/loco-merged --served-model-name loco --port 8002

Converting to GGUF for Ollama (llama.cpp)::

    git clone https://github.com/ggml-org/llama.cpp
    pip install -r llama.cpp/requirements.txt
    python llama.cpp/convert_hf_to_gguf.py out/loco-merged \\
        --outfile out/loco-f16.gguf --outtype f16
    # build llama.cpp (cmake -B build && cmake --build build --config Release), then:
    ./llama.cpp/build/bin/llama-quantize out/loco-f16.gguf out/loco-Q4_K_M.gguf Q4_K_M

    # Ollama: point a Modelfile at the GGUF (see ollama/Modelfile in this repo)
    #   FROM ./loco-Q4_K_M.gguf
    ollama create loco -f ollama/Modelfile
    ollama run loco
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model (bf16/fp16).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adapter", required=True, help="Directory containing the trained adapter.")
    p.add_argument("--base", default=None,
                   help="Base model id/path (defaults to base_model_name_or_path in adapter_config.json).")
    p.add_argument("--output-dir", required=True, help="Where to write the merged model.")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto = bf16 if supported by the chosen device, else fp16 on GPU / bf16 on CPU.")
    p.add_argument("--device-map", default="cpu", help="'cpu' (safe, RAM-bound) or 'auto' (GPU).")
    p.add_argument("--max-shard-size", default="5GB")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer source (defaults to the adapter dir, then the base model).")
    p.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                   help="Also upload the merged model to this Hugging Face repo.")
    p.add_argument("--private", action="store_true", help="Create the hub repo as private.")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args(argv)


def resolve_base(adapter_dir: Path, base: str | None) -> str:
    if base:
        return base
    cfg_file = adapter_dir / "adapter_config.json"
    if not cfg_file.is_file():
        sys.exit(f"[error] {cfg_file} not found; pass --base explicitly.")
    name = json.loads(cfg_file.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if not name:
        sys.exit("[error] adapter_config.json has no base_model_name_or_path; pass --base.")
    return name


def resolve_dtype(choice: str, device_map: str):
    import torch

    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if choice == "fp32":
        return torch.float32
    if device_map != "cpu" and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_dir():
        sys.exit(f"[error] adapter directory not found: {adapter_dir}")
    base = resolve_base(adapter_dir, args.base)
    dtype = resolve_dtype(args.dtype, args.device_map)
    print(f"[merge] base={base} adapter={adapter_dir} dtype={dtype} device_map={args.device_map}")

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), torch_dtype=dtype)
    model = model.merge_and_unload()
    model.config.use_cache = True

    tok_source = args.tokenizer
    if tok_source is None:
        tok_source = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").is_file() else base
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=args.trust_remote_code)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(str(out))
    print(f"[merge] saved merged model to {out.resolve()}")

    if args.push_to_hub:
        print(f"[hub] uploading to {args.push_to_hub} (private={args.private}) ...")
        model.push_to_hub(args.push_to_hub, private=args.private,
                          safe_serialization=True, max_shard_size=args.max_shard_size)
        tokenizer.push_to_hub(args.push_to_hub, private=args.private)
        print(f"[hub] done: https://huggingface.co/{args.push_to_hub}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[next] GGUF/Ollama conversion steps are in this script's docstring "
          "(python merge_lora.py --help shows only flags).")


if __name__ == "__main__":
    main()
