#!/usr/bin/env python
"""Fast-LoCo: QLoRA DPO that teaches an R1-distill model to *think less*.

LoCo ("Think Less, Locally") applies Direct Preference Optimization so the model
prefers a concise ``<think>`` trace over a redundant one that reaches the *same*
correct answer. This script is a lightweight, single-GPU re-implementation of
that recipe for ``deepseek-ai/DeepSeek-R1-Distill-Qwen-7B``:

* 4-bit NF4 quantized base (QLoRA, double quantization) + LoRA adapters.
* ``trl.DPOTrainer`` with an *implicit* reference model (the same weights with
  the adapter disabled), so no second copy of the model is held in VRAM.
* Paged 8-bit AdamW, gradient checkpointing, bf16/fp16 autodetection.

Dataset: local JSONL, one object per line (see ``CONTRACT.md``)::

    {"prompt": "<question>",
     "chosen": "<think>\\nconcise reasoning\\n</think>\\n\\n... \\\\boxed{X}",
     "rejected": "<think>\\nlong redundant reasoning\\n</think>\\n\\n... \\\\boxed{X}"}

``prompt`` is the raw user question; the tokenizer's chat template is applied
here (``add_generation_prompt=True``). The R1-distill template already appends
``<think>\\n`` to the generation prompt, in which case the leading ``<think>\\n``
of each completion is stripped so the tag is not duplicated.

VRAM guide (7B, QLoRA r=16, batch 1, grad-accum 8, gradient checkpointing):

* 24 GB: defaults (``--max-length 4096``).
* 16 GB: ``--max-length 2048 --max-prompt-length 512``.
* 12 GB: ``--max-length 1536 --max-prompt-length 384 --lora-r 8``.

Examples::

    # CPU-only sanity check: tokenizer + dataset, no model weights downloaded
    python train_dpo_lite.py --data data/sample_pairs.jsonl --dry-run

    # Train (24 GB GPU)
    python train_dpo_lite.py --data data/pairs.jsonl --output-dir out/loco-lora

    # Train (16 GB GPU) with a 5% eval split
    python train_dpo_lite.py --data data/pairs.jsonl --max-length 2048 \\
        --max-prompt-length 512 --eval-split 0.05
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import inspect
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA DPO (LoCo-style concise-reasoning preference) for R1-distill models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("model / data")
    g.add_argument("--model", default=DEFAULT_MODEL, help="Base model (HF id or local path).")
    g.add_argument("--tokenizer", default=None, help="Tokenizer id/path (defaults to --model).")
    g.add_argument("--data", required=True, help="JSONL with prompt/chosen/rejected fields.")
    g.add_argument("--eval-split", type=float, default=0.0,
                   help="Fraction of pairs held out for evaluation (0 disables).")
    g.add_argument("--system-prompt", default=None,
                   help="Optional system message. DeepSeek recommends none for R1-distill models.")
    g.add_argument("--output-dir", default="out/loco-lora", help="Where the adapter is saved.")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--trust-remote-code", action="store_true")

    g = p.add_argument_group("quantization / attention")
    g.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                   help="Disable 4-bit loading (plain LoRA in bf16/fp16; needs much more VRAM).")
    g.add_argument("--flash-attn", action="store_true",
                   help="Use flash_attention_2 if the flash-attn package is installed (else sdpa).")

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--target-modules", nargs="+",
                   default=["q_proj", "v_proj", "k_proj", "o_proj"],
                   help="Module names to adapt (add gate_proj up_proj down_proj for more capacity).")

    g = p.add_argument_group("DPO / optimization")
    g.add_argument("--beta", type=float, default=0.1, help="DPO temperature.")
    g.add_argument("--loss-type", default="sigmoid", help="DPO loss variant (sigmoid, ipo, ...).")
    g.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=5e-6)
    g.add_argument("--epochs", type=float, default=1.0)
    g.add_argument("--max-steps", type=int, default=-1, help="Overrides --epochs when > 0.")
    g.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    g.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    g.add_argument("--lr-scheduler", default="cosine")
    g.add_argument("--warmup-ratio", type=float, default=0.1)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--max-length", type=int, default=4096,
                   help="Max prompt+completion tokens (use 2048 on 16 GB GPUs).")
    g.add_argument("--max-prompt-length", type=int, default=1024)
    g.add_argument("--optim", default="paged_adamw_8bit")
    g.add_argument("--logging-steps", type=int, default=5)
    g.add_argument("--save-steps", type=int, default=100)
    g.add_argument("--eval-steps", type=int, default=100)
    g.add_argument("--save-total-limit", type=int, default=2)
    g.add_argument("--report-to", default="none", help="none | wandb | tensorboard ...")
    g.add_argument("--resume-from-checkpoint", default=None)

    g = p.add_argument_group("misc")
    g.add_argument("--dry-run", action="store_true",
                   help="Only load tokenizer + dataset, print a formatted sample and length stats "
                        "(CPU only, no model weights).")
    g.add_argument("--num-samples-print", type=int, default=1,
                   help="How many formatted samples to print in --dry-run.")
    args = p.parse_args(argv)

    if not 0.0 <= args.eval_split < 1.0:
        p.error("--eval-split must be in [0, 1).")
    if args.max_prompt_length >= args.max_length:
        p.error("--max-prompt-length must be smaller than --max-length.")
    return args


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_pairs(path: str) -> list[dict[str, str]]:
    """Load and validate a prompt/chosen/rejected JSONL file."""
    file = Path(path)
    if not file.is_file():
        sys.exit(f"[error] dataset not found: {file}")

    rows: list[dict[str, str]] = []
    errors: list[str] = []
    with file.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(obj, dict):
                errors.append(f"line {lineno}: expected a JSON object")
                continue
            missing = [k for k in REQUIRED_FIELDS if not isinstance(obj.get(k), str) or not obj[k].strip()]
            if missing:
                errors.append(f"line {lineno}: missing/empty string field(s): {', '.join(missing)}")
                continue
            if obj["chosen"].strip() == obj["rejected"].strip():
                errors.append(f"line {lineno}: chosen and rejected are identical")
                continue
            rows.append({k: obj[k] for k in REQUIRED_FIELDS})

    if errors:
        print(f"[warn] skipped {len(errors)} invalid line(s):", file=sys.stderr)
        for err in errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
    if not rows:
        sys.exit("[error] no valid pairs found in dataset.")
    return rows


def build_prompt(tokenizer: Any, question: str, system_prompt: str | None) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def template_opens_think(tokenizer: Any, system_prompt: str | None) -> bool:
    """True if the chat template already ends the generation prompt with ``<think>``."""
    probe = build_prompt(tokenizer, "ping", system_prompt)
    return probe.rstrip().endswith(THINK_OPEN)


def trl_appends_eos() -> bool:
    """Whether the installed TRL DPOTrainer appends EOS to completions itself.

    TRL >= 0.12 adds ``eos_token_id`` to chosen/rejected ids in ``tokenize_row``;
    if we also appended the EOS *text* it would be duplicated.
    """
    try:
        from trl import DPOTrainer

        fn = getattr(DPOTrainer, "tokenize_row", None)
        return fn is not None and "eos_token" in inspect.getsource(fn)
    except Exception:  # noqa: BLE001 - TRL missing or source unavailable
        return False


def format_completion(text: str, strip_think_open: bool, eos_text: str | None) -> str:
    out = text.lstrip()
    if strip_think_open and out.startswith(THINK_OPEN):
        out = out[len(THINK_OPEN):]
        out = out[1:] if out.startswith("\n") else out
    elif not strip_think_open and not out.startswith(THINK_OPEN) and THINK_CLOSE in out:
        # Template does not pre-insert <think>: make sure the trace opens properly.
        out = f"{THINK_OPEN}\n{out}"
    out = out.rstrip()
    if eos_text and not out.endswith(eos_text):
        out += eos_text
    return out


def format_pairs(rows: list[dict[str, str]], tokenizer: Any, system_prompt: str | None,
                 append_eos_text: bool) -> tuple[list[dict[str, str]], bool]:
    strip = template_opens_think(tokenizer, system_prompt)
    eos_text = tokenizer.eos_token if append_eos_text else None
    formatted = [
        {
            "prompt": build_prompt(tokenizer, r["prompt"], system_prompt),
            "chosen": format_completion(r["chosen"], strip, eos_text),
            "rejected": format_completion(r["rejected"], strip, eos_text),
        }
        for r in rows
    ]
    return formatted, strip


def think_part(completion: str) -> str:
    """Text of the reasoning trace (before ``</think>``), tags removed."""
    head = completion.split(THINK_CLOSE, 1)[0] if THINK_CLOSE in completion else completion
    return head.replace(THINK_OPEN, "")


def _stats(values: list[int]) -> str:
    if not values:
        return "n/a"
    return (f"min={min(values)}  mean={statistics.mean(values):.0f}  "
            f"median={statistics.median(values):.0f}  max={max(values)}")


def print_dataset_report(formatted: list[dict[str, str]], tokenizer: Any,
                         max_length: int, max_prompt_length: int) -> None:
    def ntok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    p_len = [ntok(r["prompt"]) for r in formatted]
    c_len = [ntok(r["chosen"]) for r in formatted]
    r_len = [ntok(r["rejected"]) for r in formatted]
    ct_len = [ntok(think_part(r["chosen"])) for r in formatted]
    rt_len = [ntok(think_part(r["rejected"])) for r in formatted]

    print("\n=== token length stats ===")
    print(f"pairs               : {len(formatted)}")
    print(f"prompt              : {_stats(p_len)}")
    print(f"chosen (completion) : {_stats(c_len)}")
    print(f"rejected (completion): {_stats(r_len)}")
    print(f"chosen  <think>     : {_stats(ct_len)}")
    print(f"rejected <think>    : {_stats(rt_len)}")
    if sum(rt_len):
        print(f"think-token reduction chosen vs rejected: {(1 - sum(ct_len) / sum(rt_len)) * 100:.1f}%")

    over_prompt = sum(1 for x in p_len if x > max_prompt_length)
    over_total = sum(1 for p, c, r in zip(p_len, c_len, r_len) if p + max(c, r) > max_length)
    print(f"prompts > max_prompt_length ({max_prompt_length}): {over_prompt}")
    print(f"pairs   > max_length ({max_length}) (will be truncated): {over_total}")
    if over_total:
        print("  tip: truncation cuts the end of long *rejected* traces, which weakens the "
              "length signal; raise --max-length or filter those pairs.")


# --------------------------------------------------------------------------- #
# Version-tolerant helpers
# --------------------------------------------------------------------------- #
def accepted_params(callable_obj: Any) -> set[str]:
    try:
        return set(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError):
        return set()


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any], label: str) -> dict[str, Any]:
    params = accepted_params(callable_obj)
    if not params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(callable_obj).parameters.values()
    ):
        return kwargs
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        print(f"[info] {label}: ignoring unsupported field(s) for this version: {', '.join(dropped)}")
    return kept


def pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                        trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        sys.exit("[error] tokenizer has no chat template; pass --tokenizer with one.")
    return tok


def dry_run(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args)
    rows = load_pairs(args.data)
    # In dry-run we do not need TRL; show what training would see.
    append_eos_text = not trl_appends_eos()
    formatted, stripped = format_pairs(rows, tokenizer, args.system_prompt, append_eos_text)

    print(f"tokenizer           : {args.tokenizer or args.model}")
    print(f"template opens think: {stripped} "
          f"({'leading <think> stripped from completions' if stripped else 'completions kept as-is'})")
    print(f"EOS handling        : "
          f"{'appended as text' if append_eos_text else 'appended by TRL DPOTrainer'} "
          f"(eos={tokenizer.eos_token!r})")
    for i, sample in enumerate(formatted[: max(0, args.num_samples_print)]):
        print(f"\n=== formatted sample #{i} ===")
        print("--- prompt ---")
        print(sample["prompt"])
        print("--- chosen ---")
        print(sample["chosen"])
        print("--- rejected ---")
        print(sample["rejected"])
    print_dataset_report(formatted, tokenizer, args.max_length, args.max_prompt_length)
    print("\n[dry-run] OK - dataset and chat template look valid. No model was loaded.")


def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        sys.exit("[error] CUDA GPU required for training (bitsandbytes 4-bit). "
                 "Use --dry-run to validate data on CPU.")
    set_seed(args.seed)

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    gpu = torch.cuda.get_device_properties(0)
    vram_gb = gpu.total_memory / 1024**3
    print(f"[gpu] {gpu.name}  {vram_gb:.1f} GB  dtype={'bf16' if use_bf16 else 'fp16'}")
    if vram_gb < 20 and args.max_length > 2048:
        print("[tip] <20 GB VRAM detected: consider --max-length 2048 --max-prompt-length 512.")

    # ---- tokenizer + data ------------------------------------------------- #
    tokenizer = load_tokenizer(args)
    rows = load_pairs(args.data)
    formatted, stripped = format_pairs(rows, tokenizer, args.system_prompt,
                                       append_eos_text=not trl_appends_eos())
    print(f"[data] {len(formatted)} pairs; template opens <think>: {stripped}")

    dataset = Dataset.from_list(formatted)
    eval_dataset = None
    if args.eval_split > 0 and len(dataset) >= 10:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        dataset, eval_dataset = split["train"], split["test"]
        print(f"[data] train={len(dataset)} eval={len(eval_dataset)}")
    elif args.eval_split > 0:
        print("[warn] dataset too small for an eval split (<10 pairs); training on all of it.")

    # ---- model ------------------------------------------------------------ #
    attn_impl = "sdpa"
    if args.flash_attn:
        if importlib.util.find_spec("flash_attn") is not None:
            attn_impl = "flash_attention_2"
        else:
            print("[warn] --flash-attn requested but flash-attn is not installed; using sdpa.")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": compute_dtype,
        "attn_implementation": attn_impl,
        "device_map": {"": torch.cuda.current_device()},
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    print(f"[model] loading {args.model} (4bit={args.load_in_4bit}, attn={attn_impl})")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    gc_kwargs = {"use_reentrant": False}
    if args.load_in_4bit:
        prep = {"use_gradient_checkpointing": True}
        if "gradient_checkpointing_kwargs" in accepted_params(prepare_model_for_kbit_training):
            prep["gradient_checkpointing_kwargs"] = gc_kwargs
        model = prepare_model_for_kbit_training(model, **prep)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- DPO config (filtered to what this TRL version supports) ---------- #
    cfg_params = accepted_params(DPOConfig)
    cfg: dict[str, Any] = {
        "output_dir": args.output_dir,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lr_scheduler_type": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_length - args.max_prompt_length,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "optim": args.optim,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": gc_kwargs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "seed": args.seed,
        "remove_unused_columns": False,
        "precompute_ref_log_probs": False,
    }
    strategy_key = "eval_strategy" if "eval_strategy" in cfg_params else "evaluation_strategy"
    if eval_dataset is not None:
        cfg[strategy_key] = "steps"
        cfg["eval_steps"] = args.eval_steps
    else:
        cfg[strategy_key] = "no"
    dpo_args = DPOConfig(**filter_kwargs(DPOConfig, cfg, "DPOConfig"))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": None,  # PEFT: reference = base weights with adapter disabled
        "args": dpo_args,
        "train_dataset": dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_params = accepted_params(DPOTrainer.__init__)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_params})

    print("[train] starting DPO. VRAM tips: lower --max-length, --lora-r or add --flash-attn "
          "if you hit OOM; keep batch 1 and raise --grad-accum instead of batch size.")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ---- save ------------------------------------------------------------- #
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    meta = {
        "cli_args": vars(args),
        "template_opens_think": stripped,
        "num_train_pairs": len(dataset),
        "num_eval_pairs": len(eval_dataset) if eval_dataset is not None else 0,
        "dtype": "bf16" if use_bf16 else "fp16",
        "attn_implementation": attn_impl,
        "versions": {n: pkg_version(n) for n in
                     ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate")},
    }
    (out / "training_args.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if torch.cuda.is_available():
        print(f"[gpu] peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    print(f"[done] adapter + tokenizer saved to {out.resolve()}")
    print(f"       next: python merge_lora.py --adapter {out} --output-dir out/loco-merged")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.dry_run:
        dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
