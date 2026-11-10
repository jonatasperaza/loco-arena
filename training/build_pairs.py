#!/usr/bin/env python
"""Build DPO training pairs from base model generations.

In the LoCo methodology ("Think Less, Locally"), we sample multiple generations
for the same prompt from the base model (DeepSeek-R1-Distill-Qwen-7B), filter for
those that reach the *correct* final answer, and construct preference pairs:
  - chosen: the shortest correct completion (most concise reasoning)
  - rejected: the longest correct completion (most redundant reasoning)

Usage::

    # Generate pairs from problems.jsonl against local vLLM / Ollama
    python build_pairs.py \\
        --input-file data/problems.jsonl \\
        --output-file data/loco_pairs.jsonl \\
        --base-url http://localhost:8001/v1 \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --samples-per-prompt 5 \\
        --min-ratio 1.4 \\
        --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

BOXED_REGEX = re.compile(r"\\boxed\{([^{}]+)\}")
THINK_REGEX = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract concise vs redundant reasoning pairs for DPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", required=True, help="Input JSONL with 'prompt' and 'answer' fields.")
    parser.add_argument("--output-file", required=True, help="Output JSONL with 'prompt', 'chosen', 'rejected'.")
    parser.add_argument("--base-url", default="http://localhost:8001/v1", help="OpenAI-compatible server URL.")
    parser.add_argument("--api-key", default="EMPTY", help="Bearer API key.")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="Model name.")
    parser.add_argument("--samples-per-prompt", "-n", type=int, default=5, help="Number of completions per prompt.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max completion tokens.")
    parser.add_argument("--min-ratio", type=float, default=1.5,
                        help="Minimum ratio (len(rejected_think) / len(chosen_think)) to accept pair.")
    parser.add_argument("--concurrency", type=int, default=4, help="Max parallel API requests.")
    return parser.parse_args()


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from the completion text."""
    matches = BOXED_REGEX.findall(text)
    if not matches:
        return None
    ans = matches[-1].strip()
    # Normalize LaTeX \text{...}
    ans = re.sub(r"^\\text\{([^{}]+)\}$", r"\1", ans)
    return ans.strip()


def normalize_answer(ans: str) -> str:
    """Normalize mathematical / textual answers for robust comparison."""
    norm = ans.strip().lower()
    norm = re.sub(r"\s+", " ", norm)
    # Strip common dollar signs or units for numeric checks
    norm = norm.replace("$", "").replace(",", "")
    return norm


def extract_think_and_answer(text: str) -> tuple[str, str]:
    """Split completion into thinking trace and final answer."""
    m = THINK_REGEX.search(text)
    if m:
        think = m.group(1).strip()
        answer = text[m.end() :].strip()
        return think, answer
    # Fallback if closing tag only
    if "</think>" in text:
        parts = text.split("</think>", 1)
        return parts[0].replace("<think>", "").strip(), parts[1].strip()
    return "", text.strip()


async def sample_completion(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> str | None:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with semaphore:
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content", "")
                reasoning = msg.get("reasoning_content")
                if reasoning and "</think>" not in content:
                    return f"<think>\n{reasoning}\n</think>\n\n{content}"
                return content
            print(f"[warn] Server returned HTTP {resp.status_code}: {resp.text[:100]}", file=sys.stderr)
            return None
        except Exception as exc:
            print(f"[warn] Request failed: {exc}", file=sys.stderr)
            return None


async def process_prompt(
    item: dict[str, Any],
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, str] | None:
    prompt = item.get("prompt", "").strip()
    expected_ans = item.get("answer", "").strip()
    if not prompt or not expected_ans:
        return None

    norm_expected = normalize_answer(expected_ans)

    tasks = [
        sample_completion(
            client=client,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            semaphore=semaphore,
        )
        for _ in range(args.samples_per_prompt)
    ]

    completions = await asyncio.gather(*tasks)

    # Filter correct completions
    correct_completions: list[tuple[int, str]] = []
    for comp in completions:
        if not comp:
            continue
        boxed = extract_boxed_answer(comp)
        if boxed and normalize_answer(boxed) == norm_expected:
            think, _ = extract_think_and_answer(comp)
            # Estimate think length in words / rough tokens
            think_len = len(think.split())
            if think_len > 0:
                correct_completions.append((think_len, comp))

    if len(correct_completions) < 2:
        return None

    # Sort by think length: shortest first
    correct_completions.sort(key=lambda x: x[0])
    shortest_len, chosen_text = correct_completions[0]
    longest_len, rejected_text = correct_completions[-1]

    ratio = longest_len / max(1, shortest_len)
    if ratio < args.min_ratio:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


async def main() -> None:
    args = parse_args()

    in_path = Path(args.input_file)
    if not in_path.is_file():
        sys.exit(f"Input file not found: {in_path}")

    items = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"[build_pairs] Loaded {len(items)} prompts from {in_path}")
    print(f"[build_pairs] Sampling {args.samples_per_prompt} completions per prompt against {args.base_url}...")

    semaphore = asyncio.Semaphore(args.concurrency)
    out_pairs = []

    async with httpx.AsyncClient() as client:
        tasks = [process_prompt(it, client, args, semaphore) for it in items]
        results = await asyncio.gather(*tasks)

    for res in results:
        if res:
            out_pairs.append(res)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in out_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n[build_pairs] Summary:")
    print(f"  Processed Prompts: {len(items)}")
    print(f"  Valid Pairs Built: {len(out_pairs)}")
    if out_pairs:
        chosen_lens = [len(extract_think_and_answer(p['chosen'])[0].split()) for p in out_pairs]
        rejected_lens = [len(extract_think_and_answer(p['rejected'])[0].split()) for p in out_pairs]
        avg_chosen = sum(chosen_lens) / len(chosen_lens)
        avg_rejected = sum(rejected_lens) / len(rejected_lens)
        reduction = (1.0 - (avg_chosen / avg_rejected)) * 100.0
        print(f"  Avg Think Words (Chosen):   {avg_chosen:.1f}")
        print(f"  Avg Think Words (Rejected): {avg_rejected:.1f}")
        print(f"  Estimated Think Reduction:  {reduction:.1f}%")
        print(f"  Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
