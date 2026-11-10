"""Inference streaming engine supporting both 'live' (OpenAI-compatible) and 'demo' (zero-GPU simulator) modes.

Conforms strictly to CONTRACT.md schemas and event ordering.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from typing import AsyncGenerator

import httpx

from app.config import Settings
from app.metrics import MetricsTracker
from app.schemas import (
    DoneEvent,
    ErrorEvent,
    ModelEvent,
    ModelRole,
    Phase,
    PhaseEvent,
    StartEvent,
    TokenEvent,
)
from app.think_parser import ThinkParser

# Realistic demo thinking templates for presets and general queries
DEMO_PRESETS_THINK = {
    "compare-911-99": {
        "baseline": (
            "Let me look at the two numbers: 9.11 and 9.9.\n"
            "At first glance, 11 is bigger than 9, so some people might think 9.11 is greater.\n"
            "Wait, let me double-check the place values carefully.\n"
            "Both numbers have the same integer part: 9.\n"
            "Now look at the tenths digit. In 9.11, the tenths digit is 1.\n"
            "In 9.9, the tenths digit is 9.\n"
            "Since 9 > 1, 9.9 must be greater.\n"
            "Wait, let me make sure I am not confusing version numbering like v9.11 vs v9.9.\n"
            "In standard mathematical decimals, 9.9 is 9 + 9/10 = 9 + 90/100 = 9.90.\n"
            "And 9.11 is 9 + 11/100.\n"
            "Comparing 9.90 to 9.11, 90/100 is clearly greater than 11/100.\n"
            "Let me re-verify: 9.90 - 9.11 = 0.79 > 0.\n"
            "Yes, 9.9 is strictly greater than 9.11.\n"
            "Let me ensure no edge cases in the wording."
        ),
        "loco": (
            "We compare 9.11 and 9.9.\n"
            "Both have integer part 9. Comparing the tenths place: 9.9 has tenths digit 9, while 9.11 has tenths digit 1.\n"
            "Since 9.90 > 9.11, 9.9 is greater."
        ),
        "answer": "The greater number is 9.9. In decimal place value, 9.9 equals 9.90, which is larger than 9.11 by 0.79.\n\nFinal answer: \\boxed{9.9}",
    },
    "strawberry-r-count": {
        "baseline": (
            "Let's count the occurrences of the letter 'r' in the word 'strawberry'.\n"
            "Let me spell out the word letter by letter:\n"
            "Letter 1: s\n"
            "Letter 2: t\n"
            "Letter 3: r -> That's one 'r'.\n"
            "Letter 4: a\n"
            "Letter 5: w\n"
            "Letter 6: b\n"
            "Letter 7: e\n"
            "Letter 8: r -> That's two 'r's.\n"
            "Letter 9: r -> That's three 'r's.\n"
            "Letter 10: y\n"
            "Wait, let me re-read the word slowly to ensure I didn't miscount.\n"
            "Word: s-t-r-a-w-b-e-r-r-y.\n"
            "Is there an 'r' in straw? s - t - r - a - w. Yes, position 3.\n"
            "Is there an 'r' in berry? b - e - r - r - y. Positions 8 and 9.\n"
            "Wait, could 'r' be case-sensitive? The prompt says 'letter 'r''.\n"
            "Total positions with 'r': 3, 8, 9. Total count = 3.\n"
            "Let me double-check whether any other letters are 'r'. No, only 3."
        ),
        "loco": (
            "Breaking down 'strawberry' character by character:\n"
            "s-t-r-a-w-b-e-r-r-y\n"
            "Positions with 'r': 3 ('r'), 8 ('r'), 9 ('r').\n"
            "Total count is 3."
        ),
        "answer": "The letter 'r' appears 3 times in the word 'strawberry' (at indices 3, 8, and 9).\n\nFinal answer: \\boxed{3}",
    },
    "bat-and-ball": {
        "baseline": (
            "Let the total cost be $1.10.\n"
            "Bat + Ball = $1.10.\n"
            "Bat = Ball + $1.00.\n"
            "Wait, the intuitive answer people jump to is 10 cents, but let's check:\n"
            "If Ball = $0.10, then Bat = $1.10, so total = $1.20, which is wrong.\n"
            "Let me set up algebraic equations formally:\n"
            "Let x be the price of the ball in dollars.\n"
            "Then the bat costs x + 1.00.\n"
            "Equation: x + (x + 1.00) = 1.10\n"
            "2x + 1.00 = 1.10\n"
            "Subtract 1.00 from both sides: 2x = 0.10.\n"
            "Divide by 2: x = 0.05.\n"
            "Wait, let me check the question carefully: does it ask for dollars or cents?\n"
            "'How much does the ball cost in cents?'\n"
            "x = $0.05 = 5 cents.\n"
            "Let me double-check: Bat = $1.05, Ball = $0.05.\n"
            "Bat - Ball = $1.05 - $0.05 = $1.00 (correct).\n"
            "Bat + Ball = $1.05 + $0.05 = $1.10 (correct)."
        ),
        "loco": (
            "Let b be the cost of the ball. The bat costs b + 1.00.\n"
            "b + (b + 1.00) = 1.10 => 2b = 0.10 => b = 0.05 dollars = 5 cents."
        ),
        "answer": "The ball costs 5 cents ($0.05).\n\nVerification: Bat ($1.05) + Ball ($0.05) = $1.10, and Bat costs exactly $1.00 more than Ball.\n\nFinal answer: \\boxed{5}",
    },
    "powers-mod-7": {
        "baseline": (
            "We want to find 2^2025 mod 7.\n"
            "Since 7 is prime, by Fermat's Little Theorem, 2^(7-1) = 2^6 = 1 (mod 7).\n"
            "Wait, does the power of 2 have a smaller order modulo 7?\n"
            "Let's compute consecutive powers of 2 mod 7:\n"
            "2^1 = 2 mod 7\n"
            "2^2 = 4 mod 7\n"
            "2^3 = 8 = 1 mod 7\n"
            "So the order of 2 modulo 7 is 3, not 6! The cycle length is 3.\n"
            "Now we need to compute 2025 mod 3.\n"
            "Let's sum the digits of 2025: 2 + 0 + 2 + 5 = 9.\n"
            "Since 9 is divisible by 3, 2025 is divisible by 3.\n"
            "Specifically: 2025 / 3 = 675 with remainder 0.\n"
            "Wait, let me double-check the division:\n"
            "3 * 600 = 1800, remainder 225. 3 * 75 = 225. 1800 + 225 = 2025. Exact!\n"
            "So 2^2025 = (2^3)^675 = 1^675 = 1 (mod 7).\n"
            "Wait, let me verify if 2^0 = 1.\n"
            "Yes, 2^3 = 8 = 1 (mod 7). Any multiple of 3 exponent gives 1 mod 7."
        ),
        "loco": (
            "Notice that 2^3 = 8 = 1 (mod 7). Thus the powers of 2 cycle with period 3.\n"
            "Sum of digits of 2025 is 2 + 0 + 2 + 5 = 9, so 2025 = 0 (mod 3).\n"
            "Hence 2^2025 = (2^3)^675 = 1^675 = 1 (mod 7)."
        ),
        "answer": "The remainder is 1.\n\nExplanation: 2^3 = 8 = 1 (mod 7). Since 2025 is evenly divisible by 3 (2025 = 3 * 675), 2^2025 = (2^3)^675 = 1^675 = 1 (mod 7).\n\nFinal answer: \\boxed{1}",
    },
    "knights-and-knaves": {
        "baseline": (
            "We have two inhabitants A and B on the island.\n"
            "Knights always tell the truth; Knaves always lie.\n"
            "Inhabitant A says: 'At least one of us is a knave.'\n"
            "Let's test both hypotheses for A:\n"
            "Case 1: Suppose A is a Knave.\n"
            "Then A's statement must be false.\n"
            "The negation of 'At least one of us is a knave' is 'Neither of us is a knave', meaning both A and B are knights.\n"
            "But we assumed A is a knave! A cannot be both a knight and a knave.\n"
            "This is a contradiction, so Case 1 is impossible.\n"
            "Case 2: Suppose A is a Knight.\n"
            "Then A tells the truth, so the statement 'At least one of us is a knave' must be TRUE.\n"
            "Since A is a knight, A is not a knave.\n"
            "For 'at least one of us is a knave' to hold true, B must be a knave.\n"
            "Let me check if this leads to any contradiction with B:\n"
            "B is a knave. A is a knight. A's statement is true. Everything is consistent.\n"
            "Wait, let me double check the logical formulation:\n"
            "A <=> (not A or not B).\n"
            "If A=False: False <=> (True or ...), which is False <=> True (contradiction).\n"
            "If A=True: True <=> (False or not B) => not B is True => B=False (Knave).\n"
            "Deduction holds solid."
        ),
        "loco": (
            "If A were a knave, 'at least one is a knave' would be false, implying both are knights (contradiction).\n"
            "Thus A is a knight and speaks the truth. For at least one to be a knave, B must be a knave."
        ),
        "answer": "A is a knight and B is a knave.\n\nFinal answer: \\boxed{\\text{A is a knight, B is a knave}}",
    },
}


def _match_preset(prompt: str) -> tuple[str, str, str] | None:
    p = prompt.lower()
    if "9.11" in p and "9.9" in p:
        d = DEMO_PRESETS_THINK["compare-911-99"]
        return d["baseline"], d["loco"], d["answer"]
    if "strawberry" in p and "r" in p:
        d = DEMO_PRESETS_THINK["strawberry-r-count"]
        return d["baseline"], d["loco"], d["answer"]
    if "bat" in p and "ball" in p:
        d = DEMO_PRESETS_THINK["bat-and-ball"]
        return d["baseline"], d["loco"], d["answer"]
    if "2025" in p and ("mod 7" in p or "divided by 7" in p):
        d = DEMO_PRESETS_THINK["powers-mod-7"]
        return d["baseline"], d["loco"], d["answer"]
    if "knight" in p and "knave" in p:
        d = DEMO_PRESETS_THINK["knights-and-knaves"]
        return d["baseline"], d["loco"], d["answer"]
    return None


async def run_demo_stream(
    model_role: ModelRole,
    model_name: str,
    prompt: str,
    delay_scale: float = 1.0,
) -> AsyncGenerator[ModelEvent, None]:
    """Simulate streaming for a single model in demo mode."""
    start_time = time.perf_counter()
    tracker = MetricsTracker(start_time=start_time)

    # 1. Emit StartEvent
    yield StartEvent(
        model=model_role,
        model_name=model_name,
        t_ms=0.0,
    )

    preset_data = _match_preset(prompt)
    if preset_data:
        baseline_think, loco_think, answer_text = preset_data
        think_text = baseline_think if model_role == "baseline" else loco_think
    else:
        # Fallback reasoning generated for custom user prompts
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        if model_role == "baseline":
            think_text = (
                f"Analyzing the problem: '{prompt[:60]}...'\n"
                "Let me examine all constraints and assumptions step by step.\n"
                "First consideration: what is the direct objective?\n"
                "Wait, let me double check if there are alternative interpretations.\n"
                "Evaluating approach 1: formal rigorous derivation.\n"
                "Checking intermediate state consistency...\n"
                "Wait, let me re-verify this step again to eliminate any potential flaws.\n"
                "Everything appears consistent, proceeding to final answer formulation."
            )
        else:
            think_text = (
                f"Analyzing: '{prompt[:60]}...'\n"
                "Direct derivation from given conditions leads to consistent result."
            )
        answer_text = (
            f"Based on the analysis of the prompt, the deduction is verified.\n\n"
            f"Final answer: \\boxed{{{h % 100}}}"
        )

    # Simulated TTFT (Time to first token)
    base_ttft = 0.25 if model_role == "loco" else 0.45
    ttft_delay = (base_ttft + random.uniform(0.05, 0.15)) * delay_scale
    if ttft_delay > 0:
        await asyncio.sleep(ttft_delay)

    # Stream Think phase word by word
    think_words = think_text.split()
    for word in think_words:
        token_str = word + " "
        tracker.record_token("think", 1)
        yield TokenEvent(
            model=model_role,
            phase="think",
            text=token_str,
            metrics=tracker.snapshot(),
        )
        if delay_scale > 0:
            # Baseline ~30 tok/s (0.033s), LoCo ~40 tok/s (0.025s) with jitter
            w_delay = (0.025 if model_role == "loco" else 0.033) * delay_scale
            await asyncio.sleep(w_delay * random.uniform(0.8, 1.2))

    # Phase transition to Answer
    tracker.record_phase_switch()
    yield PhaseEvent(
        model=model_role,
        phase="answer",
        metrics=tracker.snapshot(),
    )

    if delay_scale > 0:
        await asyncio.sleep(0.08 * delay_scale)

    # Stream Answer phase word by word
    answer_words = answer_text.split()
    for word in answer_words:
        token_str = word + " "
        tracker.record_token("answer", 1)
        yield TokenEvent(
            model=model_role,
            phase="answer",
            text=token_str,
            metrics=tracker.snapshot(),
        )
        if delay_scale > 0:
            w_delay = 0.022 * delay_scale
            await asyncio.sleep(w_delay * random.uniform(0.8, 1.2))

    # Done Event
    tracker.record_done()
    yield DoneEvent(
        model=model_role,
        metrics=tracker.snapshot(),
    )


async def run_live_stream(
    model_role: ModelRole,
    model_name: str,
    base_url: str,
    api_key: str,
    prompt: str,
    settings: Settings,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> AsyncGenerator[ModelEvent, None]:
    """Execute live streaming against an OpenAI-compatible endpoint."""
    start_time = time.perf_counter()
    tracker = MetricsTracker(start_time=start_time)
    parser = ThinkParser()

    yield StartEvent(
        model=model_role,
        model_name=model_name,
        t_ms=0.0,
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature if temperature is not None else settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": max_tokens if max_tokens is not None else settings.max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    yield ErrorEvent(
                        model=model_role,
                        message=f"HTTP {resp.status_code}: {err_body.decode('utf-8', errors='ignore')[:300]}",
                    )
                    return

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # Handle reasoning_content (vLLM reasoning parser or Ollama)
                    reasoning_delta = delta.get("reasoning_content")
                    if reasoning_delta:
                        segments = parser.feed_reasoning(reasoning_delta)
                        for seg in segments:
                            tracker.record_token("think", 1)
                            yield TokenEvent(
                                model=model_role,
                                phase="think",
                                text=seg.text,
                                metrics=tracker.snapshot(),
                            )

                    # Handle regular content
                    content_delta = delta.get("content")
                    if content_delta:
                        segments = parser.feed(content_delta)
                        for seg in segments:
                            if seg.transition:
                                tracker.record_phase_switch()
                                yield PhaseEvent(
                                    model=model_role,
                                    phase="answer",
                                    metrics=tracker.snapshot(),
                                )
                            elif seg.text:
                                tracker.record_token(seg.phase, 1)
                                yield TokenEvent(
                                    model=model_role,
                                    phase=seg.phase,
                                    text=seg.text,
                                    metrics=tracker.snapshot(),
                                )

                # Flush remaining buffer at finish
                remaining_segs = parser.finish()
                for seg in remaining_segs:
                    if seg.text:
                        tracker.record_token(seg.phase, 1)
                        yield TokenEvent(
                            model=model_role,
                            phase=seg.phase,
                            text=seg.text,
                            metrics=tracker.snapshot(),
                        )

        tracker.record_done()
        yield DoneEvent(
            model=model_role,
            metrics=tracker.snapshot(),
        )

    except Exception as exc:
        yield ErrorEvent(
            model=model_role,
            message=str(exc),
        )
