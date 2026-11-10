# loco-arena — Interface Contract (source of truth)

All components MUST conform to this file. Do not change it without updating every consumer.

## Directory tree

```
loco-arena/
├── CONTRACT.md                 # this file
├── README.md                   # [Agent 4]
├── LICENSE                     # [Agent 4] Apache-2.0
├── .env.example                # [Agent 4]
├── .gitignore                  # [Agent 4]
├── .dockerignore               # [Agent 4]
├── docker-compose.yml          # [Agent 4]
├── Makefile                    # [Agent 4]
├── ollama/
│   ├── Modelfile               # [Agent 4] LoCo model
│   └── Modelfile.baseline      # [Agent 4] baseline R1-Distill-Qwen-7B
├── backend/                    # [Agent 1]
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── presets.json
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app, routes, SSE
│   │   ├── config.py           # env-driven settings
│   │   ├── schemas.py          # pydantic models
│   │   ├── think_parser.py     # incremental <think>...</think> parser
│   │   ├── metrics.py          # TTFT / tok/s / counters
│   │   └── inference.py        # async OpenAI-compatible streaming + demo simulator
│   └── tests/
│       ├── test_think_parser.py
│       └── test_api.py
├── frontend/                   # [Agent 2]  (no build step: HTML + Tailwind CDN + vanilla JS)
│   ├── Dockerfile              # nginx:alpine
│   ├── nginx.conf              # serves static + proxies /api -> backend:8000 (SSE-safe)
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── training/                   # [Agent 3]
    ├── requirements.txt
    ├── train_dpo_lite.py
    ├── merge_lora.py
    ├── build_pairs.py          # turn baseline generations into (prompt, chosen, rejected) JSONL
    └── data/sample_pairs.jsonl
```

## Environment variables (backend)

| Var | Default | Meaning |
|---|---|---|
| `ARENA_MODE` | `demo` | `demo` = built-in simulator (no GPU), `live` = call OpenAI-compatible servers |
| `BASELINE_BASE_URL` | `http://localhost:8001/v1` | OpenAI-compatible base URL (vLLM / Ollama `/v1` / llama.cpp) |
| `BASELINE_MODEL` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | model name sent in request |
| `BASELINE_API_KEY` | `EMPTY` | bearer token |
| `LOCO_BASE_URL` | `http://localhost:8002/v1` | |
| `LOCO_MODEL` | `MuyaoYuan/LoCo-7B` | |
| `LOCO_API_KEY` | `EMPTY` | |
| `MAX_TOKENS` | `8192` | default max new tokens |
| `TEMPERATURE` | `0.6` | R1 recommended |
| `TOP_P` | `0.95` | |
| `REQUEST_TIMEOUT` | `600` | seconds |
| `CORS_ORIGINS` | `*` | comma separated |
| `PRESETS_PATH` | `presets.json` (relative to backend/) | |

Ollama on the host from Docker: `http://host.docker.internal:11434/v1`.

## HTTP API (backend listens on :8000, all routes prefixed `/api`)

- `GET /api/health` → `{"status":"ok","mode":"demo|live","baseline_model":str,"loco_model":str}`
- `GET /api/presets` → `[{"id":str,"title":str,"category":"math|logic","prompt":str,"expected_answer":str,"why_overthinks":str}]` (exactly 5)
- `GET /api/arena/stream?prompt=<str>&max_tokens=<int, optional>&temperature=<float, optional>` → `text/event-stream`
  - GET (so the browser `EventSource` can consume it). `prompt` required, 1..8000 chars.
  - Both models are run **concurrently**; their events are interleaved on one stream.
  - Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`.
  - Every event is a default SSE `message` event: `data: <json>\n\n`. A `: ping` comment every 15s.

### SSE event payloads (`type` discriminator; `model` ∈ `"baseline" | "loco"`)

```jsonc
{"type":"start","model":"baseline","model_name":"deepseek-ai/...","t_ms":0}
{"type":"token","model":"loco","phase":"think","text":"Let me","metrics":{...Metrics}}
{"type":"phase","model":"loco","phase":"answer","metrics":{...Metrics}}   // emitted once when </think> closes
{"type":"done","model":"loco","metrics":{...Metrics}}
{"type":"error","model":"baseline","message":"..."}                     // that model is finished after this
{"type":"arena_done","summary":{...Summary}}                               // ALWAYS last event; client closes EventSource
```

`phase` is `"think"` while inside `<think>…</think>` and `"answer"` afterwards. If the model output does not open with `<think>` (R1-distill chat templates often pre-insert `<think>\n` in the prompt), the parser starts in `think` phase and switches to `answer` on `</think>`; if no `</think>` ever appears, everything counts as `think`... EXCEPT when the first 64 chars contain no tag at all AND the stream ends — then still `think` (conservative). Tags themselves are never included in `text`.

**Metrics** (all numbers; `null` allowed where noted):
```jsonc
{
  "ttft_ms": 123.4 | null,        // time from request start to first non-empty token
  "elapsed_ms": 4567.8,
  "think_tokens": 512,
  "answer_tokens": 64,
  "total_tokens": 576,
  "tokens_per_sec": 41.2,         // total_tokens / (elapsed since first token)
  "think_end_ms": 3900.1 | null   // when </think> was seen
}
```
Token counting: use `usage` from the final chunk if the server provides `completion_tokens` (then scale think/answer proportionally only for the final `done` metrics); otherwise count 1 per streamed content delta (vLLM/Ollama emit ~1 token per delta).

**Summary**:
```jsonc
{
  "baseline": {...Metrics} | null,
  "loco": {...Metrics} | null,
  "think_token_savings_pct": 43.1 | null,   // (1 - loco.think/baseline.think)*100
  "total_token_savings_pct": 40.2 | null,
  "time_savings_pct": 38.0 | null,
  "winner": "loco" | "baseline" | "tie" | null   // who finished first
}
```

## Training data format (JSONL, one object per line)

```json
{"prompt": "<question>", "chosen": "<think>\nconcise reasoning\n</think>\n\nfinal answer \\boxed{X}", "rejected": "<think>\nlong redundant reasoning\n</think>\n\nfinal answer \\boxed{X}"}
```
`prompt` is the raw user question (the trainer applies the chat template). `chosen`/`rejected` are the assistant completions.

## Ports
backend `8000`, frontend `8080` (nginx), vLLM baseline `8001`, vLLM loco `8002`, Ollama `11434`.
