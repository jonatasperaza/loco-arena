# loco-arena — developer shortcuts.
# Windows: use WSL / Git Bash with GNU make, or the PowerShell equivalents in README.md.

PYTHON       ?= python
COMPOSE      ?= docker compose
TRAIN_DATA   ?= data/sample_pairs.jsonl
TRAIN_OUT    ?= out/loco-lora
MERGED_OUT   ?= out/loco-merged
TRAIN_ARGS   ?=
MERGE_ARGS   ?=
OLLAMA_NAME  ?= loco

.DEFAULT_GOAL := help
.PHONY: help up up-gpu up-ollama down logs test dev-backend dev-frontend \
        train-dry train merge ollama-create

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start the arena in demo mode (no GPU) -> http://localhost:8080
	$(COMPOSE) up --build -d
	@echo "Arena: http://localhost:8080   API: http://localhost:8000/api/health"

up-gpu: ## Start arena + two vLLM servers (requires ARENA_MODE=live etc. in .env)
	$(COMPOSE) --profile gpu up --build -d
	@echo "vLLM model download/load can take several minutes: make logs"

up-ollama: ## Start arena + Ollama container (requires ARENA_MODE=live etc. in .env)
	$(COMPOSE) --profile ollama up --build -d

down: ## Stop every service (all profiles)
	$(COMPOSE) --profile gpu --profile ollama down

logs: ## Follow logs of all running services
	$(COMPOSE) --profile gpu --profile ollama logs -f --tail=200

test: ## Run backend tests
	cd backend && $(PYTHON) -m pytest -q

dev-backend: ## Run the FastAPI backend with auto-reload on :8000
	cd backend && $(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend: ## Serve the static frontend on :8080 (talks to the dev backend)
	@echo "Open http://localhost:8080/?api=http://localhost:8000"
	cd frontend && $(PYTHON) -m http.server 8080

train-dry: ## Validate the dataset + tokenizer on CPU (no model weights)
	cd training && $(PYTHON) train_dpo_lite.py --data $(TRAIN_DATA) --dry-run $(TRAIN_ARGS)

train: ## QLoRA DPO training (GPU). Override TRAIN_DATA / TRAIN_ARGS
	cd training && $(PYTHON) train_dpo_lite.py --data $(TRAIN_DATA) --output-dir $(TRAIN_OUT) $(TRAIN_ARGS)

merge: ## Merge the LoRA adapter into full HF weights
	cd training && $(PYTHON) merge_lora.py --adapter $(TRAIN_OUT) --output-dir $(MERGED_OUT) $(MERGE_ARGS)

ollama-create: ## Register LoCo (needs ollama/loco-7b-q4_k_m.gguf) and the baseline in Ollama
	cd ollama && ollama create $(OLLAMA_NAME) -f Modelfile
	ollama pull deepseek-r1:7b
	cd ollama && ollama create r1-baseline -f Modelfile.baseline
