# Fast-LoCo Trainer

Lightweight single-GPU post-training pipeline for **LoCo ("Think Less, Locally")** on `DeepSeek-R1-Distill-Qwen-7B`.

LoCo applies **Direct Preference Optimization (DPO)** to eliminate 40%–45% of redundant `<think>` reasoning tokens while maintaining or increasing accuracy. Both the chosen and rejected completions reach the *identical* correct answer in `\boxed{...}`, but the chosen completion does so with concise, direct reasoning instead of circular verification loops.

---

## ⚡ Quickstart

### 1. Requirements

```bash
cd training
pip install -r requirements.txt
```

> **Note on CUDA:** If using CUDA on Linux/Windows, install PyTorch with CUDA 12.1+ support:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### 2. Dry-Run (CPU Validation, Zero Weights Download)

Test your dataset formatting and inspect prompt templating + token statistics without loading GPU models:

```bash
python train_dpo_lite.py --data data/sample_pairs.jsonl --dry-run
```

Output highlights:
- Validates JSONL structure (`prompt`, `chosen`, `rejected`).
- Formats prompts with R1 chat template and strips duplicate `<think>\n` prefill.
- Reports sequence length distributions and token stats.

---

## 🚀 Training (QLoRA 4-bit)

`train_dpo_lite.py` is optimized for consumer GPUs (16 GB to 24 GB VRAM) using:
- **BitsAndBytes 4-bit NF4** base model quantization + double quantization.
- **LoRA** (`r=16`, `lora_alpha=32`, `target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]`).
- **Implicit reference model** in `trl.DPOTrainer` (`ref_model=None`), saving ~7 GB VRAM.
- **Paged 8-bit AdamW** optimizer + gradient checkpointing.

### 24 GB GPU (RTX 3090 / 4090):
```bash
python train_dpo_lite.py \
    --data data/sample_pairs.jsonl \
    --output-dir out/loco-lora \
    --max-length 4096 \
    --max-prompt-length 1024 \
    --learning-rate 5e-6 \
    --lora-r 16
```

### 16 GB GPU (RTX 4080 / 4070 Ti Super):
```bash
python train_dpo_lite.py \
    --data data/sample_pairs.jsonl \
    --output-dir out/loco-lora \
    --max-length 2048 \
    --max-prompt-length 512 \
    --learning-rate 5e-6 \
    --lora-r 16
```

---

## 🔄 Merging LoRA Weights

Once training completes, merge the LoRA adapter into the base model weights to produce a standalone 16-bit Hugging Face checkpoint:

```bash
python merge_lora.py \
    --adapter out/loco-lora \
    --output-dir out/loco-merged \
    --device-map cpu
```

---

## 🦙 Export to GGUF / Ollama

To run your trained LoCo model locally via Ollama:

1. **Convert to GGUF using llama.cpp:**
   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   pip install -r llama.cpp/requirements.txt
   python llama.cpp/convert_hf_to_gguf.py out/loco-merged --outfile out/loco-7b-f16.gguf --outtype f16
   ```

2. **Quantize to Q4_K_M:**
   ```bash
   ./llama.cpp/build/bin/llama-quantize out/loco-7b-f16.gguf out/loco-7b-q4_k_m.gguf Q4_K_M
   ```

3. **Register and Run with Ollama:**
   ```bash
   # From project root
   ollama create loco -f ollama/Modelfile
   ollama run loco
   ```

---

## 🛠️ Building Custom Datasets (`build_pairs.py`)

If you have a set of math/logic problems with ground truth answers in `problems.jsonl`:
```json
{"prompt": "Which is greater: 9.11 or 9.9?", "answer": "9.9"}
```

Run the automatic pair miner against a running instance of `DeepSeek-R1-Distill-Qwen-7B`:
```bash
python build_pairs.py \
    --input-file data/problems.jsonl \
    --output-file data/loco_pairs.jsonl \
    --base-url http://localhost:8001/v1 \
    --samples-per-prompt 5 \
    --min-ratio 1.5
```
It extracts verified-correct generations, assigns the shortest trace as `chosen`, the longest as `rejected`, and outputs validated DPO training pairs.
