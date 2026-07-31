# Directives for LLM Execution: VQA Multimodal Kaggle Project Pipeline

## 1. Executive Summary & Objective
Generate a single, robust, self-contained, and production-ready Jupyter Notebook (`.ipynb`) tailored for Google Colab Free (NVIDIA T4 GPU, ~15GB VRAM) or local CPU fallback.
The notebook implements an end-to-end Visual Question Answering (VQA) pipeline for digital logic circuits.

The primary goal is **methodological demonstration and comparative analysis** of multimodal reasoning enhancement techniques rather than pure Kaggle leaderboarding.

---

## 2. Technical Environment & Hardware Constraints

1. **Primary Target Environment**: Google Colab Free (T4 GPU, 15GB VRAM, 12GB System RAM, Python 3.11+).
2. **Local Fallback Option**: Ryzen 7 / 16GB RAM (CPU-only mode).
3. **Hardware Mitigation Rules (CRITICAL FOR VRAM CONSTRAINTS)**:
   - **Quantization**: Always load Qwen VLM using 4-bit NormalFloat quantization (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`).
   - **Image Resolution Bounds**: Set `min_pixels = 256 * 28 * 28` and `max_pixels = 512 * 28 * 28` inside `AutoProcessor` to prevent VRAM spikes during high-res image grid tokenization.
   - **Generation Limits**: Bound `max_new_tokens` (default: 128 for CoT/Reflexion, 16 for Baseline/Prompt Eng).
   - **Explicit Garbage Collection**: Call `clear_vram()` at the beginning and end of every batch/experiment section.

---

## 3. Directory & Persistence Architecture

The notebook must automatically mount Google Drive and enforce the following persistent structure under `/content/drive/MyDrive/vqa_project/`:

```
vqa_project/
├── data/
│   ├── raw/                  # Stores original .zip dataset
│   ├── processed/            # Stores JSONL splits (train.jsonl, val.jsonl, test.jsonl)
│   └── kaggle_dataset/       # Extracted images and raw jsonl files
├── output/
│   ├── checkpoints/          # Intermediate experiment predictions
│   │   ├── exp1_baseline.jsonl
│   │   ├── exp2_prompt.jsonl
│   │   ├── exp3_cot.jsonl
│   │   ├── exp4_self_consistency.jsonl
│   │   └── exp5_ttc_reflexion.jsonl
│   ├── submissions/          # Final CSV outputs formatted for Kaggle
│   │   ├── submission_baseline.csv
│   │   ├── submission_prompt.csv
│   │   ├── submission_cot.csv
│   │   ├── submission_self_consistency.csv
│   │   ├── submission_ttc.csv
│   │   └── submission_lora.csv
│   └── models/               # Saved QLoRA adapters and checkpoints
```

---

## 4. Detailed Data Processing Guidelines

Implement data preparation matching the logic of `prepare_dataset.py`:
1. **Unzipping**: Auto-detect `kaggle_dataset.zip` or existing `/kaggle_dataset/` directory.
2. **Record Structuring**: Transform raw JSON/JSONL into standard format:
   ```json
   {"index": 0, "image": "kaggle_dataset/train_dataset/images/circuit_0.png", "question": "...", "answer": "..."}
   ```
3. **Deterministic Validation Split**:
   - Total train dataset: 2,000 samples.
   - Shuffle `train_dataset` using seed `42`.
   - Split: 90% (`train.jsonl` = 1,800 items) and 10% (`val.jsonl` = 200 items).
   - Test set: 1,000 samples (`test.jsonl`).
4. **Validation First Policy**: Experiments 1 through 5 **must evaluate accuracy on `val.jsonl` first**. If accuracy calculation is confirmed valid, run inference on `test.jsonl` to output the submission CSV.

---

## 5. Core Helper Functions Specification

The notebook must define centralized, reusable functions to avoid code redundancy:

### 5.1 Memory Management
```python
import gc
import torch

def clear_vram():
    """Forces garbage collection and empties CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 5.2 Answer Parsing & Regex Extraction
```python
import re

def clean_answer(text: str) -> str:
    """
    Extracts canonical target tokens (True, False, or single/multi-digit integers)
    from raw generation strings, handling CoT reasoning dumps.
    """
    text_clean = text.strip()

    # Priority 1: Check for explicit "Final Answer: [value]" syntax
    cot_match = re.search(r"Final Answer:\s*([^\n\.]+)", text_clean, re.IGNORECASE)
    if cot_match:
        text_clean = cot_match.group(1).strip()

    # Priority 2: Extract boolean or digit tokens
    match = re.search(r"\b(True|False|[0-9]+)\b", text_clean, re.IGNORECASE)
    if not match:
        return text_clean

    token = match.group(1)
    if token.lower() == "true":
        return "True"
    if token.lower() == "false":
        return "False"
    return token
```

### 5.3 Resilient Checkpoint & Incremental Progress Engine
Implement an incremental JSONL writer/loader:
- Before running inference on index $i$, check if $i$ already exists in `checkpoint_file.jsonl`.
- If present, skip model execution and load previous response.
- Append new outputs line-by-line to disk instantly (`flush()`) to survive Colab session disconnects.

---

## 6. Notebook Section-by-Section Implementation Plan

Each experiment must be completely modular, isolated, and callable independently.

### Section 1: Global Setup & Hardware Auto-Detection
- Install required packages: `pip install -q transformers accelerate bitsandbytes peft datasets trl pillow pandas tqdm`.
- Mount Google Drive.
- Define global configuration variables:
  ```python
  DEBUG = False             # Set to True to test pipelines on 10 samples
  DEBUG_SAMPLES = 10
  MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # Fallback: "Qwen/Qwen2-VL-2B-Instruct" or "vikhyatk/moondream2"
  SEED = 42
  BATCH_SIZE = 1
  ```

### Section 2: Dataset Extraction & Preparation
- Execute dataset extraction and JSONL generation (`train.jsonl`, `val.jsonl`, `test.jsonl`).
- Display image-question-answer samples using `PIL` and `matplotlib`.

### Section 3: Model & Processor Initialization
- Load `Qwen2VLForConditionalGeneration` with 4-bit quantization and `AutoProcessor`.
- Wrap model load inside a single singleton function to avoid duplicate model loads in VRAM.

---

### Section 4: Experiment 1 - Baseline (Zero-Shot)
- **Concept**: Minimal contextual prompting.
- **System Prompt**: `"Answer the question about the image."`
- **User Prompt**: `{question}`
- **Evaluation**: Measure accuracy on `val.jsonl`. Generate `submission_baseline.csv`.

---

### Section 5: Experiment 2 - Prompt Engineering (Domain-Specific)
- **Concept**: Provide explicit role instruction and target format constraints.
- **System Prompt**:
  ```text
  You are an expert in digital logic circuits. Answer the question about the circuit shown in the image.
  If the answer is a number, respond with only the digit.
  If the answer is a boolean, respond with only 'True' or 'False'.
  ```
- **Evaluation**: Measure accuracy on `val.jsonl`. Generate `submission_prompt.csv`.

---

### Section 6: Experiment 3 - Chain-of-Thought (CoT) Prompting
- **Concept**: Force the model to reason through step-by-step logic circuit tracing.
- **System Prompt**: Same domain prompt as Exp 2.
- **User Prompt**: `"{question}\nAnalyze the circuit logic step-by-step. Conclude your reasoning with 'Final Answer: [your answer]'."`
- **Output Parsing**: Pass raw text through `clean_answer()`.
- **Evaluation**: Measure accuracy on `val.jsonl`. Generate `submission_cot.csv`.

---

### Section 7: Experiment 4 - Self-Consistency (Majority Voting)
- **Concept**: Sample $N=3$ independent generations per sample with non-zero temperature, then aggregate via majority vote.
- **Parameters**: `temperature = 0.7`, `top_p = 0.9`, $N=3$ samples per item.
- **Aggregation Logic**:
  ```python
  from collections import Counter
  votes = [clean_answer(gen) for gen in n_generations]
  final_answer = Counter(votes).most_common(1)[0][0]
  ```
- **Evaluation**: Measure accuracy on `val.jsonl`. Generate `submission_self_consistency.csv`.

---

### Section 8: Experiment 5 - Test-Time Compute (TTC / Reflexion)
- **Concept**: Two-stage generation where the model critiques its own initial answer.
- **Stage 1 (Generation)**: Run CoT inference to obtain `initial_answer` and `reasoning`.
- **Stage 2 (Critique & Correction)**:
  - **Prompt**:
    ```text
    Question: {question}
    Initial Analysis: {reasoning}
    Proposed Answer: {initial_answer}

    Double-check the circuit diagram carefully. Are there any false logic gate recognitions or miscounted inputs/outputs?
    Provide a corrected breakdown and finish with 'Final Answer: [your final answer]'."
    ```
- **Evaluation**: Measure accuracy on `val.jsonl`. Generate `submission_ttc.csv`.

---

### Section 9: Experiment 6 - Fine-Tuning with QLoRA
- **Technique**: Parameter-Efficient Fine-Tuning using PEFT + QLoRA on 1,800 training samples.
- **Data Collator**: Implement prompt-masking collator (`QwenVLCollator`) to compute loss **only on assistant responses**, ignoring prompt tokens (masking with `-100`).
- **LoRA Hyperparameters**:
  ```python
  LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      task_type="CAUSAL_LM"
  )
  ```
- **Training Hyperparameters**:
  - `per_device_train_batch_size = 1`
  - `gradient_accumulation_steps = 8`
  - `learning_rate = 2e-4`
  - `num_train_epochs = 2` or `3`
  - `bf16 = True` (or `fp16 = True` depending on Colab GPU allocation)
  - `gradient_checkpointing = True`
- **Inference**: Load base model + saved LoRA adapter (`PeftModel.from_pretrained`), infer on test set, generate `submission_lora.csv`.

---

### Section 10: Quantitative Comparison & Summary

Generate a consolidated Pandas summary table and bar charts comparing all techniques:

| Experiment | Validation Accuracy (%) | Avg Inference Time / Image (s) | Total Generated Tokens | Compute Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **1. Baseline** | ... | ... | ... | 1x (Base) |
| **2. Prompt Engineering** | ... | ... | ... | 1x |
| **3. Chain-of-Thought** | ... | ... | ... | ~2.5x |
| **4. Self-Consistency (N=3)**| ... | ... | ... | ~3x |
| **5. TTC (Reflexion)** | ... | ... | ... | ~2.2x |
| **6. QLoRA Fine-Tuning** | ... | ... | ... | High (Training cost) |

---

## 7. Execution Robustness Checklist for LLM Code Generator

When generating the notebook code, ensure:
1. Every cell is **idempotent** (can be re-executed safely without re-downloading datasets or corrupting existing output CSVs).
2. All progress loops use `tqdm` progress bars with explicit description labels (`desc="Running Exp 3 CoT"`).
3. System calls use Python's `subprocess` or native OS operations rather than unreliable bash commands.
4. Regex matching inside `clean_answer()` is resilient against lower/upper case variations (`True`, `TRUE`, `true`, `1`, `0`).
5. Memory cleanup code (`clear_vram()`) is explicitly placed after every major loop.
