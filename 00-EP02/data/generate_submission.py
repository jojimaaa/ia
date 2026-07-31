"""
Roda inferência do modelo fine-tuned (base + adapter LoRA) no test set e
gera output/submissions/submission.csv.

Uso:
    uv run python models/infer_submission.py
"""

import json
import re
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_DIR = ROOT / "output" / "models" / "qwen" / "lora_adapter"
TEST_JSONL = ROOT / "data" / "processed" / "test.jsonl"
SUBMISSION_PATH = ROOT / "output" / "submissions" / "submission.csv"

SYSTEM_PROMPT = (
    "You are an expert in digital logic circuits. Answer the question about "
    "the circuit shown in the image. If the answer is a number, respond with "
    "only the digit. If the answer is a boolean, respond with only 'True' or 'False'."
)

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_answer(text: str) -> str:
    """Extrai só o token relevante (True/False/dígito) da geração do modelo."""
    match = re.search(r"\b(True|False|[0-9])\b", text.strip(), re.IGNORECASE)
    if not match:
        return text.strip()
    token = match.group(1)
    if token.lower() == "true":
        return "True"
    if token.lower() == "false":
        return "False"
    return token


def main() -> None:
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    test_items = load_jsonl(TEST_JSONL)

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSION_PATH.open("w", encoding="utf-8") as out_f:
        out_f.write("index,answer\n")
        for ex in test_items:
            image = Image.open(ROOT / ex["image"]).convert("RGB")
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["question"]},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(
                model.device
            )

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=8)

            generated_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
            output_text = processor.batch_decode(
                generated_trimmed, skip_special_tokens=True
            )[0]

            answer = clean_answer(output_text)
            out_f.write(f"{ex['index']},{answer}\n")

            if ex["index"] % 100 == 0:
                print(f"[{ex['index']}] {ex['question']} -> {answer}")

    print(f"Submissão salva em {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
