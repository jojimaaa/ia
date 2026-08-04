"""
Roda inferência do modelo fine-tuned (base + adapter LoRA) no test set e gera
output/submissions/submission_lora.csv.

Antes de submeter, mede a acurácia na validação — se o número não fizer sentido,
não gasta 1.000 inferências no teste.

Uso:
    uv run python data/generate_submission.py            # val + test
    uv run python data/generate_submission.py --val-only # só valida
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import resolve_model_class  # noqa: E402
from vqa import (  # noqa: E402
    CHECKPOINT_DIR,
    ROOT,
    SUBMISSION_DIR,
    TEST_JSONL,
    VAL_JSONL,
    clean_answer,
    format_score,
    load_jsonl,
    majority_answer,
    question_kind,
    run_predictions,
    score,
    write_submission,
)

BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_DIR = ROOT / "output" / "models" / "qwen" / "lora_adapter"

SYSTEM_PROMPT = (
    "You are an expert in digital logic circuits. Answer the question about "
    "the circuit shown in the image. If the answer is a number, respond with "
    "only the digit. If the answer is a boolean, respond with only 'True' or 'False'."
)

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28
MAX_NEW_TOKENS = 8


def load_model():
    """Base em 4-bit NF4 + adapter LoRA, na mesma precisão usada no treino.

    A T4 do Colab é Turing e não tem bf16 nativo; carregar em bf16 ali funciona
    mas é emulado e lento. Detecta e usa fp16 quando bf16 não é suportado.
    """
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    backend_tag = f"cuda-4bit-{'bf16' if use_bf16 else 'fp16'}-lora"

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    base_model = resolve_model_class().from_pretrained(
        BASE_MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    print(f"adapter carregado de {ADAPTER_DIR} ({backend_tag})")
    return model, processor, backend_tag


def make_predictor(model, processor):
    def predict(item: dict) -> dict:
        image = Image.open(ROOT / item["image"]).convert("RGB")
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["question"]},
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
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        kind = question_kind(item["question"])
        return {
            "raw": raw,
            "answer": clean_answer(
                raw,
                kind=kind,
                question=item["question"],
                default=majority_answer(kind),
            ),
            "new_tokens": int(trimmed.shape[1]),
        }

    return predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-only", action="store_true")
    args = parser.parse_args()

    model, processor, backend_tag = load_model()
    predict = make_predictor(model, processor)
    meta = {"backend": backend_tag}

    val = load_jsonl(VAL_JSONL)
    val_preds = run_predictions(
        val,
        predict,
        CHECKPOINT_DIR / "exp6_lora_val.jsonl",
        desc="exp6 LoRA (val)",
        meta=meta,
    )
    print(format_score(score(val_preds, val), "val — QLoRA fine-tuned"))

    if args.val_only:
        return

    test = load_jsonl(TEST_JSONL)
    test_preds = run_predictions(
        test,
        predict,
        CHECKPOINT_DIR / "exp6_lora_test.jsonl",
        desc="exp6 LoRA (test)",
        meta=meta,
    )
    write_submission(test_preds, SUBMISSION_DIR / "submission_lora.csv")


if __name__ == "__main__":
    main()
