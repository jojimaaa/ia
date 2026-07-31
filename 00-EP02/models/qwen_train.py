"""
Fine-tuning do Qwen2-VL-2B-Instruct com QLoRA (4-bit) para responder perguntas
sobre circuitos lógicos (contagem de portas / saída do circuito) a partir de
uma imagem.

Pensado para rodar em uma única GPU de ~16GB (ex.: T4 do Colab gratuito).

Uso (depois de rodar data/prepare_dataset.py):
    uv run python models/train_qwen.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = ROOT / "output" / "models" / "qwen"

SYSTEM_PROMPT = (
    "You are an expert in digital logic circuits. Answer the question about "
    "the circuit shown in the image. If the answer is a number, respond with "
    "only the digit. If the answer is a boolean, respond with only 'True' or 'False'."
)

# Limita a resolução/tokens de imagem para caber na VRAM de uma T4 (16GB).
# Aumente se tiver mais VRAM disponível (ex.: A100).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(jsonl_path: Path) -> Dataset:
    return Dataset.from_list(load_jsonl(jsonl_path))


def _find_subsequence(seq: list[int], sub: list[int]) -> int | None:
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i : i + m] == sub:
            return i
    return None


@dataclass
class QwenVLCollator:
    processor: AutoProcessor

    def __call__(self, batch: list[dict]):
        texts = []
        images = []
        for ex in batch:
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
                {"role": "assistant", "content": ex["answer"]},
            ]
            text = self.processor.apply_chat_template(conversation, tokenize=False)
            texts.append(text)
            images.append(image)

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Calcula a loss só na resposta do assistant, mascarando o resto do prompt.
        assistant_marker = self.processor.tokenizer(
            "assistant\n", add_special_tokens=False
        )["input_ids"]
        for i, ids in enumerate(inputs["input_ids"]):
            start = _find_subsequence(ids.tolist(), assistant_marker)
            if start is not None:
                labels[i, : start + len(assistant_marker)] = -100

        inputs["labels"] = labels
        return inputs


def main() -> None:
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = build_dataset(ROOT / "data" / "processed" / "train.jsonl")
    val_ds = build_dataset(ROOT / "data" / "processed" / "val.jsonl")

    collator = QwenVLCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Adapter LoRA salvo em {adapter_dir}")


if __name__ == "__main__":
    main()
