"""
Experimento 6 — fine-tuning do Qwen2-VL-2B-Instruct com QLoRA (4-bit).

Treina um adapter LoRA sobre os 1.800 exemplos de treino para responder
perguntas sobre circuitos lógicos. A loss é calculada só na resposta do
assistant; o prompt é mascarado com -100, senão o modelo aprenderia a repetir a
pergunta.

Exige CUDA: `bitsandbytes` não quantiza em 4-bit na CPU. Em CPU seria LoRA fp32
puro, ~8GB só de pesos, e o treino levaria dias — use uma T4 do Colab.

Retomada: o Trainer salva a cada `--save-steps` e, por padrão, esta rotina
retoma automaticamente do último checkpoint encontrado em --output-dir. Uma
sessão de Colab que cai no meio de um treino de horas não custa nada além do
tempo já gasto.

Uso:
    uv run python models/qwen_train.py --limit 20      # smoke test rápido
    uv run python models/qwen_train.py                 # treino completo
    uv run python models/qwen_train.py --no-resume     # ignora checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import MAX_PIXELS, MIN_PIXELS, MODEL_ID, resolve_model_class  # noqa: E402
from vqa import ROOT, TRAIN_JSONL, VAL_JSONL, load_jsonl  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "output" / "models" / "qwen"

# Mesmo prompt do experimento 2, para que a comparação entre fine-tuning e
# prompt engineering isole o efeito do treino e não o do texto do sistema.
SYSTEM_PROMPT = (
    "You are an expert in digital logic circuits. Answer the question about the "
    "circuit shown in the image.\n"
    "If the answer is a number, respond with only the digit.\n"
    "If the answer is a boolean, respond with only 'True' or 'False'."
)


def _find_subsequence(seq: list[int], sub: list[int]) -> int | None:
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i : i + m] == sub:
            return i
    return None


@dataclass
class QwenVLCollator:
    """Monta o batch e mascara tudo que não é a resposta do assistant."""

    processor: object

    def __call__(self, batch: list[dict]):
        from PIL import Image

        texts, images = [], []
        for ex in batch:
            images.append(Image.open(ROOT / ex["image"]).convert("RGB"))
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
            texts.append(self.processor.apply_chat_template(conversation, tokenize=False))

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

        # Sem esta máscara a loss inclui o prompt e o modelo aprende a repetir a
        # pergunta em vez de responder.
        marker = self.processor.tokenizer("assistant\n", add_special_tokens=False)[
            "input_ids"
        ]
        for i, ids in enumerate(inputs["input_ids"]):
            start = _find_subsequence(ids.tolist(), marker)
            if start is not None:
                labels[i, : start + len(marker)] = -100

        inputs["labels"] = labels
        return inputs


def last_checkpoint(output_dir: Path) -> Path | None:
    """Checkpoint mais recente salvo pelo Trainer, se houver."""
    if not output_dir.is_dir():
        return None
    checkpoints = [
        p for p in output_dir.glob("checkpoint-*") if (p / "trainer_state.json").exists()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name.split("-")[1]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--limit", type=int, default=None,
                   help="usa só os N primeiros exemplos (smoke test)")
    p.add_argument("--no-resume", action="store_true",
                   help="começa do zero, ignorando checkpoints em --output-dir")
    p.add_argument("--no-4bit", action="store_true",
                   help="sem quantização; exige bem mais VRAM")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA indisponível. QLoRA 4-bit exige GPU: bitsandbytes não quantiza "
            "em CPU, e LoRA fp32 num 2B levaria dias. Use uma T4 do Colab."
        )

    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )

    model_cls = resolve_model_class()
    if args.no_4bit:
        model = model_cls.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:
        from transformers import BitsAndBytesConfig

        model = model_cls.from_pretrained(
            args.model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_records = load_jsonl(TRAIN_JSONL)
    val_records = load_jsonl(VAL_JSONL)
    if args.limit:
        train_records = train_records[: args.limit]
        val_records = val_records[: max(2, args.limit // 10)]
    print(f"treino: {len(train_records)} exemplos, validação: {len(val_records)}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(val_records),
        data_collator=QwenVLCollator(processor=processor),
    )

    resume = None if args.no_resume else last_checkpoint(args.output_dir)
    if resume:
        print(f"retomando de {resume}")
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    adapter_dir = args.output_dir / "lora_adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"adapter LoRA salvo em {adapter_dir}")


if __name__ == "__main__":
    main()
