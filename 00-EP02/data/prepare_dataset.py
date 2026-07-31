"""
Extrai o dataset da Kaggle (zip contendo kaggle_dataset/) e gera jsonl
prontos para o fine-tuning do Qwen2-VL: data/processed/{train,val,test}.jsonl

Cada registro fica no formato:
    {"index": 0, "image": "kaggle_dataset/train_dataset/images/circuit_0.png",
     "question": "...", "answer": "..."}   # answer ausente no test

Uso:
    # se você ainda tem o .zip:
    uv run python data/prepare_dataset.py --zip-path /caminho/pcs-3838-pcs-5022-2026-parte-2.zip

    # se o kaggle_dataset/ já está extraído na raiz do repo:
    uv run python data/prepare_dataset.py --no-extract
"""

import argparse
import json
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = ROOT / "kaggle_dataset"
PROCESSED_DIR = ROOT / "data" / "processed"


def extract_zip(zip_path: Path) -> None:
    print(f"Extraindo {zip_path} -> {ROOT}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ROOT)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_records(items: list[dict], images_dir: Path, has_answer: bool) -> list[dict]:
    records = []
    for item in items:
        idx = item["index"]
        image_path = images_dir / f"circuit_{idx}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        rec = {
            "index": idx,
            "image": str(image_path.relative_to(ROOT)),
            "question": item["question"],
        }
        if has_answer:
            rec["answer"] = str(item["answer"])
        records.append(rec)
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Salvo {len(records)} registros em {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=str, default=None,
                         help="Caminho para o .zip do dataset Kaggle")
    parser.add_argument("--no-extract", action="store_true",
                         help="Pula a extração, assume que kaggle_dataset/ já existe na raiz")
    parser.add_argument("--val-frac", type=float, default=0.1,
                         help="Fração do treino separada para validação")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.no_extract:
        if args.zip_path is None:
            raise ValueError("Informe --zip-path ou use --no-extract se já estiver extraído")
        extract_zip(Path(args.zip_path))

    train_json_path = KAGGLE_DIR / "train_dataset" / "text" / "questions_with_answers.jsonl"
    test_json_path = KAGGLE_DIR / "test_dataset" / "text" / "questions.jsonl"
    train_images_dir = KAGGLE_DIR / "train_dataset" / "images"
    test_images_dir = KAGGLE_DIR / "test_dataset" / "images"

    train_items = load_jsonl(train_json_path)
    test_items = load_jsonl(test_json_path)

    train_records = build_records(train_items, train_images_dir, has_answer=True)
    test_records = build_records(test_items, test_images_dir, has_answer=False)

    random.seed(args.seed)
    random.shuffle(train_records)
    n_val = int(len(train_records) * args.val_frac)
    val_records = train_records[:n_val]
    train_records = train_records[n_val:]

    save_jsonl(train_records, PROCESSED_DIR / "train.jsonl")
    save_jsonl(val_records, PROCESSED_DIR / "val.jsonl")
    save_jsonl(test_records, PROCESSED_DIR / "test.jsonl")


if __name__ == "__main__":
    main()
