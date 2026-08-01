"""
Pontua qualquer arquivo de predições contra um split com label.

Aceita o .jsonl de checkpoint gerado por vqa.run_predictions ou um .csv no
formato de submissão. Serve para reavaliar um experimento sem rodar inferência
de novo — inclusive para testar mudança no clean_answer sobre gerações antigas,
que é o caso comum quando se ajusta o parsing dos experimentos de CoT.

Uso:
    uv run python scripts/score.py output/checkpoints/exp0_constant_val.jsonl
    uv run python scripts/score.py output/checkpoints/exp3_cot_val.jsonl --reparse
    uv run python scripts/score.py qualquer.csv --split train
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqa import (  # noqa: E402
    TRAIN_JSONL,
    VAL_JSONL,
    clean_answer,
    format_score,
    load_jsonl,
    majority_answer,
    question_kind,
    score,
)

SPLITS = {"val": VAL_JSONL, "train": TRAIN_JSONL}


def load_predictions(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return [
                {"index": int(row["index"]), "answer": row["answer"].strip()}
                for row in csv.DictReader(f)
            ]
    return load_jsonl(path)


def reparse(predictions: list[dict], references: list[dict]) -> list[dict]:
    """Reaplica clean_answer sobre o campo "raw" das gerações já salvas."""
    question_by_index = {r["index"]: r["question"] for r in references}
    out = []
    for p in predictions:
        raw = p.get("raw")
        if raw is None:
            out.append(p)
            continue
        question = question_by_index.get(p["index"], p.get("question"))
        kind = question_kind(question)
        out.append(
            {
                **p,
                "answer": clean_answer(
                    raw, kind=kind, question=question, default=majority_answer(kind)
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help=".jsonl de checkpoint ou .csv")
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="reaplica clean_answer no campo 'raw' antes de pontuar",
    )
    args = parser.parse_args()

    references = load_jsonl(SPLITS[args.split])
    predictions = load_predictions(args.predictions)

    if args.reparse:
        before = score(predictions, references)["accuracy"]
        predictions = reparse(predictions, references)
        after = score(predictions, references)["accuracy"]
        print(f"reparse: {before:.2f}% -> {after:.2f}%\n")

    print(format_score(score(predictions, references), f"{args.predictions.name} @ {args.split}"))


if __name__ == "__main__":
    main()
