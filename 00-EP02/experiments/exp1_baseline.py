"""
Experimento 1 — baseline zero-shot.

Prompt mínimo, sem papel nem restrição de formato. Mede o que o modelo faz
cru, para os experimentos seguintes terem contra o que ser comparados.

Uso:
    uv run python experiments/exp1_baseline.py --dry-run --limit 20
    uv run python experiments/exp1_baseline.py                     # val
    uv run python experiments/exp1_baseline.py --split train --sample 500
    uv run python experiments/exp1_baseline.py --test              # + submissão
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _runner import build_argparser, run_experiment  # noqa: E402
from vqa import ROOT, clean_answer, majority_answer, question_kind  # noqa: E402

NAME = "exp1_baseline"
SYSTEM_PROMPT = "Answer the question about the image."
MAX_NEW_TOKENS = 16


def make_predict(backend):
    def predict(item: dict) -> dict:
        raw, n_tokens = backend.ask(
            ROOT / item["image"],
            system=SYSTEM_PROMPT,
            user=item["question"],
            max_new_tokens=MAX_NEW_TOKENS,
        )
        kind = question_kind(item["question"])
        return {
            "raw": raw,
            "answer": clean_answer(
                raw, kind=kind, question=item["question"], default=majority_answer(kind)
            ),
            "new_tokens": n_tokens,
        }

    return predict


if __name__ == "__main__":
    run_experiment(NAME, make_predict, build_argparser(__doc__).parse_args())
