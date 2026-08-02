"""
Experimento 2 — prompt engineering de domínio.

Mesmo custo de compute do baseline (1x): muda só o texto do sistema, dando papel
de especialista e restringindo o formato da saída. Isola quanto do erro do
baseline era formatação em vez de raciocínio.

Uso:
    uv run python experiments/exp2_prompt.py --dry-run --limit 20
    uv run python experiments/exp2_prompt.py --split train --sample 500
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _runner import build_argparser, run_experiment  # noqa: E402
from vqa import ROOT, clean_answer, majority_answer, question_kind  # noqa: E402

NAME = "exp2_prompt"

SYSTEM_PROMPT = (
    "You are an expert in digital logic circuits. Answer the question about the "
    "circuit shown in the image.\n"
    "If the answer is a number, respond with only the digit.\n"
    "If the answer is a boolean, respond with only 'True' or 'False'."
)
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
