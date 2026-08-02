"""
Experimento 3 — Chain-of-Thought.

Força o rastreamento porta a porta antes de responder, e exige a conclusão num
marcador fixo ("Final Answer:") para o parsing não depender de adivinhação.

É aqui que o parsing passa a importar: o raciocínio está cheio de números e
booleanos distratores (nomes de porta, valores de nó intermediário), então
`clean_answer` prioriza o trecho depois do marcador. Sem isso a acurácia mede o
parser, não o modelo.

Custo ~2,5x o baseline em tokens gerados.

Uso:
    uv run python experiments/exp3_cot.py --dry-run --limit 20
    uv run python experiments/exp3_cot.py --split train --sample 500
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _runner import build_argparser, run_experiment  # noqa: E402
from vqa import ROOT, clean_answer, majority_answer, question_kind  # noqa: E402

NAME = "exp3_cot"

SYSTEM_PROMPT = (
    "You are an expert in digital logic circuits. Answer the question about the "
    "circuit shown in the image.\n"
    "If the answer is a number, respond with only the digit.\n"
    "If the answer is a boolean, respond with only 'True' or 'False'."
)

COT_SUFFIX = (
    "\nAnalyze the circuit logic step-by-step. "
    "Conclude your reasoning with 'Final Answer: [your answer]'."
)
MAX_NEW_TOKENS = 128


def cot_user_prompt(question: str) -> str:
    return f"{question}{COT_SUFFIX}"


def make_predict(backend):
    def predict(item: dict) -> dict:
        raw, n_tokens = backend.ask(
            ROOT / item["image"],
            system=SYSTEM_PROMPT,
            user=cot_user_prompt(item["question"]),
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
