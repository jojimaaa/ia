"""
Experimento 5 — test-time compute / Reflexion.

Duas etapas: o modelo responde com CoT e depois critica a própria resposta
olhando o diagrama de novo, com atenção aos erros típicos desta tarefa (porta
lida errada, entrada contada errada).

A imagem vai nas DUAS etapas de propósito: a crítica é sobre percepção, não
sobre o texto do raciocínio. Criticar sem a figura só produziria coerência
interna, que não é o gargalo aqui.

Custo ~2,2x o baseline, mais barato que o self-consistency por gastar 2 gerações
em vez de 3.

Uso:
    uv run python experiments/exp5_reflexion.py --dry-run --limit 20
    uv run python experiments/exp5_reflexion.py --split train --sample 500
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _runner import build_argparser, run_experiment  # noqa: E402
from exp3_cot import SYSTEM_PROMPT, cot_user_prompt  # noqa: E402
from vqa import ROOT, clean_answer, majority_answer, question_kind  # noqa: E402

NAME = "exp5_reflexion"

MAX_NEW_TOKENS = 128


def critique_prompt(question: str, reasoning: str, initial_answer: str | None) -> str:
    return (
        f"Question: {question}\n"
        f"Initial Analysis: {reasoning}\n"
        f"Proposed Answer: {initial_answer}\n\n"
        "Double-check the circuit diagram carefully. Are there any false logic "
        "gate recognitions or miscounted inputs/outputs? Provide a corrected "
        "breakdown and finish with 'Final Answer: [your final answer]'."
    )


def make_predict(backend):
    def predict(item: dict) -> dict:
        kind = question_kind(item["question"])
        image = ROOT / item["image"]

        # Etapa 1 — geração com CoT.
        reasoning, tokens_1 = backend.ask(
            image,
            system=SYSTEM_PROMPT,
            user=cot_user_prompt(item["question"]),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        initial = clean_answer(reasoning, kind=kind, question=item["question"])

        # Etapa 2 — crítica e correção, com a imagem de novo.
        revised, tokens_2 = backend.ask(
            image,
            system=SYSTEM_PROMPT,
            user=critique_prompt(item["question"], reasoning, initial),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        final = clean_answer(revised, kind=kind, question=item["question"])

        # Se a revisão não parseia, a etapa 1 ainda vale mais que a constante.
        answer = final if final is not None else initial
        if answer is None:
            answer = majority_answer(kind)

        return {
            "raw": revised,
            "raw_stage1": reasoning,
            "initial_answer": initial,
            "answer": answer,
            "changed": initial != answer,  # a crítica mudou de opinião?
            "new_tokens": tokens_1 + tokens_2,
        }

    return predict


if __name__ == "__main__":
    run_experiment(NAME, make_predict, build_argparser(__doc__).parse_args())
