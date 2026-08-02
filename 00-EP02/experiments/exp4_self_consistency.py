"""
Experimento 4 — self-consistency (voto de maioria).

N=3 gerações independentes de CoT com amostragem (temperature 0.7, top_p 0.9),
agregadas por voto de maioria. A ideia é que erros de rastreamento sejam
inconsistentes entre amostras enquanto a resposta certa se repete.

Custo ~3x o CoT (que já é ~2,5x o baseline), o mais caro dos experimentos de
inferência.

Detalhes que mudam o número:
  - votos que não parseiam são DESCARTADOS da eleição em vez de virarem a
    constante majoritária; se virassem, a constante ganharia a eleição só por
    aparecer em todo voto inválido.
  - se nenhum voto parseia, aí sim cai na constante.
  - empate resolve pelo voto que apareceu primeiro (Counter.most_common preserva
    ordem de inserção), o que é determinístico dada a seed.

Uso:
    uv run python experiments/exp4_self_consistency.py --dry-run --limit 20
    uv run python experiments/exp4_self_consistency.py --split train --sample 500
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _runner import build_argparser, run_experiment  # noqa: E402
from exp3_cot import SYSTEM_PROMPT, cot_user_prompt  # noqa: E402
from vqa import ROOT, clean_answer, majority_answer, question_kind  # noqa: E402

NAME = "exp4_self_consistency"

N_SAMPLES = 3
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 128


def make_predict(backend):
    def predict(item: dict) -> dict:
        kind = question_kind(item["question"])
        user = cot_user_prompt(item["question"])

        raws, votes, tokens = [], [], 0
        for _ in range(N_SAMPLES):
            raw, n_tokens = backend.ask(
                ROOT / item["image"],
                system=SYSTEM_PROMPT,
                user=user,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            raws.append(raw)
            tokens += n_tokens
            parsed = clean_answer(raw, kind=kind, question=item["question"])
            if parsed is not None:
                votes.append(parsed)

        if votes:
            answer, agreement = Counter(votes).most_common(1)[0]
        else:
            answer, agreement = majority_answer(kind), 0

        return {
            "raw": raws[0],
            "raws": raws,
            "votes": votes,
            "agreement": agreement,  # quantos dos N concordaram com a resposta
            "answer": answer,
            "new_tokens": tokens,
        }

    return predict


if __name__ == "__main__":
    run_experiment(NAME, make_predict, build_argparser(__doc__).parse_args())
