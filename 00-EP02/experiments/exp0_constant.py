"""
Experimento 0 — baseline constante, sem modelo nenhum.

Serve de piso de referência: qualquer técnica com GPU tem que bater isso para
significar algo. Responde sempre a constante majoritária do treino para cada
tipo de pergunta (contagem -> "0", saída -> "True").

Também mede a dummy sugerida no enunciado (saída -> True, contagem -> 1) para
deixar explícito no relatório que trocar 1 por 0 vale ~7 pontos de graça.

Uso:
    uv run python experiments/exp0_constant.py

Gera:
    output/checkpoints/exp0_constant_val.jsonl
    output/submissions/submission_constant.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqa import (  # noqa: E402
    CHECKPOINT_DIR,
    COUNT,
    OUTPUT,
    SUBMISSION_DIR,
    TEST_JSONL,
    VAL_JSONL,
    format_score,
    load_jsonl,
    majority_answer,
    question_kind,
    run_predictions,
    score,
    write_submission,
)

# A dummy do enunciado, para comparação.
ENUNCIADO_DUMMY = {OUTPUT: "True", COUNT: "1"}


def make_predictor(table: dict[str, str]):
    def predict(item: dict) -> dict:
        return {"answer": table[question_kind(item["question"])]}

    return predict


def main() -> None:
    val = load_jsonl(VAL_JSONL)
    test = load_jsonl(TEST_JSONL)

    majority = {COUNT: majority_answer(COUNT), OUTPUT: majority_answer(OUTPUT)}
    print(f"constante majoritária do treino: {majority}")
    print(f"dummy do enunciado:              {ENUNCIADO_DUMMY}\n")

    # Referência: a dummy do enunciado, na validação.
    dummy_preds = [
        {"index": it["index"], "answer": ENUNCIADO_DUMMY[question_kind(it["question"])]}
        for it in val
    ]
    print(format_score(score(dummy_preds, val), "val — dummy do enunciado (True / 1)"))
    print()

    # O baseline que vamos submeter.
    val_preds = run_predictions(
        val,
        make_predictor(majority),
        CHECKPOINT_DIR / "exp0_constant_val.jsonl",
        desc="exp0 constante (val)",
    )
    print(format_score(score(val_preds, val), "val — constante majoritária (True / 0)"))
    print()

    test_preds = run_predictions(
        test,
        make_predictor(majority),
        CHECKPOINT_DIR / "exp0_constant_test.jsonl",
        desc="exp0 constante (test)",
    )
    write_submission(test_preds, SUBMISSION_DIR / "submission_constant.csv")


if __name__ == "__main__":
    main()
