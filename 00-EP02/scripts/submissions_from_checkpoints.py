"""
Gera os CSVs de submissão a partir dos checkpoints de teste já existentes.

Os arquivos `*_test.jsonl` em output/checkpoints/ já contêm as respostas dos
1.000 itens do teste — falta só virar CSV. Este script faz isso sem carregar o
modelo, então roda em qualquer máquina, sem GPU e sem cota.

Serve para o caso comum de ter rodado a inferência numa sessão que morreu antes
de escrever a submissão, ou de querer o CSV numa máquina diferente da que rodou.

Recusa escrever submissão incompleta: o Kaggle corrige por correspondência exata
sobre os 1.000 índices, e um CSV faltando linhas é rejeitado ou pontua como erro.

Uso:
    uv run python scripts/submissions_from_checkpoints.py
    uv run python scripts/submissions_from_checkpoints.py exp4_self_consistency_test.jsonl
    uv run python scripts/submissions_from_checkpoints.py --checkpoint-dir /caminho/do/drive
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqa import (  # noqa: E402
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    TEST_JSONL,
    clamp_to_support,
    load_jsonl,
    question_kind,
    write_submission,
)


def submission_name(checkpoint: Path) -> str:
    """exp3_cot_test.jsonl -> submission_exp3_cot.csv"""
    stem = checkpoint.stem
    if stem.endswith("_test"):
        stem = stem[: -len("_test")]
    return f"submission_{stem}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*",
                        help="arquivos de checkpoint; vazio = todos os *_test.jsonl")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--out-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--clamp", action="store_true",
                        help="troca resposta fora do suporte do treino (ex.: contagem 10, "
                             "cujo máximo observado é 7) pela constante majoritária")
    args = parser.parse_args()

    expected = {item["index"] for item in load_jsonl(TEST_JSONL)}

    if args.names:
        checkpoints = [args.checkpoint_dir / n for n in args.names]
    else:
        checkpoints = sorted(args.checkpoint_dir.glob("*_test.jsonl"))

    if not checkpoints:
        raise SystemExit(f"nenhum *_test.jsonl em {args.checkpoint_dir}")

    written, skipped = 0, 0
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            print(f"[falta]      {checkpoint.name}")
            skipped += 1
            continue

        records = load_jsonl(checkpoint)
        by_index = {r["index"]: r for r in records}
        missing = expected - by_index.keys()
        extra = by_index.keys() - expected

        if missing:
            print(
                f"[incompleto] {checkpoint.name}: {len(by_index)}/{len(expected)} itens, "
                f"faltam {len(missing)} (ex.: {sorted(missing)[:5]}) — não gerei o CSV"
            )
            skipped += 1
            continue
        if extra:
            print(f"[aviso]      {checkpoint.name}: {len(extra)} índices fora do teste, ignorados")

        predictions = [by_index[i] for i in sorted(expected)]
        if args.clamp:
            kinds = {r["index"]: question_kind(r["question"]) for r in load_jsonl(TEST_JSONL)}
            changed = 0
            for p in predictions:
                fixed = clamp_to_support(p.get("answer"), kinds[p["index"]])
                if fixed != p.get("answer"):
                    changed += 1
                p["answer"] = fixed
            if changed:
                print(f"[clamp]      {checkpoint.name}: {changed} respostas fora do suporte trocadas")
        blank = [p["index"] for p in predictions if p.get("answer") in (None, "")]
        if blank:
            print(
                f"[vazio]      {checkpoint.name}: {len(blank)} respostas vazias "
                f"(ex.: {blank[:5]}) — não gerei o CSV"
            )
            skipped += 1
            continue

        backends = {r.get("backend") for r in records}
        if len(backends) > 1:
            print(f"[aviso]      {checkpoint.name}: backends misturados {sorted(map(str, backends))}")

        write_submission(predictions, args.out_dir / submission_name(checkpoint))
        written += 1

    print(f"\n{written} submissões geradas, {skipped} puladas")


if __name__ == "__main__":
    main()
