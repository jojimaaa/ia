"""
Arnês comum dos experimentos 1 a 5.

Cada experimento vira um arquivo curto que só declara os prompts e como agregar
a geração; toda a repetição (CLI, escolha do split de avaliação, checkpoint,
score, submissão) mora aqui.

Política de validação primeiro: por padrão só avalia, e nada de teste. A
submissão só é gerada com --test explícito, depois de olhar a acurácia.

Sobre o split: o val tem 200 amostras, o que dá IC95 de ~±6,9 pontos — largo
demais para separar as 5 técnicas, que provavelmente caem dentro desse
intervalo umas das outras. Os experimentos 1 a 5 não treinam, então avaliar em
amostras do treino não vaza label nenhum e reduz o ruído:

    --split train --sample 500     IC95 ~±4,4 pontos
    --split train                  IC95 ~±2,2 pontos (1.800 itens, caro)

O experimento 6 (QLoRA) é o único obrigado ao val, porque para ele o treino é
dado visto.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import clear_vram, load_backend, set_seed  # noqa: E402
from vqa import (  # noqa: E402
    CHECKPOINT_DIR,
    COUNT,
    OUTPUT,
    SUBMISSION_DIR,
    TEST_JSONL,
    TRAIN_JSONL,
    VAL_JSONL,
    format_score,
    load_jsonl,
    question_kind,
    run_predictions,
    score,
    write_submission,
)

SPLITS = {"val": VAL_JSONL, "train": TRAIN_JSONL}


def build_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--split", choices=SPLITS, default="val",
                   help="split de avaliação (default: val)")
    p.add_argument("--sample", type=int, default=None,
                   help="amostra estratificada de N itens do split (reduz custo)")
    p.add_argument("--limit", type=int, default=None,
                   help="usa só os N primeiros itens (debug rápido)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test", action="store_true",
                   help="além de avaliar, roda o teste e escreve a submissão")
    p.add_argument("--dry-run", action="store_true",
                   help="backend falso, sem GPU: valida a mecânica do experimento")
    p.add_argument("--model-id", default=None)
    p.add_argument("--no-4bit", action="store_true",
                   help="carrega sem quantização (mais VRAM)")
    p.add_argument("--allow-mixed-backend", action="store_true",
                   help="retoma checkpoint gerado em outro backend (invalida a comparação)")
    p.add_argument("--mirror-dir", type=Path, default=None,
                   help="espelha o checkpoint neste diretório a cada --mirror-every itens "
                        "(use o Drive no Colab: escrever direto nele não é durável)")
    p.add_argument("--mirror-every", type=int, default=25)
    return p


def stratified_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Amostra n itens mantendo a proporção entre contagem e saída."""
    by_kind: dict[str, list[dict]] = {COUNT: [], OUTPUT: []}
    for it in items:
        by_kind[question_kind(it["question"])].append(it)

    rng = random.Random(seed)
    picked: list[dict] = []
    for kind, group in by_kind.items():
        share = round(n * len(group) / len(items))
        picked.extend(rng.sample(group, min(share, len(group))))
    rng.shuffle(picked)
    return picked


def resolve_eval_items(args) -> tuple[list[dict], str]:
    """Itens de avaliação e um sufixo que identifica o split no nome do checkpoint."""
    items = load_jsonl(SPLITS[args.split])
    tag = args.split
    if args.sample:
        items = stratified_sample(items, args.sample, args.seed)
        tag = f"{args.split}{len(items)}"
    if args.limit:
        items = items[: args.limit]
        tag = f"{tag}-lim{args.limit}"
    if args.dry_run:
        tag = f"{tag}-dry"
    return items, tag


def run_experiment(
    name: str,
    make_predict: Callable[[object], Callable[[dict], dict]],
    args,
) -> dict:
    """Carrega o backend, avalia e (com --test) gera a submissão.

    `make_predict(backend)` devolve a função que recebe um item do dataset e
    responde o dict do experimento (no mínimo {"answer": ...}).
    """
    kwargs = {"dry_run": args.dry_run, "four_bit": not args.no_4bit}
    if args.model_id:
        kwargs["model_id"] = args.model_id

    clear_vram()
    set_seed(args.seed)
    backend = load_backend(**kwargs)
    predict = make_predict(backend)

    meta = {"backend": backend.tag}

    def mirror_for(filename: str):
        return args.mirror_dir / filename if args.mirror_dir else None

    eval_items, tag = resolve_eval_items(args)
    eval_name = f"{name}_{tag}.jsonl"
    eval_preds = run_predictions(
        eval_items,
        predict,
        CHECKPOINT_DIR / eval_name,
        desc=f"{name} ({tag})",
        meta=meta,
        allow_mixed_backend=args.allow_mixed_backend,
        mirror_path=mirror_for(eval_name),
        mirror_every=args.mirror_every,
    )
    metrics = score(eval_preds, eval_items)
    print()
    print(format_score(metrics, f"{name} @ {tag} (n={len(eval_items)})"))
    total_tokens = sum(p.get("new_tokens", 0) for p in eval_preds)
    if total_tokens:
        print(f"  tokens gerados: {total_tokens} ({total_tokens / len(eval_preds):.1f}/item)")
    clear_vram()

    if args.test:
        suffix = "_dry" if args.dry_run else ""
        test_name = f"{name}_test{suffix}.jsonl"
        test_preds = run_predictions(
            load_jsonl(TEST_JSONL),
            predict,
            CHECKPOINT_DIR / test_name,
            desc=f"{name} (test)",
            meta=meta,
            allow_mixed_backend=args.allow_mixed_backend,
            mirror_path=mirror_for(test_name),
            mirror_every=args.mirror_every,
        )
        write_submission(test_preds, SUBMISSION_DIR / f"submission_{name}{suffix}.csv")
        clear_vram()

    return metrics
