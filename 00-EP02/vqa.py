"""
Núcleo compartilhado do EP02 (VQA sobre circuitos lógicos digitais).

Concentra o que todos os experimentos precisam, para que cada experimento
seja só "qual prompt / qual estratégia de geração":

  - question_kind()    distingue pergunta de contagem de pergunta de saída
  - clean_answer()     extrai a resposta canônica da geração do modelo
  - majority_answer()  constante de maior frequência no treino (fallback)
  - run_predictions()  loop com checkpoint incremental (sobrevive queda de sessão)
  - score()            acurácia total E separada por tipo de pergunta
  - write_submission() CSV no formato exigido pelo Kaggle

A correção do Kaggle é por correspondência exata, então toda resposta que sai
daqui é uma string já normalizada: "True", "False" ou um inteiro sem zeros à
esquerda.

Autoteste (asserções do clean_answer + estatísticas do treino):
    uv run python vqa.py
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_JSONL = PROCESSED_DIR / "train.jsonl"
VAL_JSONL = PROCESSED_DIR / "val.jsonl"
TEST_JSONL = PROCESSED_DIR / "test.jsonl"

OUTPUT_DIR = ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

COUNT = "count"    # "How many <gate> gates are in this logic circuit?"  -> inteiro
OUTPUT = "output"  # "What is the output of this logic circuit, given..." -> True/False


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #


def load_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: Iterable[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Tipo de pergunta
# --------------------------------------------------------------------------- #


def question_kind(question: str) -> str:
    """COUNT ou OUTPUT. Levanta erro em pergunta fora dos dois formatos conhecidos.

    O dataset é fechado e só contém essas duas famílias (verificado nos 2.000 de
    treino e nos 1.000 de teste); classificar errado em silêncio seria pior que
    falhar, porque muda o parsing da resposta.
    """
    q = question.strip()
    if q.startswith("What is the output"):
        return OUTPUT
    if q.startswith("How many"):
        return COUNT
    raise ValueError(f"Pergunta em formato desconhecido: {question!r}")


def asked_gate(question: str) -> str | None:
    """Tipo de porta perguntado numa pergunta de contagem (and, nor, xor, ...)."""
    m = re.search(r"How many (\w+) gates", question, re.IGNORECASE)
    return m.group(1).lower() if m else None


# --------------------------------------------------------------------------- #
# Parsing da resposta
# --------------------------------------------------------------------------- #

_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}

# "Final Answer:", "final answer -", "FINAL ANSWER" ...
_FINAL_MARKER = re.compile(r"final\s*answer\s*[:\-–]?", re.IGNORECASE)

# Ecos da própria pergunta: "x0=False, x1=True". Precisam morrer antes de
# procurar True/False, senão o parser lê a entrada do circuito como resposta.
_INPUT_ASSIGNMENT = re.compile(r"\bx\d+\s*=\s*(?:true|false)\b", re.IGNORECASE)


def _strip_question_artifacts(text: str, question: str | None = None) -> str:
    """Remove o eco da pergunta e as atribuições x<i>=<bool> da geração."""
    if question:
        pos = text.find(question.strip())
        if pos != -1:
            text = text[pos + len(question.strip()):]
    return _INPUT_ASSIGNMENT.sub(" ", text)


def _last(matches: Sequence[str]) -> str | None:
    return matches[-1] if matches else None


def _extract_bool(segment: str) -> str | None:
    tok = _last(re.findall(r"\b(true|false)\b", segment, re.IGNORECASE))
    if tok:
        return "True" if tok.lower() == "true" else "False"
    tok = _last(re.findall(r"\b([01])\b", segment))
    if tok:
        return "True" if tok == "1" else "False"
    tok = _last(re.findall(r"\b(yes|no)\b", segment, re.IGNORECASE))
    if tok:
        return "True" if tok.lower() == "yes" else "False"
    return None


def _extract_count(segment: str) -> str | None:
    tok = _last(re.findall(r"\b(\d+)\b", segment))
    if tok:
        return str(int(tok))  # normaliza "02" -> "2"
    words = "|".join(_WORD_NUMBERS)
    tok = _last(re.findall(rf"\b({words})\b", segment, re.IGNORECASE))
    if tok:
        return _WORD_NUMBERS[tok.lower()]
    return None


def clean_answer(
    text: str,
    kind: str | None = None,
    question: str | None = None,
    default: str | None = None,
) -> str | None:
    """Extrai a resposta canônica de uma geração livre do modelo.

    `kind` (COUNT/OUTPUT) restringe o que é aceito: numa pergunta de contagem só
    dígito conta, numa de saída só booleano. Isso descarta de graça uma classe
    inteira de erro (modelo responder "True" para "how many").

    Estratégia, na ordem:
      1. tira o eco da pergunta e os "x0=False" (senão a entrada vira resposta);
      2. se houver "Final Answer:", usa o trecho depois da ÚLTIMA ocorrência
         (na Reflexion a resposta corrigida é a última);
      3. dentro do trecho, pega a ÚLTIMA ocorrência válida — geração termina na
         conclusão, não começa nela;
      4. se o trecho não tiver nada válido, tenta o texto inteiro;
      5. se nada parseia, devolve `default` (None se não informado).

    Devolver `default` em vez de texto cru importa: com correspondência exata uma
    resposta malformada é erro garantido, e a constante majoritária do treino
    acerta ~52% (saída) / ~38% (contagem).
    """
    if kind is None and question is not None:
        kind = question_kind(question)

    cleaned = _strip_question_artifacts(text or "", question)

    markers = list(_FINAL_MARKER.finditer(cleaned))
    segments = []
    if markers:
        tail = cleaned[markers[-1].end():]
        first_line = tail.splitlines()[0] if tail.splitlines() else tail
        segments.append(first_line)
        segments.append(tail)
    segments.append(cleaned)

    if kind == COUNT:
        extract = _extract_count
    elif kind == OUTPUT:
        extract = _extract_bool
    else:
        def extract(s: str) -> str | None:
            return _extract_bool(s) or _extract_count(s)

    for segment in segments:
        parsed = extract(segment)
        if parsed is not None:
            return parsed
    return default


# --------------------------------------------------------------------------- #
# Baseline constante (derivada do treino, nunca da validação)
# --------------------------------------------------------------------------- #

_majority_cache: dict[str, str] | None = None


def majority_answer(kind: str, train_path: Path | str = TRAIN_JSONL) -> str:
    """Resposta mais frequente no treino para aquele tipo de pergunta.

    Lê `data/processed/train.jsonl` (1.800 amostras), não o arquivo bruto de
    2.000 — a validação fica fora, então usar isso como fallback não vaza label.
    """
    global _majority_cache
    if _majority_cache is None:
        counters: dict[str, Counter] = {COUNT: Counter(), OUTPUT: Counter()}
        for rec in load_jsonl(train_path):
            counters[question_kind(rec["question"])][rec["answer"]] += 1
        _majority_cache = {k: c.most_common(1)[0][0] for k, c in counters.items()}
    return _majority_cache[kind]


# --------------------------------------------------------------------------- #
# Loop de inferência com checkpoint incremental
# --------------------------------------------------------------------------- #


def _progress(iterable, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm
    except ImportError:
        print(f"{desc}: {total if total is not None else '?'} itens")
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def load_checkpoint(path: Path | str) -> dict[int, dict]:
    """Índices já processados. Ignora linha truncada por sessão morta no meio."""
    path = Path(path)
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # escrita interrompida; será refeita
            if "index" in rec:
                done[rec["index"]] = rec
    return done


def run_predictions(
    items: Sequence[dict],
    predict: Callable[[dict], dict],
    checkpoint_path: Path | str,
    desc: str = "inferência",
) -> list[dict]:
    """Roda `predict` em cada item, pulando o que já está no checkpoint.

    `predict(item)` devolve um dict com no mínimo {"answer": str}; pode incluir
    "raw" (geração bruta) e o que mais o experimento quiser registrar. Cada
    resultado é gravado e o buffer esvaziado na hora, então uma queda de sessão
    do Colab custa no máximo um item.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_checkpoint(checkpoint_path)
    pending = [it for it in items if it["index"] not in done]
    if done:
        print(f"{desc}: {len(done)} já no checkpoint, {len(pending)} restando")

    with checkpoint_path.open("a", encoding="utf-8") as f:
        for item in _progress(pending, desc, total=len(pending)):
            started = time.perf_counter()
            result = predict(item)
            record = {
                "index": item["index"],
                "question": item["question"],
                "elapsed_s": round(time.perf_counter() - started, 4),
                **result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done[item["index"]] = record

    return [done[it["index"]] for it in items if it["index"] in done]


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


def score(predictions: Sequence[dict], references: Sequence[dict]) -> dict:
    """Acurácia por correspondência exata, total e separada por tipo de pergunta.

    A separação não é enfeite: as duas metades têm pisos muito diferentes
    (contagem ~38%, saída ~52%), então um ganho na contagem pode esconder uma
    perda na saída e levar à conclusão errada sobre a técnica.
    """
    pred_by_index = {p["index"]: p for p in predictions}
    buckets: dict[str, dict] = {}
    missing = 0

    for ref in references:
        kind = question_kind(ref["question"])
        bucket = buckets.setdefault(
            kind, {"n": 0, "correct": 0, "elapsed_s": 0.0, "timed": 0}
        )
        pred = pred_by_index.get(ref["index"])
        if pred is None:
            missing += 1
            continue
        bucket["n"] += 1
        bucket["correct"] += int(str(pred.get("answer")) == str(ref["answer"]))
        if "elapsed_s" in pred:
            bucket["elapsed_s"] += pred["elapsed_s"]
            bucket["timed"] += 1

    total_n = sum(b["n"] for b in buckets.values())
    total_correct = sum(b["correct"] for b in buckets.values())
    total_elapsed = sum(b["elapsed_s"] for b in buckets.values())
    total_timed = sum(b["timed"] for b in buckets.values())

    by_kind = {}
    for kind, b in buckets.items():
        by_kind[kind] = {
            "n": b["n"],
            "correct": b["correct"],
            "accuracy": (b["correct"] / b["n"] * 100) if b["n"] else 0.0,
            "avg_s": (b["elapsed_s"] / b["timed"]) if b["timed"] else None,
        }

    return {
        "n": total_n,
        "correct": total_correct,
        "accuracy": (total_correct / total_n * 100) if total_n else 0.0,
        "avg_s": (total_elapsed / total_timed) if total_timed else None,
        "missing": missing,
        "by_kind": by_kind,
    }


def format_score(metrics: dict, title: str = "resultado") -> str:
    lines = [f"=== {title} ==="]
    for kind in (COUNT, OUTPUT):
        k = metrics["by_kind"].get(kind)
        if not k:
            continue
        avg = f"  {k['avg_s']:.2f}s/item" if k["avg_s"] else ""
        lines.append(
            f"  {kind:<7} {k['accuracy']:6.2f}%  ({k['correct']}/{k['n']}){avg}"
        )
    avg = f"  {metrics['avg_s']:.2f}s/item" if metrics["avg_s"] else ""
    lines.append(
        f"  {'TOTAL':<7} {metrics['accuracy']:6.2f}%  "
        f"({metrics['correct']}/{metrics['n']}){avg}"
    )
    if metrics["missing"]:
        lines.append(f"  !! {metrics['missing']} itens sem predição")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Submissão
# --------------------------------------------------------------------------- #


def write_submission(predictions: Sequence[dict], path: Path | str) -> Path:
    """CSV com as colunas index,answer exigidas pelo Kaggle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(predictions, key=lambda p: p["index"])

    blank = [p["index"] for p in rows if p.get("answer") in (None, "")]
    if blank:
        raise ValueError(
            f"{len(blank)} respostas vazias (ex.: índices {blank[:5]}). "
            "Passe um `default` ao clean_answer — vazio é erro garantido."
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "answer"])
        for p in rows:
            writer.writerow([p["index"], p["answer"]])
    print(f"Submissão com {len(rows)} linhas salva em {path}")
    return path


# --------------------------------------------------------------------------- #
# Autoteste
# --------------------------------------------------------------------------- #


def _selftest_clean_answer() -> None:
    cases = [
        # (texto, kind, esperado, comentário)
        ("True", OUTPUT, "True", "resposta seca"),
        ("false", OUTPUT, "False", "caixa baixa normaliza"),
        ("2", COUNT, "2", "dígito seco"),
        ("There are 12 and gates.", COUNT, "12", "multi-dígito (o bug antigo cortava em 1)"),
        ("02", COUNT, "2", "zero à esquerda"),
        ("zero", COUNT, "0", "número escrito"),
        ("Yes", OUTPUT, "True", "yes/no"),
        ("1", OUTPUT, "True", "1/0 como booleano"),
        ("Final Answer: [True]", OUTPUT, "True", "colchetes"),
        (
            "Step 1: gate 3 is AND. Step 2: gate 4 is OR.\nFinal Answer: 5",
            COUNT, "5", "CoT: marcador vence os números do raciocínio",
        ),
        (
            "I see gate 1, gate 2, so there are 3 nor gates",
            COUNT, "3", "CoT sem marcador: última ocorrência é a conclusão",
        ),
        (
            "Final Answer: True\nOn reflection the NAND was misread.\nFinal Answer: False",
            OUTPUT, "False", "Reflexion: vale o ÚLTIMO marcador",
        ),
        (
            "given the inputs x0=False, x1=False, x2=False. Final Answer: True",
            OUTPUT, "True", "eco de x<i>=<bool> não vira resposta",
        ),
        ("given x0=False, x1=True", OUTPUT, None, "só eco da pergunta -> default"),
        ("no idea", COUNT, None, "nada parseável -> default"),
    ]
    for text, kind, expected, note in cases:
        got = clean_answer(text, kind=kind)
        assert got == expected, f"{note}: clean_answer({text!r}, {kind}) = {got!r}, esperado {expected!r}"

    # `default` cobre o caso não parseável, garantindo linha válida na submissão.
    assert clean_answer("no idea", kind=COUNT, default="0") == "0"

    # Eco literal da pergunta é descartado.
    question = (
        "What is the output of this logic circuit, given the inputs "
        "x0=True, x1=False?"
    )
    assert clean_answer(f"{question} False", question=question) == "False"

    print(f"clean_answer: {len(cases) + 2} asserções OK")


def _print_dataset_stats() -> None:
    if not TRAIN_JSONL.exists():
        print(f"(sem {TRAIN_JSONL}; rode data/prepare_dataset.py primeiro)")
        return
    for name, path in (("train", TRAIN_JSONL), ("val", VAL_JSONL), ("test", TEST_JSONL)):
        if not path.exists():
            continue
        recs = load_jsonl(path)
        kinds = Counter(question_kind(r["question"]) for r in recs)
        print(f"{name:<5} n={len(recs):<5} count={kinds[COUNT]:<5} output={kinds[OUTPUT]}")
    print(
        f"constante majoritária do treino: "
        f"{COUNT}={majority_answer(COUNT)!r}  {OUTPUT}={majority_answer(OUTPUT)!r}"
    )


if __name__ == "__main__":
    _selftest_clean_answer()
    _print_dataset_stats()
