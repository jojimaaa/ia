"""
PCS 3838 - Inteligência Artificial - Trabalho Prático 1
AutoML / busca de hiperparâmetros para o Oblique Random Forest (JojiForest).

O que faz:
  - Gera várias combinações de hiperparâmetros (grid ou random search).
  - Avalia cada combinação com k-fold estratificado (ou holdout) sobre o X_train.
  - Salva, INCREMENTALMENTE, todos os parâmetros usados + acurácia (média±desvio
    entre folds) + tempos num CSV. Pode interromper (Ctrl-C) e retomar depois:
    combinações já avaliadas são puladas.
  - Ao final imprime um leaderboard ordenado por acurácia média de validação.

Por que k-fold: a floresta é NÃO determinística (bagging/subespaço sorteados sem
semente fixa em oRF.JojiForest), então a variação entre execuções (~±0.01-0.02 de
acc, ver tiebreak_results.csv) é da mesma ordem que a diferença entre configs. O
k-fold com média±desvio é o jeito correto de comparar configs sem se enganar com
ruído. Se estiver com pressa, use CV_FOLDS = 1 (holdout estratificado único).

Uso:
    python automl.py
Retomar (mesmo comando; ele lê o CSV e pula o que já foi feito):
    python automl.py

IMPORTANTE (Windows): JojiForest já paraleliza internamente com multiprocessing.
Por isso este script roda as configs em SÉRIE no processo principal (pools
aninhados quebram no spawn do Windows) e todo o trabalho fica sob
`if __name__ == '__main__'`. Não envolva a busca em outro Pool.
"""

import itertools
import os
import random
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Console do Windows costuma usar cp1252 e quebra em caracteres fora do Latin-1.
# Força UTF-8 na saída para acentos e símbolos não derrubarem a execução.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Reusa as classes/funções do arquivo principal (fonte única da verdade).
# Importar oRF NÃO executa o main() dele (está sob __main__), é seguro.
from oRF import JojiForest, exportPredictions, loadFile


# =============================================================================
# CONFIGURAÇÃO DA BUSCA
# =============================================================================

DATA_PATH = 'data.npz'
RESULTS_CSV = 'automl_results.csv'      # resultados (append incremental; base do resume)

# ---- ESTRATÉGIA DE BUSCA ----------------------------------------------------
SEARCH_MODE = 'random'        # 'random' (amostra MAX_TRIALS combos) | 'grid' (todas)
MAX_TRIALS = 25               # nº de combinações no modo random
SEED = 42                     # semente p/ folds e p/ amostragem de combos (reprodutível)

# ---- VALIDAÇÃO --------------------------------------------------------------
CV_FOLDS = 3                  # k do k-fold estratificado. Use 1 => holdout único (rápido)
HOLDOUT_TEST_SIZE = 0.25      # fração de validação quando CV_FOLDS == 1
TIME_BUDGET_MIN = 180         # tempo máx total (min); para de iniciar novas configs após isso. None = sem limite

# ---- O QUE GERAR NO FINAL ---------------------------------------------------
GENERATE_SUBMISSION_FOR_BEST = False  # treina a melhor config no X_train inteiro e gera CSV de submissão

# ---- ESPAÇO DE BUSCA --------------------------------------------------------
# Cada eixo é uma lista de opções. O produto cartesiano (filtrado) é o universo
# de combinações. No modo 'random' amostramos MAX_TRIALS combos distintos dele.
SPLIT_METHODS = [
    {'type': 'orthogonal'},
    {'type': 'PCA', 'c': 5},
    {'type': 'PCA', 'c': 10},
    {'type': 'SVM', 'C': 1.0, 'kernel': 'linear'},
    {'type': 'SVM', 'C': 1.0, 'kernel': 'rbf'},
    {'type': 'SVM', 'C': 1.0, 'kernel': 'poly'},
]
GAIN_METHODS = ['gini', 'entropy']
MAX_DEPTHS = [4, 8, 12]
LDA_OPTIONS = [None, 2]              # None desliga; 2 = nº de componentes LDA (máx = nº classes - 1 = 2)
TREE_COUNTS = [100]
FEATURES_PER_TREE = [15, 20, 25]    # <= 34 (nº de features do dataset)
SAMPLES_FRACTIONS = [0.8]
REPEATED_SAMPLING = [True]

N_FEATURES = 34                     # nº de features do dataset (para validar FEATURES_PER_TREE)


# =============================================================================
# ESPAÇO DE BUSCA
# =============================================================================
def build_search_space():
    """Produto cartesiano dos eixos, já filtrado para combinações válidas."""
    space = []
    for (split, gain, depth, lda, n_trees, feats, frac, boot) in itertools.product(
        SPLIT_METHODS, GAIN_METHODS, MAX_DEPTHS, LDA_OPTIONS,
        TREE_COUNTS, FEATURES_PER_TREE, SAMPLES_FRACTIONS, REPEATED_SAMPLING,
    ):
        # features sorteadas por árvore não pode exceder o nº de features
        if feats > N_FEATURES:
            continue
        # PCA testa 'c' componentes do subespaço de tamanho `feats`: exige c <= feats
        if split['type'] == 'PCA' and split['c'] > feats:
            continue

        space.append({
            'split': dict(split),   # cópia (evita alias entre configs)
            'gain': gain,
            'max_depth': depth,
            'lda': lda,
            'tree_count': n_trees,
            'features_per_tree': feats,
            'samples_fraction': frac,
            'repeated_sampling': boot,
        })
    return space


def split_label(split: dict) -> str:
    """Rótulo curto e legível do método de split (ex.: 'PCA-c5', 'SVM-rbf-C1.0')."""
    t = split['type']
    if t == 'orthogonal':
        return 'orthogonal'
    if t == 'PCA':
        return f"PCA-c{split['c']}"
    if t == 'SVM':
        return f"SVM-{split['kernel']}-C{split['C']}"
    return t


def config_key(cfg: dict) -> str:
    """Chave estável e única de uma config (usada para deduplicar e retomar)."""
    return (
        f"{split_label(cfg['split'])}|gain={cfg['gain']}|depth={cfg['max_depth']}"
        f"|lda={cfg['lda']}|trees={cfg['tree_count']}|feats={cfg['features_per_tree']}"
        f"|frac={cfg['samples_fraction']}|boot={cfg['repeated_sampling']}"
    )


# =============================================================================
# AVALIAÇÃO DE UMA CONFIG (k-fold)
# =============================================================================
def _fit_predict_fold(cfg, X_tr, y_tr, X_val):
    """Treina uma floresta no fold de treino e prevê o de validação.

    LDA (quando ligado) é ajustado DENTRO de cada árvore, sobre o bag — que vem
    só do fold de treino. Logo não há vazamento do fold de validação.
    """
    n_tr = len(X_tr)
    samples_per_tree = max(1, int(n_tr * cfg['samples_fraction']))
    samples_per_tree = min(samples_per_tree, n_tr)

    forest = JojiForest(
        featuresPerTree=cfg['features_per_tree'],
        samplesPerTree=samples_per_tree,
        repeatedSampling=cfg['repeated_sampling'],
        treeCount=cfg['tree_count'],
        maxDepth=cfg['max_depth'],
        gainMethod=cfg['gain'],
        splitMethod=cfg['split'],
        lda_components=cfg['lda'],
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    forest.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_hat = forest.predict(X_val)
    t_pred = time.perf_counter() - t0

    return np.asarray(y_hat), t_fit, t_pred


def evaluate_config(cfg, X, y):
    """Roda k-fold (ou holdout) e devolve um dict de métricas agregadas."""
    if CV_FOLDS >= 2:
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        folds = list(splitter.split(X, y))
    else:
        # Holdout estratificado único (CV_FOLDS == 1)
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        tr_idx, val_idx = train_test_split(
            idx, test_size=HOLDOUT_TEST_SIZE, random_state=SEED,
            shuffle=True, stratify=y,
        )
        folds = [(tr_idx, val_idx)]

    accs, fit_times, pred_times = [], [], []
    for tr_idx, val_idx in folds:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        y_hat, t_fit, t_pred = _fit_predict_fold(cfg, X_tr, y_tr, X_val)
        acc = float(np.mean(y_hat == y_val))

        accs.append(acc)
        fit_times.append(t_fit)
        pred_times.append(t_pred)

    accs = np.array(accs)
    return {
        'acc_mean': float(accs.mean()),
        'acc_std': float(accs.std()),
        'acc_folds': '|'.join(f'{a:.4f}' for a in accs),
        'fit_s_mean': float(np.mean(fit_times)),
        'pred_s_mean': float(np.mean(pred_times)),
    }


# =============================================================================
# PERSISTÊNCIA (append incremental + resume)
# =============================================================================
CSV_COLUMNS = [
    'timestamp', 'split', 'gain', 'max_depth', 'lda', 'tree_count',
    'features_per_tree', 'samples_fraction', 'repeated_sampling',
    'cv_folds', 'acc_mean', 'acc_std', 'acc_folds',
    'fit_s_mean', 'pred_s_mean', 'total_s', 'status', 'config_key',
]


def load_done_keys(csv_path: str) -> set:
    """Chaves de configs já avaliadas COM SUCESSO (para retomar sem repetir)."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')  # tolera linha final truncada
    except Exception as e:
        # Não zere o resume em silêncio: preserva o arquivo e aborta com aviso.
        bak = csv_path + '.corrupt'
        os.replace(csv_path, bak)
        raise SystemExit(
            f"[!] {csv_path} corrompido ({e}). Movido para {bak}. "
            f"Conserte/remova a última linha malformada e rode de novo."
        )
    if 'config_key' not in df.columns or 'status' not in df.columns:
        return set()
    done = df[df['status'] == 'ok']['config_key'].astype(str)
    return set(done.tolist())


def append_result(csv_path: str, row: dict):
    """Anexa uma linha ao CSV (cria com cabeçalho se não existir)."""
    df = pd.DataFrame([{c: row.get(c, '') for c in CSV_COLUMNS}])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)


def row_from(cfg: dict, metrics: dict | None, total_s: float, status: str) -> dict:
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'split': split_label(cfg['split']),
        'gain': cfg['gain'],
        'max_depth': cfg['max_depth'],
        'lda': cfg['lda'],
        'tree_count': cfg['tree_count'],
        'features_per_tree': cfg['features_per_tree'],
        'samples_fraction': cfg['samples_fraction'],
        'repeated_sampling': cfg['repeated_sampling'],
        'cv_folds': CV_FOLDS,
        'total_s': round(total_s, 1),
        'status': status,
        'config_key': config_key(cfg),
    }
    if metrics is not None:
        row.update({
            'acc_mean': round(metrics['acc_mean'], 4),
            'acc_std': round(metrics['acc_std'], 4),
            'acc_folds': metrics['acc_folds'],
            'fit_s_mean': round(metrics['fit_s_mean'], 1),
            'pred_s_mean': round(metrics['pred_s_mean'], 1),
        })
    return row


# =============================================================================
# LEADERBOARD / SUBMISSÃO
# =============================================================================
def print_leaderboard(csv_path: str, top: int = 15):
    if not os.path.exists(csv_path):
        print('Nenhum resultado para exibir.')
        return None
    df = pd.read_csv(csv_path)
    ok = df[df['status'] == 'ok'].copy()
    if ok.empty:
        print('Nenhuma config avaliada com sucesso ainda.')
        return None

    # Deduplica por config_key (rede de segurança caso o resume tenha reexecutado
    # uma config): mantém a melhor acc_mean vista para cada config.
    ok = ok.sort_values('acc_mean', ascending=False)
    ok = ok.drop_duplicates('config_key', keep='first').reset_index(drop=True)
    print('\n' + '=' * 78)
    print(f'LEADERBOARD (top {top}) — ordenado por acc_mean de validação')
    print('=' * 78)
    cols = ['split', 'gain', 'max_depth', 'lda', 'features_per_tree',
            'acc_mean', 'acc_std', 'fit_s_mean', 'pred_s_mean']
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(ok[cols].head(top).to_string(index=False))
    return ok.iloc[0]


def generate_submission_for_best(best_row, X_train, y_train, X_test):
    """Reconstrói a melhor config, treina no X_train INTEIRO e gera o CSV Kaggle."""
    # Reconstrói o split a partir do rótulo salvo
    label = str(best_row['split'])
    if label == 'orthogonal':
        split = {'type': 'orthogonal'}
    elif label.startswith('PCA-c'):
        split = {'type': 'PCA', 'c': int(label.split('c')[1])}
    elif label.startswith('SVM-'):
        _, kernel, c = label.split('-')
        split = {'type': 'SVM', 'kernel': kernel, 'C': float(c[1:])}
    else:
        print(f'Split desconhecido no leaderboard: {label}; pulando submissão.')
        return

    n_full = len(X_train)
    frac = float(best_row['samples_fraction'])
    lda = None if str(best_row['lda']) in ('None', 'nan', '') else int(float(best_row['lda']))

    forest = JojiForest(
        featuresPerTree=int(best_row['features_per_tree']),
        samplesPerTree=int(n_full * frac),
        repeatedSampling=bool(best_row['repeated_sampling']),
        treeCount=int(best_row['tree_count']),
        maxDepth=int(best_row['max_depth']),
        gainMethod=str(best_row['gain']),
        splitMethod=split,
        lda_components=lda,
        n_jobs=-1,
    )
    print('\nTreinando a melhor config no X_train inteiro para submissão...')
    forest.fit(X_train, y_train)
    y_submit = forest.predict(X_test)
    path = exportPredictions(y_submit, outDir='outputs')
    print(f'CSV de submissão gerado: {path}  ({len(y_submit)} linhas)')


# =============================================================================
# MAIN
# =============================================================================
def main():
    X_train, y_train, X_test = loadFile(DATA_PATH)

    space = build_search_space()

    if SEARCH_MODE == 'random':
        rng = random.Random(SEED)
        rng.shuffle(space)
        space = space[:MAX_TRIALS]
    # 'grid' => usa o espaço inteiro

    done = load_done_keys(RESULTS_CSV)
    pending = [cfg for cfg in space if config_key(cfg) not in done]

    print(f"\nModo de busca : {SEARCH_MODE}")
    print(f"Universo total: {len(build_search_space())} combinações válidas")
    print(f"Selecionadas  : {len(space)}  |  já feitas: {len(done)}  |  a rodar: {len(pending)}")
    print(f"Validação     : {'holdout ' + str(HOLDOUT_TEST_SIZE) if CV_FOLDS < 2 else str(CV_FOLDS) + '-fold estratificado'}")
    print(f"Resultados em : {RESULTS_CSV}")
    if TIME_BUDGET_MIN:
        print(f"Orçamento     : {TIME_BUDGET_MIN} min (para de iniciar novas configs após isso)")

    t_start = time.perf_counter()
    for i, cfg in enumerate(pending, 1):
        elapsed_min = (time.perf_counter() - t_start) / 60.0
        if TIME_BUDGET_MIN and elapsed_min > TIME_BUDGET_MIN:
            print(f"\n[!] Orçamento de {TIME_BUDGET_MIN} min esgotado — parando. "
                  f"Rode de novo depois para continuar de onde parou.")
            break

        key = config_key(cfg)
        print(f"\n[{i}/{len(pending)}] {key}")
        t0 = time.perf_counter()

        # Avalia dentro do try; grava a linha e imprime FORA dele, para que uma
        # falha de I/O de console não gere uma segunda linha (dupla escrita).
        metrics, status, err, tb = None, 'ok', None, None
        try:
            metrics = evaluate_config(cfg, X_train, y_train)
        except KeyboardInterrupt:
            print("\n[!] Interrompido pelo usuário. Progresso salvo — rode de novo para continuar.")
            raise
        except Exception as e:
            # Captura o traceback AQUI: fora do except o sys.exc_info() já foi
            # limpo (Python 3.11+) e traceback.print_exc() imprimiria 'NoneType'.
            status, err, tb = f'erro: {e}', e, traceback.format_exc()

        total_s = time.perf_counter() - t0
        append_result(RESULTS_CSV, row_from(cfg, metrics, total_s, status))

        if status == 'ok':
            print(f"    acc={metrics['acc_mean']:.4f} +/- {metrics['acc_std']:.4f} "
                  f"(folds: {metrics['acc_folds']})  "
                  f"fit~{metrics['fit_s_mean']:.1f}s pred~{metrics['pred_s_mean']:.1f}s "
                  f"total={total_s:.1f}s")
        else:
            print(f"    [ERRO] {err}")
            print(tb)

    best = print_leaderboard(RESULTS_CSV)

    if GENERATE_SUBMISSION_FOR_BEST and best is not None:
        generate_submission_for_best(best, X_train, y_train, X_test)


if __name__ == '__main__':
    main()
