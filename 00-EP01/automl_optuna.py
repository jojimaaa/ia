"""
PCS 3838 - Inteligência Artificial - Trabalho Prático 1
AutoML com Optuna (TPE) para o Oblique Random Forest (JojiForest).

Busca inteligente (Bayesian/TPE) sobre TODOS os hiperparâmetros:
  - maxDepth, gainMethod (gini/entropy)
  - splitType: orthogonal | PCA | SVM-{linear,rbf,poly,sigmoid}  (todos os kernels)
  - PCA: c ∈ {5,10,15,20}      | SVM: C ∈ [0.01, 100] (log)
  - lda_components ∈ {None,1,2}
  - featuresPerTree, samplesFraction (bagging), treeCount

Diferenças em relação ao snippet original do Optuna (para não otimizar ruído):
  1) k-fold estratificado DENTRO do objective (a floresta é não-determinística,
     então avaliar num único holdout faz o TPE perseguir ruído). Com PRUNING:
     trials ruins são cortados cedo (economiza tempo).
  2) samplesFraction (0.3–1.0) em vez de samplesPerTree absoluto: cada fold de
     treino é menor que n_train, então um valor absoluto até n_train estouraria
     o ValueError de JojiForest.fit no fold. A fração é convertida por fold.
  3) storage SQLite => RESUME de graça (rode de novo e ele continua até a meta).
  4) timeout (orçamento de tempo) + sampler com seed (busca reprodutível).

Uso:
    python automl_optuna.py
Retomar (mesmo comando; continua de onde parou até N_TRIALS_TARGET):
    python automl_optuna.py

IMPORTANTE (Windows): JojiForest já paraleliza com multiprocessing internamente.
Os trials rodam em SÉRIE no processo principal (n_jobs=1) e todo o trabalho fica
sob `if __name__ == '__main__'`. Não use Optuna n_jobs>1 aqui (pools aninhados).
"""

import sys
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split

from oRF import JojiForest, accuracyAndError, exportPredictions, loadFile

# Console do Windows (cp1252) quebra em símbolos fora do Latin-1: força UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass


# =============================================================================
# CONFIGURAÇÃO DA BUSCA
# =============================================================================
DATA_PATH = 'data.npz'
STORAGE = 'sqlite:///automl_optuna.db'   # base do resume (não apagar entre runs)
STUDY_NAME = 'jojiforest'
RESULTS_CSV = 'automl_optuna_results.csv'  # snapshot legível de todos os trials

N_TRIALS_TARGET = 200        # meta TOTAL de trials (resume continua até aqui)
TIME_BUDGET_MIN = None       # orçamento de tempo por execução (min). None = sem limite (roda até a meta)
SEED = 42                    # semente do sampler e dos folds (busca reprodutível)

CV_FOLDS = 3                 # k do k-fold estratificado no objective. 1 = holdout (rápido)
HOLDOUT_TEST_SIZE = 0.25     # usado quando CV_FOLDS == 1
USE_PRUNING = True           # corta trials ruins cedo (MedianPruner sobre os folds)

GENERATE_SUBMISSION_FOR_BEST = False  # ao final, treina a melhor config no X_train inteiro

# ---- FAIXAS DO ESPAÇO DE BUSCA ---------------------------------------------
DEPTH_RANGE = (2, 10)                 # maxDepth (int)
TREES_RANGE = (10, 200)               # treeCount (int)
SAMPLES_FRACTION_RANGE = (0.3, 1.0)   # fração de bagging por árvore (float)
FEATS_MIN = 1                         # featuresPerTree mínimo (máx = nº de features, dinâmico)
SPLIT_TYPES = ['orthogonal', 'PCA', 'SVM-linear', 'SVM-rbf', 'SVM-poly', 'SVM-sigmoid']
GAIN_CHOICES = ['gini', 'entropy']
LDA_CHOICES = [None, 1, 2]
PCA_C_CHOICES = [5, 10, 15, 20]
SVM_C_RANGE = (0.01, 100.0)           # log-uniform


# =============================================================================
# PARÂMETROS -> MODELO
# =============================================================================
def suggest_params(trial, n_features: int) -> dict:
    """Sorteia um conjunto de hiperparâmetros (espaço condicional do Optuna)."""
    params = {
        'maxDepth': trial.suggest_int('maxDepth', *DEPTH_RANGE),
        'gainMethod': trial.suggest_categorical('gainMethod', GAIN_CHOICES),
        'splitType': trial.suggest_categorical('splitType', SPLIT_TYPES),
        'lda_components': trial.suggest_categorical('lda_components', LDA_CHOICES),
        'featuresPerTree': trial.suggest_int('featuresPerTree', FEATS_MIN, n_features),
        'samplesFraction': trial.suggest_float('samplesFraction', *SAMPLES_FRACTION_RANGE),
        'treeCount': trial.suggest_int('treeCount', *TREES_RANGE),
    }
    # Parâmetros condicionais (só existem para o tipo de split escolhido)
    if params['splitType'] == 'PCA':
        params['pca_c'] = trial.suggest_categorical('pca_c', PCA_C_CHOICES)
    elif params['splitType'].startswith('SVM-'):
        params['svm_C'] = trial.suggest_float('svm_C', *SVM_C_RANGE, log=True)
    return params


def build_split_method(params: dict) -> dict:
    """Constrói o dict splitMethod que JojiForest/JojiTree esperam."""
    st = params['splitType']
    if st == 'orthogonal':
        return {'type': 'orthogonal'}
    if st == 'PCA':
        return {'type': 'PCA', 'c': int(params['pca_c'])}
    # SVM-<kernel>
    kernel = st.replace('SVM-', '')
    return {'type': 'SVM', 'C': float(params['svm_C']), 'kernel': kernel}


def make_forest(params: dict, samples_per_tree: int) -> JojiForest:
    return JojiForest(
        featuresPerTree=int(params['featuresPerTree']),
        samplesPerTree=int(samples_per_tree),
        repeatedSampling=True,
        treeCount=int(params['treeCount']),
        maxDepth=int(params['maxDepth']),
        gainMethod=params['gainMethod'],
        splitMethod=build_split_method(params),
        lda_components=params['lda_components'],
        n_jobs=-1,
    )


# =============================================================================
# AVALIAÇÃO (k-fold com pruning)
# =============================================================================
def cv_accuracy(params: dict, X, y, trial=None) -> float:
    """Acurácia média de validação em k-fold estratificado.

    samplesFraction é convertido em contagem absoluta POR FOLD (o fold de treino
    é menor que o dataset inteiro). Reporta a média corrente ao trial a cada fold
    para permitir pruning de trials ruins.
    """
    if CV_FOLDS >= 2:
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        folds = list(splitter.split(X, y))
    else:
        idx = np.arange(len(y))
        tr_idx, val_idx = train_test_split(
            idx, test_size=HOLDOUT_TEST_SIZE, random_state=SEED, shuffle=True, stratify=y)
        folds = [(tr_idx, val_idx)]

    accs = []
    for step, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        n_tr = len(X_tr)
        samples_per_tree = min(n_tr, max(1, int(n_tr * params['samplesFraction'])))

        forest = make_forest(params, samples_per_tree)
        forest.fit(X_tr, y_tr)
        y_hat = forest.predict(X_val)
        acc, _ = accuracyAndError(np.asarray(y_hat), y_val)
        accs.append(acc)

        # Pruning: reporta a média corrente; corta se estiver claramente ruim
        if trial is not None:
            trial.report(float(np.mean(accs)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(accs))


# =============================================================================
# CALLBACK / RESULTADOS
# =============================================================================
def make_callback():
    def _cb(study, trial):
        state = trial.state.name
        val = f"{trial.value:.4f}" if trial.value is not None else "----"
        best = f"{study.best_value:.4f}" if study.best_trial else "----"
        # resumo compacto dos params principais
        p = trial.params
        split = p.get('splitType', '?')
        extra = ''
        if split == 'PCA':
            extra = f" c={p.get('pca_c')}"
        elif str(split).startswith('SVM-'):
            extra = f" C={p.get('svm_C', 0):.3g}"
        print(f"[trial {trial.number:>3}] {state:<8} acc={val}  best={best}  "
              f"| {split}{extra} depth={p.get('maxDepth')} gain={p.get('gainMethod')} "
              f"lda={p.get('lda_components')} feats={p.get('featuresPerTree')} "
              f"frac={p.get('samplesFraction', 0):.2f} trees={p.get('treeCount')}")
    return _cb


def export_and_report(study):
    # snapshot legível de todos os trials
    df = study.trials_dataframe()
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nTrials salvos em: {RESULTS_CSV}  (storage: {STORAGE})")

    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    if not completed:
        print("Nenhum trial COMPLETE ainda.")
        return None

    print("\n" + "=" * 70)
    print(f"MELHOR ACURÁCIA (validação {CV_FOLDS}-fold): {study.best_value:.4f}")
    print("MELHORES PARÂMETROS:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # importância dos hiperparâmetros
    try:
        print("\nIMPORTÂNCIA DOS HIPERPARÂMETROS:")
        for param, imp in optuna.importance.get_param_importances(study).items():
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"  (não foi possível calcular importâncias: {e})")

    return study.best_params


def generate_submission_for_best(best_params, X_train, y_train, X_test):
    n_full = len(X_train)
    samples_per_tree = min(n_full, max(1, int(n_full * best_params['samplesFraction'])))
    forest = make_forest(best_params, samples_per_tree)
    print("\nTreinando a melhor config no X_train inteiro para submissão...")
    forest.fit(X_train, y_train)
    y_submit = forest.predict(X_test)
    path = exportPredictions(y_submit, outDir='outputs')
    print(f"CSV de submissão gerado: {path}  ({len(y_submit)} linhas)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train, X_test = loadFile(DATA_PATH)
    _, m = X_train.shape

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
              if USE_PRUNING and CV_FOLDS >= 2 else optuna.pruners.NopPruner())

    study = optuna.create_study(
        direction='maximize',
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,      # <== RESUME: reusa o estudo salvo no SQLite
        sampler=sampler,
        pruner=pruner,
    )

    n_existing = len(study.trials)
    remaining = max(0, N_TRIALS_TARGET - n_existing)

    print(f"\nEstudo    : {STUDY_NAME}  (trials já feitos: {n_existing})")
    print(f"Meta      : {N_TRIALS_TARGET} trials totais  ->  faltam {remaining}")
    print(f"Validação : {'holdout ' + str(HOLDOUT_TEST_SIZE) if CV_FOLDS < 2 else str(CV_FOLDS) + '-fold estratificado'}"
          f"  | pruning: {USE_PRUNING and CV_FOLDS >= 2}")
    if TIME_BUDGET_MIN:
        print(f"Orçamento : {TIME_BUDGET_MIN} min por execução")

    def objective(trial):
        params = suggest_params(trial, m)
        return cv_accuracy(params, X_train, y_train, trial=trial)

    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=(TIME_BUDGET_MIN * 60 if TIME_BUDGET_MIN else None),
            callbacks=[make_callback()],
            catch=(Exception,),   # um trial que falhar (ex.: SVM ruim) não derruba a busca
            gc_after_trial=True,
        )
    else:
        print("Meta de trials já atingida — nada a rodar (só relatório).")

    best_params = export_and_report(study)

    if GENERATE_SUBMISSION_FOR_BEST and best_params is not None:
        generate_submission_for_best(best_params, X_train, y_train, X_test)


if __name__ == '__main__':
    main()
