import contextlib
import glob
import logging
import os
import random
import re
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from modeling.utils import (
    FRAMES_DIR,
    OUTPUT_DIR,
    UEQ_COLUMNS,
    ModelingDataset,
    ensure_output_dir,
    load_modeling_dataset,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("grid_search_debug.log"),
        logging.StreamHandler(),
    ],
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGETS = list(UEQ_COLUMNS)
USE_SUBJECT_ZSCORE = True
def _build_scenarios():
    base_metrics = ("p95",)
    scenarios = {
        "P95_Emotions": {
            "emotion_metrics": base_metrics,
            "humor_prefixes": (),
            "include_controls": False,
            "use_target_encoding": False,
        },
        "P95_Emotions_PHA": {
            "emotion_metrics": base_metrics,
            "humor_prefixes": ("PHA",),
            "include_controls": False,
            "use_target_encoding": False,
        },
        "P95_Emotions_Controls": {
            "emotion_metrics": base_metrics,
            "humor_prefixes": (),
            "include_controls": True,
            "use_target_encoding": False,
        },
        "P95_Emotions_PHA_Controls": {
            "emotion_metrics": base_metrics,
            "humor_prefixes": ("PHA",),
            "include_controls": True,
            "use_target_encoding": False,
        },
        "P95_Emotions_PHA_PHD": {
            "emotion_metrics": base_metrics,
            "humor_prefixes": ("PHA", "PHD"),
            "include_controls": False,
            "use_target_encoding": False,
        },
        "P95_Emotions_PHA_PHD_Controls": {
            "emotion_metrics": base_metrics,
            "humor_prefixes": ("PHA", "PHD"),
            "include_controls": True,
            "use_target_encoding": False,
        },
    }
    return scenarios


SCENARIOS = _build_scenarios()

# Definição dos Regressores a serem testados
REGRESSORS = {
    "Baseline_Dummy": DummyRegressor(strategy="mean"),
    "Lasso": Lasso(alpha=0.1, random_state=42),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "BayesianRidge": BayesianRidge(),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    "SVR_Linear": SVR(kernel='linear', C=1.0),
    "SVR_RBF": SVR(kernel='rbf', C=1.0, gamma='scale'),
    "KernelRidge": KernelRidge(alpha=1.0, kernel="rbf"),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1),
    # "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    # "XGBRegressor": xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, eval_metric='mae'),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    # "MLPRegressor": MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    "PLSRegression": PLSRegression(n_components=2),
}

def _parse_env_list(env_name: str, cast_fn=str):
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    try:
        return [cast_fn(x) for x in items]
    except ValueError:
        logging.warning(f"Valor inválido em {env_name}='{raw}'. Ignorando.")
        return None

def get_window_size(filename: str) -> int:
    match = re.search(r"nwindows(\d+)", filename)
    return int(match.group(1)) if match else 0

def _prepare_dataset(summary_file: str, target: str, params: dict) -> ModelingDataset | None:
    dataset = load_modeling_dataset(
        summary_name=summary_file,
        target_column=target,
        emotion_metrics=params["emotion_metrics"],
        humor_prefixes=params["humor_prefixes"],
        include_controls=params["include_controls"],
        drop_neutral=False,
        zscore_by_subject=USE_SUBJECT_ZSCORE,
    )

    if dataset.X.shape[1] == 0:
        return None

    # --- FILTRAGEM DE OUTLIERS ---
    exclude_mask = ~np.isin(dataset.groups, [21])
    dataset = ModelingDataset(
        X=dataset.X[exclude_mask],
        y=dataset.y[exclude_mask],
        groups=dataset.groups[exclude_mask],
        games=dataset.games[exclude_mask],
        feature_names=dataset.feature_names,
        dataframe=dataset.dataframe,
    )
    return dataset

def _maybe_set_n_jobs(model, parallel_enabled: bool) -> None:
    if parallel_enabled and hasattr(model, "n_jobs"):
        with contextlib.suppress(Exception):
            model.n_jobs = 1

def _evaluate_regressor(
    win_size: int,
    target: str,
    scenario_name: str,
    params: dict,
    regressor_name: str,
    regressor_instance,
    dataset: ModelingDataset,
    parallel_enabled: bool,
) -> dict:
    try:
        X = dataset.X
        y = dataset.y
        groups = dataset.groups
        games = dataset.games

        # Loop de Validação Cruzada (LOSO)
        logo = LeaveOneGroupOut()
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # --- Target Encoding do Jogo (Opcional) ---
            if params.get("use_target_encoding"):
                train_games = games[train_idx]
                test_games = games[test_idx]

                df_train = pd.DataFrame({'g': train_games, 'y': y_train})
                game_means = df_train.groupby('g')['y'].mean().to_dict()
                global_mean = y_train.mean()

                feat_train = np.array([game_means.get(g, global_mean) for g in train_games]).reshape(-1, 1)
                feat_test = np.array([game_means.get(g, global_mean) for g in test_games]).reshape(-1, 1)

                X_train = np.hstack([X_train, feat_train])
                X_test = np.hstack([X_test, feat_test])

            # Escalonamento
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = clone(regressor_instance)
            _maybe_set_n_jobs(model, parallel_enabled)

            try:
                model.fit(X_train_s, y_train)
            except np.linalg.LinAlgError:
                # Fallback: Adicionar jitter para resolver problemas de convergência SVD (ex: BayesianRidge)
                jit = 1e-6
                X_train_jit = X_train_s + np.random.normal(0, jit, X_train_s.shape)
                model.fit(X_train_jit, y_train)

            preds = model.predict(X_test_s)
            if preds.ndim > 1:
                preds = preds.ravel()

            y_true_all.extend(y_test)
            y_pred_all.extend(preds)

        mae = mean_absolute_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)

        return {
            "ok": True,
            "result": {
                "WindowSize": win_size,
                "Target": target,
                "Scenario": scenario_name,
                "Regressor": regressor_name,
                "MAE": mae,
                "R2": r2,
                "FeatureCount": X.shape[1],
            },
        }
    except (ValueError, np.linalg.LinAlgError) as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
            "context": (win_size, target, scenario_name, regressor_name),
        }
    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
            "context": (win_size, target, scenario_name, regressor_name),
        }

def run_comprehensive_search():
    logging.info(">>> Iniciando Grid Search Experimental (Janelas x Features x Regressores)...")
    ensure_output_dir()

    # 1. Identificar arquivos disponíveis
    files = sorted(glob.glob(str(FRAMES_DIR / "summary_max_nwindows*.csv")))
    window_sizes = sorted([get_window_size(f) for f in files])
    # Filtrar apenas janelas válidas (>0)
    window_sizes = [w for w in window_sizes if w > 0]

    # Filtros opcionais por variáveis de ambiente
    env_windows = _parse_env_list("GRID_WINDOWS", int)
    if env_windows:
        window_sizes = [w for w in window_sizes if w in env_windows]

    env_scenarios = _parse_env_list("GRID_SCENARIOS", str)
    if env_scenarios:
        scenarios = {k: v for k, v in SCENARIOS.items() if k in env_scenarios}
    else:
        scenarios = SCENARIOS

    env_regressors = _parse_env_list("GRID_REGRESSORS", str)
    if env_regressors:
        regressors = {k: v for k, v in REGRESSORS.items() if k in env_regressors}
    else:
        regressors = REGRESSORS

    env_targets = _parse_env_list("GRID_TARGETS", str)
    targets = env_targets or TARGETS

    n_jobs_raw = os.environ.get("GRID_N_JOBS", "1").strip()
    try:
        n_jobs = max(1, int(n_jobs_raw))
    except ValueError:
        logging.warning(f"GRID_N_JOBS inválido: '{n_jobs_raw}'. Usando 1.")
        n_jobs = 1
    parallel_enabled = n_jobs > 1

    logging.info(f"Janelas encontradas: {window_sizes}")
    logging.info(f"Cenários: {list(scenarios.keys())}")
    logging.info(f"Regressores: {list(regressors.keys())}")
    logging.info(f"Targets: {targets}")
    logging.info(f"Paralelização (GRID_N_JOBS): {n_jobs}")

    results = []
    error_counts = defaultdict(int)
    success_count = 0

    # Loop Principal
    total_steps = len(window_sizes) * len(scenarios) * len(targets) * len(regressors)
    step = 0

    dataset_cache = {}

    for win_size in window_sizes:
        summary_file = f"summary_max_nwindows{win_size}.csv"

        for target in targets:
            for scenario_name, params in scenarios.items():
                cache_key = (win_size, target, scenario_name)
                if cache_key not in dataset_cache:
                    dataset_cache[cache_key] = _prepare_dataset(summary_file, target, params)

                dataset = dataset_cache[cache_key]
                if dataset is None or dataset.X.shape[1] == 0:
                    continue

                if parallel_enabled:
                    tasks = [
                        delayed(_evaluate_regressor)(
                            win_size,
                            target,
                            scenario_name,
                            params,
                            regressor_name,
                            regressor_instance,
                            dataset,
                            parallel_enabled,
                        )
                        for regressor_name, regressor_instance in regressors.items()
                    ]
                    outputs = Parallel(n_jobs=n_jobs, backend="threading")(tasks)
                else:
                    outputs = [
                        _evaluate_regressor(
                            win_size,
                            target,
                            scenario_name,
                            params,
                            regressor_name,
                            regressor_instance,
                            dataset,
                            parallel_enabled,
                        )
                        for regressor_name, regressor_instance in regressors.items()
                    ]

                for output in outputs:
                    step += 1
                    if output.get("ok"):
                        results.append(output["result"])
                        success_count += 1
                    else:
                        error_type = output.get("error_type", "UnknownError")
                        error_counts[error_type] += 1
                        context = output.get("context", ("?", "?", "?", "?"))
                        logging.error(
                            f"Erro ({error_type}) em W{context[0]}-{context[1]}-{context[2]}-{context[3]}: {output.get('error')}"
                        )

                    if step % 50 == 0 or step == total_steps:
                        print(f"[{step}/{total_steps}] Win: {win_size} | Target: {target} | Scenario: {scenario_name}", end="\r")

    logging.info("\n>>> Processamento concluído. Gerando relatórios...")

    # Estatísticas de execução
    logging.info(f"Experimentos bem-sucedidos: {success_count}/{total_steps}")
    logging.info(f"Total de erros: {sum(error_counts.values())}")
    if error_counts:
        logging.info("Erros por tipo:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {error_type}: {count}")

    # Salvar Tabela
    df_results = pd.DataFrame(results)
    if len(df_results) == 0:
        logging.error("ALERTA: Nenhum resultado foi gerado! Verifique os logs de erro acima.")
        return

    csv_path = OUTPUT_DIR / "grid_search_experiments.csv"
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Tabela salva em: {csv_path} ({len(df_results)} linhas)")

    # Gerar Gráficos
    generate_plots(df_results)

def generate_plots(df):
    sns.set_style("whitegrid")

    # Plot 1: MAE por Tamanho de Janela (Facetado por Target e Regressor)
    g = sns.FacetGrid(df, col="Target", row="Regressor", height=3, aspect=1.5, sharey=False)
    g.map_dataframe(sns.lineplot, x="WindowSize", y="MAE", hue="Scenario", marker="o", linewidth=2)
    g.add_legend(title="Features")
    g.set_axis_labels("Número de Janelas", "MAE (Erro Médio Absoluto)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.suptitle("Performance Preditiva por Configuração", y=1.02)

    plot_path = OUTPUT_DIR / "plot_grid_search_mae_all_models.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico MAE salvo em: {plot_path}")

    # Plot 2: Melhor R2 por Regressor e Target (cenário completo)
    full_scenario = "P95_Emotions_PHA_Controls"
    df_best_r2 = df[df["Scenario"] == full_scenario].groupby(["Regressor", "Target"])["R2"].max().unstack(level=0)
    if not df_best_r2.empty:
        plt.figure(figsize=(14, 8))
        sns.heatmap(df_best_r2, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
        plt.title("Melhor R² por Regressor e Target (Cenário: Emoções + PHA + Controles)")
        plt.xlabel("Regressor")
        plt.ylabel("Target")
        plt.tight_layout()
        plot_path_heatmap = OUTPUT_DIR / "plot_grid_search_heatmap_best_r2.png"
        plt.savefig(plot_path_heatmap, bbox_inches='tight')
        plt.close()
        print(f"Heatmap do melhor R² salvo em: {plot_path_heatmap}")

if __name__ == "__main__":
    run_comprehensive_search()
