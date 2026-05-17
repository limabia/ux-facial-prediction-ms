#!/usr/bin/env python3
"""
Script para análise segmentada de desempenho preditivo.
Avalia o modelo Lasso P95 (melhor configuração) em subgrupos:
1. Por Gênero (Masculino vs Feminino)
2. Por Jogo (TunelRush, CatsDrop, PacMan)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from modeling.utils import OUTPUT_DIR, PUEQ_GROUPS, REPO_ROOT, UEQ_COLUMNS, load_modeling_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Normalizacao por sujeito foi escolhida como padrao.
USE_SUBJECT_ZSCORE = True
BEST_WINDOW = 2
BEST_SCENARIO_FALLBACK = "P95_Only"
TARGETS = list(UEQ_COLUMNS)
EXCLUDED_PARTICIPANTS = [21]  # Outlier com variância zero


def run_loso_experiment(X, y, groups):
    """Executa LOSO com Lasso e retorna métricas."""
    if len(np.unique(groups)) < 2:
        return {"mae": np.nan, "r2": np.nan, "n_samples": len(y)}

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Lasso(alpha=0.1, random_state=42, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    return {
        "mae": mean_absolute_error(y_true_all, y_pred_all),
        "r2": r2_score(y_true_all, y_pred_all),
        "n_samples": len(y)
    }


def resolve_best_scenario(target: str) -> str:
    grid_path = OUTPUT_DIR / "grid_search_experiments.csv"
    if not grid_path.exists():
        return BEST_SCENARIO_FALLBACK
    df = pd.read_csv(grid_path)
    subset = df[df["Target"] == target].copy()
    if subset.empty:
        return BEST_SCENARIO_FALLBACK
    best = subset.sort_values("MAE").iloc[0]
    return str(best["Scenario"])


def _scenario_params(scenario: str):
    """Resolve parâmetros do cenário de features."""
    include_controls = "Controls" in scenario
    mood_prefixes = ("PHA",) if "PHA" in scenario else ()
    return include_controls, mood_prefixes


def analyze_by_gender(target_name: str):
    """Analisa desempenho por gênero."""
    logging.info(f"=== Análise por Gênero ({target_name}) ===")
    scenario = resolve_best_scenario(target_name)
    include_controls, mood_prefixes = _scenario_params(scenario)

    dataset = load_modeling_dataset(
        summary_name=f"summary_max_nwindows{BEST_WINDOW}.csv",
        target_column=target_name,
        emotion_metrics=("p95",),
        humor_prefixes=mood_prefixes,
        include_controls=include_controls,
        drop_neutral=False,
        zscore_by_subject=USE_SUBJECT_ZSCORE,
    )

    df = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    df["y"] = dataset.y
    df["group"] = dataset.groups
    df["game"] = dataset.games

    # Carregar dados brutos para pegar o sexo
    raw_df = pd.read_csv(REPO_ROOT / "data/interim/frames" / f"summary_max_nwindows{BEST_WINDOW}.csv")

    # Merge para pegar sexo
    df["idPessoa"] = dataset.groups
    sex_mapping = raw_df.groupby("idPessoa")["sexo"].first().to_dict()
    df["sexo"] = df["idPessoa"].map(sex_mapping)

    # Remover outliers
    df = df[~df["idPessoa"].isin(EXCLUDED_PARTICIPANTS)]

    results = []
    for sex in ["masculino", "feminino"]:
        subset = df[df["sexo"] == sex]
        if len(subset) < 5:
            logging.warning(f"Poucos dados para {sex}: {len(subset)}")
            continue

        X = subset.drop(columns=["y", "group", "game", "idPessoa", "sexo"]).values
        y = subset["y"].values
        groups = subset["idPessoa"].values

        metrics = run_loso_experiment(X, y, groups)
        results.append({
            "Segmento": f"Gênero: {sex.capitalize()}",
            "Target": target_name,
            "N": metrics["n_samples"],
            "MAE": metrics["mae"],
            "R2": metrics["r2"]
        })
        logging.info(f"  {sex}: N={metrics['n_samples']}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")

    return results


def analyze_by_game(target_name: str):
    """Analisa desempenho por jogo."""
    logging.info(f"=== Análise por Jogo ({target_name}) ===")
    scenario = resolve_best_scenario(target_name)
    include_controls, mood_prefixes = _scenario_params(scenario)

    dataset = load_modeling_dataset(
        summary_name=f"summary_max_nwindows{BEST_WINDOW}.csv",
        target_column=target_name,
        emotion_metrics=("p95",),
        humor_prefixes=mood_prefixes,
        include_controls=include_controls,
        drop_neutral=False,
        zscore_by_subject=USE_SUBJECT_ZSCORE,
    )

    df = pd.DataFrame(dataset.X, columns=dataset.feature_names)
    df["y"] = dataset.y
    df["group"] = dataset.groups
    df["game"] = dataset.games

    # Remover outliers
    df = df[~df["group"].isin(EXCLUDED_PARTICIPANTS)]

    results = []
    for game in ["TunelRush", "CatsDrop", "PacMan"]:
        subset = df[df["game"] == game]
        if len(subset) < 5:
            logging.warning(f"Poucos dados para {game}: {len(subset)}")
            continue

        X = subset.drop(columns=["y", "group", "game"]).values
        y = subset["y"].values
        groups = subset["group"].values

        metrics = run_loso_experiment(X, y, groups)
        results.append({
            "Segmento": f"Jogo: {game}",
            "Target": target_name,
            "N": metrics["n_samples"],
            "MAE": metrics["mae"],
            "R2": metrics["r2"]
        })
        logging.info(f"  {game}: N={metrics['n_samples']}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")

    return results


def analyze_overall(target_name: str):
    """Baseline: todos os dados juntos."""
    logging.info(f"=== Baseline (Todos os dados) ({target_name}) ===")
    scenario = resolve_best_scenario(target_name)
    include_controls, mood_prefixes = _scenario_params(scenario)

    dataset = load_modeling_dataset(
        summary_name=f"summary_max_nwindows{BEST_WINDOW}.csv",
        target_column=target_name,
        emotion_metrics=("p95",),
        humor_prefixes=mood_prefixes,
        include_controls=include_controls,
        drop_neutral=False,
        zscore_by_subject=USE_SUBJECT_ZSCORE,
    )

    X = dataset.X
    y = dataset.y
    groups = dataset.groups

    # Remover outliers
    mask = ~np.isin(groups, EXCLUDED_PARTICIPANTS)
    X, y, groups = X[mask], y[mask], groups[mask]

    metrics = run_loso_experiment(X, y, groups)
    logging.info(f"  Overall: N={metrics['n_samples']}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")

    return [{
        "Segmento": "Baseline (Todos)",
        "Target": target_name,
        "N": metrics["n_samples"],
        "MAE": metrics["mae"],
        "R2": metrics["r2"]
    }]


def aggregate_by_scale(results_df: pd.DataFrame) -> pd.DataFrame:
    agg_rows = []
    for segmento in results_df["Segmento"].unique():
        for scale_name, items in PUEQ_GROUPS.items():
            subset = results_df[
                (results_df["Segmento"] == segmento)
                & (results_df["Target"].isin(items))
            ]
            if subset.empty:
                continue
            agg_rows.append(
                {
                    "Segmento": segmento,
                    "Target": scale_name,
                    "N": int(subset["N"].mean()),
                    "MAE": subset["MAE"].mean(),
                    "R2": subset["R2"].mean(),
                }
            )

    return pd.DataFrame(agg_rows)


if __name__ == "__main__":
    all_results = []

    for target_name in TARGETS:
        all_results.extend(analyze_overall(target_name))
        all_results.extend(analyze_by_gender(target_name))
        all_results.extend(analyze_by_game(target_name))

    # Salvar CSV por item
    results_items_df = pd.DataFrame(all_results)
    items_path = OUTPUT_DIR / "segmented_analysis_results_items.csv"
    results_items_df.to_csv(items_path, index=False)
    logging.info(f"\nResultados por item salvos em: {items_path}")

    # Agregar por escala UEQ
    results_df = aggregate_by_scale(results_items_df)
    output_path = OUTPUT_DIR / "segmented_analysis_results.csv"
    results_df.to_csv(output_path, index=False)
    logging.info(f"Resultados agregados salvos em: {output_path}")

    # Imprimir tabela formatada
    print("\n" + "="*60)
    print("RESULTADOS DA ANÁLISE SEGMENTADA (AGREGADO)")
    print("="*60)
    print(results_df.to_string(index=False))
