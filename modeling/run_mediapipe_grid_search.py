#!/usr/bin/env python3
"""
Grid Search de Modelos com Features MediaPipe.

Executa experimentos de regressão LOSO comparando múltiplos modelos
usando features extraídas do MediaPipe (blendshapes faciais).

Modelos testados: Baseline, Lasso, Ridge, ElasticNet, SVR, XGBoost, RF.
Targets: PUEQ1-8, Pragmatic, Hedonic, Overall.

Uso:
    python modeling/run_mediapipe_grid_search.py

Saídas:
    - exploration/outputs/mediapipe_results/mediapipe_grid_search_results.csv
    - exploration/outputs/mediapipe_results/performance_comparison.png
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data/processed/mediapipe_modeling_dataset.csv"
OUTPUT_DIR = REPO_ROOT / "exploration/outputs/mediapipe_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    """Carrega dataset e retorna DataFrame, features e targets."""
    if not DATASET_PATH.exists():
        logging.error(f"Dataset não encontrado: {DATASET_PATH}")
        return None, None, None

    df = pd.read_csv(DATASET_PATH)

    # Feature Columns: containing _mean_w, _max_w, _std_w, _p95_w (windowed features)
    feature_cols = [c for c in df.columns if "_w0" in c]

    # Target Cols
    target_cols = [f"PUEQ{i}" for i in range(1, 9)] + ["Pragmatic", "Hedonic", "Overall"]

    return df, feature_cols, target_cols

def run_experiments() -> None:
    """Executa grid search com múltiplos modelos e targets."""
    logging.info(">>> Iniciando Grid Search MediaPipe...")

    df, feature_cols, target_cols = load_data()
    if df is None:
        return

    logging.info(f"Dataset carregado: {len(df)} amostras, {len(feature_cols)} features")

    models = {
        "Baseline": DummyRegressor(strategy="mean"),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "SVR_RBF": SVR(C=1.0, epsilon=0.1),
        "XGBTuned": xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        "RF": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    }

    results = []

    for target in target_cols:
        if target not in df.columns:
            continue

        y = df[target].values
        groups = df["idPessoa"].values
        X = df[feature_cols].values

        # Remove NaNs
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        groups = groups[mask]

        # Excluir outliers (P21 tem variância zero)
        outlier_mask = groups != 21
        X = X[outlier_mask]
        y = y[outlier_mask]
        groups = groups[outlier_mask]

        logo = LeaveOneGroupOut()

        for name, model in models.items():
            y_true_all = []
            y_pred_all = []

            # Pipeline: Scale -> PCA -> Model
            if name == "Baseline":
                pipeline = model
            else:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.95)),
                    ("model", model),
                ])

            try:
                for train_idx, test_idx in logo.split(X, y, groups):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)

                    y_true_all.extend(y_test)
                    y_pred_all.extend(preds)

                mae = mean_absolute_error(y_true_all, y_pred_all)
                r2 = r2_score(y_true_all, y_pred_all)

                logging.info(f"Target: {target} | Model: {name} | R²: {r2:.4f} | MAE: {mae:.4f}")

                results.append({
                    "Target": target,
                    "Model": name,
                    "R2": r2,
                    "MAE": mae,
                })

            except Exception as e:
                logging.error(f"Erro {target}-{name}: {e}")

    # Salvar resultados
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "mediapipe_grid_search_results.csv", index=False)

    # Gerar gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x="Target", y="R2", hue="Model")
    plt.title("MediaPipe Features Performance (PCA 95%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_comparison.png")
    plt.close()

    logging.info(">>> Grid Search MediaPipe concluído.")

if __name__ == "__main__":
    run_experiments()
