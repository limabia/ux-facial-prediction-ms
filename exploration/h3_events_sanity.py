#!/usr/bin/env python3
"""
H3: Sanity Check com Features Baseadas em Eventos.

Valida se features derivadas de eventos faciais (razões, máximos) conseguem
discriminar o contexto de jogo ou predizer a experiência do usuário.

Metodologia:
1. Usa dataset de eventos (mediapipe_events_dataset.csv).
2. Classificação LOSO para predizer o jogo.
3. Regressão LOSO para predizer UEQ Overall.

Uso:
    python exploration/h3_events_sanity.py

Saídas:
    - Logs com acurácia de classificação e R² de regressão.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data/processed/mediapipe_events_dataset.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    """Executa sanity check com features de eventos."""
    logging.info(">>> H3: Iniciando sanity check com features de eventos...")

    if not DATASET_PATH.exists():
        logging.error(f"Dataset não encontrado: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    # Feature Cols
    X_cols = [c for c in df.columns if c.endswith(("_ratio", "_max"))]
    y_col = "jogo"

    X = df[X_cols].values
    y = df[y_col].values
    groups = df["idPessoa"].values

    logging.info(f"Features (N={len(X_cols)}): {X_cols[:5]}...")

    # 1. Classificação (Contexto de Jogo)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    logo = LeaveOneGroupOut()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    y_true, y_pred = [], []
    for train_idx, test_idx in logo.split(X, y_enc, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        y_true.extend(y_test)
        y_pred.extend(pred)

    acc = accuracy_score(y_true, y_pred)
    logging.info(f"Classificação (Jogo) - Acurácia: {acc:.4f}")

    # 2. Regressão (UEQ Overall)
    if "Overall" not in df.columns:
        logging.warning("Coluna 'Overall' não encontrada. Pulando regressão.")
        return

    y_reg = df["Overall"].values
    pipeline_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    y_true_r, y_pred_r = [], []
    for train_idx, test_idx in logo.split(X, y_reg, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_reg[train_idx], y_reg[test_idx]
        pipeline_reg.fit(X_train, y_train)
        pred = pipeline_reg.predict(X_test)
        y_true_r.extend(y_test)
        y_pred_r.extend(pred)

    r2 = r2_score(y_true_r, y_pred_r)
    logging.info(f"Regressão (UEQ Overall) - R²: {r2:.4f}")

    logging.info(">>> H3: Sanity check concluído.")

if __name__ == "__main__":
    main()
