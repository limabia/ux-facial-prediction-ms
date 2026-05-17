#!/usr/bin/env python3
"""
Sanity Check: Classificação de Jogo com MediaPipe Features.

Valida se features faciais MediaPipe conseguem distinguir o contexto de jogo,
servindo como sanity check para verificar se há sinal discriminativo nos dados.

Metodologia:
- Classificação LOSO com Random Forest.
- Report de acurácia, matriz de confusão e top features.

Uso:
    python exploration/sanity_check_predict_game.py

Saídas:
    - Logs com métricas de classificação e features mais importantes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data/processed/mediapipe_modeling_dataset.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    """Executa sanity check de classificação de jogo."""
    logging.info(">>> Sanity Check: Classificação de Jogo com MediaPipe...")

    if not DATASET_PATH.exists():
        logging.error(f"Dataset não encontrado: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    X_cols = [c for c in df.columns if c.endswith(("_mean", "_max", "_std"))]
    y_col = "jogo"

    X = df[X_cols].values
    y = df[y_col].values
    groups = df["idPessoa"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    logging.info(f"Classes: {classes}")

    logo = LeaveOneGroupOut()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in logo.split(X, y_enc, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    acc = accuracy_score(y_true_all, y_pred_all)
    logging.info(f"Acurácia Geral: {acc:.4f}")
    logging.info("\nClassification Report:\n" + classification_report(y_true_all, y_pred_all, target_names=classes))
    logging.info("\nMatriz de Confusão:\n" + str(confusion_matrix(y_true_all, y_pred_all)))

    # Feature Importance
    rf = pipeline.named_steps["clf"]
    rf.fit(X, y_enc)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    logging.info("\nTop 10 Features discriminativas:")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        logging.info(f"{i+1}. {X_cols[idx]} ({importances[idx]:.4f})")

    logging.info(">>> Sanity check concluído.")

if __name__ == "__main__":
    main()
