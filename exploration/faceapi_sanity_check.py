#!/usr/bin/env python3
"""
Sanity Check: Classificação de Jogo com Face-API Features.

Valida se features de emoções do Face-API (pipeline original) conseguem
distinguir o contexto de jogo, servindo como baseline de comparação.

Metodologia:
- Usa summary com janelas (nwindows1).
- Classificação LOSO com Random Forest.
- Report de acurácia e matriz de confusão.

Uso:
    python exploration/faceapi_sanity_check.py

Saídas:
    - Logs com métricas de classificação.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data/interim/frames/summary_max_nwindows1.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    """Executa sanity check com features Face-API."""
    logging.info(">>> Sanity Check: Classificação de Jogo com Face-API...")

    if not DATASET_PATH.exists():
        logging.error(f"Dataset não encontrado: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    # Emoções do Face-API
    emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
    X_cols = [c for c in df.columns if any(c.startswith(e) for e in emotions) and "w01" in c]
    y_col = "jogo"

    logging.info(f"Features Face-API selecionadas: {len(X_cols)}")

    df_clean = df.dropna(subset=[*X_cols, y_col])

    X = df_clean[X_cols].values
    y = df_clean[y_col].values
    groups = df_clean["idPessoa"].values

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
    logging.info(f"Acurácia Face-API: {acc:.4f}")
    logging.info("\n" + classification_report(y_true_all, y_pred_all, target_names=classes))
    logging.info("\nMatriz de Confusão:\n" + str(confusion_matrix(y_true_all, y_pred_all)))

    logging.info(">>> Sanity check Face-API concluído.")

if __name__ == "__main__":
    main()
