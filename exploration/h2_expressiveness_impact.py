#!/usr/bin/env python3
"""
H2: Impacto da Expressividade Facial na Predição de Contexto de Jogo.

Este script testa a hipótese de que participantes mais expressivos (maior
variabilidade nos blendshapes) permitem predições de contexto (jogo) mais
precisas do que participantes com expressões faciais mais neutras.

Metodologia:
1. Calcula a expressividade de cada participante (média dos STDs dos blendshapes).
2. Divide os participantes em dois grupos pelo mediano.
3. Treina modelos de classificação (LOSO) para cada subgrupo.
4. Compara a acurácia entre os grupos.

Uso:
    python exploration/h2_expressiveness_impact.py

Saídas:
    - Logs com acurácia por subgrupo (High/Low Expressiveness).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data/processed/mediapipe_modeling_dataset.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_subgroup(sub_df: pd.DataFrame, name: str) -> float:
    """Treina um classificador LOSO para um subgrupo e retorna a acurácia."""
    X_cols = [c for c in sub_df.columns if c.endswith(("_mean", "_max", "_std"))]
    X = sub_df[X_cols].values
    y = sub_df["jogo"].values
    groups = sub_df["idPessoa"].values

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
    logging.info(f"[{name}] Accuracy: {acc:.4f} (N={len(sub_df)})")
    return acc


def main() -> None:
    """Executa a análise de impacto da expressividade."""
    logging.info(">>> H2: Iniciando análise de impacto da expressividade...")

    if not DATASET_PATH.exists():
        logging.error(f"Dataset não encontrado: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    # 1. Calcula expressividade (média dos STDs de cada participante)
    std_cols = [c for c in df.columns if c.endswith("_std")]
    user_expressiveness = df.groupby("idPessoa")[std_cols].mean().mean(axis=1)

    median_expr = user_expressiveness.median()
    logging.info(f"Mediana da Expressividade: {median_expr:.6f}")

    # Divide participantes pelo mediano
    expressive_users = user_expressiveness[user_expressiveness >= median_expr].index
    stiff_users = user_expressiveness[user_expressiveness < median_expr].index

    logging.info(f"Participantes Expressivos (N={len(expressive_users)})")
    logging.info(f"Participantes Neutros (N={len(stiff_users)})")

    # 2. Treina e compara subgrupos
    logging.info("--- Comparando Subgrupos ---")

    df_exp = df[df["idPessoa"].isin(expressive_users)]
    train_subgroup(df_exp, "High Expressiveness")

    df_stiff = df[df["idPessoa"].isin(stiff_users)]
    train_subgroup(df_stiff, "Low Expressiveness")

    logging.info(">>> H2: Análise concluída.")

if __name__ == "__main__":
    main()
