#!/usr/bin/env python3
"""
H4: Predição Multimodal de Experiência do Usuário.

Compara três cenários de features para predizer escalas UEQ:
1. Somente MediaPipe (blendshapes faciais).
2. Somente PHA/PHD (humor auto-reportado).
3. Combinado (MediaPipe + PHA + Metadados demográficos).

Metodologia:
- Validação LOSO (Leave-One-Subject-Out).
- Random Forest Regressor para cada cenário.
- Métricas: R² por target (Pragmatic, Hedonic, Overall).

Uso:
    python exploration/h4_multimodal_prediction.py

Saídas:
    - Logs com R² por cenário e target.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data/processed/mediapipe_modeling_dataset.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_age(x) -> float:
    """Extrai idade numérica de strings como '26 anos'."""
    try:
        return int(str(x).split()[0])
    except (ValueError, IndexError):
        return np.nan


def main() -> None:
    """Executa comparação de cenários multimodais."""
    logging.info(">>> H4: Iniciando predição multimodal...")

    if not DATASET_PATH.exists():
        logging.error(f"Dataset não encontrado: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)

    # Define Feature Sets
    mp_cols = [c for c in df.columns if c.endswith(("_mean", "_max", "_std"))]

    pha_cols = [f"PHA{i}" for i in range(1, 8)] + [f"PHD{i}" for i in range(1, 8)]
    pha_cols = [c for c in pha_cols if c in df.columns]

    cat_cols = ["sexo", "familiaridade_com_jogos", "formacao_academica"]
    target_cols = ["Overall", "Pragmatic", "Hedonic"]

    # Preprocessing
    df["idade_num"] = df["idade"].apply(parse_age)

    for target in target_cols:
        if target not in df.columns:
            continue

        logging.info(f"--- Target: {target} ---")

        y = df[target].values
        groups = df["idPessoa"].values

        # Cenário 1: MediaPipe Only
        X_mp = df[mp_cols].values
        r2_mp = run_cv(X_mp, y, groups, StandardScaler())
        logging.info(f"  [MP Only] R²: {r2_mp:.4f}")

        # Cenário 2: PHA Only
        X_pha = df[pha_cols].values
        imp = SimpleImputer(strategy="median")
        X_pha = imp.fit_transform(X_pha)
        r2_pha = run_cv(X_pha, y, groups, StandardScaler())
        logging.info(f"  [PHA Only] R²: {r2_pha:.4f}")

        # Cenário 3: Combinado
        X_combined_df = df[mp_cols + pha_cols + cat_cols + ["idade_num"]]

        numeric_features = mp_cols + pha_cols + ["idade_num"]
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, cat_cols),
        ])

        r2_comb = run_cv_pipeline(X_combined_df, y, groups, preprocessor)
        logging.info(f"  [Combined] R²: {r2_comb:.4f}")

    logging.info(">>> H4: Predição multimodal concluída.")


def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, scaler) -> float:
    """Executa validação LOSO com arrays numpy."""
    logo = LeaveOneGroupOut()
    pipeline = Pipeline([
        ("scaler", scaler),
        ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    y_true, y_pred = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Handle NaN y
        if np.isnan(y_train).any():
            mask = ~np.isnan(y_train)
            X_train = X_train[mask]
            y_train = y_train[mask]

        if len(y_train) == 0:
            continue

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        y_true.extend(y_test)
        y_pred.extend(pred)

    return r2_score(y_true, y_pred)


def run_cv_pipeline(X_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, preprocessor) -> float:
    """Executa validação LOSO com DataFrame e preprocessor."""
    logo = LeaveOneGroupOut()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    y_true, y_pred = [], []
    for train_idx, test_idx in logo.split(X_df, y, groups):
        X_train = X_df.iloc[train_idx]
        X_test = X_df.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Drop NaNs in target
        mask = ~np.isnan(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        if len(y_train) == 0:
            continue

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        y_true.extend(y_test)
        y_pred.extend(pred)

    return r2_score(y_true, y_pred)

if __name__ == "__main__":
    main()
