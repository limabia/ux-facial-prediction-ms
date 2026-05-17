"""Funções utilitárias para modelagem (carregamento, anti-leakage, z-score)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config

REPO_ROOT = config.REPO_ROOT
FRAMES_DIR = config.FRAMES_DIR
OUTPUT_DIR = config.OUTPUT_DIR
EMOTIONS = config.EMOTIONS
CONTROL_COLUMNS = config.CONTROL_COLUMNS
UEQ_COLUMNS = config.UEQ_COLUMNS
TARGET_PREFIXES = config.TARGET_PREFIXES
UEQ_SCALES = config.UEQ_SCALES
PUEQ_GROUPS = config.PUEQ_GROUPS


@dataclass
class ModelingDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    games: np.ndarray
    feature_names: list[str]
    dataframe: pd.DataFrame


def resolve_summary_path(name: str) -> Path:
    path = FRAMES_DIR / name
    if path.exists():
        return path
    raise FileNotFoundError(f"Summary '{name}' não encontrado.")


def ensure_output_dir() -> None:
    config.ensure_dirs()


def _is_emotion_column(col: str) -> bool:
    """Verifica se a coluna é de emoção (ex: happy_mean_w01, sad_max_w05)."""
    col_lower = col.lower()
    return any(e in col_lower for e in EMOTIONS) and "_w" in col_lower


def _is_target_column(col: str) -> bool:
    """Verifica se a coluna é um target (PUEQ*, UEQ_*)."""
    return any(col.startswith(p) for p in TARGET_PREFIXES)


def _is_humor_column(col: str, prefixes: tuple[str, ...]) -> bool:
    """Verifica se a coluna é de humor pré/pós-jogo."""
    return any(col.startswith(p) for p in prefixes)


def _one_hot_categorical(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Gera one-hot encoding para variáveis categóricas nominais."""
    return pd.get_dummies(series.astype("category"), prefix=prefix, dummy_na=True)


def _parse_age(val) -> float:
    if pd.isna(val):
        return np.nan
    digits = "".join(c for c in str(val) if c.isdigit())
    return float(digits) if digits else np.nan


def zscore_by_group(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Aplica z-score por sujeito (grupo) nas features."""
    Xz = X.copy()
    for g in np.unique(groups):
        idx = groups == g
        if idx.sum() < 2:
            continue
        mu = Xz[idx].mean(axis=0)
        sigma = Xz[idx].std(axis=0, ddof=0)
        sigma[sigma == 0] = 1.0
        Xz[idx] = (Xz[idx] - mu) / sigma
    return Xz


def load_modeling_dataset(
    summary_name: str = "summary_max_nwindows20.csv",
    emotion_metrics: tuple[str, ...] = ("mean",),
    include_controls: bool = False,
    humor_prefixes: tuple[str, ...] = ("PHA",),
    target_column: str | list[str] = "UEQ_Overall",
    drop_neutral: bool = False,
    zscore_by_subject: bool = False,
) -> ModelingDataset:
    df = pd.read_csv(resolve_summary_path(summary_name))

    pueq_cols = [c for c in df.columns if c.startswith("PUEQ")]
    if "UEQ_Overall" not in df.columns and pueq_cols:
        df["UEQ_Overall"] = df[pueq_cols].mean(axis=1)
    if "UEQ_Pragmatic" not in df.columns:
        df["UEQ_Pragmatic"] = df[[c for c in pueq_cols if c in ["PUEQ1", "PUEQ2", "PUEQ3", "PUEQ4"]]].mean(axis=1)
    if "UEQ_Hedonic" not in df.columns:
        df["UEQ_Hedonic"] = df[[c for c in pueq_cols if c in ["PUEQ5", "PUEQ6", "PUEQ7", "PUEQ8"]]].mean(axis=1)

    features: list[str] = []

    for col in df.columns:
        if _is_emotion_column(col) and any(f"_{m}_" in col for m in emotion_metrics):
            if drop_neutral and "neutral" in col.lower():
                continue
            features.append(col)

    features.extend(col for col in df.columns if _is_humor_column(col, humor_prefixes))

    if include_controls:
        for col in CONTROL_COLUMNS:
            if col not in df.columns:
                continue
            if col == "idade":
                df[col] = df[col].apply(_parse_age)
                features.append(col)
            elif is_numeric_dtype(df[col]):
                features.append(col)
            else:
                dummies = _one_hot_categorical(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                features.extend(dummies.columns.tolist())

    # Anti-leakage: features não podem ser targets nem prefixos PUEQ/UEQ_*.
    targets_check = [target_column] if isinstance(target_column, str) else target_column
    for f in features:
        assert f not in targets_check, f"LEAKAGE: '{f}' é um target."
        assert not _is_target_column(f), f"LEAKAGE: '{f}' é um target (padrão PUEQ/UEQ_)."

    X = df[features].astype(float).values
    y = df[target_column].astype(float).values
    groups = df["idPessoa"].values
    games = df["jogo"].values

    if zscore_by_subject:
        X = zscore_by_group(X, groups)

    return ModelingDataset(
        X=X,
        y=y,
        groups=groups,
        games=games,
        feature_names=features,
        dataframe=df,
    )
