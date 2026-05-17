#!/usr/bin/env python3
"""
Gera dataset de modelagem a partir de features MediaPipe.

Consolida os CSVs de blendshapes extraídos pelo MediaPipe, calcula
estatísticas por janela temporal (mean, max, std, p95) e faz merge com
scores UEQ e metadados demográficos.

Uso:
    python modeling/generate_mediapipe_dataset.py

Saídas:
    - data/processed/mediapipe_modeling_dataset.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DATASET = REPO_ROOT / "data/processed/mediapipe_modeling_dataset.csv"
MP_DIR = REPO_ROOT / "data/processed/mediapipe"
UEQ_DIR = REPO_ROOT / "data/raw/ueq"
N_WINDOWS = 3  # Match Face-API pipeline (w01, w02, w03)

def load_ueq_all() -> dict:
    """Carrega todos os arquivos UEQ e cria mapa (ParticipantID, Game) -> Scores."""
    ueq_map = {}
    for game in ["CatsDrop", "PacMan", "TunelRush"]:
        path = UEQ_DIR / f"UEQ-{game}.csv"
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                pid = row["idPessoa"]
                # Pragmatic = mean(1..4), Hedonic = mean(5..8), Overall = mean(1..8)
                prag = row[["PUEQ1", "PUEQ2", "PUEQ3", "PUEQ4"]].mean()
                hedo = row[["PUEQ5", "PUEQ6", "PUEQ7", "PUEQ8"]].mean()
                overall = row[["PUEQ1", "PUEQ2", "PUEQ3", "PUEQ4", "PUEQ5", "PUEQ6", "PUEQ7", "PUEQ8"]].mean()

                ueq_map[(pid, game)] = {
                    "Pragmatic": prag,
                    "Hedonic": hedo,
                    "Overall": overall,
                    **{f"PUEQ{i}": row[f"PUEQ{i}"] for i in range(1, 9)},
                    **{f"PHA{i}": row.get(f"PHA{i}", np.nan) for i in range(1, 8)},
                    **{f"PHD{i}": row.get(f"PHD{i}", np.nan) for i in range(1, 8)},
                    "sexo": row.get("sexo", "unknown"),
                    "idade": row.get("idade", "unknown"),
                    "familiaridade_com_jogos": row.get("familiaridade_com_jogos", "unknown"),
                    "formacao_academica": row.get("formacao_academica", "unknown"),
                }
    return ueq_map

def load_metadata_master() -> dict:
    """Carrega metadados e scores de humor do summary existente."""
    path = REPO_ROOT / "data/interim/frames/summary_max_nwindows1.csv"
    meta_map = {}
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            try:
                pid = int(row["idPessoa"])
            except (ValueError, TypeError):
                continue

            game = row["jogo"]

            affect = {f"PHA{i}": row.get(f"PHA{i}", np.nan) for i in range(1, 8)}
            affect.update({f"PHD{i}": row.get(f"PHD{i}", np.nan) for i in range(1, 8)})

            meta = {
                "sexo": row.get("sexo", "unknown"),
                "idade": row.get("idade", "unknown"),
                "familiaridade_com_jogos": row.get("familiaridade_com_jogos", "unknown"),
                "formacao_academica": row.get("formacao_academica", "unknown"),
            }

            meta_map[(pid, game)] = {**affect, **meta}

    return meta_map

def process_features() -> None:
    """Processa arquivos MediaPipe e gera dataset consolidado."""
    print(">>> Iniciando geração do dataset MediaPipe...")

    mp_files = list(MP_DIR.glob("*.csv"))
    print(f"Arquivos encontrados: {len(mp_files)}")

    if not mp_files:
        print(f"Nenhum arquivo encontrado em {MP_DIR}")
        return

    ueq_map = load_ueq_all()
    meta_map = load_metadata_master()

    data_points = []

    for csv_file in mp_files:
        basename = csv_file.name
        # Parse filename: {id}-{Game}-{Name}-{Date}.csv
        try:
            parts = basename.split("-")
            pid = int(parts[0])
            game = parts[1]
        except (ValueError, IndexError):
            print(f"Pulando arquivo com nome inválido: {basename}")
            continue

        if (pid, game) not in ueq_map:
            print(f"Aviso: Sem UEQ para P{pid} {game}")
            continue

        targets = ueq_map[(pid, game)]

        df = pd.read_csv(csv_file)

        ignore = {"frame", "timestamp_ms", "face_detected"}
        feature_cols = [c for c in df.columns if c not in ignore]

        if df.empty or "face_detected" not in df.columns:
            continue

        active_df = df[df["face_detected"]]

        if active_df.empty:
            print(f"Nenhuma face detectada em {basename}")
            continue

        row = {
            "idPessoa": pid,
            "jogo": game,
            "file": basename,
            "frames_total": len(df),
            "frames_active": len(active_df),
        }

        row.update(targets)

        if (pid, game) in meta_map:
            row.update(meta_map[(pid, game)])
        else:
            print(f"Aviso: Sem metadados para P{pid} {game}")
            for i in range(1, 8):
                row[f"PHA{i}"] = np.nan
                row[f"PHD{i}"] = np.nan
            row["sexo"] = "unknown"
            row["idade"] = "unknown"
            row["familiaridade_com_jogos"] = "unknown"
            row["formacao_academica"] = "unknown"

        # Compute stats per window
        window_size = len(active_df) // N_WINDOWS

        for w_idx in range(N_WINDOWS):
            start_idx = w_idx * window_size
            end_idx = (w_idx + 1) * window_size if w_idx < N_WINDOWS - 1 else len(active_df)

            window_df = active_df.iloc[start_idx:end_idx]

            if window_df.empty:
                for col in feature_cols:
                    row[f"{col}_mean_w{w_idx+1:02d}"] = np.nan
                    row[f"{col}_max_w{w_idx+1:02d}"] = np.nan
                    row[f"{col}_std_w{w_idx+1:02d}"] = np.nan
                    row[f"{col}_p95_w{w_idx+1:02d}"] = np.nan
                continue

            stats = window_df[feature_cols].agg(["mean", "max", "std"])

            for col in feature_cols:
                row[f"{col}_mean_w{w_idx+1:02d}"] = stats.loc["mean", col]
                row[f"{col}_max_w{w_idx+1:02d}"] = stats.loc["max", col]
                row[f"{col}_std_w{w_idx+1:02d}"] = stats.loc["std", col]
                row[f"{col}_p95_w{w_idx+1:02d}"] = window_df[col].quantile(0.95)

        data_points.append(row)

    if not data_points:
        print("Nenhum dado válido gerado.")
        return

    # Garante diretório de saída
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(data_points)
    out_df.to_csv(OUTPUT_DATASET, index=False)
    print(f"Dataset salvo com {len(out_df)} sessões em {OUTPUT_DATASET}")
    print(f"Shape: {out_df.shape}")

if __name__ == "__main__":
    process_features()
