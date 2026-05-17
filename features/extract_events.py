#!/usr/bin/env python3
"""
Extrai features baseadas em eventos dos dados MediaPipe.

Define thresholds para Action Units (AUs) e calcula a proporção de frames
onde cada AU ultrapassa o threshold, além do valor máximo.

Uso:
    python features/extract_events.py

Saídas:
    - data/processed/mediapipe_events_dataset.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

MP_DIR = REPO_ROOT / "data/processed/mediapipe"
OUTPUT_DATASET = REPO_ROOT / "data/processed/mediapipe_events_dataset.csv"
UEQ_DIR = REPO_ROOT / "data/raw/ueq"

# Thresholds para eventos (AUs)
THRESHOLDS = {
    "mouthSmileLeft": 0.5,
    "mouthSmileRight": 0.5,
    "browDownLeft": 0.5,
    "browDownRight": 0.5,
    "jawOpen": 0.3,
    "eyeSquintLeft": 0.4,
    "eyeSquintRight": 0.4,
    "mouthPucker": 0.5,
    "mouthDimpleLeft": 0.5,
    "mouthDimpleRight": 0.5,
}


def load_ueq_all() -> dict:
    """Carrega todos os arquivos UEQ."""
    ueq_map = {}
    for game in ["CatsDrop", "PacMan", "TunelRush"]:
        path = UEQ_DIR / f"UEQ-{game}.csv"
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                pid = row["idPessoa"]
                prag = row[["PUEQ1", "PUEQ2", "PUEQ3", "PUEQ4"]].mean()
                hedo = row[["PUEQ5", "PUEQ6", "PUEQ7", "PUEQ8"]].mean()
                overall = row[["PUEQ1", "PUEQ2", "PUEQ3", "PUEQ4", "PUEQ5", "PUEQ6", "PUEQ7", "PUEQ8"]].mean()
                ueq_map[(pid, game)] = {
                    "Pragmatic": prag,
                    "Hedonic": hedo,
                    "Overall": overall,
                    "PUEQ1": row["PUEQ1"],
                    "PUEQ7": row["PUEQ7"],
                }
    return ueq_map


def extract_events() -> None:
    """Extrai features de eventos dos dados MediaPipe."""
    print(">>> Extraindo features de eventos...")

    mp_files = list(MP_DIR.glob("*.csv"))
    if not mp_files:
        print(f"Nenhum arquivo encontrado em {MP_DIR}")
        return

    ueq_map = load_ueq_all()
    data_points = []

    for csv_file in mp_files:
        basename = csv_file.name
        try:
            parts = basename.split("-")
            pid = int(parts[0])
            game = parts[1]
        except (ValueError, IndexError):
            continue

        if (pid, game) not in ueq_map:
            continue

        df = pd.read_csv(csv_file)
        if df.empty or "face_detected" not in df.columns:
            continue

        active_df = df[df["face_detected"]]
        if active_df.empty:
            continue

        row = {"idPessoa": pid, "jogo": game}
        row.update(ueq_map[(pid, game)])

        for feature, thresh in THRESHOLDS.items():
            if feature not in active_df.columns:
                continue

            count = (active_df[feature] > thresh).sum()
            ratio = count / len(active_df)
            max_val = active_df[feature].max()

            row[f"{feature}_ratio"] = ratio
            row[f"{feature}_max"] = max_val

        data_points.append(row)

    if not data_points:
        print("Nenhum dado válido gerado.")
        return

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(data_points)
    out_df.to_csv(OUTPUT_DATASET, index=False)
    print(f"Dataset de eventos salvo: {len(out_df)} linhas em {OUTPUT_DATASET}")


if __name__ == "__main__":
    extract_events()
