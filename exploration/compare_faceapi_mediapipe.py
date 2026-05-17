#!/usr/bin/env python3
"""
Comparação Visual: Face-API vs MediaPipe.

Gera gráficos comparando as séries temporais de emoções/blendshapes
extraídos pelo Face-API e pelo MediaPipe para o mesmo vídeo.

Uso:
    python exploration/compare_faceapi_mediapipe.py

Saídas:
    - exploration/outputs/comparison_*.png (um por vídeo)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "exploration/outputs"


def load_faceapi(folder_path: Path) -> pd.DataFrame:
    """Carrega dados do Face-API de uma pasta de JSONs."""
    json_files = sorted(folder_path.glob("frame_*.json"))
    print(f"Carregando {len(json_files)} arquivos Face-API de {folder_path}...")

    rows = []
    for jf in json_files:
        try:
            frame_idx = int(jf.stem.replace("frame_", ""))
        except ValueError:
            continue

        try:
            data = json.loads(jf.read_text())
            if data and len(data) > 0:
                expressions = data[0].get("expressions", {})
                row = {"frame": frame_idx}
                for k, v in expressions.items():
                    row[f"faceapi_{k}"] = v
                rows.append(row)
        except json.JSONDecodeError:
            pass

    return pd.DataFrame(rows)


def load_mediapipe(csv_path: Path) -> pd.DataFrame:
    """Carrega dados do MediaPipe de um CSV."""
    return pd.read_csv(csv_path)


def compare_video(faceapi_folder: Path, mediapipe_path: Path, output_img: Path) -> None:
    """Gera gráfico comparativo para um vídeo."""
    print(f"Comparando {faceapi_folder.name} vs {mediapipe_path.name}")

    df_api = load_faceapi(faceapi_folder)
    df_mp = load_mediapipe(mediapipe_path)

    if df_api.empty:
        print("Aviso: DataFrame Face-API vazio.")
        return

    merged = pd.merge(df_mp, df_api, on="frame", how="inner")

    if merged.empty:
        print("Aviso: Nenhum frame em comum encontrado.")
        return

    _fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Happy vs Smile
    axs[0].plot(merged["frame"], merged["faceapi_happy"], label="FaceAPI Happy", alpha=0.7)
    axs[0].plot(merged["frame"], merged.get("mouthSmileLeft", 0), label="MP SmileLeft", alpha=0.5, linestyle="--")
    axs[0].plot(merged["frame"], merged.get("mouthSmileRight", 0), label="MP SmileRight", alpha=0.5, linestyle="--")
    axs[0].set_title("Happiness / Smile")
    axs[0].legend()

    # 2. Surprise vs BrowUp
    axs[1].plot(merged["frame"], merged.get("faceapi_surprised", 0), label="FaceAPI Surprise", alpha=0.7)
    axs[1].plot(merged["frame"], merged.get("browOuterUpLeft", 0), label="MP BrowOuterUpLeft", alpha=0.5, linestyle="--")
    axs[1].plot(merged["frame"], merged.get("browInnerUp", 0), label="MP BrowInnerUp", alpha=0.5, linestyle="--")
    axs[1].set_title("Surprise / Brow Up")
    axs[1].legend()

    # 3. Neutral vs Activity
    axs[2].plot(merged["frame"], merged["faceapi_neutral"], label="FaceAPI Neutral", alpha=0.7)
    axs[2].plot(merged["frame"], merged.get("jawOpen", 0), label="MP JawOpen", alpha=0.5, linestyle="--")
    axs[2].plot(merged["frame"], merged.get("eyeBlinkLeft", 0), label="MP BlinkLeft", alpha=0.5, linestyle="--")
    axs[2].set_title("Neutral vs Activity")
    axs[2].legend()

    plt.xlabel("Frame")
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"Gráfico salvo em {output_img}")


def main() -> None:
    """Executa comparação para todos os arquivos MediaPipe disponíveis."""
    print(">>> Comparando Face-API vs MediaPipe...")

    mp_files = sorted(OUTPUT_DIR.glob("mediapipe_*.csv"))

    if not mp_files:
        print("Nenhum arquivo MediaPipe encontrado.")
        return

    for mp_file in mp_files:
        basename = mp_file.stem.replace("mediapipe_", "")
        faceapi_folder = REPO_ROOT / "data/interim/frames" / basename

        if faceapi_folder.exists():
            output_img = OUTPUT_DIR / f"comparison_{basename}.png"
            compare_video(faceapi_folder, mp_file, output_img)
        else:
            print(f"Pasta Face-API não encontrada para {basename}")

    print(">>> Comparação concluída.")

if __name__ == "__main__":
    main()
