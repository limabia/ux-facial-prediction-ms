#!/usr/bin/env python3
"""
Extração completa de blendshapes MediaPipe para todos os vídeos.

Processa todos os vídeos em data/raw/ usando o MediaPipe FaceLandmarker
e salva CSVs individuais com blendshapes frame-a-frame.

Requisitos:
    - Modelo face_landmarker.task em exploration/

Uso:
    python features/extract_mediapipe_full.py

Saídas:
    - data/processed/mediapipe/*.csv (um por vídeo)
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = REPO_ROOT / "exploration/face_landmarker.task"
OUTPUT_DIR = REPO_ROOT / "data/processed/mediapipe"
RAW_DIR = REPO_ROOT / "data/raw"
LOG_FILE = REPO_ROOT / "exploration/outputs/extraction_full.log"

# Setup logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def process_video(video_path: Path) -> str:
    """Processa um vídeo e extrai blendshapes."""
    output_csv = OUTPUT_DIR / f"{video_path.stem}.csv"

    if output_csv.exists():
        logging.info(f"Pulando {video_path.name}, já existe.")
        return f"Skipped: {video_path.name}"

    logging.info(f"Processando {video_path.name}...")

    try:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Erro ao abrir {video_path}")
            return f"Error: {video_path.name}"

        data_rows = []

        with FaceLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            last_timestamp_ms = -1

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                frame_idx += 1
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                )

                try:
                    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as e:
                    logging.warning(f"Erro no frame {frame_idx} de {video_path.name}: {e}")
                    continue

                row = {"frame": frame_idx, "timestamp_ms": timestamp_ms, "face_detected": False}

                if detection_result.face_blendshapes:
                    row["face_detected"] = True
                    blendshapes = detection_result.face_blendshapes[0]
                    for shape in blendshapes:
                        row[shape.category_name] = shape.score

                data_rows.append(row)

        cap.release()

        if data_rows:
            df = pd.DataFrame(data_rows)
            df.to_csv(output_csv, index=False)
            logging.info(f"Salvos {len(df)} frames em {output_csv}")
            return f"Success: {video_path.name} ({len(df)} frames)"
        else:
            logging.warning(f"Nenhum frame processado para {video_path.name}")
            return f"Empty: {video_path.name}"

    except Exception as e:
        logging.error(f"Falha ao processar {video_path.name}: {e}")
        return f"Failed: {video_path.name}"


def main() -> None:
    """Executa extração para todos os vídeos."""
    logging.info(">>> Iniciando extração completa MediaPipe...")

    if not MODEL_PATH.exists():
        logging.error(f"Modelo não encontrado: {MODEL_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extensions = ["*.mov", "*.mp4", "*.avi", "*.mkv"]
    videos = []
    for ext in extensions:
        videos.extend(RAW_DIR.glob(ext))
        videos.extend(RAW_DIR.glob(f"*/{ext}"))

    videos = sorted(set(videos))
    logging.info(f"Encontrados {len(videos)} vídeos")

    for vid in videos:
        res = process_video(vid)
        print(res)

    logging.info(">>> Extração completa concluída.")


if __name__ == "__main__":
    main()
