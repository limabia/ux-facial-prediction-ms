#!/usr/bin/env python3
"""Conversor de expressões faciais JSON para CSV.

Lê pastas de sessão em ``data/interim/frames/`` (configurável via env
``DATA_INTERIM_DIR``), processa os JSONs gerados pelo Face-API.js e gera
um ``expressions.csv`` consolidado por sessão.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from modeling import config

EMOTIONS = ("neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised")
FRAME_NUMBER_RE = re.compile(r"frame_(\d+)\.json")


def _resolve_frames_dir() -> Path:
    env = os.getenv("DATA_INTERIM_DIR")
    if env:
        candidate = Path(env).resolve()
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return config.FRAMES_DIR


FRAMES_DIR = _resolve_frames_dir()


def extract_frame_number(filename: str) -> int:
    match = FRAME_NUMBER_RE.search(filename)
    return int(match.group(1)) if match else -1


def process_experiment_folder(experiment_path: Path) -> None:
    print(f"Processando: {experiment_path.name}")

    json_files = sorted(
        (f for f in os.listdir(experiment_path) if f.endswith(".json")),
        key=extract_frame_number,
    )

    rows = []
    for jf in json_files:
        try:
            with (experiment_path / jf).open() as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"  Erro ao processar {jf}: {exc}")
            continue

        if not (data and isinstance(data, list) and data[0].get("expressions")):
            print(f"  Aviso: {jf} sem dados de expressões válidos")
            continue

        expr = data[0]["expressions"]
        rows.append({"frame": Path(jf).stem, **{e: expr.get(e, 0) for e in EMOTIONS}})

    if not rows:
        print("  Nenhum dado de expressão encontrado")
        return

    csv_path = experiment_path / "expressions.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", *EMOTIONS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV criado: {csv_path} ({len(rows)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        type=Path,
        help="Caminho para uma pasta de sessão específica (padrão: processa todas).",
    )
    args = parser.parse_args()

    if args.folder:
        folder = args.folder.resolve()
        if not folder.is_dir():
            print(f"Erro: pasta {folder} não encontrada.", file=sys.stderr)
            sys.exit(1)
        process_experiment_folder(folder)
        return

    if not FRAMES_DIR.is_dir():
        print(
            f"Erro: diretório {FRAMES_DIR} não encontrado. "
            "Defina DATA_INTERIM_DIR se moveu os dados.",
            file=sys.stderr,
        )
        sys.exit(1)

    folders = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]
    if not folders:
        print("Nenhuma pasta de sessão encontrada.", file=sys.stderr)
        sys.exit(1)

    print(f"Encontradas {len(folders)} pastas de sessão.")
    for folder in folders:
        process_experiment_folder(folder)
    print("Processamento concluído.")


if __name__ == "__main__":
    main()
