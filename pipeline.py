"""Orquestra o pipeline Face-API end-to-end por vídeo.

Processa cada `.mov` em `data/raw/` extraindo frames, rodando inferência
no módulo Node (face_expression_recognition), consolidando JSON → CSV e
apagando os JPGs intermediários para economizar disco. Ao final, dispara
`make summaries` para gerar os datasets por janela.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from modeling import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_command(command: str, cwd: Path | None = None, env: dict | None = None) -> None:
    logger.info("Executando: %s", command)
    try:
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)
        subprocess.run(command, shell=True, check=True, cwd=cwd, env=cmd_env)
    except subprocess.CalledProcessError as exc:
        logger.error("Erro ao executar comando: %s", exc)
        sys.exit(1)


class Pipeline:
    def __init__(self) -> None:
        self.repo_root = config.REPO_ROOT
        self.raw_dir = config.DATA_RAW_DIR
        self.frames_base_dir = config.FRAMES_DIR
        self.venv_activate = f"source {self.repo_root}/.venv/bin/activate &&"

    def get_videos(self) -> list[Path]:
        return sorted(self.raw_dir.glob("*.mov"))

    def _find_frames_dir(self, video_name: str) -> Path | None:
        candidates = list(self.frames_base_dir.glob(f"{video_name}*"))
        if not candidates:
            return None
        exact = self.frames_base_dir / video_name
        return exact if exact.exists() else max(candidates, key=os.path.getmtime)

    def process_video(self, video_path: Path, index: int, total: int) -> None:
        video_name = video_path.stem
        logger.info("[%d/%d] %s", index, total, video_name)

        existing = self._find_frames_dir(video_name)
        if existing and (existing / "expressions.csv").exists():
            logger.info("Já processado (expressions.csv encontrado). Pulando.")
            return

        logger.info("Extraindo frames...")
        run_command(
            f"{self.venv_activate} python frames_extract/main.py '{video_path}'",
            cwd=self.repo_root,
        )

        frames_dir = self._find_frames_dir(video_name)
        if frames_dir is None:
            logger.error("Pasta de frames não encontrada para %s após extração.", video_name)
            return

        logger.info("Inferindo emoções (Face-API)...")
        run_command(
            f"export SINGLE_FOLDER='{frames_dir}' && make infer",
            cwd=self.repo_root,
        )

        logger.info("Gerando CSV de expressões...")
        run_command(
            f"{self.venv_activate} python expressions_csv_exporter/json_to_csv.py "
            f"--folder '{frames_dir}'",
            cwd=self.repo_root,
        )

        deleted = 0
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img in frames_dir.glob(ext):
                img.unlink()
                deleted += 1
        logger.info("Removidos %d frames de imagem (JSONs preservados).", deleted)

    def run(self, limit: int | None = None) -> None:
        videos = self.get_videos()
        if not videos:
            logger.warning("Nenhum vídeo encontrado em data/raw/.")
            sys.exit(0)

        if limit:
            videos = videos[:limit]
            logger.info("Limitando a %d vídeo(s).", limit)

        logger.info("Iniciando pipeline para %d vídeos.", len(videos))
        for i, video_path in enumerate(videos, 1):
            self.process_video(video_path, i, len(videos))

        logger.info("Gerando summaries finais...")
        run_command(f"{self.venv_activate} make summaries", cwd=self.repo_root)
        logger.info("Pipeline concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="Número máximo de vídeos para processar (para testes).",
    )
    args = parser.parse_args()
    Pipeline().run(limit=args.limit)
