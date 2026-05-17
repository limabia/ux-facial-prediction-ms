"""Extrai os primeiros 180 s de frames de um vídeo .mov.

Uso:
    python frames_extract/main.py <caminho.mov>
    python frames_extract/main.py            # menu curses para escolher de data/raw/
"""

import curses
import sys
from curses import wrapper
from pathlib import Path

import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
FRAMES_DIR = REPO_ROOT / "data" / "interim" / "frames"
DURACAO_SEGUNDOS = 180


def list_mov_files() -> list[Path]:
    if not RAW_DIR.is_dir():
        return []
    return sorted(RAW_DIR.glob("*.mov"))


def pick_from_menu(files: list[Path]) -> Path:
    def _menu(stdscr: "curses._CursesWindow") -> Path:
        cursor = 0
        while True:
            stdscr.clear()
            stdscr.addstr(f"Selecione um arquivo em: {RAW_DIR}\n", curses.A_BOLD)
            for idx, f in enumerate(files):
                style = curses.A_REVERSE if idx == cursor else curses.A_NORMAL
                stdscr.addstr(f"{'>' if idx == cursor else ' '} {f.name}\n", style)
            stdscr.refresh()
            key = stdscr.getch()
            if key == curses.KEY_UP and cursor > 0:
                cursor -= 1
            elif key == curses.KEY_DOWN and cursor < len(files) - 1:
                cursor += 1
            elif key == 10:  # Enter
                return files[cursor]

    return wrapper(_menu)


def extract_frames(video_path: Path) -> None:
    output_dir = FRAMES_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = min(int(fps * DURACAO_SEGUNDOS), total_frames)
    print(f"FPS={fps}, total={total_frames}, alvo={target_frames}")

    count = 0
    with tqdm(total=target_frames, desc="Extraindo frames", unit="frame") as pbar:
        while count < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(output_dir / f"frame_{count:04d}.jpg"), frame)
            count += 1
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Frames extraídos: {count} → {output_dir}")


def main() -> None:
    if len(sys.argv) > 1:
        video = Path(sys.argv[1]).resolve()
    else:
        files = list_mov_files()
        if not files:
            print(f"Nenhum arquivo .mov encontrado em '{RAW_DIR}'.")
            sys.exit(1)
        video = pick_from_menu(files)
    extract_frames(video)


if __name__ == "__main__":
    main()
