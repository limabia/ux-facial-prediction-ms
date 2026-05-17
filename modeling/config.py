from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
FRAMES_DIR = REPO_ROOT / "data" / "interim" / "frames"
OUTPUT_DIR = REPO_ROOT / "exploration" / "outputs"

EMOTIONS = ("happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral")

CONTROL_COLUMNS = (
    "sexo",
    "idade",
    "formacao_academica",
    "familiaridade_com_jogos",
    "usa_oculos",
    "jogo",
)

UEQ_COLUMNS = tuple(f"PUEQ{i}" for i in range(1, 9))
HUMOR_PREFIXES = ("PHA",)
HUMOR_COLUMNS = tuple(f"{prefix}{i}" for prefix in HUMOR_PREFIXES for i in range(1, 8))

TARGET_PREFIXES = ("PUEQ", "UEQ_")
UEQ_SCALES = ("UEQ_Overall", "UEQ_Pragmatic", "UEQ_Hedonic")

# Agrupamento dos itens UEQ-S em escalas (pragmática 1-4, hedônica 5-8).
PUEQ_GROUPS: dict[str, tuple[str, ...]] = {
    "UEQ_Pragmatic": tuple(f"PUEQ{i}" for i in range(1, 5)),
    "UEQ_Hedonic": tuple(f"PUEQ{i}" for i in range(5, 9)),
    "UEQ_Overall": UEQ_COLUMNS,
}


def ensure_dirs() -> None:
    for d in [FRAMES_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
