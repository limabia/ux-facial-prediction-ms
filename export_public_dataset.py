"""Exporta a versão pública (anonimizada) do dataset FaceUX para ``dataset/``.

Copia apenas dados derivados e anonimizados, espelhando o layout de ``data/``
para que o pipeline rode direto sobre a pasta pública:

    DATA_RAW_DIR=dataset/raw DATA_INTERIM_DIR=dataset/interim/frames \\
        python expressions_csv_exporter/build_dataset_by_windows.py

O que é feito:
- questionários e controle: usa as cópias ``-anonimizado`` (geradas por
  ``anonymize_data.py``) e as publica com o nome original esperado pelos scripts;
- ``data_realizacao`` é reduzida a mês/ano (MM/AAAA), pois a data exata,
  combinada com sexo e idade, funciona como quase-identificador;
- pastas de sessão ``<NN>-<Jogo>-<Participante>-<AAAA-MM-DD>`` viram
  ``<NN>-<Jogo>`` e só o ``expressions.csv`` é copiado (sem JSONs/frames);
- summaries ``summary_max_nwindows{1..20}.csv`` são copiados com a mesma
  redução de data e a mesma normalização de rótulos do controle (os summaries
  de ``data/interim`` foram montados a partir do controle não normalizado);
- ao final, varre nomes de arquivos e conteúdos procurando qualquer nome de
  participante (coluna ``nome`` do controle original e pastas de sessão) e
  aborta se encontrar; gera ``SHA256SUMS`` para conferência de integridade.

Uso:
    python export_public_dataset.py
"""

import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd

from anonymize_data import normalize_labels

REPO_ROOT = Path(__file__).resolve().parent
DATA_RAW = REPO_ROOT / "data" / "raw"
FRAMES = REPO_ROOT / "data" / "interim" / "frames"
OUT = REPO_ROOT / "dataset"

GAMES = ("CatsDrop", "PacMan", "TunelRush")
N_WINDOWS = range(1, 21)
N_SESSIONS = 90
# Subpastas geradas por este script; README/LICENSE em dataset/ são preservados.
GENERATED = ("raw", "interim", "SHA256SUMS")


def anonimizado(path: Path) -> Path:
    """Caminho da cópia anonimizada; falha se ``anonymize_data.py`` não rodou."""
    anon = path.with_name(f"{path.stem}-anonimizado{path.suffix}")
    if not anon.exists():
        raise FileNotFoundError(f"{anon} não encontrado. Rode antes: python anonymize_data.py")
    return anon


def month_year(series: pd.Series) -> pd.Series:
    """DD/MM/AAAA -> MM/AAAA."""
    return pd.to_datetime(series, format="%d/%m/%Y").dt.strftime("%m/%Y")


def clean_output() -> None:
    for name in GENERATED:
        target = OUT / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def export_questionnaires() -> None:
    raw_out = OUT / "raw"
    (raw_out / "ueq").mkdir(parents=True)
    (raw_out / "humor").mkdir(parents=True)

    control = pd.read_csv(anonimizado(DATA_RAW / "controle-experimentos.csv"), dtype=str)
    control["data_realizacao"] = month_year(control["data_realizacao"])
    control.to_csv(raw_out / "controle-experimentos.csv", index=False)

    for sub in ("ueq", "humor"):
        for f in sorted((DATA_RAW / sub).glob("*.csv")):
            if "-anonimizado" not in f.stem:
                shutil.copy2(anonimizado(f), raw_out / sub / f.name)


def export_sessions() -> None:
    frames_out = OUT / "interim" / "frames"
    seen: set[str] = set()
    for session in sorted(d for d in FRAMES.iterdir() if d.is_dir()):
        pid, game = session.name.split("-")[:2]
        if game not in GAMES:
            raise ValueError(f"Jogo desconhecido na pasta {session.name}")
        public_name = f"{pid}-{game}"
        if public_name in seen:
            raise ValueError(f"Sessão duplicada: {public_name}")
        seen.add(public_name)
        (frames_out / public_name).mkdir(parents=True)
        shutil.copy2(session / "expressions.csv", frames_out / public_name / "expressions.csv")

    if len(seen) != N_SESSIONS:
        raise ValueError(f"Esperadas {N_SESSIONS} sessões, encontradas {len(seen)}")

    for n in N_WINDOWS:
        summary = pd.read_csv(FRAMES / f"summary_max_nwindows{n}.csv", dtype={"data_realizacao": str})
        summary["data_realizacao"] = month_year(summary["data_realizacao"])
        summary = normalize_labels(summary)
        summary.to_csv(frames_out / f"summary_max_nwindows{n}.csv", index=False)


def participant_names() -> set[str]:
    """Nomes completos e partes (>= 3 letras) do controle original e das pastas."""
    names = set(pd.read_csv(DATA_RAW / "controle-experimentos.csv")["nome"].dropna().str.strip())
    names |= {d.name.split("-")[2] for d in FRAMES.iterdir() if d.is_dir()}
    tokens = set()
    for name in names:
        tokens.add(name)
        # Pastas usam CamelCase sem espaço (ex.: NomeSobrenome).
        tokens.update(p for p in re.split(r"\s+|(?<=[a-zà-ú])(?=[A-ZÀ-Ú])", name) if len(p) >= 3)
    return tokens


def check_no_pii() -> None:
    pattern = re.compile(
        r"\b(" + "|".join(sorted(map(re.escape, participant_names()), key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    hits = []
    for f in OUT.rglob("*"):
        rel = f.relative_to(OUT).as_posix()
        if pattern.search(rel):
            hits.append(f"nome de arquivo: {rel}")
        if f.is_file() and f.suffix == ".csv" and (m := pattern.search(f.read_text(encoding="utf-8"))):
            hits.append(f"conteúdo de {rel}: '{m.group(0)}'")
    if hits:
        raise RuntimeError("Possível PII no dataset público:\n  " + "\n  ".join(hits))


def write_checksums() -> None:
    lines = []
    for f in sorted(p for p in (OUT / "raw").rglob("*.csv")) + sorted((OUT / "interim").rglob("*.csv")):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{digest}  {f.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    clean_output()
    export_questionnaires()
    export_sessions()
    check_no_pii()
    write_checksums()
    n_files = sum(1 for _ in OUT.rglob("*.csv"))
    print(f"[OK] {n_files} CSVs exportados para {OUT.relative_to(REPO_ROOT)}/ (verificação de PII sem ocorrências)")


if __name__ == "__main__":
    main()
