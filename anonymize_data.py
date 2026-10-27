"""Gera versões anonimizadas dos CSVs de questionários/controle.

Mantém os arquivos originais intactos e cria cópias com sufixo
``-anonimizado`` em que a pessoa é identificada APENAS por um id de dois
dígitos (01, 02, ...). Remove a coluna de nome (PII) e normaliza espaços
em branco e variações de rótulo categórico.

Uso:
    python anonymize_data.py
"""

from pathlib import Path

import pandas as pd

from modeling.utils import normalize_labels, strip_strings

DATA_RAW = Path(__file__).resolve().parent / "data" / "raw"


def pad_id(series: pd.Series) -> pd.Series:
    """Converte ids inteiros (1, 2, ...) em strings de 2 dígitos (01, 02)."""
    return series.astype(int).astype(str).str.zfill(2)


def anonymize_controle(path: Path) -> None:
    df = pd.read_csv(path)

    # id -> 01, 02, ...  e remove a coluna de nome (PII)
    df["id"] = pad_id(df["id"])
    df = df.drop(columns=[c for c in ("nome",) if c in df.columns])
    df = normalize_labels(df)

    out = path.with_name(f"{path.stem}-anonimizado{path.suffix}")
    df.to_csv(out, index=False)
    print(f"  {path.name} -> {out.name}  (coluna 'nome' removida; {len(df)} linhas)")


def anonymize_idpessoa(path: Path) -> None:
    """Para UEQ e humor: já usam apenas idPessoa; padroniza para 01, 02..."""
    df = pd.read_csv(path)
    df = strip_strings(df)
    if "idPessoa" in df:
        df["idPessoa"] = pad_id(df["idPessoa"])
    out = path.with_name(f"{path.stem}-anonimizado{path.suffix}")
    df.to_csv(out, index=False)
    print(f"  {path.name} -> {out.name}  ({len(df)} linhas)")


def main() -> None:
    print("Controle:")
    anonymize_controle(DATA_RAW / "controle-experimentos.csv")

    print("UEQ:")
    for f in sorted(DATA_RAW.glob("ueq/UEQ-*.csv")):
        if "-anonimizado" not in f.stem:
            anonymize_idpessoa(f)

    print("Humor:")
    for f in sorted(DATA_RAW.glob("humor/*.csv")):
        if "-anonimizado" not in f.stem:
            anonymize_idpessoa(f)


if __name__ == "__main__":
    main()
