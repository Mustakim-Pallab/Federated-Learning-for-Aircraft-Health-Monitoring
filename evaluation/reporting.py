from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> None:
    lines = []
    columns = [str(column) for column in df.columns]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        values = [str(row[column]) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
