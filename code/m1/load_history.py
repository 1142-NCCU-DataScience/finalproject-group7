"""
M1 — Historical snapshot loader.

Reads every `data/youbike/youbike_YYYYMMDD_HHMMSS.json` snapshot and
returns a single long-format DataFrame with the minimal set of columns
M3 needs. We deliberately drop the human-readable string fields
(`sna`, `aren`, `sareaen`, ...) to keep the in-memory footprint
predictable across ~3k snapshots × ~1k stations.

The filename's `YYYYMMDD_HHMMSS` is treated as the canonical tick
timestamp (Asia/Taipei). The per-record `mday` / `infoTime` /
`srcUpdateTime` fields are kept for auditability — they tend to be a
few minutes behind the filename tick because the API itself updates
unevenly per station.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_HISTORY_DIR = Path(__file__).resolve().parents[2] / "data" / "youbike"

# Filename pattern: youbike_20260424_092839.json
_FNAME_RE = re.compile(r"^youbike_(\d{8})_(\d{6})\.json$")

# Columns we keep per record. Everything else (sna, sareaen, snaen,
# aren, ar, infoDate, updateTime, available_return_bikes) is dropped
# to keep the in-memory footprint small.
KEEP_FIELDS = [
    "sno",
    "act",
    "Quantity",
    "available_rent_bikes",
    "latitude",
    "longitude",
    "mday",
    "infoTime",
    "srcUpdateTime",
]


def _tick_from_filename(name: str) -> pd.Timestamp:
    m = _FNAME_RE.match(name)
    if m is None:
        raise ValueError(f"unexpected filename: {name!r}")
    date_part, time_part = m.groups()
    naive = pd.Timestamp(f"{date_part} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}")
    return naive.tz_localize("Asia/Taipei")


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single snapshot JSON into a slim DataFrame."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame.from_records(records, columns=KEEP_FIELDS)
    df["tick"] = _tick_from_filename(path.name)
    return df


def load_history(
    history_dir: str | Path | None = None,
    files: Iterable[str | Path] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load all snapshots in `history_dir` (or an explicit list of files).

    Parameters
    ----------
    history_dir : Path-like, optional
        Directory containing youbike_*.json snapshots. Defaults to
        `<repo>/data/youbike`.
    files : iterable of paths, optional
        Use this explicit set of files instead of scanning a directory
        (used by tests).
    limit : int, optional
        Read at most N snapshot files (after sorting by filename).
        Useful for smoke tests on a subset.

    Returns
    -------
    pd.DataFrame
        Long-format frame with columns:
            sno, act, Quantity, available_rent_bikes,
            latitude, longitude, mday, infoTime, srcUpdateTime, tick
        where `tick` is the tz-aware filename timestamp (Asia/Taipei).
        Sorted by (sno, tick).
    """
    if files is None:
        history_dir = Path(history_dir) if history_dir is not None else DEFAULT_HISTORY_DIR
        paths = sorted(history_dir.glob("youbike_*.json"))
    else:
        paths = [Path(f) for f in files]

    if limit is not None:
        paths = paths[:limit]

    if not paths:
        raise FileNotFoundError(f"No snapshot files found (dir={history_dir!s}, files={files!r})")

    frames = [_read_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["sno", "tick"]).reset_index(drop=True)
    return df
