"""
(用於預存資料測試，驗證模型輸出)
讀取已儲存的 snapshot
→ 合併成 history
→ 取最後一個 timestamp 當作 current snapshot
→ 計算 spatial_lag_shortage / LISA
→ 建立 17 維 feature
→ 呼叫 M4 Predictor 載入 XGBoost model
→ 輸出 pred_prob
→ 存成 latest_from_saved_snapshots.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# Path setup
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from m2.lisa import compute_lisa
from m3.build_features import (
    FEATURE_COLUMNS,
    build_features,
    load_station_static,
    validate_features,
)
from m4.predictor import ShortagePredictor


# Config
SNAPSHOT_DIR = REPO_ROOT / "data" / "raw_snapshots"
MODEL_PATH = REPO_ROOT / "models" / "xgb_v1_20260520.pkl"
OUTPUT_PATH = REPO_ROOT / "predictions" / "latest_from_saved_snapshots.json"  # 測試用

TOP_K = 20  # 測試結果只印出 t+60 後缺車機率最高的前 K 個站點


def load_saved_snapshots(snapshot_dir: Path) -> pd.DataFrame:
    """
    Load saved YouBike snapshot CSV files.
    """
    files = sorted(snapshot_dir.glob("youbike_*.csv"))

    if not files:
        raise FileNotFoundError(f"No snapshot files found in: {snapshot_dir}")

    print(f"[load] found {len(files)} snapshot files")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)

    print(f"[load] raw rows = {len(raw):,}")
    print(f"[load] columns = {raw.columns.tolist()}")

    return raw


def normalize_snapshot_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert saved CSV snapshot format into the format expected by M1/M2/M3.

    M1 cleaner / M3 feature builder expect at least:
        sno, timestamp, shortage_rate, available_bikes, total_capacity

    M2 lisa expects:
        sno, shortage_rate

    station_static provides coordinates for LISA, so snapshot lat/lng is not required by M2.
    """
    df = raw.copy()

    rename_map = {
        # If some files are raw official API format, support them too.
        "Quantity": "total_capacity",
        "available_rent_bikes": "available_bikes",
        "available_return_bikes": "available_slots",
        "latitude": "lat",
        "longitude": "lng",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = [
        "timestamp",
        "sno",
        "total_capacity",
        "available_bikes",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sno"] = df["sno"].astype(str)

    # 與正式 M1/M3 pipeline 相容：如果某些欄位不存在，補上對應欄位
    if "tick" not in df.columns:
        df["tick"] = df["timestamp"]

    if "Quantity" not in df.columns and "total_capacity" in df.columns:
        df["Quantity"] = df["total_capacity"]

    if "available_rent_bikes" not in df.columns and "available_bikes" in df.columns:
        df["available_rent_bikes"] = df["available_bikes"]

    if "available_return_bikes" not in df.columns and "available_slots" in df.columns:
        df["available_return_bikes"] = df["available_slots"]

    if "latitude" not in df.columns and "lat" in df.columns:
        df["latitude"] = df["lat"]

    if "longitude" not in df.columns and "lng" in df.columns:
        df["longitude"] = df["lng"]

    df["total_capacity"] = pd.to_numeric(df["total_capacity"], errors="coerce")
    df["available_bikes"] = pd.to_numeric(df["available_bikes"], errors="coerce")

    if "act" not in df.columns:
        df["act"] = "1"

    if "shortage_rate" not in df.columns:
        df["shortage_rate"] = 1.0 - df["available_bikes"] / df["total_capacity"]

    df["shortage_rate"] = pd.to_numeric(df["shortage_rate"], errors="coerce").clip(0, 1)

    # cleaner 可能會需要這些欄位存在；沒有的就補空字串
    for col in ["sna", "sarea", "ar"]:
        if col not in df.columns:
            df[col] = ""

    return df


def prepare_history() -> pd.DataFrame:
    """
    Load snapshots and prepare history for smoke test.

    The saved snapshots must contain at least 7 timestamps to produce lag_60min.
    """
    raw = load_saved_snapshots(SNAPSHOT_DIR)
    raw = normalize_snapshot_columns(raw)

    print("[clean] using normalized saved snapshots directly for smoke test...")

    cleaned = raw.copy()

    # 確保必要欄位型態正確
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])
    cleaned["sno"] = cleaned["sno"].astype(str)

    cleaned["total_capacity"] = pd.to_numeric(cleaned["total_capacity"], errors="coerce")
    cleaned["available_bikes"] = pd.to_numeric(cleaned["available_bikes"], errors="coerce")
    cleaned["shortage_rate"] = pd.to_numeric(cleaned["shortage_rate"], errors="coerce")

    # 保留啟用中且容量有效的站
    if "act" in cleaned.columns:
        cleaned = cleaned[cleaned["act"].astype(str) == "1"].copy()

    cleaned = cleaned[cleaned["total_capacity"] > 0].copy()

    # 如果 shortage_rate 有缺，就重新算
    cleaned["shortage_rate"] = cleaned["shortage_rate"].fillna(
        1.0 - cleaned["available_bikes"] / cleaned["total_capacity"]
    )
    cleaned["shortage_rate"] = cleaned["shortage_rate"].clip(0, 1)

    # build_features contract: sorted by sno, timestamp
    cleaned = cleaned.sort_values(["sno", "timestamp"]).reset_index(drop=True)

    print(f"[clean] cleaned rows = {len(cleaned):,}")
    print(f"[clean] timestamps = {cleaned['timestamp'].nunique()}")
    print(f"[clean] stations = {cleaned['sno'].nunique()}")
    print(f"[clean] time range = {cleaned['timestamp'].min()} ~ {cleaned['timestamp'].max()}")

    return cleaned


def build_latest_features(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build 17-dimensional feature matrix for the latest timestamp.

    Returns:
        latest_full:
            rows at the latest timestamp, including sno/timestamp and FEATURE_COLUMNS
        latest_lisa:
            LISA result for latest timestamp, including moran_type and moran_p_value
    """
    station_static = load_station_static()
    station_static["sno"] = station_static["sno"].astype(str)

    print("[lisa] computing spatial lag for every timestamp...")

    spatial_parts = []
    lisa_latest = None
    latest_time = cleaned["timestamp"].max()  # find the latest timestamp

    # 對每個 timestamp 計算 LISA
    for ts, snap in cleaned.groupby("timestamp", sort=False):
        lisa_df = compute_lisa(snap, station_static)
        spatial_parts.append(lisa_df["spatial_lag_shortage"])

        if ts == latest_time:
            lisa_latest = lisa_df.copy()

    spatial_lag = pd.concat(spatial_parts).reindex(cleaned.index)

    if lisa_latest is None:
        raise RuntimeError("Failed to compute LISA for latest timestamp.")

    print("[m3] building features...")
    features = build_features(cleaned, station_static, spatial_lag)

    full = cleaned[["sno", "timestamp", "sna", "sarea", "ar"]].copy()
    full = pd.concat([full, features], axis=1)

    latest_full = full[full["timestamp"] == latest_time].copy()

    print(f"[m3] latest timestamp = {latest_time}")
    print(f"[m3] latest rows = {len(latest_full):,}")

    print("[m3] feature NaN count at latest timestamp:")
    print(latest_full[FEATURE_COLUMNS].isna().sum())

    latest_full = latest_full.dropna(subset=FEATURE_COLUMNS).copy()

    print(f"[m3] valid rows for prediction = {len(latest_full):,}")

    if len(latest_full) == 0:
        raise ValueError(
            "No valid rows for prediction. "
            "Most likely lag_60min is still NaN. "
            "You need at least 7 timestamps per station."
        )

    # Validate exact feature order.
    X = latest_full[FEATURE_COLUMNS].copy()
    validate_features(X, strict=True)

    # M4 stricter check: XGBoost inference should not receive NaN.
    if X.isna().any().any():
        bad = X.isna().sum()
        bad = bad[bad > 0]
        raise ValueError(f"NaN exists in feature matrix:\n{bad}")

    return latest_full, lisa_latest


def predict(latest_full: pd.DataFrame) -> pd.DataFrame:
    """
    Use M4 Predictor to load pretrained XGBoost model and predict pred_prob.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"[model] using M4 predictor: {MODEL_PATH}")

    predictor = ShortagePredictor(MODEL_PATH)
    pred_df = predictor.predict(latest_full)

    return pred_df


def attach_lisa_output(pred_df: pd.DataFrame, lisa_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Attach moran_type and moran_p_value to prediction results.
    """
    lisa_cols = lisa_latest[["sno", "moran_type", "moran_p_value"]].copy()
    lisa_cols["sno"] = lisa_cols["sno"].astype(str)

    out = pred_df.merge(lisa_cols, on="sno", how="left")

    return out


def write_latest_json(pred_df: pd.DataFrame) -> None:
    """
    Write a latest.json-like file for Shiny / middleware smoke test.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    update_time = pred_df["timestamp"].max()

    output = {
        "model_version": MODEL_PATH.stem,
        "update_time": str(update_time),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "saved_snapshots_smoke_test",
        "n_stations": int(len(pred_df)),
        "predictions": [],
    }

    for _, row in pred_df.sort_values("pred_prob", ascending=False).iterrows():
        item = {
            "sno": str(row["sno"]),
            "sna": str(row.get("sna", "")),
            "sarea": str(row.get("sarea", "")),
            "shortage_rate": float(row["shortage_rate"]),
            "available_bikes": int(row["available_bikes"]),
            "total_capacity": int(row["total_capacity"]),
            "spatial_lag_shortage": float(row["spatial_lag_shortage"]),
            "distance_to_mrt": float(row["distance_to_mrt"]),
            "moran_type": str(row.get("moran_type", "NS")),
            "moran_p_value": float(row.get("moran_p_value", 1.0)),
            "pred_prob": float(row["pred_prob"]),
        }

        output["predictions"].append(item)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[write] saved: {OUTPUT_PATH}")


def print_top_k(pred_df: pd.DataFrame, k: int = TOP_K) -> None:
    show_cols = [
        "timestamp",
        "sno",
        "sna",
        "sarea",
        "available_bikes",
        "total_capacity",
        "shortage_rate",
        "spatial_lag_shortage",
        "distance_to_mrt",
        "moran_type",
        "moran_p_value",
        "pred_prob",
    ]

    print(f"\n=== Top {k} Predicted Shortage Risk Stations ===")
    print(
        pred_df.sort_values("pred_prob", ascending=False)[show_cols]
        .head(k)
        .to_string(index=False)
    )


def main() -> None:
    cleaned = prepare_history()
    latest_full, lisa_latest = build_latest_features(cleaned)
    pred_df = predict(latest_full)
    pred_df = attach_lisa_output(pred_df, lisa_latest)

    print_top_k(pred_df, TOP_K)
    write_latest_json(pred_df)

    print("\n[done] realtime prediction smoke test completed.")


if __name__ == "__main__":
    main()