from __future__ import annotations

import json
import sys
import time
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from loguru import logger


import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

# Path setup
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

TZ_TPE = timezone(timedelta(hours=8))

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
API_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"

# XGBoost model
MODEL_PATH = REPO_ROOT / "models" / "xgb_v1_20260520.pkl"

# runtime 資料夾：存線上驗證過程的中間檔
RUNTIME_DIR = REPO_ROOT / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

LIVE_HISTORY_PATH = RUNTIME_DIR / "live_history.csv"
PENDING_PATH = RUNTIME_DIR / "pending_predictions.csv"
VERIFIED_PATH = RUNTIME_DIR / "verified_predictions.csv"
METRICS_PATH = RUNTIME_DIR / "online_metrics.json"

# 預測 60 分鐘後是否缺車
PREDICTION_HORIZON_MINUTES = 60

# y 的定義：
# actual_y = 1 if future shortage_rate > 0.8 else 0
SHORTAGE_THRESHOLD = 0.8

# features 是以 10 分鐘為一個 tick。
# 每 10 分鐘跑一次
EXPECTED_INTERVAL_MINUTES = 10

# 避免用間隔太久的資料產生 lag
MAX_HISTORY_GAP_MINUTES = 20

TOP_K_LIST = [10, 20]

def parse_taipei_time(s):
    """
    Robustly parse mixed timezone / string timestamps and normalize to Asia/Taipei.
    統一先用 utc=True 解析再轉回 Asia/Taipei
    """
    return pd.to_datetime(s, utc=True).dt.tz_convert("Asia/Taipei")

# Fetch + normalize current API snapshot
def fetch_current_snapshot() -> pd.DataFrame:
    """
    從 YouBike 即時 API 抓取目前所有站點狀態
    這裡只保留 act == 1 且 Quantity > 0 的站點。
    """
    now = datetime.now(ZoneInfo("Asia/Taipei")).replace(second=0, microsecond=0)

    response = requests.get(API_URL, timeout=10)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)

    # 轉 numeric
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["available_rent_bikes"] = pd.to_numeric(df["available_rent_bikes"], errors="coerce")
    df["available_return_bikes"] = pd.to_numeric(df["available_return_bikes"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # 只保留啟用中且容量有效的站
    df = df[(df["act"].astype(str) == "1") & (df["Quantity"] > 0)].copy()

    # 統一成後續 feature pipeline 使用的欄位名稱
    df = df.rename(
        columns={
            "Quantity": "total_capacity",
            "available_rent_bikes": "available_bikes",
            "available_return_bikes": "available_slots",
            "latitude": "lat",
            "longitude": "lng",
        }
    )

    df["timestamp"] = now
    df["sno"] = df["sno"].astype(str)

    # 即時缺車率
    df["shortage_rate"] = 1.0 - df["available_bikes"] / df["total_capacity"]
    df["shortage_rate"] = df["shortage_rate"].clip(0, 1)

    keep_cols = [
        "timestamp",
        "sno",
        "sna",
        "sarea",
        "ar",
        "lat",
        "lng",
        "total_capacity",
        "available_bikes",
        "available_slots",
        "shortage_rate",
        "srcUpdateTime",
    ]

    return df[keep_cols].copy()

def load_live_history() -> pd.DataFrame:
    """
    讀取 runtime/live_history.csv
    如果檔案不存在，代表目前還沒有 history，回傳空 DataFrame
    """
    if not LIVE_HISTORY_PATH.exists():
        return pd.DataFrame()

    hist = pd.read_csv(LIVE_HISTORY_PATH)
    hist["timestamp"] = parse_taipei_time(hist["timestamp"])
    hist["sno"] = hist["sno"].astype(str)

    return hist


def update_live_history(current_df: pd.DataFrame) -> pd.DataFrame:
    """
    把目前這一輪 API snapshot 加進 live_history.csv
    每次執行 script 都會：
        1. 讀取舊 history
        2. append current snapshot
        3. 去除重複 timestamp+sno
        4. 只保留最近 24 個 timestamp，避免檔案過大
    """
    hist = load_live_history()

    hist_cols = [
        "timestamp",
        "sno",
        "sna",
        "sarea",
        "ar",
        "total_capacity",
        "available_bikes",
        "shortage_rate",
    ]

    current_hist = current_df[hist_cols].copy()

    if len(hist) > 0:
        hist = pd.concat([hist, current_hist], ignore_index=True)
    else:
        hist = current_hist

    hist["timestamp"] = parse_taipei_time(hist["timestamp"])
    hist["sno"] = hist["sno"].astype(str)

    hist = hist.drop_duplicates(subset=["timestamp", "sno"], keep="last")
    hist = hist.sort_values(["sno", "timestamp"])

    # 只保留最近 24 個 timestamp，避免檔案太大
    recent_times = sorted(hist["timestamp"].dropna().unique())[-24:]
    hist = hist[hist["timestamp"].isin(recent_times)].copy()

    hist.to_csv(LIVE_HISTORY_PATH, index=False, encoding="utf-8-sig")

    return hist


def has_continuous_recent_history(hist: pd.DataFrame) -> bool:
    """
    檢查目前是否有足夠且連續的 history 來產生 lag_60min
    """
    if len(hist) == 0:
        return False

    times = sorted(parse_taipei_time(hist["timestamp"]).dropna().unique())

    if len(times) < 7:
        return False

    recent = times[-7:]
    gaps = [
        (recent[i] - recent[i - 1]).total_seconds() / 60
        for i in range(1, len(recent))
    ]

    return all(g <= MAX_HISTORY_GAP_MINUTES for g in gaps)


def align_snapshot_with_station_static(
    current_df: pd.DataFrame,
    hist: pd.DataFrame,
    station_static: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    將即時 API snapshot 與 station_static 對齊
    """
    station_static = station_static.copy()
    station_static["sno"] = station_static["sno"].astype(str)

    valid_snos = set(station_static["sno"])

    current_df = current_df.copy()
    hist = hist.copy()

    current_df["sno"] = current_df["sno"].astype(str)
    hist["sno"] = hist["sno"].astype(str)

    missing_current = sorted(set(current_df["sno"]) - valid_snos)

    if missing_current:
        print(
            f"[station_static] dropping {len(missing_current)} live station(s) "
            f"not found in station_static: {missing_current[:10]}"
        )

    current_df = current_df[current_df["sno"].isin(valid_snos)].copy()
    hist = hist[hist["sno"].isin(valid_snos)].copy()

    return current_df, hist


def build_current_features(
    current_df: pd.DataFrame,
    hist: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    建立即時推論用的 17 維 feature matrix

    1. 檢查是否有足夠 history 產生 lag_60min
    2. 載入 station_static
    3. 如果即時 API 有新站但 station_static 沒有，先 drop
    4. 用 M2 compute_lisa() 算 current snapshot 的 spatial lag / LISA
    5. 對 history 每個 timestamp 計算 spatial_lag_shortage
    6. 呼叫 M3 build_features() 產生與訓練一致的 17 個 feature
    7. 只取 current timestamp 的 feature
    8. 檢查 feature 欄位順序與 NaN
    """
    if not has_continuous_recent_history(hist):
        raise RuntimeError(
            "Not enough continuous recent history for lag_60min. "
            "You need at least 7 recent timestamps, ideally collected every 10 minutes."
        )

    station_static = load_station_static()
    station_static["sno"] = station_static["sno"].astype(str)

    # 避免即時 API 出現 station_static 沒有的新站導致 compute_lisa() 失敗
    current_df, hist = align_snapshot_with_station_static(
        current_df,
        hist,
        station_static,
    )

    if len(current_df) == 0:
        raise RuntimeError("No live stations left after aligning with station_static.")

    current_time = current_df["timestamp"].max()

    # 給 M2 LISA 用的是 current snapshot
    lisa_current = compute_lisa(current_df, station_static)

    # 給 M3 build_features 用的是整段 recent history 的 spatial lag
    spatial_parts = []
    for _, snap in hist.groupby("timestamp", sort=False):
        lisa_df = compute_lisa(snap, station_static)
        spatial_parts.append(lisa_df["spatial_lag_shortage"])

    if not spatial_parts:
        raise RuntimeError("No spatial lag can be computed from history.")

    spatial_lag = pd.concat(spatial_parts).reindex(hist.index)

    hist_sorted = hist.sort_values(["sno", "timestamp"]).reset_index(drop=True)
    spatial_lag = spatial_lag.reset_index(drop=True)

    features = build_features(hist_sorted, station_static, spatial_lag)

    full = hist_sorted[["sno", "timestamp", "sna", "sarea", "ar"]].copy()
    full = pd.concat([full, features], axis=1)

    current_features = full[full["timestamp"] == current_time].copy()

    current_features = current_features.dropna(subset=FEATURE_COLUMNS).copy()

    if len(current_features) == 0:
        raise RuntimeError("Current feature matrix is empty after dropping NaN.")

    X = current_features[FEATURE_COLUMNS].copy()
    validate_features(X, strict=True)

    if X.isna().any().any():
        bad = X.isna().sum()
        bad = bad[bad > 0]
        raise RuntimeError(f"NaN in current feature matrix:\n{bad}")

    return current_features, lisa_current



def predict_current(current_features: pd.DataFrame) -> pd.DataFrame:
    """
    使用 M4 Predictor 載入預訓練 XGBoost model
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"[model] using M4 predictor: {MODEL_PATH}")

    predictor = ShortagePredictor(MODEL_PATH)
    pred_df = predictor.predict(current_features)

    return pred_df


def save_pending_predictions(pred_df: pd.DataFrame) -> None:
    """
    將本輪預測結果存入 pending_predictions.csv。
    """
    predict_time = pred_df["timestamp"].max()
    target_time = predict_time + timedelta(minutes=PREDICTION_HORIZON_MINUTES)

    save_cols = [
        "timestamp",
        "sno",
        "sna",
        "sarea",
        "available_bikes",
        "total_capacity",
        "shortage_rate",
        "pred_prob",
        "model_version",
    ]

    out = pred_df[save_cols].copy()
    out = out.rename(
        columns={
            "timestamp": "predict_time",
            "available_bikes": "predict_available_bikes",
            "total_capacity": "predict_total_capacity",
            "shortage_rate": "predict_shortage_rate",
        }
    )

    out["target_time"] = target_time

    if PENDING_PATH.exists():
        pending = pd.read_csv(PENDING_PATH)
        pending["predict_time"] = parse_taipei_time(pending["predict_time"])
        pending["target_time"] = parse_taipei_time(pending["target_time"])
        pending["sno"] = pending["sno"].astype(str)

        pending = pd.concat([pending, out], ignore_index=True)
    else:
        pending = out

    pending["predict_time"] = parse_taipei_time(pending["predict_time"])
    pending["target_time"] = parse_taipei_time(pending["target_time"])
    pending["sno"] = pending["sno"].astype(str)

    pending = pending.drop_duplicates(
        subset=["predict_time", "target_time", "sno"],
        keep="last",
    )

    pending.to_csv(PENDING_PATH, index=False, encoding="utf-8-sig")

    print(f"[pending] saved predictions: {len(out):,}")
    print(f"[pending] predict_time = {predict_time}")
    print(f"[pending] target_time  = {target_time}")


def verify_due_predictions(current_df: pd.DataFrame) -> None:
    """
    檢查 pending_predictions.csv 中是否有已到期的預測。

    到期條件：
        target_time <= current timestamp

    若到期：
        用目前 API 的 shortage_rate 產生 actual_y
        actual_y = 1 if actual_shortage_rate > 0.8 else 0

    然後將結果寫入 verified_predictions.csv。
    """
    if not PENDING_PATH.exists():
        print("[verify] no pending_predictions.csv yet")
        return

    now = current_df["timestamp"].max()

    pending = pd.read_csv(PENDING_PATH)
    pending["predict_time"] = parse_taipei_time(pending["predict_time"])
    pending["target_time"] = parse_taipei_time(pending["target_time"])
    pending["sno"] = pending["sno"].astype(str)

    due = pending[pending["target_time"] <= now].copy()
    not_due = pending[pending["target_time"] > now].copy()

    if len(due) == 0:
        print("[verify] no predictions are due yet")
        pending.to_csv(PENDING_PATH, index=False, encoding="utf-8-sig")
        return

    actual = current_df[
        ["sno", "timestamp", "available_bikes", "total_capacity", "shortage_rate"]
    ].copy()

    actual = actual.rename(
        columns={
            "timestamp": "actual_time",
            "available_bikes": "actual_available_bikes",
            "total_capacity": "actual_total_capacity",
            "shortage_rate": "actual_shortage_rate",
        }
    )

    actual["actual_time"] = parse_taipei_time(actual["actual_time"])
    actual["actual_y"] = (actual["actual_shortage_rate"] > SHORTAGE_THRESHOLD).astype(int)

    verified = due.merge(actual, on="sno", how="left")

    verified = verified.dropna(subset=["actual_y", "pred_prob"]).copy()

    verified["pred_label_05"] = (verified["pred_prob"] >= 0.5).astype(int)
    verified["correct_05"] = (verified["pred_label_05"] == verified["actual_y"]).astype(int)

    if VERIFIED_PATH.exists():
        old = pd.read_csv(VERIFIED_PATH)
        old["predict_time"] = parse_taipei_time(old["predict_time"])
        old["target_time"] = parse_taipei_time(old["target_time"])

        if "actual_time" in old.columns:
            old["actual_time"] = parse_taipei_time(old["actual_time"])

        old["sno"] = old["sno"].astype(str)

        verified_all = pd.concat([old, verified], ignore_index=True)
    else:
        verified_all = verified

    verified_all["predict_time"] = parse_taipei_time(verified_all["predict_time"])
    verified_all["target_time"] = parse_taipei_time(verified_all["target_time"])

    if "actual_time" in verified_all.columns:
        verified_all["actual_time"] = parse_taipei_time(verified_all["actual_time"])

    verified_all["sno"] = verified_all["sno"].astype(str)

    verified_all = verified_all.drop_duplicates(
        subset=["predict_time", "target_time", "sno"],
        keep="last",
    )

    verified_all.to_csv(VERIFIED_PATH, index=False, encoding="utf-8-sig")
    not_due.to_csv(PENDING_PATH, index=False, encoding="utf-8-sig")

    print(f"[verify] verified rows this round: {len(verified):,}")
    print(f"[verify] total verified rows: {len(verified_all):,}")


def precision_at_k(df: pd.DataFrame, k: int) -> float:
    """
    計算 Precision@K
    對每個 predict_time：
        取 pred_prob 最高的前 K 個站
        看其中 actual_y = 1 的比例
    """
    values = []

    for predict_time, group in df.groupby("predict_time"):
        top_k = group.sort_values("pred_prob", ascending=False).head(k)

        if len(top_k) == 0:
            continue

        values.append(top_k["actual_y"].mean())

    if len(values) == 0:
        return float("nan")

    return float(np.mean(values))


def report_metrics() -> None:
    """
    根據 verified_predictions.csv 計算線上驗證指標。

    輸出：
        accuracy_05
        precision_05
        recall_05
        f1_05
        auc_roc
        brier
        precision_at_10
        precision_at_20
        confusion_matrix_05

    寫入 runtime/online_metrics.json。
    """
    if not VERIFIED_PATH.exists():
        print("[metrics] no verified_predictions.csv yet")
        return

    df = pd.read_csv(VERIFIED_PATH)

    if len(df) == 0:
        print("[metrics] verified_predictions.csv is empty")
        return

    if "predict_time" in df.columns:
        df["predict_time"] = parse_taipei_time(df["predict_time"])

    if "target_time" in df.columns:
        df["target_time"] = parse_taipei_time(df["target_time"])

    if "actual_time" in df.columns:
        df["actual_time"] = parse_taipei_time(df["actual_time"])

    df = df.dropna(subset=["pred_prob", "actual_y"]).copy()

    if len(df) == 0:
        print("[metrics] no valid verified rows")
        return

    y_true = df["actual_y"].astype(int)
    y_prob = df["pred_prob"].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "n_verified_rows": int(len(df)),
        "positive_rate": float(y_true.mean()),
        "accuracy_05": float(accuracy_score(y_true, y_pred)),
        "precision_05": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_05": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_05": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_10": precision_at_k(df, 10),
        "precision_at_20": precision_at_k(df, 20),
    }

    if y_true.nunique() == 2:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    else:
        metrics["auc_roc"] = None
        metrics["brier"] = None

    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix_05"] = cm

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Online Verification Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"[metrics] saved: {METRICS_PATH}")

def get_current_iso_time() -> str:
    return datetime.now(TZ_TPE).isoformat()

def _git_commit_and_push(repo_path: str, max_retries: int = 2):
    commands = [
        ["git", "add", "predictions/latest.json"],
        ["git", "commit", "-m", f"chore: auto-update predictions {get_current_iso_time()}"],
        ["git", "push", "origin", "main"]
    ]
    
    for cmd in commands:
        if cmd[1] == "push":
            for attempt in range(max_retries + 1):
                result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.success("Git Push successful!")
                    break
                else:
                    logger.warning(f"Git Push failed (Attempt {attempt+1}/{max_retries+1}): {result.stderr}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
            else:
                logger.error("Git Push ultimately failed. Retaining local changes for the next update cycle.")
        else:
            subprocess.run(cmd, cwd=repo_path, capture_output=True)

def save_latest_json_local(
    pred_df: pd.DataFrame = None, 
    model_version: str = "unknown", 
    output_dir: str | Path = "predictions",
    is_success: bool = True
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "latest.json"
    
    current_time = get_current_iso_time()
    
    old_data = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read old latest.json: {e}")

    if is_success and pred_df is not None and not pred_df.empty:
        health_status = "ok"
        last_success_at = current_time
        update_time = str(pred_df["timestamp"].max()) if "timestamp" in pred_df.columns else current_time
        
        predictions_list = []
        for _, row in pred_df.iterrows():
            item = {
                "sno": str(row["sno"]),
                "lat": float(row.get("lat", row.get("latitude", 0.0))), 
                "lng": float(row.get("lng", row.get("longitude", 0.0))),
                "shortage_rate": float(row["shortage_rate"]),
                "available_bikes": int(row["available_bikes"]),
                "total_capacity": int(row["total_capacity"]),
                "moran_type": str(row.get("moran_type", "NS")),
                "moran_p_value": float(row.get("moran_p_value", 1.0)),
                "pred_prob": float(row["pred_prob"]),
            }
            predictions_list.append(item)
    else:
        last_success_at = old_data.get("last_success_at", current_time)
        update_time = old_data.get("update_time", last_success_at)
        predictions_list = old_data.get("predictions", [])
        model_version = old_data.get("model_version", model_version)
        
        try:
            last_success_dt = datetime.fromisoformat(last_success_at)
            delay_mins = (datetime.now(TZ_TPE) - last_success_dt).total_seconds() / 60
            if delay_mins < 15:
                health_status = "ok"
            elif delay_mins < 30:
                health_status = "stale"
            else:
                health_status = "degraded"
        except ValueError:
            health_status = "degraded"
            logger.error("Timestamp parsing error for fallback.")

    output_payload = {
        "model_version": model_version,
        "update_time": update_time,
        "last_attempt_at": current_time,
        "last_success_at": last_success_at,
        "health_status": health_status,
        "n_stations": len(predictions_list),
        "predictions": predictions_list
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Successfully saved local JSON to {json_path} (Status: {health_status})")

    # _git_commit_and_push(output_path.parent)
    return json_path

def run_once() -> None:
    """
    單次線上驗證流程。

    執行一次會：
        1. 抓目前 API snapshot
        2. 更新 live_history
        3. 驗證已到期的 pending predictions
        4. 若 history 足夠，建立 current features
        5. 呼叫 M4 predictor 產生 pred_prob
        6. 存入 pending_predictions.csv
        7. 印出目前 Top 20 高風險站
        8. 重新計算 online metrics
    """
    print("[api] fetching current snapshot...")
    current_df = fetch_current_snapshot()

    print(f"[api] current rows = {len(current_df):,}")
    print(f"[api] current timestamp = {current_df['timestamp'].max()}")

    print("[history] updating live history...")
    hist = update_live_history(current_df)

    print(f"[history] timestamps = {hist['timestamp'].nunique()}")
    print(f"[history] rows = {len(hist):,}")

    print("[verify] checking due predictions...")
    verify_due_predictions(current_df)

    try:
        print("[feature] building current features...")
        current_features, lisa_current = build_current_features(current_df, hist)

        print(f"[feature] valid current rows = {len(current_features):,}")

        print("[model] predicting current shortage probability...")
        pred_df = predict_current(current_features)
        
        lisa_cols = lisa_current[["sno", "moran_type", "moran_p_value"]].copy()
        lisa_cols["sno"] = lisa_cols["sno"].astype(str)
        pred_df = pred_df.merge(lisa_cols, on="sno", how="left")

        loc_cols = current_df[["sno", "lat", "lng"]].copy()
        loc_cols["sno"] = loc_cols["sno"].astype(str)
        pred_df = pred_df.merge(loc_cols, on="sno", how="left")

        save_pending_predictions(pred_df)

        model_ver = pred_df["model_version"].iloc[0] if "model_version" in pred_df.columns else "xgb_v1_20260520"
        save_latest_json_local(
            pred_df=pred_df,
            model_version=model_ver,
            is_success=True
        )

        print("\n=== Top 20 Current Predicted Risk Stations ===")
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
            "pred_prob",
        ]

        print(
            pred_df.sort_values("pred_prob", ascending=False)[show_cols]
            .head(20)
            .to_string(index=False)
        )

    except RuntimeError as e:
        print(f"[feature/model] skipped current prediction: {e}")
        save_latest_json_local(is_success=False)

    report_metrics()


if __name__ == "__main__":
    run_once()


# while true; do
#   echo "==============================" | tee -a logs/online_verify.log
#   date | tee -a logs/online_verify.log
#   python code/online_verify_once.py 2>&1 | tee -a logs/online_verify.log
#   echo "Sleep 600 seconds..." | tee -a logs/online_verify.log
#   sleep 600
# done