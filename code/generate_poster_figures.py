"""
Produce all statistical figures for the project poster.

Outputs land in results/poster_figures/:
    01_feature_importance.png
    02_calibration_plot.png
    03_precision_at_k.png
    04_hour_weekday_heatmap.png
    05_baseline_comparison.png
    06_global_moran_over_time.png
    07_hotspot_persistence.png
    summary_metrics.json

Run from repo root:
    python code/generate_poster_figures.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from m2.lisa import compute_lisa, compute_spatial_lag  # noqa: E402
from m3.build_features import FEATURE_COLUMNS  # noqa: E402

warnings.filterwarnings("ignore")

# Try to get Chinese font on Windows
for font in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = REPO_ROOT / "data" / "training_dataset.csv"
STATION_CSV = REPO_ROOT / "data" / "youbike_station.csv"
MODEL_PATH = REPO_ROOT / "models" / "xgb_v1_20260520.pkl"
OUT_DIR = REPO_ROOT / "results" / "poster_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Visual theme — align with architecture doc colors
COLOR_PRIMARY = "#0288d1"  # Producer blue
COLOR_ACCENT = "#f57c00"   # Middleware orange
COLOR_HOTSPOT = "#d32f2f"  # HH red
COLOR_OK = "#388e3c"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Load data + model
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    log(f"Loading {DATA_PATH.name} ...")
    dtype = {
        "sno": "string",
        "available_bikes": "int32",
        "total_capacity": "int32",
        "is_rush_hour": "int8",
        "is_weekend": "int8",
        "y": "int8",
    }
    for c in (
        "shortage_rate", "lag_10min", "lag_20min", "lag_30min", "lag_60min",
        "delta_10min", "delta_30min",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "spatial_lag_shortage", "distance_to_mrt",
    ):
        dtype[c] = "float32"

    df = pd.read_csv(DATA_PATH, dtype=dtype, parse_dates=["timestamp"])
    log(f"  loaded {len(df):,} rows, mem ~{df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    return df


def time_split(df: pd.DataFrame, test_days: float = 3.5):
    df = df.sort_values("timestamp").reset_index(drop=True)
    t_max = df["timestamp"].max()
    split_ts = t_max - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] < split_ts]
    test = df[df["timestamp"] >= split_ts]
    log(f"  train: {len(train):,} rows ({train['timestamp'].min()} → {split_ts})")
    log(f"  test : {len(test):,} rows ({split_ts} → {t_max})")
    return train, test, split_ts


# ---------------------------------------------------------------------------
# 1. Feature importance
# ---------------------------------------------------------------------------

def fig_feature_importance(model) -> dict:
    log("Figure 1: feature importance")
    imp = model.feature_importances_
    order = np.argsort(imp)
    cols = np.array(FEATURE_COLUMNS)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(cols[order], imp[order], color=COLOR_PRIMARY)
    # Highlight spatial_lag_shortage in red
    for i, name in enumerate(cols[order]):
        if name == "spatial_lag_shortage":
            bars[i].set_color(COLOR_HOTSPOT)
    ax.set_xlabel("Feature importance (gain)", fontsize=12)
    ax.set_title("XGBoost Feature Importance (17 features)", fontsize=14, pad=12)
    ax.grid(axis="x", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "01_feature_importance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  -> {out.name}")

    return {
        "top5": [(c, float(v)) for c, v in
                 sorted(zip(cols, imp), key=lambda x: -x[1])[:5]],
    }


# ---------------------------------------------------------------------------
# 2. Calibration plot
# ---------------------------------------------------------------------------

def fig_calibration(y_true, y_prob) -> dict:
    log("Figure 2: calibration plot")
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5, label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "o-", color=COLOR_PRIMARY, linewidth=2.5,
            markersize=8, label=f"XGBoost (Brier = {brier:.4f})")
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed frequency", fontsize=12)
    ax.set_title("Calibration Plot — t+60min Shortage Prediction", fontsize=13, pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "02_calibration_plot.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  -> {out.name}")
    return {"brier_test": float(brier)}


# ---------------------------------------------------------------------------
# 3. Precision@k
# ---------------------------------------------------------------------------

def fig_precision_at_k(test_df: pd.DataFrame, probs: np.ndarray) -> dict:
    log("Figure 3: Precision@k")
    df = test_df[["timestamp", "y"]].copy()
    df["prob"] = probs
    ks = [5, 10, 20, 50]
    means = {}
    for k in ks:
        # for each timestamp, take top-k stations by prob
        per_ts = df.groupby("timestamp", sort=False, group_keys=False).apply(
            lambda g: g.nlargest(min(k, len(g)), "prob")["y"].mean()
        )
        means[k] = float(per_ts.mean())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [str(k) for k in ks]
    ys = [means[k] for k in ks]
    bars = ax.bar(xs, ys, color=COLOR_PRIMARY, width=0.55)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=12, fontweight="bold")
    ax.axhline(0.299, color=COLOR_ACCENT, linestyle="--", linewidth=1.5,
               label="Base rate P(y=1) = 0.299")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k (top stations to dispatch)", fontsize=12)
    ax.set_ylabel("Precision@k", fontsize=12)
    ax.set_title("Precision@k — Business-relevant Dispatch Metric", fontsize=13, pad=12)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "03_precision_at_k.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  -> {out.name}  {means}")
    return {"precision_at_k": means}


# ---------------------------------------------------------------------------
# 4. Hour × Weekday heatmap
# ---------------------------------------------------------------------------

def fig_hour_weekday_heatmap(df: pd.DataFrame) -> dict:
    log("Figure 4: Hour x Weekday heatmap")
    s = df[["timestamp", "shortage_rate"]].copy()
    s["hour"] = s["timestamp"].dt.hour
    s["dow"] = s["timestamp"].dt.dayofweek  # 0 = Mon
    pivot = s.groupby(["dow", "hour"])["shortage_rate"].mean().unstack("hour")
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   vmin=pivot.values.min(), vmax=pivot.values.max())
    ax.set_yticks(range(7))
    ax.set_yticklabels(dow_labels, fontsize=11)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=10)
    ax.set_xlabel("Hour of day", fontsize=12)
    ax.set_title("Average Shortage Rate by Hour × Weekday",
                 fontsize=13, pad=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean shortage_rate", fontsize=11)

    # Annotate rush-hour windows
    for h in (7, 8, 17, 18):
        ax.axvspan(h - 0.5, h + 0.5, ymin=0, ymax=5/7,
                   facecolor="none", edgecolor="black", linewidth=1.2, alpha=0.6)
    plt.tight_layout()
    out = OUT_DIR / "04_hour_weekday_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  -> {out.name}")
    return {
        "weekday_morning_peak_avg": float(pivot.iloc[:5, 7:10].mean().mean()),
        "weekend_avg": float(pivot.iloc[5:, :].mean().mean()),
    }


# ---------------------------------------------------------------------------
# 5. Baseline comparison
# ---------------------------------------------------------------------------

def fig_baseline_comparison(test_df: pd.DataFrame, xgb_prob: np.ndarray) -> dict:
    log("Figure 5: baseline comparison")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler

    y_test = test_df["y"].values
    persistence_pred = (test_df["shortage_rate"] > 0.8).astype(float).values
    persistence_prob = test_df["shortage_rate"].clip(0, 1).values

    # Logistic regression on a subsample of train (memory-friendly)
    # We refit a quick LR on test_df earlier portion as proxy — but a cleaner choice:
    # use a 100k random sample of test for a "weak baseline" demo? Better: refit on
    # train data already passed in to caller via globals — keep simple here by
    # subsampling 200k rows of test for LR using only single feature shortage_rate
    # ...actually the user already has a *full* XGBoost, so LR baseline should also
    # see the same features. We'll fit it externally and pass probs in.
    # For simplicity: persistence vs XGB only here.
    auc_xgb = roc_auc_score(y_test, xgb_prob)
    auc_pers = roc_auc_score(y_test, persistence_prob)
    brier_xgb = brier_score_loss(y_test, xgb_prob)
    brier_pers = brier_score_loss(y_test, persistence_prob)

    models = ["Persistence\n(shortage_rate now)", "XGBoost\n(17 features)"]
    aucs = [auc_pers, auc_xgb]
    briers = [brier_pers, brier_xgb]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    bars1 = ax1.bar(models, aucs, color=[COLOR_ACCENT, COLOR_PRIMARY], width=0.55)
    for b, v in zip(bars1, aucs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel("AUC-ROC", fontsize=12)
    ax1.set_title("Discrimination", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    bars2 = ax2.bar(models, briers, color=[COLOR_ACCENT, COLOR_PRIMARY], width=0.55)
    for b, v in zip(bars2, briers):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(briers) * 1.3)
    ax2.set_ylabel("Brier score (lower is better)", fontsize=12)
    ax2.set_title("Calibration", fontsize=13)
    ax2.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    plt.suptitle("XGBoost vs Persistence Baseline (held-out test set)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "05_baseline_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  -> {out.name}")
    return {
        "auc_xgb": float(auc_xgb), "auc_persistence": float(auc_pers),
        "brier_xgb": float(brier_xgb), "brier_persistence": float(brier_pers),
        "auc_lift": float(auc_xgb - auc_pers),
    }


# ---------------------------------------------------------------------------
# 6. Global Moran's I across the day
# ---------------------------------------------------------------------------

def fig_global_moran(df: pd.DataFrame, station_static: pd.DataFrame) -> dict:
    log("Figure 6: Global Moran's I over time")
    from esda.moran import Moran
    from libpysal.weights import KNN

    # Pick one full weekday with complete snapshot coverage (~144 / day)
    norm = df["timestamp"].dt.normalize()
    snapshots_per_day = (
        df[["timestamp"]].assign(day=norm)
        .groupby("day")["timestamp"].nunique()
    )
    # Prefer a Monday (dow=0) with >= 144 snapshots
    candidates = snapshots_per_day[snapshots_per_day >= 144]
    weekday_candidates = candidates[
        pd.to_datetime(candidates.index).dayofweek == 0
    ]
    target_day = (weekday_candidates.index[0] if len(weekday_candidates)
                  else candidates.index[0])
    target_day = pd.Timestamp(target_day)
    day = df[norm == target_day]
    log(f"  using day {target_day.date()} "
        f"({day['timestamp'].nunique()} snapshots)")

    # Pick snapshots every 1 hour from 0:00 to 23:00 (24 points)
    ts_per_hour = (
        day.groupby(day["timestamp"].dt.hour)["timestamp"].min().sort_index()
    )

    hours = []
    morans = []
    pvals = []

    for hour, ts in ts_per_hour.items():
        snap = day[day["timestamp"] == ts][
            ["sno", "shortage_rate"]
        ].drop_duplicates("sno")
        snap = snap.merge(
            station_static[["sno", "latitude", "longitude"]], on="sno", how="left"
        )
        snap = snap.dropna(subset=["latitude", "longitude"])
        coords = snap[["latitude", "longitude"]].to_numpy(float)
        w = KNN.from_array(coords, k=6)
        w.transform = "R"
        x = snap["shortage_rate"].to_numpy(float)
        m = Moran(x, w, permutations=999)
        hours.append(int(hour))
        morans.append(float(m.I))
        pvals.append(float(m.p_sim))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(hours, morans, "o-", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=8, label="Global Moran's I")
    ax.axhline(0, color="gray", linewidth=1, linestyle=":")
    # Rush hours shading
    ax.axvspan(7, 9, alpha=0.15, color=COLOR_HOTSPOT, label="Morning rush 7–9")
    ax.axvspan(17, 19, alpha=0.15, color=COLOR_ACCENT, label="Evening rush 17–19")
    ax.set_xlabel("Hour of day", fontsize=12)
    ax.set_ylabel("Global Moran's I", fontsize=12)
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(f"Global Moran's I across {target_day.date()} "
                 f"(k-NN, k=6, 999 permutations)", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "06_global_moran_over_time.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  -> {out.name}")
    return {
        "moran_min": float(min(morans)),
        "moran_max": float(max(morans)),
        "moran_mean": float(np.mean(morans)),
        "all_significant": bool(all(p < 0.05 for p in pvals)),
        "target_day": str(target_day.date()),
    }


# ---------------------------------------------------------------------------
# 7. Hotspot persistence (% time each station was HH)
# ---------------------------------------------------------------------------

def fig_hotspot_persistence(df: pd.DataFrame, station_static: pd.DataFrame,
                            n_samples: int = 60) -> dict:
    log("Figure 7: Hotspot persistence (sampled LISA across timestamps)")
    timestamps = np.array(sorted(df["timestamp"].unique()))
    # Sample evenly across the 27-day window
    idx = np.linspace(0, len(timestamps) - 1, n_samples).astype(int)
    sample_ts = timestamps[idx]
    log(f"  computing LISA on {len(sample_ts)} timestamps ...")

    # Tally HH counts per sno
    hh_counts: dict[str, int] = {}
    total_seen: dict[str, int] = {}

    for i, ts in enumerate(sample_ts, 1):
        snap = df[df["timestamp"] == ts][["sno", "shortage_rate"]]
        if snap.empty:
            continue
        lisa = compute_lisa(snap, station_static)
        for sno, mtype in zip(lisa["sno"], lisa["moran_type"]):
            total_seen[sno] = total_seen.get(sno, 0) + 1
            if mtype == "HH":
                hh_counts[sno] = hh_counts.get(sno, 0) + 1
        if i % 10 == 0:
            log(f"    {i}/{len(sample_ts)}")

    rows = []
    for sno, n_seen in total_seen.items():
        n_hh = hh_counts.get(sno, 0)
        rows.append({"sno": sno, "n_seen": n_seen, "n_hh": n_hh,
                     "hh_ratio": n_hh / n_seen if n_seen else 0.0})
    persistence = pd.DataFrame(rows)

    # Save CSV for poster appendix
    persistence_path = OUT_DIR / "07_hotspot_persistence_per_station.csv"
    persistence.sort_values("hh_ratio", ascending=False).to_csv(
        persistence_path, index=False
    )

    bins = [0, 0.05, 0.10, 0.30, 0.50, 1.0]
    labels = ["<5%\n(rare)", "5–10%\n(occasional)",
              "10–30%\n(intermittent)", "30–50%\n(frequent)",
              "≥50%\n(chronic)"]
    counts, _ = np.histogram(persistence["hh_ratio"], bins=bins)
    colors = ["#cccccc", "#fdd49e", "#fdae6b", "#f16913", COLOR_HOTSPOT]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, counts, color=colors, width=0.55)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + max(counts) * 0.01,
                f"{int(v)}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("% of time station classified as HH (high–high hotspot)",
                  fontsize=12)
    ax.set_ylabel("Number of stations", fontsize=12)
    ax.set_title(f"Hotspot Persistence "
                 f"({len(sample_ts)} snapshots, "
                 f"{len(persistence)} stations)",
                 fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "07_hotspot_persistence.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  -> {out.name}, CSV: {persistence_path.name}")
    return {
        "n_chronic_hotspots_ge50pct": int((persistence["hh_ratio"] >= 0.5).sum()),
        "n_frequent_hotspots_30_50pct": int(
            ((persistence["hh_ratio"] >= 0.3) & (persistence["hh_ratio"] < 0.5)).sum()
        ),
        "top5_chronic_snos": persistence.nlargest(5, "hh_ratio")[
            ["sno", "hh_ratio"]
        ].to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    summary: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    log(f"Loading model {MODEL_PATH.name}")
    model = joblib.load(MODEL_PATH)
    summary.update(fig_feature_importance(model))

    df = load_data()
    station_static = pd.read_csv(STATION_CSV, dtype={"sno": "string"})

    train_df, test_df, split_ts = time_split(df, test_days=3.5)
    summary["test_split_ts"] = str(split_ts)
    summary["n_test"] = int(len(test_df))

    # Predict on test
    log("Predicting on test set ...")
    Xtest = test_df[FEATURE_COLUMNS].astype("float32")
    probs = model.predict_proba(Xtest)[:, 1]

    from sklearn.metrics import roc_auc_score
    summary["test_auc"] = float(roc_auc_score(test_df["y"], probs))
    log(f"  test AUC = {summary['test_auc']:.4f}")

    summary.update(fig_calibration(test_df["y"].values, probs))
    summary.update(fig_precision_at_k(test_df, probs))
    summary.update(fig_hour_weekday_heatmap(df))
    summary.update(fig_baseline_comparison(test_df, probs))
    summary.update(fig_global_moran(df, station_static))
    summary.update(fig_hotspot_persistence(df, station_static, n_samples=60))

    out_json = OUT_DIR / "summary_metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log(f"summary -> {out_json.name}")
    log("Done.")


if __name__ == "__main__":
    main()
