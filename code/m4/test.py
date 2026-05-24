# use training csv to test
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


MODEL_PATH = Path("models/xgb_v1_20260520.pkl")
DATA_PATH = Path("dataset/training_dataset.csv")


FEATURE_COLUMNS = [
    "shortage_rate",
    "available_bikes",
    "total_capacity",
    "lag_10min",
    "lag_20min",
    "lag_30min",
    "lag_60min",
    "delta_10min",
    "delta_30min",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_rush_hour",
    "is_weekend",
    "spatial_lag_shortage",
    "distance_to_mrt",
]


def precision_at_k(df, prob_col="pred_prob", label_col="y", k=10):
    results = []




    for timestamp, group in df.groupby("timestamp"):
        top_k = group.sort_values(prob_col, ascending=False).head(k)




        if len(top_k) == 0:
            continue


        precision = top_k[label_col].mean()


        results.append({
            "timestamp": timestamp,
            "k": k,
            "precision_at_k": precision,
            "n_positive_in_top_k": int(top_k[label_col].sum()),
            "n_samples": len(top_k),
        })


    result_df = pd.DataFrame(results)


    if len(result_df) == 0:
        return np.nan, result_df


    return result_df["precision_at_k"].mean(), result_df




def main():
    df = pd.read_csv(DATA_PATH)


    required_cols = ["sno", "timestamp", "y"] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sno"] = df["sno"].astype(str)


    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    df["y"] = pd.to_numeric(df["y"], errors="coerce")


    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["y"]).copy()
    after = len(df)


    print(f"Rows before dropna: {before}")
    print(f"Rows after dropna: {after}")


    df["y"] = df["y"].astype(int)


    print("\nLabel distribution:")
    print(df["y"].value_counts())
    print(df["y"].value_counts(normalize=True))


    df = df.sort_values("timestamp").reset_index(drop=True)


    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()


    print("\nTest time range:")
    print(test_df["timestamp"].min(), "~", test_df["timestamp"].max())


    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["y"]


    model = joblib.load(MODEL_PATH)


    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]
    test_df["pred_label"] = (test_df["pred_prob"] >= 0.5).astype(int)


    auc = roc_auc_score(y_test, test_df["pred_prob"])
    brier = brier_score_loss(y_test, test_df["pred_prob"])
    acc = accuracy_score(y_test, test_df["pred_label"])
    precision = precision_score(y_test, test_df["pred_label"], zero_division=0)
    recall = recall_score(y_test, test_df["pred_label"], zero_division=0)
    f1 = f1_score(y_test, test_df["pred_label"], zero_division=0)


    print("\n=== Overall Test Metrics ===")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Brier:     {brier:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, test_df["pred_label"]))


    print("\n=== Classification Report ===")
    print(classification_report(y_test, test_df["pred_label"], zero_division=0))


    p_at_10, p10_df = precision_at_k(test_df, k=10)
    p_at_20, p20_df = precision_at_k(test_df, k=20)


    print("\n=== Dispatch Metrics ===")
    print(f"Precision@10: {p_at_10:.4f}")
    print(f"Precision@20: {p_at_20:.4f}")


    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)


    pred_path = out_dir / "test_predictions.csv"
    metrics_path = out_dir / "test_metrics.json"
    p10_path = out_dir / "precision_at_10_by_time.csv"
    p20_path = out_dir / "precision_at_20_by_time.csv"


    test_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    p10_df.to_csv(p10_path, index=False, encoding="utf-8-sig")
    p20_df.to_csv(p20_path, index=False, encoding="utf-8-sig")


    metrics = {
        "model_path": str(MODEL_PATH),
        "data_path": str(DATA_PATH),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(test_df["timestamp"].max()),
        "n_test_samples": int(len(test_df)),
        "positive_rate": float(y_test.mean()),
        "auc_roc": float(auc),
        "brier": float(brier),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_at_10": float(p_at_10),
        "precision_at_20": float(p_at_20),
    }


    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


    print("\nSaved outputs:")
    print(pred_path)
    print(metrics_path)
    print(p10_path)
    print(p20_path)


    print("\n=== Top 20 Predicted Risk Stations ===")
    top20 = test_df.sort_values("pred_prob", ascending=False).head(20)
    show_cols = [
        "timestamp",
        "sno",
        "shortage_rate",
        "available_bikes",
        "total_capacity",
        "spatial_lag_shortage",
        "distance_to_mrt",
        "pred_prob",
        "y",
    ]
    print(top20[show_cols])


if __name__ == "__main__":
    main()


