import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
 
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

# load data and set model path
DATA_PATH = Path("./dataset/training_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# columns
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
 
def main():
    df = pd.read_csv(DATA_PATH)

    # check columns
    required_cols = ["sno", "timestamp", "y"] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
 
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sno"] = df["sno"].astype(str)

    # change type (若出現無法轉成數值的資料則紀錄nan)
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # remove NA
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["y"]).copy()
    after = len(df)
    print(f"Rows before dropna: {before}")
    print(f"Rows after dropna:  {after}")
 
    df["y"] = df["y"].astype(int)
 
    print("\nLabel distribution:")
    print(df["y"].value_counts())
 
    if df["y"].nunique() < 2:
        raise ValueError("y 只有一個類別，無法訓練二元分類模型。")
 
    # sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
 
    X = df[FEATURE_COLUMNS]
    y = df["y"]
 
    print("\nTraining data shape:", X.shape)
    print("Time range:", df["timestamp"].min(), "~", df["timestamp"].max())
    print("Stations:", df["sno"].nunique())

    # may encounter data imbalance problem ('y' column)
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nClass balance — neg: {neg}, pos: {pos}, scale_pos_weight: {scale_pos_weight:.2f}")

    params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "scale_pos_weight": scale_pos_weight, # deal with data imbalance
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    }

    # timeseriessplit cross validation
    # 確保訓練集永遠在驗證集的前面
    tscv = TimeSeriesSplit(n_splits=5)

    auc_scores       = []
    brier_scores     = []
    acc_scores       = []
    precision_scores = []
    recall_scores    = []
    best_iterations  = []


    print("\n Cross-Validation")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # use early stop to make sure achieve convergence
        model = XGBClassifier(**params, early_stopping_rounds=30)

        # fit the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = model.best_iteration if hasattr(model, "best_iteration") else params["n_estimators"]
        best_iterations.append(best_iter)
 
        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
 
        auc   = roc_auc_score(y_val, val_prob)
        brier = brier_score_loss(y_val, val_prob)
        acc   = accuracy_score(y_val, val_pred)
        prec  = precision_score(y_val, val_pred, zero_division=0)
        rec   = recall_score(y_val, val_pred, zero_division=0)
 
        auc_scores.append(auc)
        brier_scores.append(brier)
        acc_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
 
        print(
            f"Fold {fold}: "
            f"AUC={auc:.4f}, "
            f"Brier={brier:.4f}, "
            f"Acc={acc:.4f}, "
            f"Precision={prec:.4f}, "
            f"Recall={rec:.4f}, "
            f"best_iter={best_iter}"
        )
   
    print("\n  CV Summary")
    print(f"AUC       mean ± std: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Brier     mean ± std: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")
    print(f"Accuracy  mean ± std: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Precision mean ± std: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall    mean ± std: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"best_iter mean:       {int(np.mean(best_iterations))}")


    # get the best one and retrain using all the data
    best_n = max(1, int(np.mean(best_iterations)))
    print(f"\n Training Final Model (n_estimators={best_n}) ")
 
    final_params = {**params, "n_estimators": best_n}
    final_params.pop("eval_metric", None)
 
    final_model = XGBClassifier(**final_params)
    final_model.fit(X, y)


    # save model
    train_end     = df["timestamp"].max().strftime("%Y%m%d")
    model_version = f"xgb_v1_{train_end}"
 
    model_path    = MODEL_DIR / f"{model_version}.pkl"
    metadata_path = MODEL_DIR / f"{model_version}.json"
 
    joblib.dump(final_model, model_path)
 
    metadata = {
        "version":            model_version,
        "data_path":          str(DATA_PATH),
        "train_start":        str(df["timestamp"].min()),
        "train_end":          str(df["timestamp"].max()),
        "n_samples":          int(len(df)),
        "n_stations":         int(df["sno"].nunique()),
        "n_features":         len(FEATURE_COLUMNS),
        "features":           FEATURE_COLUMNS,
        "label_column":       "y",
        "model":              "XGBClassifier",
        "hyperparams":        final_params,
        "scale_pos_weight":   float(scale_pos_weight),
        "best_iter_cv_mean":  best_n,
        "cv_auc_mean":        float(np.mean(auc_scores)),
        "cv_brier_mean":      float(np.mean(brier_scores)),
        "cv_accuracy_mean":   float(np.mean(acc_scores)),
        "cv_precision_mean":  float(np.mean(precision_scores)),
        "cv_recall_mean":     float(np.mean(recall_scores)),
    }
 
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
 
    print(f"\n Model saved:    {model_path}")
    print(f"Metadata saved: {metadata_path}")
 
 
if __name__ == "__main__":
    main()




