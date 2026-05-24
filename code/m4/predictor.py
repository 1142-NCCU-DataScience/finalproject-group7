"""
1. 載入已訓練好的 XGBoost model (under ./models )
2. 確認輸入 feature 是否符合 17 個欄位
3. 呼叫 model.predict_proba()
4. 輸出 pred_prob (60 分鐘後缺車的機率)
"""

from pathlib import Path

import joblib
import pandas as pd

from m3.build_features import FEATURE_COLUMNS, validate_features


class ShortagePredictor:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        self.model_version = self.model_path.stem

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        X = feature_df[FEATURE_COLUMNS].copy()

        validate_features(X, strict=True)

        if X.isna().any().any():
            na_counts = X.isna().sum()
            bad = na_counts[na_counts > 0]
            raise ValueError(f"NaN in online feature matrix:\n{bad}")

        out = feature_df.copy()
        out["pred_prob"] = self.model.predict_proba(X)[:, 1]
        out["model_version"] = self.model_version

        return out