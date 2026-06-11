[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xfVbwuLD)
# [Group7] YouBike 站點投放與缺車率的空間相關性分析
本專案旨在解決共享單車因通勤時段產生的「需求不對稱」問題
透過空間統計學識別缺車群聚（Spatial Clustering），並結合機器學習模型預測未來 60 分鐘的缺車風險，提供即時調度決策支援

## Demo

![Demo](results/demo/demo.gif)

## Contributors
|組員|系級|學號|工作分配|
|-|-|-|-|
|陳則銓|統計四|111304015|空間分析| 
|鄭荏鍇|統計四|111304055|空間分析| 
|俞廷翰|統計三|112304043|空間分析| 
|徐鈺蓉|資科碩一|114753201|模型訓練| 
|林陽一|資科碩一|114753208|特徵工程| 
|陳劭瑋|資科碩一|114753212|資料收集、特徵工程、海報製作| 
|黃晉澄|資科碩一|114753216|前端展現、資料發布、github管理| 

## Quick start
Execute the following command every 10 minutes for online verification and prediction:
```bash
python code/online_verify_once.py
```

Use the following command to start shiny app:
```R
shiny run --reload code/app.py
```

## Folder organization and its related description
idea by Noble WS (2009) [A Quick Guide to Organizing Computational Biology Projects.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424) PLoS Comput Biol 5(7): e1000424.

### docs
* Your presentation, 1142_DS-FP_groupID.ppt/pptx/pdf (i.e.,1142_DS-FP_group1.ppt), by **06.09**
* Any related document for the project, i.e.,
  * discussion log
  * software user guide

### data
* Input
  * Source https://data.taipei/dataset/detail?id=c6bc8aed-557d-41d5-bfb1-8da24f78f2fb
  * Format json
  * Size   1743 站 × 每天抓取 144 次 × 共 27 天（約 3.5 GB）

### code
* Analysis steps
整體分為「離線訓練」與「線上推論」兩條 pipeline，共用同一套 17 維特徵定義：
  - 資料採集與清理（M1）：從 YouBike 2.0 Open API 抓取約 1700 個站點快照，過濾 Quantity > 0、act = 1，計算缺車率
  shortage_rate = 1 − available_rent_bikes / Quantity
  - 空間分析（M2, m2/lisa.py）：以 k-NN（k = 6）row-normalized 權重矩陣 W 計算：
    - 連續空間 lag spatial_lag_shortage = W @ shortage_rate（餵模型，特徵 #16）
    - Local Moran's I（LISA） → HH / LL / LH / HL / NS quadrant + pseudo p-value（999 次 conditional permutation；給前端地圖著色，不餵模型，以避免 data leakage）
  - 特徵工程（M3, m3/build_features.py）：時間 lag（10 / 20 / 30 / 60 分）、變化率（delta）、時間週期特徵（hour 與 day-of-week 的 sin / cos、is_rush_hour、is_weekend）、distance_to_mrt（站點到最近捷運站的 Haversine 距離）
  - 標籤定義：站點 i 在時間 t，若未來 60 分鐘之 shortage_rate > 0.8 則 y = 1
  - 模型訓練與評估（M4, m4/train.py）：產出 model.pkl 與 metadata.json（包含 model_version）
  - 線上推論（m4/predictor.py）：載入模型，驗證 17 個欄位的順序與 NaN 狀態，輸出 pred_prob，並經由 GitHub repository 中介層提供給 Shiny 前端

* Which method or package do you use?
  - 預測模型：使用 XGBoost（XGBClassifier, objective=binary: logistic）作為二元分類模型，預測未來 60 分鐘缺車機率
  - 空間統計：PySAL 生態系 - libpysal.weights.KNN（權重矩陣）、esda.moran.Moran_Local（LISA）
  - 評估/前處理：使用 scikit-learn 的 TimeSeriesSplit 與 metrics；LogisticRegression/StandardScaler 僅用於 baseline 比較
  - 資料處理：pandas / numpy、模型序列化 joblib
* How do you perform training and evaluation?
  - m4/train.py：使用 TimeSeriesSplit(n_splits=5) 進行時序交叉驗證，確保每一折的訓練資料都早於驗證資料，每折用 early_stopping_rounds = 30
  找收斂點，再以各折 best_iteration 的平均值（397 棵樹）在全部資料上重訓最終模型
  - m4/test.py / 海報圖：額外用時序 80/20 chronological holdout（前 80% 訓練、後 20% 完全分離測試），報告 held-out test 表現
  - 類別不平衡：正負比約 1:2.35，以 scale_pos_weight = neg/pos（負樣本數 / 正樣本數）調整正類別權重
  - Train-Serve Skew 防護：離線用 groupby('sno').shift(k)、線上用 ring buffer，兩條路徑以 code/test_feature_parity.py 做 np.allclose(atol = 1e-9) 等價性測試，並驗證切點前 6 筆 lag 為 NaN（防切點洩漏）
* What is a null model for comparison?
  Persistence baseline（持續性模型）：直接用「當下的 shortage_rate」當作 60 分鐘後缺車機率的預測（persistence_prob = shortage_rate, code/generate_poster_figures.py:266）

### results
* What is your performance?
  - 5-fold TimeSeriesSplit CV（metadata）：AUC 0.917、Brier 0.115、Recall 0.853、Precision 0.692
  - 最重要特徵：shortage_rate（0.497）≫ lag_10min（0.193）> lag_20min（0.097）- 即時供需與短期 lag 為主要預測訊號
  - 空間分析：Global Moran's I 在整段時窗介於 0.146-0.344（mean = 0.279），全部時間點皆達顯著水準（顯示缺車現象具有空間群聚特性）；LISA 找出 4 個「慢性熱點」（≥50% 時間為 HH）、116 個「頻繁熱點」（30-50%）
* Is the improvement significant?
  - AUC 提升 0.032，Persistence 本身 AUC 已達 0.877（缺車具有強自相關性，因此 baseline 並不弱），所以 3.2pp 的排序能力提升是在高基準上的增益
  - 真正關鍵在機率校準：Brier 從 0.230 降到 0.127、Persistence 把原始缺車率當機率，校準極差；XGBoost 同時改善排序與機率準確度，這對「Top-k 調度場景（Precision@10 = 0.978）」直接轉化為調度命中率

## References
* Packages you use
  Packages: XGBoost、PySAL(libpysal、esda)、scikit-learn、pandas、numpy、joblib；線上系統另用 Shiny for Python、requests、tenacity、loguru、APScheduler
* Related publications
  - Anselin (1995) - LISA, Geographical Analysis 27(2)
  - Moran (1950) - Spatial autocorrelation, Biometrika 37
  - Rey & Anselin (2007) - PySAL, Review of Regional Studies 37(1)
  - Chen & Guestrin (2016) - XGBoost, KDD
  - Brier (1950) - Probability forecast verification, Monthly Weather Review 78(1)
  - Faghih-Imani & Eluru (2016) - Spatio-temporal bike-sharing demand, J. Transport Geography 54
  - YouBike 2.0 Open API（台北市交通局、data.taipei）
