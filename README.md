### 使用 Finlab 台股資料庫，整合 技術指標 + 籌碼指標，透過 XGBoost 預測隔日股價漲跌，並建立簡易的量化交易策略回測模型。

#### -資料取得
從 Finlab 下載每日股價、成交量、三大法人、融資融券等資料。

#### -特徵工程
技術指標：MA、RSI、MACD、成交量波動
籌碼指標：外資、投信、自營商近十日買賣超
融資 / 融券變化
報酬率與 lag features

#### -模型訓練
Train / Validation / Test = 70% / 15% / 15%
使用 XGBoost（二元分類）預測隔日漲跌
評估指標：AUC

#### -回測與績效圖示
<img width="858" height="451" alt="image" src="https://github.com/user-attachments/assets/6da72f3d-bdf1-46f0-84dd-803a29c9ee79" />

##### Market Ann Return / Vol / Sharpe: -0.243459910855927 0.17599802297442343 -1.3833104790500792
##### Strategy Ann Return / Vol / Sharpe: -0.09865465820132546 0.05126674295158096 -1.9243402369087488
##### Strategy No Trading Fee Ann Return / Vol / Sharpe: 0.04229707937438132 0.05121553989755971 0.8258641543781229

