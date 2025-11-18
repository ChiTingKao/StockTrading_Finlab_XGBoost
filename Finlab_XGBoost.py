import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from finlab import data
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# 從 finlab 取得資料
Closes = data.get('etl:adj_close')
Opens = data.get('etl:adj_open')
Highs = data.get('etl:adj_high')
Lows = data.get('etl:adj_low')
Volumes = data.get('after_market_odd_lot_trade:成交股數')
Foreign_Investors = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
Investment_Trust = data.get('institutional_investors_trading_summary:投信買賣超股數')
Dealer = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
Margin_Trading = data.get('margin_transactions:融資今日餘額')
Short_Selling = data.get('margin_transactions:融券今日餘額')


# 資料前處理一，將資料由寬表格改為窄表格
Closes = Closes.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Close')
Opens = Opens.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Open')
Highs = Highs.reset_index().melt(id_vars='date', var_name='stock_id', value_name='High')
Lows = Lows.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Low')
Volumes = Volumes.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Volume')
Foreign_Investors = Foreign_Investors.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Foreign_Investor')
Investment_Trust = Investment_Trust.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Investment_Trust')
Dealer = Dealer.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Dealer')
Margin_Trading = Margin_Trading.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Margin_Trading')
Short_Selling = Short_Selling.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Short_Selling')


# 資料前處理二，將所有表格合併
df = Closes.merge(Opens, on=['date','stock_id'], how='left')
df = df.merge(Highs, on=['date','stock_id'], how='left')
df = df.merge(Lows, on=['date','stock_id'], how='left')
df = df.merge(Volumes, on=['date','stock_id'], how='left')
df = df.merge(Foreign_Investors, on=['date','stock_id'], how='left')
df = df.merge(Investment_Trust, on=['date','stock_id'], how='left')
df = df.merge(Dealer, on=['date','stock_id'], how='left')
df = df.merge(Margin_Trading, on=['date','stock_id'], how='left')
df = df.merge(Short_Selling, on=['date','stock_id'], how='left')


# 技術指標
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal = macd_line.ewm(span=span_signal, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


# 資料前處理三，加入技術指標
# 當日報酬率
df['ret'] = df['Close'].pct_change()

# 前幾日的報酬率
for l in [1,2,3,5]:
    df[f'lag_ret_{l}'] = df.groupby('stock_id')['ret'].shift(l)

# MA
df['ma5']  = df.groupby('stock_id')['Close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['ma10'] = df.groupby('stock_id')['Close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
df['ma20'] = df.groupby('stock_id')['Close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma_ratio_5_20'] = df['ma5'] / (df['ma20'] + 1e-9)

# 成交量波動
df['vol_10'] = df.groupby('stock_id')['Volume'].transform(lambda x: x.rolling(10, min_periods=1).std())
df['vol_20'] = df.groupby('stock_id')['Volume'].transform(lambda x: x.rolling(20, min_periods=1).std())

# RSI
df['rsi_14'] = df.groupby('stock_id')['Close'].transform(lambda x: rsi(x, period=14))

# MACD
df['macd'] = df.groupby('stock_id')['Close'].transform(lambda x: macd(x)[0])
df['macd_signal'] = df.groupby('stock_id')['Close'].transform(lambda x: macd(x)[1])
df['macd_hist'] = df.groupby('stock_id')['Close'].transform(lambda x: macd(x)[2])

# 三大法人近十日買賣超
df['foreign_10'] = df.groupby('stock_id')['Foreign_Investor'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
df['investment_10'] = df.groupby('stock_id')['Investment_Trust'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
df['dealer_10'] = df.groupby('stock_id')['Dealer'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())

# 融資融券近十日
df['margin_10'] = df.groupby('stock_id')['Margin_Trading'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
df['short_10'] = df.groupby('stock_id')['Short_Selling'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())

# 目標: 隔日漲跌 (1 = 漲, 0 = 平盤或跌)
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

df.dropna(inplace=True)
print(df)


# XGBoost 預測股價
"""
技術指標 + 籌碼指標 + XGBoost
"""
# features list
features = [
    'lag_ret_1','lag_ret_2','lag_ret_3','lag_ret_5',
    'ma5','ma10','ma20','ma_ratio_5_20',
    'vol_10','vol_20',
    'rsi_14',
    'macd','macd_signal','macd_hist',
    'Foreign_Investor','Investment_Trust','Dealer',
    'foreign_10','investment_10','dealer_10',
    'Margin_Trading','Short_Selling',
    'margin_10','short_10'
]

# 確保資料排序
df = df.sort_values(['stock_id','date'])

# 取得所有日期並切割：Train 70%; Val 15%; Test 15%
all_dates = df['date'].sort_values().unique()
n_total = len(all_dates)

train_end = int(n_total*0.7)     
val_end   = int(n_total*0.85)   

train_dates = all_dates[:train_end]
val_dates   = all_dates[train_end:val_end]
test_dates  = all_dates[val_end:]

# 按日期切割資料
train_df = df.loc[df['date'].isin(train_dates)].copy()
val_df   = df.loc[df['date'].isin(val_dates)].copy()
test_df  = df.loc[df['date'].isin(test_dates)].copy()

# 生成 X, y
X_train, y_train = train_df[features], train_df['target']
X_val, y_val     = val_df[features], val_df['target']
X_test, y_test   = test_df[features], test_df['target']


# 模型訓練 (XGBoost)
# 建立 DMatrix（XGBoost 專用資料格式）
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test, label=y_test)

# 設定模型參數
params = {
    'objective': 'binary:logistic',  # 二分類
    'eval_metric': 'auc',            # 評估指標
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# 訓練模型（監控驗證集）
evallist = [(dtrain, 'train'), (dval, 'eval')]
num_round = 200

bst = xgb.train(
    params,
    dtrain,
    num_round,
    evals=evallist,
    early_stopping_rounds=20,  # 驗證集 20 次沒提升就停止
    verbose_eval=10
)

# 驗證集預測
y_val_pred = bst.predict(dval)
y_val_pred_label = (y_val_pred > 0.5).astype(int)

# 測試集預測
y_test_pred = bst.predict(dtest)
y_test_pred_label = (y_test_pred > 0.5).astype(int)

acc = accuracy_score(y_test, y_test_pred_label)
auc = roc_auc_score(y_test, y_test_pred)

print("Accuracy:", acc)
print("AUC:", auc)
print(classification_report(y_test, (y_test_pred > 0.5).astype(int)))


# 模型回測
backtest = X_test.copy()
backtest['stock_id'] = df.loc[X_test.index, 'stock_id']
backtest['date'] = df.loc[X_test.index, 'date']

# 使用標籤作為信號
backtest['pred_signal'] = y_test_pred_label

# 計算隔日報酬
backtest['next_ret'] = df.loc[backtest.index, 'lag_ret_1']

# ------------------------
# 加入手續費 / 滑點
# ------------------------
trade_cost = 0.003  # 每次交易成本 0.3%

# 計算是否有交易（signal 變化才交易）
# shift(1) 判斷昨天是否持倉
backtest['prev_signal'] = backtest.groupby('stock_id')['pred_signal'].shift(1).fillna(0)
backtest['trade'] = (backtest['pred_signal'] != backtest['prev_signal']).astype(int)

# 沒手續費的策略
backtest['strat_ret_no_fee'] = backtest['pred_signal'] * backtest['next_ret']
# 扣除手續費
backtest['strat_ret'] = backtest['pred_signal'] * backtest['next_ret'] - backtest['trade'] * trade_cost

# 每日投資組合報酬 (等權平均多支股票)
daily_strat_no_fee = backtest.groupby('date')['strat_ret_no_fee'].mean()
daily_strat = backtest.groupby('date')['strat_ret'].mean()
daily_market = backtest.groupby('date')['next_ret'].mean()

# 累積報酬
cum_strat_no_fee = (1 + daily_strat_no_fee).cumprod() - 1
cum_strat = (1 + daily_strat).cumprod() - 1
cum_market = (1 + daily_market).cumprod() - 1

# 繪圖
plt.figure(figsize=(10,5))
plt.plot(cum_strat_no_fee, label='Model Strategy No Trading Fee')
plt.plot(cum_strat, label='Model Strategy')
plt.plot(cum_market, label='Buy & Hold (Market)')
plt.legend()
plt.title("Cumulative Returns (with trading cost)")
plt.show()

# 計算年化績效
def perf_stats(returns_series):
    ann_return = (1 + returns_series).prod() ** (252 / len(returns_series)) - 1 # 年化報酬率
    ann_vol = returns_series.std() * np.sqrt(252) # 年化波動率
    sharpe = ann_return / (ann_vol + 1e-9) # 夏普比率(年化報酬率/年化波動率)
    return ann_return, ann_vol, sharpe

ann_mkt, vol_mkt, sharpe_mkt = perf_stats(daily_market)
ann_strat, vol_strat, sharpe_strat = perf_stats(daily_returns)
ann_strat_no_fee, vol_strat_no_fee, sharpe_strat_no_fee = perf_stats(daily_returns)

print("Market Ann Return / Vol / Sharpe:", ann_mkt, vol_mkt, sharpe_mkt)
print("Strategy Ann Return / Vol / Sharpe:", ann_strat, vol_strat, sharpe_strat)
print("Strategy No Trading Fee Ann Return / Vol / Sharpe:", ann_strat_no_fee, vol_strat_no_fee, sharpe_strat_no_fee)

