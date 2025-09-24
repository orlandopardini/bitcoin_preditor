
"""
btc_retrain_v4.py — Retreinamento offline noturno (calibração de hiperparâmetros)
- Lê histórico do SQLite
- Gera as mesmas features do v4
- Faz walk-forward com grid maior de hiperparâmetros
- Salva os melhores no TABELA_META (key='v4_state') para uso pelo btc_predictor_v4.py
Execute manualmente ou agende no Windows Task Scheduler.
"""

import os, sqlite3, json, numpy as np, pandas as pd
from datetime import datetime
# === DB PATH (PORTABLE) ===
import os
from pathlib import Path as _Path
DB_PATH = os.environ.get("CRYPTO_DB")
if not DB_PATH:
    _ROOT = _Path(__file__).resolve().parent
    _DBDIR = _ROOT / "GIT" / "DB"
    _DBDIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = str(_DBDIR / "cripto.sqlite")
print(f"[CONFIG] DB_PATH = {DB_PATH}")
# ===========================

SYMBOL = 'BTC/USDT'
ATIVO = SYMBOL.split('/')[0].lower()
TABELA_FONTE = f"{ATIVO}_realtime"
TABELA_META  = f"{ATIVO}_model_meta"
# BASE_PATH removed (using DB_PATH)
# DB_PATH removed (using portable DB_PATH)
RET_CLIP = 0.05
EPS = 1e-6

def meta_set(key, value):
    val = json.dumps(value) if not isinstance(value, str) else value
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {TABELA_META} (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(f"INSERT OR REPLACE INTO {TABELA_META}(key,value) VALUES(?,?)", (key, val))

def read_source():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {TABELA_FONTE} ORDER BY Datetime ASC", conn, parse_dates=['Datetime'])
    if df.empty: return df
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.dropna(subset=['Datetime'], inplace=True)
    df.sort_values('Datetime', inplace=True)
    for c in ['Price','Open','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['Price'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- Indicadores (iguais ao v4) ---
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down + EPS); return 100 - (100/(1+rs))
def macd(series, fast=12, slow=26, signal=9):
    ef, es = ema(series, fast), ema(series, slow)
    line = ef - es; sig = ema(line, signal); hist = line - sig; return line, sig, hist
def bollinger(series, window=20, nstd=2):
    ma = series.rolling(window).mean(); sd = series.rolling(window).std()
    upper, lower = ma + nstd*sd, ma - nstd*sd
    width = (upper - lower) / (ma + EPS); return upper, lower, width
def obv(price, volume):
    sign = np.sign(price.diff().fillna(0)); return (sign*volume).fillna(0).cumsum()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['ret'] = d['Price'].pct_change().clip(-RET_CLIP, RET_CLIP).fillna(0.0)
    d['vol_chg'] = d['Volume'].pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)
    d['log_vol'] = np.log1p(d['Volume']).replace([np.inf,-np.inf], 0.0)
    for lag in [1,2,3,4,5,10,15,30,60]:
        d[f'ret_lag_{lag}'] = d['ret'].shift(lag)
    for w in [5,10,20,30]:
        ema_w = ema(d['Price'], w)
        d[f'ema_gap_{w}'] = (d['Price']/ema_w - 1.0).clip(-0.2,0.2)
        d[f'ema_slope_{w}'] = ema_w.pct_change().clip(-0.1,0.1)
    for w in [5,10,20,30]:
        d[f'rstd_{w}'] = d['ret'].rolling(w).std().clip(0, 0.05)
    d['rsi_14'] = rsi(d['Price'], 14).fillna(50.0)
    macd_line, signal_line, hist = macd(d['Price'])
    d['macd'], d['macd_signal'], d['macd_hist'] = macd_line, signal_line, hist
    _, _, bb_w = bollinger(d['Price'], 20, 2)
    d['bb_width'] = bb_w.fillna(0.0)
    d['obv'] = obv(d['Price'], d['Volume']).fillna(0.0)
    d['obv_chg'] = d['obv'].pct_change().replace([np.inf,-np.inf],0.0).fillna(0.0).clip(-0.5,0.5)
    mod = d['Datetime'].dt.hour*60 + d['Datetime'].dt.minute
    d['sin_m'] = np.sin(2*np.pi*mod/1440); d['cos_m'] = np.cos(2*np.pi*mod/1440)
    d['y_ret'] = d['Price'].pct_change().shift(-1).clip(-RET_CLIP, RET_CLIP)
    d = d.dropna().reset_index(drop=True)
    return d

def walk_forward_grid(feat):
    num = feat.select_dtypes(include=[np.number]).copy()
    y = num['y_ret'].values
    X = num.drop(columns=['y_ret'], errors='ignore').values
    if len(y) < 1500:
        print("Histórico insuficiente para calibração robusta (>=1500)."); return None
    # Pequeno grid
    grid = [
        {"sgd":{"epsilon":1e-3,"alpha":1e-5},"ridge":{"alpha":1e-5},"pa":{"C":0.25}},
        {"sgd":{"epsilon":1e-3,"alpha":1e-4},"ridge":{"alpha":1e-4},"pa":{"C":0.5}},
        {"sgd":{"epsilon":1e-2,"alpha":1e-4},"ridge":{"alpha":1e-3},"pa":{"C":1.0}},
        {"sgd":{"epsilon":5e-3,"alpha":5e-5},"ridge":{"alpha":5e-5},"pa":{"C":0.75}},
    ]
    best=None; best_mae=1e9
    # 3 folds walk-forward
    folds = 3
    step = len(y)//(folds+1)
    for hps in grid:
        maes=[]
        for f in range(1, folds+1):
            start = 0
            split = f*step
            Xtr, ytr = X[start:split], y[start:split]
            Xte, yte = X[split:split+step], y[split:split+step]
            # treino incremental simples (como no v4)
            from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
            from sklearn.preprocessing import StandardScaler
            def fit_block(model):
                sc = StandardScaler(with_mean=True, with_std=True)
                sc.partial_fit(Xtr); model.partial_fit(sc.transform(Xtr), ytr)
                preds=[]; last = split-1
                for i in range(split, split+step-1):
                    xi = sc.transform(X[i:i+1])
                    preds.append(np.clip(model.predict(xi)[0], -RET_CLIP, RET_CLIP))
                    sc.partial_fit(X[i:i+1]); model.partial_fit(sc.transform(X[i:i+1]), y[i:i+1])
                return np.array(preds), y[split+1:split+step]
            m1=SGDRegressor(loss='huber', epsilon=hps["sgd"]["epsilon"], penalty='l2', alpha=hps["sgd"]["alpha"], max_iter=1, tol=None, random_state=42)
            m2=SGDRegressor(loss='squared_error', penalty='l2', alpha=hps["ridge"]["alpha"], max_iter=1, tol=None, random_state=42)
            m3=PassiveAggressiveRegressor(C=hps["pa"]["C"], max_iter=1, tol=None, random_state=42, loss='epsilon_insensitive')
            p1,y1=fit_block(m1); p2,y2=fit_block(m2); p3,y3=fit_block(m3)
            ens = (p1+p2+p3)/3.0
            mae = float(np.mean(np.abs(ens - y1)))
            maes.append(mae)
        g_mae = float(np.mean(maes))
        if g_mae < best_mae: best_mae=g_mae; best=hps
    return best

def main():
    df = read_source()
    if df.empty:
        print("Sem dados."); return
    feat = make_features(df)
    best = walk_forward_grid(feat)
    if best is None:
        print("Calibração não executada (poucos dados)."); return
    meta_set('v4_state', {"hps": best, "tuned_at": datetime.utcnow().isoformat()})
    print("Hiperparâmetros salvos em v4_state:", best)

if __name__ == "__main__":
    main()