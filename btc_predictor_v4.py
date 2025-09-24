
"""
btc_predictor_v4.py — Ensemble online + indicadores técnicos + walk-forward leve
- Modelos (incrementais):
    * SGDRegressor (loss='huber')  -> "sgd"
    * SGDRegressor (loss='squared_error', l2) -> "ridge_sgd" (ridge-like)
    * PassiveAggressiveRegressor   -> "pa"
- Ensemble: média ponderada por 1/(EWM-MAE + eps). Pesos persistidos.
- Alvo: retorno do próximo minuto (clipped ±5%), reconstruindo preço previsto.
- Indicadores: RSI, MACD(12,26,9), Bollinger(20,2), OBV, slope de EMA(5,10,20).
- Timestamps no minuto, merges e gravações defensivas.
- Walk-forward leve (opcional) na primeira execução para calibrar hiperparâmetros.

Requisitos: scikit-learn, pandas, numpy, joblib
"""

import os, time, json, sqlite3, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
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
TABELA_PREV  = f"{ATIVO}_predictions"
TABELA_META  = f"{ATIVO}_model_meta"
# BASE_PATH removed (using DB_PATH)
# DB_PATH removed (using portable DB_PATH)
ARQ_STATE = os.path.join(os.path.dirname(DB_PATH), f\'{ATIVO}_v4_state.json\')
ARQ_MODELS = os.path.join(os.path.dirname(DB_PATH), f\'{ATIVO}_v4_models.joblib\')
ARQ_SCALERS = os.path.join(os.path.dirname(DB_PATH), f\'{ATIVO}_v4_scalers.joblib\')

MIN_HISTORY = 1200   # um pouco mais para indicadores
MAX_WINDOW  = 8000
SLEEP_SEC   = 60

RET_CLIP = 0.05
EPS = 1e-6

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- BD helpers ----------------
def ensure_tables_and_migrate():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABELA_PREV} (
                Datetime TEXT PRIMARY KEY,
                Symbol TEXT,
                y_hat REAL,                -- ensemble
                y_hat_ret REAL,            -- ensemble (retorno)
                baseline REAL,
                horizon_min INTEGER,
                created_at TEXT,
                model_version TEXT,
                mae REAL,
                r2 REAL
            )
        """)
        # add ensemble components if missing
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({TABELA_PREV})").fetchall()}
        add_cols = []
        for col in ["y_hat_sgd","y_hat_ridge","y_hat_pa","w_sgd","w_ridge","w_pa"]:
            if col not in cols:
                add_cols.append(f"ALTER TABLE {TABELA_PREV} ADD COLUMN {col} REAL")
        for cmd in add_cols:
            conn.execute(cmd)
        # meta table
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABELA_META} (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # limpeza NaT
        conn.execute(f"DELETE FROM {TABELA_PREV} WHERE Datetime IS NULL OR Datetime='' OR Datetime='NaT'")

def meta_get(key, default=None):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(f"SELECT value FROM {TABELA_META} WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row[0])
    except Exception:
        return row[0]

def meta_set(key, value):
    val = json.dumps(value) if not isinstance(value, str) else value
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"INSERT OR REPLACE INTO {TABELA_META}(key,value) VALUES(?,?)", (key, val))

def read_source(limit=MAX_WINDOW):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {TABELA_FONTE} ORDER BY Datetime ASC", conn, parse_dates=['Datetime'])
    if df.empty: return df
    # limpeza
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.dropna(subset=['Datetime'], inplace=True)
    df.sort_values('Datetime', inplace=True)
    df.drop_duplicates(subset='Datetime', keep='last', inplace=True)
    for c in ['Price','Open','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['Price'], inplace=True)
    if len(df) > limit: df = df.iloc[-limit:]
    df['Minute'] = df['Datetime'].dt.floor('min')
    df.dropna(subset=['Minute'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def upsert_prediction(ts_minute, row_dict):
    if pd.isna(ts_minute): return
    ts_str = pd.Timestamp(ts_minute).floor('min').strftime("%Y-%m-%d %H:%M:00")
    fields = ["Datetime","Symbol","y_hat","y_hat_ret","baseline","horizon_min","created_at","model_version",
              "mae","r2","y_hat_sgd","y_hat_ridge","y_hat_pa","w_sgd","w_ridge","w_pa"]
    values = [
        ts_str, SYMBOL, float(row_dict.get("y_hat")) if row_dict.get("y_hat") is not None else None,
        float(row_dict.get("y_hat_ret")) if row_dict.get("y_hat_ret") is not None else None,
        float(row_dict.get("baseline")) if row_dict.get("baseline") is not None else None,
        1, datetime.utcnow().isoformat(), "v4_ensemble",
        float(row_dict.get("mae")) if row_dict.get("mae") is not None else None,
        None,
        float(row_dict.get("y_hat_sgd")) if row_dict.get("y_hat_sgd") is not None else None,
        float(row_dict.get("y_hat_ridge")) if row_dict.get("y_hat_ridge") is not None else None,
        float(row_dict.get("y_hat_pa")) if row_dict.get("y_hat_pa") is not None else None,
        float(row_dict.get("w_sgd")) if row_dict.get("w_sgd") is not None else None,
        float(row_dict.get("w_ridge")) if row_dict.get("w_ridge") is not None else None,
        float(row_dict.get("w_pa")) if row_dict.get("w_pa") is not None else None,
    ]
    placeholders = ",".join(["?"]*len(fields))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"INSERT OR REPLACE INTO {TABELA_PREV} ({','.join(fields)}) VALUES ({placeholders})", values)

# ---------------- Indicadores ----------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down + EPS)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series, window=20, nstd=2):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + nstd*sd
    lower = ma - nstd*sd
    width = (upper - lower) / (ma + EPS)
    return upper, lower, width

def obv(price, volume):
    sign = np.sign(price.diff().fillna(0))
    return (sign * volume).fillna(0).cumsum()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['ret'] = d['Price'].pct_change().clip(-RET_CLIP, RET_CLIP).fillna(0.0)
    d['vol_chg'] = d['Volume'].pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)
    d['log_vol'] = np.log1p(d['Volume']).replace([np.inf,-np.inf], 0.0)

    # lags de retorno
    for lag in [1,2,3,4,5,10,15,30,60]:
        d[f'ret_lag_{lag}'] = d['ret'].shift(lag)

    # EMAs e gaps + slopes
    for w in [5,10,20,30]:
        ema_w = ema(d['Price'], w)
        d[f'ema_gap_{w}'] = (d['Price']/ema_w - 1.0).clip(-0.2,0.2)
        d[f'ema_slope_{w}'] = ema_w.pct_change().clip(-0.1,0.1)

    # Volatilidade
    for w in [5,10,20,30]:
        d[f'rstd_{w}'] = d['ret'].rolling(w).std().clip(0, 0.05)

    # RSI
    d['rsi_14'] = rsi(d['Price'], 14).fillna(50.0)
    # MACD
    macd_line, signal_line, hist = macd(d['Price'])
    d['macd'] = macd_line; d['macd_signal'] = signal_line; d['macd_hist'] = hist
    # Bollinger
    bb_u, bb_l, bb_w = bollinger(d['Price'], 20, 2)
    d['bb_width'] = bb_w.fillna(0.0)
    # OBV
    d['obv'] = obv(d['Price'], d['Volume']).fillna(0.0)
    d['obv_chg'] = d['obv'].pct_change().replace([np.inf,-np.inf],0.0).fillna(0.0).clip(-0.5,0.5)

    # Sazonalidade minuto do dia
    minute_of_day = d['Datetime'].dt.hour * 60 + d['Datetime'].dt.minute
    d['sin_m'] = np.sin(2*np.pi*minute_of_day/1440); d['cos_m'] = np.cos(2*np.pi*minute_of_day/1440)

    # Target = retorno t->t+1
    d['y_ret'] = d['Price'].pct_change().shift(-1).clip(-RET_CLIP, RET_CLIP)

    d = d.dropna().reset_index(drop=True)
    return d

# --------------- Modelos ---------------
class OnlineReg:
    def __init__(self, model, name):
        self.model = model
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.name = name
        self.ready = False

    def _clean(self, X):
        X = np.asarray(X, dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_initial(self, X, y):
        X = self._clean(X); y = np.asarray(y, dtype=float)
        self.scaler.partial_fit(X)
        self.model.partial_fit(self.scaler.transform(X), y)
        self.ready = True

    def update(self, X, y):
        X = self._clean(X); y = np.asarray(y, dtype=float)
        self.scaler.partial_fit(X)
        self.model.partial_fit(self.scaler.transform(X), y)

    def predict(self, X):
        X = self._clean(X)
        return self.model.predict(self.scaler.transform(X))

def init_models(hps=None):
    # hyperparams default
    hps = hps or {
        "sgd": {"epsilon": 1e-3, "alpha": 5e-5},
        "ridge": {"alpha": 1e-4},
        "pa": {"C": 0.5}
    }
    m_sgd = SGDRegressor(loss='huber', epsilon=hps["sgd"]["epsilon"], penalty='l2', alpha=hps["sgd"]["alpha"],
                         learning_rate='optimal', max_iter=1, tol=None, random_state=42)
    m_ridge = SGDRegressor(loss='squared_error', penalty='l2', alpha=hps["ridge"]["alpha"],
                           learning_rate='optimal', max_iter=1, tol=None, random_state=42)
    m_pa = PassiveAggressiveRegressor(C=hps["pa"]["C"], max_iter=1, tol=None, random_state=42, loss='epsilon_insensitive')
    return {
        "sgd": OnlineReg(m_sgd, "sgd"),
        "ridge": OnlineReg(m_ridge, "ridge"),
        "pa": OnlineReg(m_pa, "pa")
    }

# --------------- Walk-forward leve ---------------
def walk_forward_calibrate(feat, max_points=2500):
    # usa uma pequena janela para decidir hiperparâmetros
    feat_num = feat.select_dtypes(include=[np.number]).copy()
    if 'y_ret' not in feat_num.columns: return None
    y = feat_num['y_ret'].values
    X = feat_num.drop(columns=['y_ret'], errors='ignore').values
    if len(y) < 400: return None
    X = X[-max_points:]; y = y[-max_points:]

    # grid pequena
    grid = [
        {"sgd":{"epsilon":1e-3,"alpha":1e-5},"ridge":{"alpha":1e-5},"pa":{"C":0.25}},
        {"sgd":{"epsilon":1e-3,"alpha":1e-4},"ridge":{"alpha":1e-4},"pa":{"C":0.5}},
        {"sgd":{"epsilon":1e-2,"alpha":1e-4},"ridge":{"alpha":1e-3},"pa":{"C":1.0}},
    ]

    best = None; best_mae = 1e9
    for hps in grid:
        mods = init_models(hps)
        # fit inicial em 300 pontos, depois atualiza + avalia
        split = 300
        for m in mods.values():
            m.fit_initial(X[:split], y[:split])
        y_pred = []
        for i in range(split, len(y)-1):
            # update com i -> y[i]
            for m in mods.values():
                m.update(X[i:i+1], y[i:i+1])
            # prever próximo retorno (i -> i+1)
            preds = {k: float(np.clip(mods[k].predict(X[i:i+1])[0], -RET_CLIP, RET_CLIP)) for k in mods}
            # ensemble simples ( média )
            y_pred.append(np.mean(list(preds.values())))
        y_true = y[split+1:]
        mae = float(np.mean(np.abs(np.array(y_pred) - y_true)))
        if mae < best_mae:
            best_mae = mae; best = hps
    return best

# --------------- Ensemble weights ---------------
class EWMWeights:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.mae = {"sgd":0.01, "ridge":0.01, "pa":0.01}  # inicia pequeno para não zero

    def update(self, errors):
        # errors: dict model->abs_error
        for k,v in errors.items():
            self.mae[k] = self.alpha*v + (1-self.alpha)*self.mae.get(k, v)

    def weights(self):
        inv = {k: 1.0/(v+EPS) for k,v in self.mae.items()}
        s = sum(inv.values())
        return {k: inv[k]/s for k in inv}

# ----------------- LOOP -----------------
def main():
    os.makedirs(BASE_PATH, exist_ok=True)
    ensure_tables_and_migrate()

    # Estado/Hyperparams
    state = meta_get('v4_state', {}) or {}
    hps = state.get('hps') or walk_forward_calibrate(make_features(read_source())) or None
    if hps is None:
        hps = {"sgd":{"epsilon":1e-3,"alpha":5e-5},"ridge":{"alpha":1e-4},"pa":{"C":0.5}}
    meta_set('v4_state', {"hps": hps})

    models = init_models(hps)
    weights = EWMWeights(alpha=0.2)
    # carregar EWM se existir
    w_saved = meta_get('v4_ewm_mae')
    if isinstance(w_saved, dict):
        weights.mae.update({k: float(v) for k,v in w_saved.items() if k in weights.mae})

    # fit inicial
    df_src = read_source(limit=MAX_WINDOW)
    if df_src.empty or len(df_src) < MIN_HISTORY:
        print(f"[aguardando] Histórico insuficiente ({len(df_src)}/{MIN_HISTORY}).")
        time.sleep(SLEEP_SEC)

    feat = make_features(df_src)
    num = feat.select_dtypes(include=[np.number]).copy()
    if 'y_ret' not in num.columns:
        print("[erro] y_ret não disponível para treino inicial."); return
    y = num['y_ret'].values
    X = num.drop(columns=['y_ret'], errors='ignore').values

    init_split = min(600, len(y)-1)
    for m in models.values():
        m.fit_initial(X[:init_split], y[:init_split])

    while True:
        try:
            df = read_source(limit=MAX_WINDOW)
            if df.empty or len(df) < MIN_HISTORY:
                print(f"[aguardando] Histórico insuficiente ({len(df)}/{MIN_HISTORY}).")
                time.sleep(SLEEP_SEC); continue

            feat = make_features(df)
            num = feat.select_dtypes(include=[np.number]).copy()
            y = num['y_ret'].values
            X = num.drop(columns=['y_ret'], errors='ignore').values

            last_minute = pd.to_datetime(df['Minute'].iloc[-1]).floor('min')
            last_price  = float(df['Price'].iloc[-1])

            # Atualiza cada modelo com a última amostra conhecida
            for m in models.values():
                m.update(X[-1:], y[-1:])

            # Predições individuais (retorno)
            preds_ret = {k: float(np.clip(models[k].predict(X[-1:])[0], -RET_CLIP, RET_CLIP)) for k in models}
            preds_price = {k: float(np.clip(last_price*(1.0+preds_ret[k]), last_price*(1-RET_CLIP), last_price*(1+RET_CLIP))) for k in models}

            # Ensemble weights -> média ponderada
            w = weights.weights()
            y_hat_ret = sum(w[k]*preds_ret[k] for k in models)
            y_hat = float(np.clip(last_price*(1.0+y_hat_ret), last_price*(1-RET_CLIP), last_price*(1+RET_CLIP)))
            baseline = last_price

            # Atualiza EWM-MAE com o minuto recém-fechado (se tivermos a previsão de "agora")
            mae_val = None
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(f"SELECT y_hat, y_hat_sgd, y_hat_ridge, y_hat_pa FROM {TABELA_PREV} WHERE Datetime=?", (last_minute.strftime('%Y-%m-%d %H:%M:00'),)).fetchone()
            if row:
                yhat_prev, yhat_sgd_prev, yhat_ridge_prev, yhat_pa_prev = row
                if yhat_prev is not None:
                    y_real = float(df.loc[df['Minute']==last_minute, 'Price'].iloc[-1])
                    mae_val = abs(y_real - float(yhat_prev))
                # atualizar EWM por modelo se existirem valores
                errs = {}
                if yhat_sgd_prev is not None:
                    errs["sgd"] = abs(y_real - float(yhat_sgd_prev))
                if yhat_ridge_prev is not None:
                    errs["ridge"] = abs(y_real - float(yhat_ridge_prev))
                if yhat_pa_prev is not None:
                    errs["pa"] = abs(y_real - float(yhat_pa_prev))
                if errs:
                    weights.update(errs)
                    meta_set('v4_ewm_mae', weights.mae)

            # Salva previsão t+1 com componentes e pesos
            ts_future = last_minute + pd.Timedelta(minutes=1)
            row = {
                "y_hat": y_hat, "y_hat_ret": y_hat_ret, "baseline": baseline, "mae": mae_val,
                "y_hat_sgd": preds_price["sgd"], "y_hat_ridge": preds_price["ridge"], "y_hat_pa": preds_price["pa"],
                "w_sgd": w["sgd"], "w_ridge": w["ridge"], "w_pa": w["pa"]
            }
            upsert_prediction(ts_future, row)

            print(f"[ok] {ts_future} | ens={y_hat:.2f} (ret={y_hat_ret:.5f}) | weights {w}")
            time.sleep(SLEEP_SEC)

        except KeyboardInterrupt:
            print("Encerrado pelo usuário."); break
        except Exception as e:
            print("[erro loop]", e); time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()