"""
btc_predictor_30min.py — Ensemble online (30-min ahead) com labels atrasadas
- Modelos (incrementais):
    * SGDRegressor (loss='huber')          -> "sgd"
    * SGDRegressor (loss='squared_error')  -> "ridge"
    * PassiveAggressiveRegressor           -> "pa"
- Ensemble: pesos ~ 1/(EWM-MAE + eps), atualizados quando o alvo de 30 min "chega".
- Alvo: retorno dos PRÓXIMOS 30 minutos (clipped), prevemos preço em t+30.
- Tabelas:
    * fonte:  btc_realtime
    * saída:  btc_predictions_h30  (Datetime = timestamp alvo, i.e., t+30)
- Observação: para treinar online sem olhar o futuro, usamos labels atrasadas:
  quando chegamos em t, usamos o preço de t e t-30 para criar a label do exemplo com features em (t-30).
Requisitos: scikit-learn, pandas, numpy, joblib
"""

import os, time, json, sqlite3, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
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

# ----------------- CONFIG -----------------
SYMBOL = 'BTC/USDT'
ATIVO = SYMBOL.split('/')[0].lower()
TABELA_FONTE   = f"{ATIVO}_realtime"
TABELA_PREV_H  = f"{ATIVO}_predictions_h30"     # NOVA tabela (t+30)
TABELA_META    = f"{ATIVO}_model_meta"
# BASE_PATH removed (using DB_PATH)
# DB_PATH removed (using portable DB_PATH)
MIN_HISTORY   = 2000
MAX_WINDOW    = 12000
SLEEP_SEC     = 60

H = 30                       # horizonte em minutos
RET_CLIP_H = 0.15            # clip mais amplo p/ 30 min
EPS = 1e-8

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------- BD -----------------
def ensure_tables():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABELA_PREV_H} (
                Datetime TEXT PRIMARY KEY,   -- timestamp ALVO (t+30)
                Symbol TEXT,
                y_hat REAL,                 -- preço previsto @ t+30
                y_hat_ret REAL,             -- retorno previsto (t->t+30)
                baseline REAL,              -- preço em t
                horizon_min INTEGER,
                created_at TEXT,
                model_version TEXT,
                y_hat_sgd REAL,
                y_hat_ridge REAL,
                y_hat_pa REAL,
                w_sgd REAL,
                w_ridge REAL,
                w_pa REAL
            )
        """)
        # limpeza básica
        conn.execute(f"DELETE FROM {TABELA_PREV_H} WHERE Datetime IS NULL OR Datetime='' OR Datetime='NaT'")

def meta_get(key, default=None):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(f"SELECT value FROM {TABELA_META} WHERE key=?", (key,)).fetchone()
    if not row: return default
    try: return json.loads(row[0])
    except Exception: return row[0]

def meta_set(key, value):
    val = json.dumps(value) if not isinstance(value, str) else value
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {TABELA_META} (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(f"INSERT OR REPLACE INTO {TABELA_META}(key,value) VALUES(?,?)", (key, val))

def read_source(limit=MAX_WINDOW):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {TABELA_FONTE} ORDER BY Datetime ASC", conn, parse_dates=['Datetime'])
    if df.empty: return df
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df.dropna(subset=['Datetime'], inplace=True)
    df.sort_values('Datetime', inplace=True)
    df.drop_duplicates(subset='Datetime', keep='last', inplace=True)
    for c in ['Price','Open','Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['Price'], inplace=True)
    if len(df) > limit: df = df.iloc[-limit:]
    df['Minute'] = df['Datetime'].dt.floor('min')
    df.dropna(subset=['Minute'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def upsert_prediction(ts_target, row):
    """ts_target = minuto alvo (t+30)."""
    if pd.isna(ts_target): return
    ts_str = pd.Timestamp(ts_target).strftime("%Y-%m-%d %H:%M:00")
    fields = ["Datetime","Symbol","y_hat","y_hat_ret","baseline","horizon_min","created_at","model_version",
              "y_hat_sgd","y_hat_ridge","y_hat_pa","w_sgd","w_ridge","w_pa"]
    values = [
        ts_str, SYMBOL,
        float(row.get("y_hat")) if row.get("y_hat") is not None else None,
        float(row.get("y_hat_ret")) if row.get("y_hat_ret") is not None else None,
        float(row.get("baseline")) if row.get("baseline") is not None else None,
        H, datetime.utcnow().isoformat(), "v4_h30",
        float(row.get("y_hat_sgd")) if row.get("y_hat_sgd") is not None else None,
        float(row.get("y_hat_ridge")) if row.get("y_hat_ridge") is not None else None,
        float(row.get("y_hat_pa")) if row.get("y_hat_pa") is not None else None,
        float(row.get("w_sgd")) if row.get("w_sgd") is not None else None,
        float(row.get("w_ridge")) if row.get("w_ridge") is not None else None,
        float(row.get("w_pa")) if row.get("w_pa") is not None else None,
    ]
    placeholders = ",".join(["?"]*len(fields))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"INSERT OR REPLACE INTO {TABELA_PREV_H} ({','.join(fields)}) VALUES ({placeholders})", values)

# ----------------- Indicadores/Features -----------------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down + EPS)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    ef = ema(series, fast); es = ema(series, slow)
    macd_line = ef - es; signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def bollinger(series, window=20, nstd=2):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + nstd*sd; lower = ma - nstd*sd
    width = (upper - lower) / (ma + EPS)
    return upper, lower, width

def obv(price, volume):
    sign = np.sign(price.diff().fillna(0))
    return (sign * volume).fillna(0).cumsum()

def make_features(df):
    """Features somente do passado (causais)."""
    d = df.copy()
    d['ret'] = d['Price'].pct_change().clip(-0.1, 0.1).fillna(0.0)
    d['vol_chg'] = d['Volume'].pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)
    d['log_vol'] = np.log1p(d['Volume']).replace([np.inf,-np.inf], 0.0)

    for lag in [1,2,3,4,5,10,15,30,60]:
        d[f'ret_lag_{lag}'] = d['ret'].shift(lag)

    for w in [5,10,20,30]:
        ema_w = ema(d['Price'], w)
        d[f'ema_gap_{w}'] = (d['Price']/ema_w - 1.0).clip(-0.2,0.2)
        d[f'ema_slope_{w}'] = ema_w.pct_change().clip(-0.1,0.1)

    for w in [5,10,20,30]:
        d[f'rstd_{w}'] = d['ret'].rolling(w).std().clip(0, 0.1)

    d['rsi_14'] = rsi(d['Price'], 14).fillna(50.0)
    macd_line, signal_line, hist = macd(d['Price'])
    d['macd'] = macd_line; d['macd_signal'] = signal_line; d['macd_hist'] = hist

    _, _, bb_w = bollinger(d['Price'], 20, 2)
    d['bb_width'] = bb_w.fillna(0.0)

    d['obv'] = obv(d['Price'], d['Volume']).fillna(0.0)
    d['obv_chg'] = d['obv'].pct_change().replace([np.inf,-np.inf],0.0).fillna(0.0).clip(-0.5,0.5)

    minute_of_day = d['Datetime'].dt.hour*60 + d['Datetime'].dt.minute
    d['sin_m'] = np.sin(2*np.pi*minute_of_day/1440)
    d['cos_m'] = np.cos(2*np.pi*minute_of_day/1440)

    # alvo FUTURO 30 min (usado só no treino offline/diagnóstico)
    d['y_ret_h'] = d['Price'].pct_change(periods=H).shift(-H).clip(-RET_CLIP_H, RET_CLIP_H)
    d = d.dropna(subset=['Minute']).reset_index(drop=True)
    return d

# --------------- OnlineReg wrapper ---------------
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
    hps = hps or {"sgd":{"epsilon":1e-3,"alpha":5e-5},
                  "ridge":{"alpha":1e-4},
                  "pa":{"C":0.5}}
    m_sgd = SGDRegressor(loss='huber', epsilon=hps["sgd"]["epsilon"], penalty='l2',
                         alpha=hps["sgd"]["alpha"], learning_rate='optimal',
                         max_iter=1, tol=None, random_state=42)
    m_ridge = SGDRegressor(loss='squared_error', penalty='l2', alpha=hps["ridge"]["alpha"],
                           learning_rate='optimal', max_iter=1, tol=None, random_state=42)
    m_pa = PassiveAggressiveRegressor(C=hps["pa"]["C"], max_iter=1, tol=None,
                                      random_state=42, loss='epsilon_insensitive')
    return {"sgd": OnlineReg(m_sgd,"sgd"),
            "ridge": OnlineReg(m_ridge,"ridge"),
            "pa": OnlineReg(m_pa,"pa")}

# --------------- Pesos EWM ---------------
class EWMWeights:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.mae = {"sgd":0.02, "ridge":0.02, "pa":0.02}  # inicia >0
    def update(self, errors):
        for k,v in errors.items():
            self.mae[k] = self.alpha*float(v) + (1-self.alpha)*self.mae.get(k, float(v))
    def weights(self):
        inv = {k: 1.0/(v+EPS) for k,v in self.mae.items()}
        s = sum(inv.values())
        return {k: inv[k]/s for k in inv}

# --------------- loop principal ---------------
def main():
    os.makedirs(BASE_PATH, exist_ok=True)
    ensure_tables()

    # Estado (hiperparâmetros/pesos)
    state = meta_get('v4_h30_state', {}) or {}
    hps = state.get('hps') or {"sgd":{"epsilon":1e-3,"alpha":5e-5},
                               "ridge":{"alpha":1e-4},
                               "pa":{"C":0.5}}
    meta_set('v4_h30_state', {"hps": hps})

    models = init_models(hps)
    weights = EWMWeights(alpha=0.2)
    w_saved = meta_get('v4_h30_ewm_mae')
    if isinstance(w_saved, dict):
        for k,v in w_saved.items():
            if k in weights.mae:
                try: weights.mae[k] = float(v)
                except: pass

    # ---------- Fit inicial offline (usando histórico) ----------
    df = read_source(limit=MAX_WINDOW)
    if df.empty or len(df) < MIN_HISTORY:
        print(f"[aguardando] Histórico insuficiente ({len(df)}/{MIN_HISTORY}).")
        time.sleep(SLEEP_SEC)

    feat = make_features(df)
    num = feat.select_dtypes(include=[np.number]).copy()
    if 'y_ret_h' not in num.columns or num['y_ret_h'].dropna().empty:
        print("[erro] y_ret_h não disponível."); return

    y_all = num['y_ret_h'].values
    X_all = num.drop(columns=['y_ret_h'], errors='ignore').values

    init_split = min(800, len(y_all)-H-1)
    for m in models.values():
        m.fit_initial(X_all[:init_split], y_all[:init_split])

    # ---------- LOOP ONLINE ----------
    while True:
        try:
            df = read_source(limit=MAX_WINDOW)
            if df.empty or len(df) < MIN_HISTORY:
                print(f"[aguardando] Histórico insuficiente ({len(df)}/{MIN_HISTORY}).")
                time.sleep(SLEEP_SEC); continue

            feat = make_features(df)
            num  = feat.select_dtypes(include=[np.number]).copy()
            y_h  = num['y_ret_h'] if 'y_ret_h' in num.columns else None
            X    = num.drop(columns=['y_ret_h'], errors='ignore')

            last_minute = df['Minute'].iloc[-1]
            last_price  = float(df['Price'].iloc[-1])
            target_minute = last_minute + timedelta(minutes=H)

            # ===== 1) Atualiza modelos com label atrasada (t-30 -> t) =====
            label_minute = last_minute - timedelta(minutes=H)
            # encontra linha de features EM label_minute
            try:
                row_mask = (df['Minute'] == label_minute)
                if row_mask.any():
                    # índice correspondente na tabela de features: usamos o último índice <= label_minute
                    idx_feat = feat.index[feat['Minute'] == label_minute]
                    if len(idx_feat):
                        i = idx_feat[-1]
                        Xi = X.iloc[i:i+1].values
                        # label real: retorno (label_minute -> last_minute)
                        price_s = df.set_index('Minute')['Price']
                        y_label = (price_s.loc[last_minute] / price_s.loc[label_minute]) - 1.0
                        y_label = float(np.clip(y_label, -RET_CLIP_H, RET_CLIP_H))
                        for m in models.values():
                            m.update(Xi, [y_label])
                        # atualiza pesos usando previsões feitas no passado para o ALVO = last_minute
                        # i.e., quando Datetime == last_minute em TABELA_PREV_H
                        with sqlite3.connect(DB_PATH) as conn:
                            row = conn.execute(f"SELECT y_hat, y_hat_sgd, y_hat_ridge, y_hat_pa FROM {TABELA_PREV_H} WHERE Datetime=?",
                                               (pd.Timestamp(last_minute).strftime('%Y-%m-%d %H:%M:00'),)).fetchone()
                        if row:
                            y_real = float(price_s.loc[last_minute])
                            yhat_prev, yh_sgd, yh_ridge, yh_pa = row
                            errs = {}
                            if yh_sgd is not None:   errs["sgd"]   = abs(y_real - float(yh_sgd))
                            if yh_ridge is not None: errs["ridge"] = abs(y_real - float(yh_ridge))
                            if yh_pa is not None:    errs["pa"]    = abs(y_real - float(yh_pa))
                            if errs: weights.update(errs); meta_set('v4_h30_ewm_mae', weights.mae)
            except Exception as e:
                print(f"[warn] update atrasado: {e}")

            # ===== 2) Predição para t+30 =====
            if len(X) == 0:
                time.sleep(SLEEP_SEC); continue
            X_last = X.iloc[[-1]].values
            preds_ret = {k: float(np.clip(models[k].predict(X_last)[0], -RET_CLIP_H, RET_CLIP_H)) for k in models}
            w = weights.weights()
            y_hat_ret = sum(w[k]*preds_ret[k] for k in models)
            y_hat = float(np.clip(last_price*(1.0+y_hat_ret), last_price*(1-RET_CLIP_H), last_price*(1+RET_CLIP_H)))

            row_out = {
                "y_hat": y_hat, "y_hat_ret": y_hat_ret, "baseline": last_price,
                "y_hat_sgd": last_price*(1.0+preds_ret["sgd"]),
                "y_hat_ridge": last_price*(1.0+preds_ret["ridge"]),
                "y_hat_pa": last_price*(1.0+preds_ret["pa"]),
                "w_sgd": w["sgd"], "w_ridge": w["ridge"], "w_pa": w["pa"]
            }
            upsert_prediction(target_minute, row_out)
            print(f"[ok] {target_minute}  y_hat={y_hat:.2f}  (ret={y_hat_ret:+.4f})  w={w}")

            time.sleep(SLEEP_SEC)
        except KeyboardInterrupt:
            print("encerrando...")
            break
        except Exception as e:
            print(f"[erro loop] {e}")
            time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()