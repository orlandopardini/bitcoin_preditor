
"""
streamlit_app_v4.py — Ensemble + KPIs + μ/σ + Alertas + Paper Trading + Export + Notícias Reddit (auto)
Execute: streamlit run streamlit_app_v4.py
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import sqlite3
from pathlib import Path
import numpy as np

# ========== CONFIG BÁSICA ==========
SYMBOL = 'BTC/USDT'
ATIVO = SYMBOL.split('/')[0].lower()
TABELA_FONTE   = f"{ATIVO}_realtime"
TABELA_PREV_1  = f"{ATIVO}_predictions"
TABELA_PREV_30 = f"{ATIVO}_predictions_h30"

st.set_page_config(page_title="BTC 1-min Forecast — v4", layout="wide")
st.title("📈 BTC/USDT — Ensemble online (v4)")

# Auto-refresh opcional (60s)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="auto_refresh_v4")
except Exception:
    st.caption("💡 Instale `streamlit-autorefresh` para atualizar a cada 60s automaticamente.")

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

# ========== HELPERS ==========
def fmt_money(x):
    try: return f"{float(x):,.2f}"
    except: return "—"

def safe_mape(y_true, y_pred):
    d = pd.DataFrame({"a":pd.to_numeric(y_true, errors="coerce"),
                      "b":pd.to_numeric(y_pred, errors="coerce")}).dropna()
    if d.empty: return np.nan
    m = d["a"] != 0
    if not m.any(): return np.nan
    return float(np.mean(np.abs((d.loc[m,'a'] - d.loc[m,'b'])/d.loc[m,'a'])))*100

def safe_rmse(y_true, y_pred):
    d = pd.DataFrame({"a":pd.to_numeric(y_true, errors="coerce"),
                      "b":pd.to_numeric(y_pred, errors="coerce")}).dropna()
    if d.empty: return np.nan
    return float(np.sqrt(np.mean((d["a"]-d["b"])**2)))

def add_mu_sigma_lines(fig, x, series, color, prefix):
    """Adiciona μ e ±σ para uma série; ignora se vazia."""
    s = pd.to_numeric(series, errors="coerce")
    if not s.notna().any() or len(x)==0: return
    mu = float(np.nanmean(s)); sd = float(np.nanstd(s))
    fig.add_scatter(x=x, y=[mu]*len(x), mode='lines', name=f'μ ({prefix})',
                    line=dict(color=color, dash='dot', width=2), opacity=0.5)
    if not np.isnan(sd):
        fig.add_scatter(x=x, y=[mu+sd]*len(x), mode='lines', name=f'+σ ({prefix})',
                        line=dict(color=color, dash='dash', width=2), opacity=0.5)
        fig.add_scatter(x=x, y=[mu-sd]*len(x), mode='lines', name=f'−σ ({prefix})',
                        line=dict(color=color, dash='dash', width=2), opacity=0.5)

@st.cache_data(ttl=15)
def load_price_and_pred():
    with sqlite3.connect(DB_PATH) as conn:
        df_p  = pd.read_sql(f"SELECT * FROM {TABELA_FONTE}  ORDER BY Datetime ASC", conn, parse_dates=['Datetime'])
        df_1  = pd.read_sql(f"SELECT * FROM {TABELA_PREV_1} ORDER BY Datetime ASC", conn, parse_dates=['Datetime'])
        try:
            df_30 = pd.read_sql(f"SELECT * FROM {TABELA_PREV_30} ORDER BY Datetime ASC", conn, parse_dates=['Datetime'])
        except Exception:
            df_30 = pd.DataFrame(columns=['Datetime'])
    for d in (df_p, df_1, df_30):
        if d.empty: continue
        d['Datetime'] = pd.to_datetime(d['Datetime'], errors='coerce')
        d.dropna(subset=['Datetime'], inplace=True)
        d.sort_values('Datetime', inplace=True)
        d.reset_index(drop=True, inplace=True)
    return df_p, df_1, df_30

df_price, df_prev, df_prev30 = load_price_and_pred()
if df_price.empty:
    st.warning("Sem dados de preço em tempo real."); st.stop()

# Chave 'Minute' (piso do minuto)
df_price['Minute'] = df_price['Datetime'].dt.floor('min')
if not df_prev.empty:
    df_prev['Minute'] = df_prev['Datetime'].dt.floor('min')
if not df_prev30.empty:
    df_prev30['Minute'] = df_prev30['Datetime'].dt.floor('min')

# ========== CONTROLES DE JANELA ==========
colA, colB = st.columns([1,1])
with colA:
    janela_h = st.selectbox("Janela (horas) — 1 min", [1,6,12,24,48,72], index=0, help="Histórico para os gráficos/KPIs do modelo 1 min.")
    pontos = janela_h*60
with colB:
    janela_h_30 = st.selectbox("Janela (horas) — 30 min", [1,6,12,24,48,72], index=1, help="Histórico para os gráficos/KPIs do modelo 30 min.")
    pontos_30 = janela_h_30*60

# ========== 1) SEÇÃO 1-MIN ==========
st.subheader("⚡ Previsão de 1 minuto (ensemble online)")

if df_prev.empty:
    st.info("Sem previsões ainda para 1 min. Deixe o preditor online rodando.")
else:
    df_m = df_prev.merge(df_price[['Minute','Price']], on='Minute', how='left', suffixes=('','_real'))
    df_m.rename(columns={'Price':'Price_real'}, inplace=True)

    df_w = df_m.iloc[-pontos:].copy() if len(df_m) > pontos else df_m.copy()
    # ==== ANTES de calcular erro / acerto_dir, garanta o Price_T ====
    # Se sua app já calculou Close_T em algum lugar, use-o como Price_T
    if 'Price_T' not in df_w.columns:
        if 'Close_T' in df_w.columns and 'Price_T' not in df_w.columns:
            df_w = df_w.copy()
            df_w.rename(columns={'Close_T': 'Price_T'}, inplace=True)

    # Alternativa: se você tem uma série de preços intradiários/diários no df_w
    # (por ex. coluna 'Close' do yfinance) e cada linha é a previsão para T+H,
    # você pode estimar o preço âncora T como um lag de H passos:
    if 'Price_T' not in df_w.columns and 'Close' in df_w.columns:
        df_w = df_w.copy()
        # ajuste H conforme seu horizonte. Se não fizer sentido no seu índice, remova este fallback.
        H = 10 if 'H' not in globals() else H
        df_w.loc[:, 'Price_T'] = df_w['Close'].shift(H)

    # Se ainda não deu, deixe explícito como NaN (evita KeyError e permite avisar o usuário)
    if 'Price_T' not in df_w.columns:
        df_w = df_w.copy()
        df_w.loc[:, 'Price_T'] = np.nan

    df_w.loc[:, 'erro']       = np.where(df_w['Price_real'].notna(), df_w['y_hat'] - df_w['Price_real'], np.nan)
    df_w.loc[:, 'erro_base']  = np.where(df_w['Price_real'].notna(), df_w['baseline'] - df_w['Price_real'], np.nan)

    # acerto_dir só faz sentido com Price_T definido + real disponível
    need_cols = ['y_hat', 'Price_real', 'Price_T']
    has_all   = all(c in df_w.columns for c in need_cols)

    if has_all:
        mask_ok = df_w[need_cols].notna().all(axis=1)
        df_w.loc[:, 'acerto_dir'] = np.where(
            mask_ok,
            np.sign(df_w['y_hat'] - df_w['Price_T']) == np.sign(df_w['Price_real'] - df_w['Price_T']),
            np.nan
        )
    else:
        # evita KeyError depois
        df_w.loc[:, 'acerto_dir'] = np.nan
        st.warning("Coluna 'Price_T' ausente — direcional não pôde ser calculado. Garanta que Price_T é criado no pipeline de previsão.")

    df_w.loc[:, 'melhor_que_base'] = np.where(
        df_w['Price_real'].notna(),
        np.abs(df_w['y_hat'] - df_w['Price_real']) < np.abs(df_w['baseline'] - df_w['Price_real']),
        np.nan
    )

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Último Preço", fmt_money(df_price['Price'].iloc[-1]))
    with c2:
        last_pred = df_w['y_hat'].dropna().iloc[-1] if df_w['y_hat'].notna().any() else np.nan
        st.metric("Última Previsão (1m)", fmt_money(last_pred) if pd.notna(last_pred) else "—")
    with c3: st.metric(f"MAE {janela_h}h", f"{float(np.abs(df_w['erro'].dropna()).mean()):,.4f}")
    with c4: st.metric(f"MAPE {janela_h}h", f"{safe_mape(df_w['Price_real'], df_w['y_hat']):,.3f}%")
    with c5: st.metric(f"RMSE {janela_h}h", f"{safe_rmse(df_w['Price_real'], df_w['y_hat']):,.4f}")

    c6,c7 = st.columns(2)
    with c6:
        acc = 100*df_w['acerto_dir'].dropna().mean() if df_w['acerto_dir'].notna().any() else np.nan
        st.metric(f"Acerto direcional {janela_h}h", f"{acc:,.1f}%" if pd.notna(acc) else "—")
    with c7:
        beat = 100*df_w['melhor_que_base'].dropna().mean() if df_w['melhor_que_base'].notna().any() else np.nan
        st.metric(f"Melhor que baseline {janela_h}h", f"{beat:,.1f}%" if pd.notna(beat) else "—")

    # Gráfico com μ/±σ
    fig = px.line()
    real_c, pred_c = "#1f77b4", "#ff7f0e"
    fig.add_scatter(x=df_w['Datetime'], y=df_w['Price_real'], mode='lines', name='Preço Real', line=dict(color=real_c))
    if df_w['y_hat'].notna().any():
        fig.add_scatter(x=df_w['Datetime'], y=df_w['y_hat'], mode='lines', name='Previsto (1m)', line=dict(color=pred_c))
    add_mu_sigma_lines(fig, df_w['Datetime'], df_w['Price_real'], real_c, 'real')
    if df_w['y_hat'].notna().any():
        add_mu_sigma_lines(fig, df_w['Datetime'], df_w['y_hat'], pred_c, 'prev')
    st.plotly_chart(fig, use_container_width=True)

    # Erro & distribuição
    if df_w['erro'].notna().any():
        st.plotly_chart(px.line(df_w, x='Datetime', y='erro', title=f'Erro (1m) {janela_h}h'), use_container_width=True)
        st.plotly_chart(px.histogram(df_w, x='erro', nbins=50, title='Distribuição do erro (1m)'), use_container_width=True)

# ========== 2) SEÇÃO 30-MIN ==========
st.subheader("🕒 Previsão de 30 minutos (ensemble online)")

if df_prev30.empty:
    st.info("Sem previsões ainda para 30 min. Deixe o preditor H=30 rodando.")
else:
    # Merge EXATO por minuto: futuro fica Price_real=NaN até o realizado chegar
    df_m30 = df_prev30.merge(df_price[['Minute','Price']], on='Minute', how='left', suffixes=('','_real'))
    df_m30.rename(columns={'Price':'Price_real'}, inplace=True)

    df_w30 = df_m30.iloc[-pontos_30:] if len(df_m30) > pontos_30 else df_m30.copy()
    df_w30['erro']       = np.where(df_w30['Price_real'].notna(), df_w30['y_hat'] - df_w30['Price_real'], np.nan)
    df_w30['erro_base']  = np.where(df_w30['Price_real'].notna(), df_w30['baseline'] - df_w30['Price_real'], np.nan)
    df_w30['acerto_dir'] = np.where(
        df_w30['Price_real'].notna(),
        np.sign(df_w30['y_hat'] - df_w30['baseline']) == np.sign(df_w30['Price_real'] - df_w30['baseline']),
        np.nan
    )
    df_w30['melhor_que_base'] = np.where(
        df_w30['Price_real'].notna(),
        np.abs(df_w30['erro']) < np.abs(df_w30['erro_base']),
        np.nan
    )

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Último Preço", fmt_money(df_price['Price'].iloc[-1]))
    with c2:
        last_pred30 = df_w30['y_hat'].dropna().iloc[-1] if df_w30['y_hat'].notna().any() else np.nan
        st.metric("Última Previsão (30m)", fmt_money(last_pred30) if pd.notna(last_pred30) else "—")
    with c3: st.metric(f"MAE {janela_h_30}h", f"{float(np.abs(df_w30['erro'].dropna()).mean()):,.4f}")
    with c4: st.metric(f"MAPE {janela_h_30}h", f"{safe_mape(df_w30['Price_real'], df_w30['y_hat']):,.3f}%")
    with c5: st.metric(f"RMSE {janela_h_30}h", f"{safe_rmse(df_w30['Price_real'], df_w30['y_hat']):,.4f}")

    c6,c7 = st.columns(2)
    with c6:
        acc30 = 100*df_w30['acerto_dir'].dropna().mean() if df_w30['acerto_dir'].notna().any() else np.nan
        st.metric(f"Acerto direcional {janela_h_30}h", f"{acc30:,.1f}%" if pd.notna(acc30) else "—")
    with c7:
        beat30 = 100*df_w30['melhor_que_base'].dropna().mean() if df_w30['melhor_que_base'].notna().any() else np.nan
        st.metric(f"Melhor que baseline {janela_h_30}h", f"{beat30:,.1f}%" if pd.notna(beat30) else "—")

    st.caption(f"⏳ Prev. H=30 aguardando realizado: **{int(df_w30['Price_real'].isna().sum())}**")

    # Gráfico com μ/±σ
    fig30 = px.line()
    real_c, pred_c = "#1f77b4", "#ff7f0e"
    fig30.add_scatter(x=df_w30['Datetime'], y=df_w30['Price_real'], mode='lines', name='Preço Real', line=dict(color=real_c))
    if df_w30['y_hat'].notna().any():
        fig30.add_scatter(x=df_w30['Datetime'], y=df_w30['y_hat'], mode='lines', name='Previsto (H=30)', line=dict(color=pred_c))
    add_mu_sigma_lines(fig30, df_w30['Datetime'], df_w30['Price_real'], real_c, 'real')
    if df_w30['y_hat'].notna().any():
        add_mu_sigma_lines(fig30, df_w30['Datetime'], df_w30['y_hat'], pred_c, 'prev')
    st.plotly_chart(fig30, use_container_width=True)

    if df_w30['erro'].notna().any():
        st.plotly_chart(px.line(df_w30, x='Datetime', y='erro', title=f'Erro (30m) {janela_h_30}h'), use_container_width=True)
        st.plotly_chart(px.histogram(df_w30, x='erro', nbins=50, title='Distribuição do erro (30m)'), use_container_width=True)

# ========== 3) ALERTAS (com base em 1-min) ==========
st.subheader("🔔 Alertas (1-min)")
colA, colB, colC = st.columns(3)
with colA:
    err_thresh = st.number_input("Erro absoluto médio limite (na janela)", min_value=0.0, value=float(np.std(df_price['Price'][-300:])*0.5 if len(df_price)>0 else 100.0))
with colB:
    acc_thresh = st.slider("Acerto direcional mínimo (%)", 0, 100, 50)
with colC:
    window_min = st.number_input("Minutos p/ alerta", min_value=5, value=60, step=5)

if not df_prev.empty:
    df_alert = df_prev.merge(df_price[['Minute','Price']], on='Minute', how='left', suffixes=('','_real'))
    df_alert.rename(columns={'Price':'Price_real'}, inplace=True)
    df_alert = df_alert.assign(
        erro=np.where(df_alert['Price_real'].notna(), df_alert['y_hat'] - df_alert['Price_real'], np.nan),
        acerto_dir=np.where(
            df_alert['Price_real'].notna(),
            np.sign(df_alert['y_hat'] - df_alert['baseline']) == np.sign(df_alert['Price_real'] - df_alert['baseline']),
            np.nan
        )
    )

    df_alert['erro'] = np.where(df_alert['Price_real'].notna(), df_alert['y_hat']-df_alert['Price_real'], np.nan)
    df_alert['acerto_dir'] = np.where(
        df_alert['Price_real'].notna(),
        np.sign(df_alert['y_hat']-df_alert['baseline']) == np.sign(df_alert['Price_real']-df_alert['baseline']),
        np.nan
    )
    mean_abs_err = float(np.abs(df_alert['erro'].dropna()).mean()) if df_alert['erro'].notna().any() else np.nan
    acc_win = 100*df_alert['acerto_dir'].dropna().mean() if df_alert['acerto_dir'].notna().any() else np.nan

    if not np.isnan(mean_abs_err) and mean_abs_err > err_thresh:
        st.error(f"Erro médio na janela ({int(window_min)}m): {mean_abs_err:,.4f} > limite {err_thresh:,.4f}")
    elif not np.isnan(acc_win) and acc_win < acc_thresh:
        st.warning(f"Acerto direcional {acc_win:,.1f}% < {acc_thresh}% na janela de {int(window_min)}m")
    else:
        st.success("Tudo OK dentro dos limites.")

# ========== 4) PAPER TRADING (1-min) ==========
st.subheader("🧪 Paper Trading — estratégia simples (1-min)")
col1, col2, col3 = st.columns(3)
with col1:
    pred_thr = st.number_input("Limiar de retorno previsto (bps)", value=5, help="Abre LONG se y_hat_ret > threshold/10,000")
with col2:
    fee_bps = st.number_input("Custo por ida/volta (bps)", value=4, help="Taxa + slippage total")
with col3:
    hold_cash = st.checkbox("Comparar com HODL (buy&hold)", value=True)

if not df_prev.empty:
    df_m = df_prev.merge(df_price[['Minute','Price']], on='Minute', how='left', suffixes=('','_real'))
    df_m.rename(columns={'Price':'Price_real'}, inplace=True)
    df_w = (df_m.iloc[-pontos:].copy() if len(df_m) > pontos else df_m.copy()).assign(
        erro=np.where(df_w['Price_real'].notna(), df_w['y_hat'] - df_w['Price_real'], np.nan),
        erro_base=np.where(df_w['Price_real'].notna(), df_w['baseline'] - df_w['Price_real'], np.nan),
        acerto_dir=np.where(
            df_w['Price_real'].notna(),
            np.sign(df_w['y_hat'] - df_w['baseline']) == np.sign(df_w['Price_real'] - df_w['baseline']),
            np.nan
        ),
        melhor_que_base=np.where(
            df_w['Price_real'].notna(),
            np.abs(df_w['erro']) < np.abs(df_w['erro_base']),
            np.nan))

    if 'y_hat_ret' in df_w.columns and df_w['y_hat_ret'].notna().any():
        thr = pred_thr/10000.0
        df_w['signal'] = (df_w['y_hat_ret'] > thr).astype(int)
        df_w.loc[:, 'ret_real'] = df_w['Price_real'].pct_change(fill_method=None).fillna(0.0)
        df_w['ret_model'] = df_w['signal'].shift(1).fillna(0.0) * df_w['ret_real']
        trade_turn = (df_w['signal'].diff().abs().fillna(0.0))
        df_w['ret_model'] -= trade_turn * (fee_bps/10000.0)
        df_w['eq_model'] = (1.0 + df_w['ret_model']).cumprod()
        df_w['eq_hold']  = (1.0 + df_w['ret_real']).cumprod()
        total_model = df_w['eq_model'].iloc[-1] - 1.0
        total_hold  = df_w['eq_hold'].iloc[-1]  - 1.0
        rollmax = df_w['eq_model'].cummax()
        mdd = float(((df_w['eq_model'] - rollmax)/rollmax).min()) if not rollmax.empty else np.nan
        st.write(f"Retorno modelo: **{100*total_model:,.2f}%** | HODL: **{100*total_hold:,.2f}%** | Max Drawdown: **{100*mdd:,.1f}%**")
        fig_e = px.line()
        fig_e.add_scatter(x=df_w['Datetime'], y=df_w['eq_model'], mode='lines', name='Equity Modelo')
        if hold_cash: fig_e.add_scatter(x=df_w['Datetime'], y=df_w['eq_hold'], mode='lines', name='Equity HODL')
        st.plotly_chart(fig_e, use_container_width=True)

# ========== 5) EXPORT ==========
st.subheader("📤 Exportar métricas (1-min)")
if not df_prev.empty:
    exp = df_prev.merge(df_price[['Minute','Price']], on='Minute', how='left', suffixes=('','_real'))
    exp.rename(columns={'Price':'Price_real'}, inplace=True)
    exp['erro'] = np.where(exp['Price_real'].notna(), exp['y_hat']-exp['Price_real'], np.nan)
    exp['erro_base'] = np.where(exp['Price_real'].notna(), exp['baseline']-exp['Price_real'], np.nan)
    exp['acerto_dir'] = np.where(
        exp['Price_real'].notna(),
        np.sign(exp['y_hat']-exp['baseline']) == np.sign(exp['Price_real']-exp['baseline']),
        np.nan
    )
    export_df = exp[['Datetime','Price_real','y_hat','erro','erro_base','acerto_dir','y_hat_ret','baseline']].copy()
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV (métricas 1-min)", data=csv_bytes, file_name="btc_metrics_1min.csv", mime="text/csv")
    try:
        import pyarrow as pa, pyarrow.parquet as pq, io
        buf = io.BytesIO(); table = pa.Table.from_pandas(export_df); pq.write_table(table, buf)
        st.download_button("Baixar Parquet (métricas 1-min)", data=buf.getvalue(), file_name="btc_metrics_1min.parquet", mime="application/octet-stream")
    except Exception:
        st.caption("Instale `pyarrow` para exportar Parquet (opcional).")

# ===== Erro e distribuição =====
if (not df_prev.empty) and ("erro" in df_w.columns) and df_w['erro'].notna().any():
    st.plotly_chart(px.line(df_w, x='Datetime', y='erro', title=f'Erro {janela_h}h'), use_container_width=True)
    st.plotly_chart(px.histogram(df_w, x='erro', nbins=50, title='Distribuição do erro'), use_container_width=True)

# ===== Heatmap hora×dia =====
if (not df_prev.empty) and ('acerto_dir' in df_w.columns): # use a versão que já tem as colunas derivadas
    df_hm = df_w.dropna(subset=['acerto_dir']).copy() if 'acerto_dir' in df_w.columns else pd.DataFrame()
else:
    # fallback seguro: evita crash e mostra mensagem amigável
    st.warning("Sem coluna 'acerto_dir' para o período selecionado. Verifique se há dados reais suficientes.")
    df_hm = pd.DataFrame()
if not df_hm.empty:
    df_hm['hora'] = df_hm['Datetime'].dt.hour
    df_hm['dia'] = df_hm['Datetime'].dt.day_name()
    pivot = df_hm.pivot_table(values='acerto_dir', index='dia', columns='hora', aggfunc='mean')*100
    st.subheader("Acerto direcional — heatmap (%, quanto maior melhor)")
    st.dataframe(pivot.round(1))

# ===== 🗂️ Atualizar histórico (yfinance) =====
st.header("🗂️ Atualizar histórico (yfinance)")

# Checa última data existente antes da atualização
def _get_last_date_in_db(db_path, tabela):
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(f"SELECT MAX(Date) FROM {tabela}")
            row = cur.fetchone()
            if not row or row[0] is None:
                return None
            dt = pd.to_datetime(row[0], errors='coerce')
            if pd.isna(dt):
                return None
            return dt.date()
    except Exception:
        return None

ticker_fix = 'BTC-USD'
tabela_fix = 'btc_historico'

last_date_before = _get_last_date_in_db(DB_PATH, tabela_fix)
colbd1, colbd2 = st.columns(2)
with colbd1:
    st.markdown(f"**Última data no banco (antes):** {last_date_before if last_date_before else '—'}")

if st.button("🔄 Atualizar histórico BTC-USD (1 clique)"):
    with st.spinner("Baixando BTC-USD diário e atualizando base..."):
        try:
            try:
                import yfinance as yf
            except ModuleNotFoundError:
                raise RuntimeError("yfinance não está instalado neste ambiente. Instale com: pip install yfinance")

            # Baixa dados diários completos
            df = yf.download(ticker_fix, start='2012-01-01', interval='1d', auto_adjust=False, progress=False)
            # Normaliza colunas (remove MultiIndex)
            if hasattr(df.columns, "levels") and len(getattr(df.columns, "levels", [])) > 1:
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()  # garantimos coluna 'Date'

            # Salva no SQLite
            with sqlite3.connect(DB_PATH) as conn:
                df.to_sql(tabela_fix, conn, if_exists='replace', index=False)
                conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{tabela_fix}_date ON {tabela_fix}(Date);")

            # Salva parquet também
            base_dir = os.path.dirname(DB_PATH)
            pq_dir = os.path.join(base_dir, f"dados_{ticker_fix.split('-')[0].lower()}",
                                  f"{ticker_fix.split('-')[0].lower()}_parquet")
            os.makedirs(pq_dir, exist_ok=True)
            df.to_parquet(os.path.join(pq_dir, 'historico.parquet'), index=False)

            # Pega a nova última data
            last_date_after = _get_last_date_in_db(DB_PATH, tabela_fix)

            with colbd2:
                st.markdown(f"**Última data no banco (depois):** {last_date_after if last_date_after else '—'}")

            # Sinalização verde/vermelha conforme "ontem" (America/Sao_Paulo)
            from datetime import datetime, timedelta
            try:
                import pytz
                tz = pytz.timezone("America/Sao_Paulo")
                today_local = datetime.now(tz).date()
            except Exception:
                today_local = datetime.now().date()
            yesterday = today_local - timedelta(days=1)

            if last_date_after is not None:
                if last_date_after == yesterday:
                    st.success(f"Atualização OK ✅ — última data = ontem ({last_date_after})")
                else:
                    diff = (today_local - last_date_after).days
                    st.error(f"Atenção ⚠️ — última data ({last_date_after}) ≠ ontem ({yesterday}). Diferença: {diff} dia(s).")
            else:
                st.warning("Não foi possível identificar a última data após a atualização.")
        except Exception as e:
            st.error(f"Falha na atualização: {e}")
else:
    with colbd2:
        st.markdown("**Última data no banco (depois):** —")

# ===== Notícias Reddit (auto, sem botão) =====
st.header("📰 Notícias Reddit")

with st.expander("Coleta automática (posts 'hot' atuais)", expanded=True):
    limite_posts = st.number_input("limite_posts por subreddit", min_value=10, max_value=5000, value=100, step=10)
    subreddits = ["Bitcoin", "CryptoCurrency"]
    ativo = "btc"

    try:
        import praw
        from textblob import TextBlob
        from datetime import datetime
    except Exception as e:
        st.error(f"Dependências ausentes: {e}. Instale com: `pip install praw textblob`")
    else:
        reddit = praw.Reddit(client_id="SfFwfkDuXJLAk5z1n2XQRg",
                             client_secret="V70NuuAcLE_fqpWkwkAAJxtDTCchGA",
                             user_agent="criptosentimento_kyruuh/0.1")
        posts = []
        for sub in subreddits:
            try:
                for post in reddit.subreddit(sub).hot(limit=int(limite_posts)):
                    titulo = (post.title or "").strip()
                    if not titulo: continue
                    pol = TextBlob(titulo).sentiment.polarity
                    data = datetime.utcfromtimestamp(getattr(post,'created_utc',0) or 0).date()
                    posts.append({"ativo": ativo, "data": str(data), "texto_original": titulo, "sentimento": float(pol)})
            except Exception as e:
                st.warning(f"Falha ao coletar de r/{sub}: {e}")

        df = pd.DataFrame(posts)
        if not df.empty:
            # Persistência com deduplicação ANTES de criar índice único
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS sentimento_reddit_posts (
                                    ativo TEXT, data DATE, texto_original TEXT, sentimento REAL)""")
                # remove duplicatas pré-existentes
                conn.execute("""DELETE FROM sentimento_reddit_posts
                                WHERE rowid NOT IN (
                                    SELECT MIN(rowid) FROM sentimento_reddit_posts
                                    GROUP BY ativo, data, texto_original
                                )""")
                # cria índice único só depois da limpeza
                try:
                    conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_sent_post_unique
                                    ON sentimento_reddit_posts(ativo, data, texto_original)""")
                except sqlite3.IntegrityError:
                    # se ainda houver duplicatas por concorrência, limpa novamente e tenta de novo
                    conn.execute("""DELETE FROM sentimento_reddit_posts
                                    WHERE rowid NOT IN (
                                        SELECT MIN(rowid) FROM sentimento_reddit_posts
                                        GROUP BY ativo, data, texto_original
                                    )""")
                    conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_sent_post_unique
                                    ON sentimento_reddit_posts(ativo, data, texto_original)""")

                # Inserir usando tabela temporária + OR IGNORE para respeitar unicidade
                df.to_sql("_tmp_sent_posts", conn, if_exists="replace", index=False)
                conn.execute("""INSERT OR IGNORE INTO sentimento_reddit_posts(ativo, data, texto_original, sentimento)
                                SELECT ativo, data, texto_original, sentimento FROM _tmp_sent_posts""")
                conn.execute("DROP TABLE _tmp_sent_posts")

                # Tabela agregada diária e índice único
                conn.execute("""CREATE TABLE IF NOT EXISTS sentimento_reddit (
                                    ativo TEXT, data DATE, media_sentimento REAL)""")
                conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_sentimento_unique
                                    ON sentimento_reddit(ativo, data)""")

                # Calcula agregação e faz UPSERT
                df['data'] = pd.to_datetime(df['data'])
                df_agg = df.groupby(['ativo','data']).agg(media_sentimento=('sentimento','mean')).reset_index()
                conn.executemany(
                    """INSERT INTO sentimento_reddit (ativo, data, media_sentimento)
                       VALUES (?, ?, ?)
                       ON CONFLICT(ativo, data) DO UPDATE SET media_sentimento=excluded.media_sentimento""",
                    [(row['ativo'], str(row['data'].date()), float(row['media_sentimento'])) for _, row in df_agg.iterrows()]
                )
            st.caption(f"{len(df)} frases armazenadas e médias diárias atualizadas.")
        else:
            st.caption("Nenhum post coletado (tente aumentar o limite).")

# --- Filtros e viz (Reddit) ---
st.subheader("Filtrar por período (Reddit)")
colf1, colf2 = st.columns(2)
with colf1:
    start_date = st.date_input("Data inicial", value=pd.to_datetime("today") - pd.Timedelta(days=7))
with colf2:
    end_date = st.date_input("Data final", value=pd.to_datetime("today"))

@st.cache_data(ttl=30)
def load_reddit_tables(start_date, end_date):
    with sqlite3.connect(DB_PATH) as conn:
        q_posts = """SELECT data, texto_original, sentimento FROM sentimento_reddit_posts WHERE data BETWEEN ? AND ? AND ativo='btc' ORDER BY data DESC"""
        q_agg = """SELECT data, media_sentimento FROM sentimento_reddit WHERE data BETWEEN ? AND ? AND ativo='btc' ORDER BY data ASC"""
        df_posts = pd.read_sql(q_posts, conn, params=(str(start_date), str(end_date)))
        df_agg = pd.read_sql(q_agg, conn, params=(str(start_date), str(end_date)))
    for d in (df_posts, df_agg):
        if 'data' in d.columns:
            # aceita strings só com data ("YYYY-MM-DD") ou datetime completo
            d['data'] = pd.to_datetime(d['data'], errors='coerce')
    return df_posts, df_agg

df_posts, df_agg = load_reddit_tables(start_date, end_date)
tabs = st.tabs(["📄 Posts coletados", "📊 Média diária"])
with tabs[0]:
    st.caption("Tabela: sentimento_reddit_posts")
    st.dataframe(df_posts, use_container_width=True)
with tabs[1]:
    st.caption("Tabela: sentimento_reddit (agregado diário)")
    st.dataframe(df_agg, use_container_width=True)
    if not df_agg.empty:
        line = px.line(df_agg, x='data', y='media_sentimento', title='Sentimento médio diário (Reddit)')
        st.plotly_chart(line, use_container_width=True)

# ===================== PAINEL ========================

def run_painel_features():
    st.header("Engenharia de Features (SQLite ➜ Tabela de saída)")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        ativo = st.text_input("Ativo", value="btc", key="pf_ativo").strip().lower()
    with c2:
        tabela_entrada = st.text_input("Tabela histórica (entrada)", value=f"{ativo}_historico", key="pf_tab_in")
    with c3:
        tabela_saida = st.text_input("Tabela de saída (features)", value=f"{ativo}_modelo", key="pf_tab_out")

    tabela_eventos = st.text_input("Tabela de eventos", value="eventos", key="pf_tab_evt")
    tabela_sentimento = st.text_input("Tabela de sentimento", value="sentimento_reddit", key="pf_tab_sent")

    st.caption(f"SQLite: `{DB_PATH}`")

    if st.button("▶️ Rodar engenharia e salvar (SQLite)", type="primary", key="pf_run"):
        try:
            # 1) Carrega dados
            with sqlite3.connect(DB_PATH, timeout=60) as conn:
                try:
                    df_hist = pd.read_sql(f"SELECT * FROM {tabela_entrada}", conn)
                except Exception:
                    st.error(f"Tabela **{tabela_entrada}** não encontrada no SQLite."); return

                def _read_opt(sql, params=None):
                    try:
                        return pd.read_sql(sql, conn, params=params)
                    except Exception:
                        return pd.DataFrame()

                df_eventos = _read_opt(f"SELECT * FROM {tabela_eventos} WHERE ativo = ?", (ativo,))
                df_sentimento = _read_opt(f"SELECT * FROM {tabela_sentimento} WHERE ativo = ?", (ativo,))

            # 2) Parse datas/colunas obrigatórias
            if "Date" in df_hist.columns: df_hist.rename(columns={"Date": "Datetime"}, inplace=True)
            if "date" in df_hist.columns: df_hist.rename(columns={"date": "Datetime"}, inplace=True)
            df_hist["Datetime"] = pd.to_datetime(df_hist["Datetime"], errors="coerce")

            if not df_eventos.empty:
                df_eventos["data"] = pd.to_datetime(df_eventos["data"], errors="coerce")
            else:
                df_eventos = pd.DataFrame(columns=["data","evento"])

            if not df_sentimento.empty:
                df_sentimento["data"] = pd.to_datetime(df_sentimento["data"], errors="coerce")
            else:
                df_sentimento = pd.DataFrame(columns=["data","media_sentimento"])

            req_cols = ["Datetime","Open","High","Low","Close","Volume"]
            miss = [c for c in req_cols if c not in df_hist.columns]
            if miss:
                st.error(f"Faltam colunas no histórico: {miss}")
                return

            # 3) Pré-processamento
            df = (df_hist[req_cols]
                  .sort_values("Datetime")
                  .dropna()
                  .reset_index(drop=True))

            # 4) Features técnicas (USANDO **ta** — nada de pandas_ta aqui)
            df["return_pct"] = df["Close"].pct_change() * 100
            df["return_log"] = np.log(df["Close"] / df["Close"].shift(1))

            df["SMA_7"]  = SMAIndicator(df["Close"], 7).sma_indicator()
            df["SMA_30"] = SMAIndicator(df["Close"], 30).sma_indicator()
            df["EMA_7"]  = EMAIndicator(df["Close"], 7).ema_indicator()
            df["EMA_30"] = EMAIndicator(df["Close"], 30).ema_indicator()
            df["RSI_14"] = RSIIndicator(df["Close"], 14).rsi()

            macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
            df["MACD"]   = macd.macd()
            df["MACD_S"] = macd.macd_signal()
            df["MACD_H"] = macd.macd_diff()

            bb = BollingerBands(df["Close"], window=20, window_dev=2)
            df["BBL_20_2.0"] = bb.bollinger_lband()
            df["BBM_20_2.0"] = bb.bollinger_mavg()
            df["BBU_20_2.0"] = bb.bollinger_hband()

            df["volatilidade_7d"] = df["Close"].rolling(7).std()
            denom = float(df["Volume"].std(skipna=True))
            df["volume_norm"] = (df["Volume"] - df["Volume"].mean()) / denom if (denom and not np.isnan(denom)) else 0.0

            # 5) Eventos e sentimento
            df["data"] = df["Datetime"].dt.normalize()

            if not df_eventos.empty and "evento" in df_eventos.columns:
                dte = df_eventos.copy()
                dte["data"] = pd.to_datetime(dte["data"], errors="coerce").dt.normalize()
                df["evento_halving"] = df["data"].isin(dte.loc[dte["evento"] == "halving", "data"]).astype(int)
                df["evento_fork"]    = df["data"].isin(dte.loc[dte["evento"] == "fork",    "data"]).astype(int)
            else:
                df["evento_halving"] = 0
                df["evento_fork"] = 0

            if not df_sentimento.empty and "media_sentimento" in df_sentimento.columns:
                dts = df_sentimento[["data","media_sentimento"]].copy()
                dts["data"] = pd.to_datetime(dts["data"], errors="coerce").dt.normalize()
                df = df.merge(dts, how="left", on="data")
                df["media_sentimento"] = df["media_sentimento"].fillna(0)

            df.drop(columns=["data"], inplace=True, errors="ignore")
            df.dropna(inplace=True)
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

            # 6) Salva SOMENTE no SQLite (sem parquet)
            with sqlite3.connect(DB_PATH, timeout=60) as conn:
                df.to_sql(tabela_saida, conn, if_exists="replace", index=False)

            st.success(f"✅ Features salvas no SQLite | tabela: {tabela_saida} | linhas: {len(df)}")
            st.dataframe(df.tail(10), use_container_width=True)

        except Exception as e:
            st.error(f"Falha: {e}")


# em algum lugar do layout:
# run_painel_features()
# ================== /PAINEL =======================================================


# ===== 📊 Estatísticas (SQLite → tabela btc_modelo) =====
st.header("📊 Estatísticas (SQLite)")

@st.cache_data(ttl=60, show_spinner=False)
def load_stats_df_from_db(db_path: str, table: str = "btc_modelo"):
    import sqlite3, pandas as pd, numpy as np
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {table} ORDER BY Datetime ASC", conn)
    except Exception as e:
        return None, f"Erro ao ler {table}: {e}"

    # Normaliza Datetime
    if 'Datetime' not in df.columns:
        if 'date' in df.columns: df.rename(columns={'date':'Datetime'}, inplace=True)
        elif 'Date' in df.columns: df.rename(columns={'Date':'Datetime'}, inplace=True)
        else:
            return None, "Coluna 'Datetime' não encontrada na tabela."

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)

    # Targets (se faltarem)
    if 'target_pct' not in df.columns:
        if 'return_pct' in df.columns:
            df['target_pct'] = df['return_pct'].shift(-1)
        elif 'ret' in df.columns:
            df['target_pct'] = df['ret'].shift(-1)

    if 'target_bin' not in df.columns and 'target_pct' in df.columns:
        df['target_bin'] = (df['target_pct'] > 0).astype('Int64')

    return df, None

df_stats, err = load_stats_df_from_db(DB_PATH, table='btc_modelo')
if err:
    st.info(f"Não foi possível carregar do SQLite: {err}")
else:
    # Filtro por data
    cdt1, cdt2 = st.columns(2)
    with cdt1:
        dmin = pd.to_datetime(df_stats['Datetime'].min()).date()
        start = st.date_input("Data inicial (estatísticas)", value=dmin)
    with cdt2:
        dmax = pd.to_datetime(df_stats['Datetime'].max()).date()
        end = st.date_input("Data final (estatísticas)", value=dmax)
    mask = (df_stats['Datetime'].dt.date >= start) & (df_stats['Datetime'].dt.date <= end)
    ds = df_stats.loc[mask].copy()

    if ds.empty:
        st.warning("Intervalo sem dados.")
    else:
        # -------- 1) Matriz de Correlação --------
        st.subheader("1) Matriz de correlação")
        st.caption("Correlação de Pearson entre variáveis numéricas.")
        num_cols = sorted([c for c in ds.select_dtypes(include=['float64','float32','int64','int32']).columns if c not in ['target_bin']])
        default_cols = [c for c in num_cols if c not in ['target_pct']][:30]
        cols_sel = st.multiselect("Variáveis para a matriz", options=num_cols, default=default_cols)
        if len(cols_sel) >= 2:
            corr = ds[cols_sel].corr().round(2)
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Greens', aspect='auto',
                                 title="Matriz de correlação (seleção dinâmica)")
            fig_corr.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Selecione pelo menos 2 variáveis.")

        # -------- 2) Série histórica com μ e ±σ --------
        st.subheader("2) Série histórica com μ e ±σ")
        st.caption("Preço histórico com média (μ) e desvio padrão (±σ).")
        price_col = 'Close' if 'Close' in ds.columns else ('Price' if 'Price' in ds.columns else None)
        if price_col:
            mu = float(np.nanmean(ds[price_col])); sd = float(np.nanstd(ds[price_col]))
            x = ds['Datetime']
            fig_p = px.line()
            green, darkg = "#2ca02c", "#006d2c"
            fig_p.add_scatter(x=x, y=ds[price_col], mode='lines', name='Preço', line=dict(color=green))
            fig_p.add_scatter(x=x, y=[mu]*len(x), mode='lines', name='μ (média)', line=dict(color=darkg, width=2, dash='dot'), opacity=0.7)
            fig_p.add_scatter(x=x, y=[mu+sd]*len(x), mode='lines', name='+σ', line=dict(color=green, dash='dash'), opacity=0.5)
            fig_p.add_scatter(x=x, y=[mu-sd]*len(x), mode='lines', name='−σ', line=dict(color=green, dash='dash'), opacity=0.5)
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Coluna de preço ('Close' ou 'Price') não encontrada na tabela.")

        # -------- 3) Distribuição do target --------
        st.subheader("3) Distribuição do target com aproximação Normal")
        st.caption("Histograma do alvo (target_pct) com curva Normal aproximada.")
        import plotly.graph_objects as go
        if 'target_pct' in ds.columns:
            vals = ds['target_pct'].dropna().values
            if len(vals) > 5:
                mu = float(np.nanmean(vals)); sd = float(np.nanstd(vals))
                hist = go.Figure()
                hist.add_histogram(x=vals, nbinsx=50, name='Observado', histnorm='probability density',
                                   marker_color="#2ca02c", opacity=0.7)
                xline = np.linspace(vals.min(), vals.max(), 200)
                pdf = (1/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((xline-mu)/sd)**2) if sd > 0 else np.zeros_like(xline)
                hist.add_scatter(x=xline, y=pdf, mode='lines', name='Normal(μ,σ)', line=dict(color="#1f77b4", dash='dot'))
                hist.update_layout(margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(hist, use_container_width=True)
            else:
                st.info("Dados insuficientes para histograma do target_pct.")
        else:
            st.info("Coluna 'target_pct' não encontrada.")

        # -------- 4) Distribuição de classes --------
        st.subheader("4) Distribuição de classes (target_bin)")
        st.caption("Contagem das classes binárias (queda=0, alta=1).")
        if 'target_bin' in ds.columns:
            vc = ds['target_bin'].value_counts().sort_index()
            labels = ['0 = Queda','1 = Alta']
            values = [int(vc.get(0,0)), int(vc.get(1,0))]
            fig_c = px.bar(x=labels, y=values, labels={'x':'Classe','y':'Contagem'}, text=values)
            fig_c.update_traces(marker_color=['#d62728', '#2ca02c'], textposition='outside')
            fig_c.update_layout(margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("Coluna 'target_bin' não encontrada.")

# ===== Tabela detalhada (componentes e pesos) =====
st.subheader("Previsões recentes (componentes do ensemble)")
cols = ['Datetime','Price_real','y_hat','y_hat_sgd','y_hat_ridge','y_hat_pa','w_sgd','w_ridge','w_pa','erro','acerto_dir']
cols = [c for c in cols if c in df_m.columns]
st.dataframe(df_m[cols].sort_values('Datetime', ascending=False).head(200), use_container_width=True)
