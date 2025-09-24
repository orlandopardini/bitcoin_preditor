# BTC Predictor & Streamlit Dashboard — Documentação Completa

Este repositório reúne **coleta em tempo real**, **modelagem/ensemble online**, **retreinamento offline** e um **dashboard Streamlit** para análise de Bitcoin.  
O projeto utiliza **SQLite** como armazenamento padrão (`GIT/DB/cripto.sqlite`) para máxima portabilidade.

> **TL;DR**  
> - Rode tudo com `python launcher.py start` (ou `docker-compose up -d --build`).  
> - Banco em `GIT/DB/cripto.sqlite` (configurável via `CRYPTO_DB`).  
> - Dashboard em `http://localhost:8501` (se subir o serviço `streamlit`).

---

## Componentes & Fluxo (Visão Geral)

```
          ┌───────────────────┐           ┌───────────────────────────┐
 Price →  │  Coletor (1 min)  │  writes   │   SQLite (cripto.sqlite)  │
(binance) │ btc_tempo_real    ├─────────▶│  btc_realtime (+ índices) │
          └───────────────────┘           └──────────┬────────────────┘
                                                     │
                          ┌───────────────────────────▼─────────────────────────┐
                          │              Preditores (online)                    │
                          │  - btc_predictor_v4.py  (h=1 min)                   │
                          │  - btc_predictor_30min_v4.py (h=30 min, label lag)  │
                          │  calculam features + ensemble (SGD/Ridge/PA)        │
                          │  escrevem btc_predictions / btc_predictions_h30     │
                          └───────────────────────────┬─────────────────────────┘
                                                      │
                           ┌───────────────────────────▼─────────────────────────┐
                           │   Retreinamento Offline (walk-forward grid)         │
                           │           btc_retrain_v4.py                          │
                           │   grava melhores hiperparâmetros em model_meta       │
                           └───────────────────────────┬──────────────────────────┘
                                                       │
                        ┌───────────────────────────────▼──────────────────────────────┐
                        │               Dashboard Streamlit                             │
                        │                 streamlit_app_v4.py                           │
                        │   - KPIs, gráficos, export CSV/Parquet                       │
                        │   - Atualização BTC-USD (yfinance) em btc_historico          │
                        │   - Coleta Reddit + TextBlob (sentimento_reddit_posts)       │
                        └───────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Banco de Dados (SQLite)

**Arquivo padrão:** `GIT/DB/cripto.sqlite` (criado automaticamente).  
**Variável de ambiente:** `CRYPTO_DB` (sobrepõe o caminho padrão).

### Tabelas principais

1) **`btc_realtime`** — preços minuto-a-minuto (coletor)  
Campos típicos:
- `Datetime` (TEXT, **ÚNICO** — índice `idx_btc_realtime_datetime`)  
- `Symbol` (TEXT) — ex.: `BTC/USDT`  
- `Price` (REAL) — preço “last”  
- `Open` (REAL) — preço de abertura do minuto (da exchange)  
- `Volume` (REAL) — volume base  
- `Return_%` (REAL, opcional) — variação percentual do minuto

2) **`btc_predictions`** — saídas do preditor de **1 min**  
Campos:
- `Datetime` (TEXT, **PRIMARY KEY**) — timestamp de validade da previsão (minuto T)  
- `Symbol` (TEXT)  
- `y_hat` (REAL) — **preço previsto** (ensemble)  
- `y_hat_ret` (REAL) — retorno previsto (ensemble)  
- `baseline` (REAL) — preço observado em T-1 (ou referência)  
- `horizon_min` (INTEGER) — 1  
- `created_at` (TEXT), `model_version` (TEXT)  
- `mae` (REAL), `r2` (REAL) — métricas opcionais  
- Componentes: `y_hat_sgd`, `y_hat_ridge`, `y_hat_pa`  
- Pesos: `w_sgd`, `w_ridge`, `w_pa`

3) **`btc_predictions_h30`** — saídas do preditor de **30 min**  
Campos (análogos ao de 1 min, mas `horizon_min = 30`):
- `Datetime` (TEXT, **PRIMARY KEY**) — **alvo t+30** (atenção)  
- `y_hat`, `y_hat_ret`, `baseline`  
- `y_hat_sgd`, `y_hat_ridge`, `y_hat_pa`  
- `w_sgd`, `w_ridge`, `w_pa`  
- `Symbol`, `created_at`, `model_version`

4) **`btc_model_meta`** — metadados/estado de modelos  
- `key` (TEXT, **PRIMARY KEY**) — ex.: `v4_state`, `v4_h30_ewm_mae`  
- `value` (TEXT/JSON)

5) **`sentimento_reddit_posts`** — posts coletados do Reddit (opcional)  
- `ativo` (TEXT), `data` (DATE), `texto_original` (TEXT), `sentimento` (REAL)  
- Índice único (`ativo, data, texto_original`).

6) **`btc_historico`** — dados diários do `yfinance` (opcional)  
- Colunas padrão do Yahoo (`Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`).  
- Índice único em `Date` (`idx_btc_historico_date`).

> **Dica:** exporte o schema com:  
> `sqlite3 GIT/DB/cripto.sqlite .schema > db_schema.sql`

---

## 🧩 Scripts & Funcionalidades (Detalhado)

### 1) `btc_tempo_real_v4.py` — **Coletor em tempo real**
- **Fonte:** `ccxt` (Binance). Intervalo padrão: **60 s**.  
- **Saída:** escreve em `btc_realtime`. Garante **índice único** por `Datetime` (dedup).  
- **Timezone:** converte o `timestamp` da exchange (UTC) para `America/Sao_Paulo` e remove tz-info (armazenando “naive” em local time).  
- **Resiliência:** se erro, espera 30 s e tenta novamente.  
- **Config:**  
  - `symbol = 'BTC/USDT'`  
  - **DB:** usa `sqlite3.connect(DB_PATH)` — recomendamos padronizar via `CRYPTO_DB` (ver seção *Portabilidade de Caminhos*).

**Principais funções:**
- `criar_indice_unico()` — cria índice `idx_<tabela>_datetime` se não existir.  
- `remover_duplicatas()` — lê, deduplica, regrava; faz limpeza de históricos duplicados.  
- `atualizar_preco()` — faz `fetch_ticker`, monta linha e `INSERT` (com índice/unique).

---

### 2) `btc_predictor_v4.py` — **Preditor online (h=1 min)**

**Objetivo:** prever o **preço do próximo minuto** via ensemble de 3 modelos incrementais:
- `SGDRegressor(loss='huber')` → **sgd**  
- `SGDRegressor(loss='squared_error')` → **ridge-like** (ridge_sgd)  
- `PassiveAggressiveRegressor` → **pa**

**Alvo/modelagem:**
- Treina no **retorno do próximo minuto** (clipped em ±5%) e reconstrói **preço previsto** `y_hat = baseline*(1 + y_hat_ret)`.  
- **Features**:
  - Retornos, lags (`ret_lag_1...60`), volatilidades (`rstd_5/10/20/30`)
  - EMAs (5/10/20/30), *gaps* vs EMA, *slopes*
  - Indicadores: **RSI**, **MACD**, **Bollinger (width)**, **OBV**  
- **Janelas:** `MIN_HISTORY=1200`, `MAX_WINDOW=8000`.  
- **Loop:** 60 s por iteração.

**Ensemble & pesos (EWM-MAE):**
- Calcula erro de cada modelo quando o **real** chega para o timestamp previsto e atualiza **pesos** ~ `1/(EWM-MAE + ε)`.  
- Persistência de **pesos/estado** em `btc_model_meta` (chaves tipo `v4_ewm_mae`).

**Saída:** tabela `btc_predictions` (campos listados na seção **Banco**).  
**Boas práticas:**
- **Dedup**: limpeza `Datetime IS NULL / '' / 'NaT'`  
- **Partial fit** incremental a cada novo minuto  
- **Cortes (clip)** para estabilidade (`RET_CLIP=0.05`)

---

### 3) `btc_predictor_30min_v4.py` — **Preditor online (h=30 min, label atrasada)**

**Diferença-chave:** ao prever `t+30`, **não usamos rótulo futuro**. O script usa **label atrasada**:  
- Em `t`, atualiza o modelo com a label que “chegou” para `t-30 → t`.  
- **Depois**, faz a **previsão** para `t+30` (usando estado mais recente).

**Config/constantes:**
- `H=30`, `RET_CLIP_H=0.15`, `MIN_HISTORY=2000`, `MAX_WINDOW=12000`  
- Indicadores/Features **iguais** ao v4 (EMAs, RSI, MACD, Bollinger, OBV, lags, volatilidades, etc.).

**Ensemble idêntico** (SGD/Ridge/PA) com **pesos pelo EWM-MAE** atualizados quando o **alvo (t+30)** realiza.  
**Saída:** `btc_predictions_h30` (PRIMARY KEY em `Datetime` **alvo**).  
**Log:** imprime `y_hat`, `y_hat_ret` e `w` (pesos) a cada iteração.

---

### 4) `btc_retrain_v4.py` — **Retreinamento offline (walk‑forward grid)**

**Objetivo:** calibrar **hiperparâmetros** dos 3 modelos do ensemble usando histórico.  
**Procedimento (resumo):**
- Lê `btc_realtime`, gera as **mesmas features** do v4 (consistência).  
- Executa **3 folds** *walk-forward* com pequeno **grid** de hiperparâmetros para cada modelo.  
- Escolhe o **melhor conjunto** por **MAE** médio e salva em `btc_model_meta` (e.g., `key='v4_state'`).

**Saída:** `btc_model_meta` (`key='v4_state'`, `value=JSON` com hiperparâmetros e timestamp `tuned_at`).  
**Uso recomendado:** agendar para rodar **1x/dia** via scheduler (ou manualmente).

---

### 5) `streamlit_app_v4.py` — **Dashboard (KPIs + Visualizações + Ferramentas)**

**Seções principais:**
1. **Preço & Previsões (1 min e 30 min)**  
   - Carrega `btc_realtime`, `btc_predictions`, `btc_predictions_h30` (cache 15 s).  
   - Alinha séries pelo **piso do minuto** (`.dt.floor('min')`).  
   - KPI/Gráficos com `plotly.express`.  
   - Linhas auxiliares de **μ** e **±σ** para séries escolhidas.  
   - Janela interativa (1, 6, 12, 24, 48, 72 horas).  
   - **Export CSV/Parquet** das métricas (requer `pyarrow`).

2. **Heatmap de acerto direcional** (hora × dia da semana)  
   - Usa `acerto_dir = sign(y_hat-baseline) == sign(real-baseline)` (média por célula).

3. **Atualização de histórico (yfinance)** — opcional  
   - Baixa `BTC-USD` (diário) em `btc_historico` com índice único em `Date`.  
   - Salva também `.parquet` ao lado do DB.  
   - *Signalização* verde/vermelha se a última data == **ontem** (no fuso `America/Sao_Paulo`).

4. **Notícias Reddit + Sentimento (praw + TextBlob)** — opcional  
   - Subreddits: `r/Bitcoin`, `r/CryptoCurrency` (configuráveis).  
   - Persiste em `sentimento_reddit_posts` com **índice único** (`ativo, data, texto_original`).

**Recursos técnicos:**
- `st.cache_data(ttl=15)` para reduzir I/O no SQLite.  
- **Autorefresh** opcional com `streamlit-autorefresh`.  
- Proteções contra `NaT`, divisões por zero, valores ausentes.

> **Atenção:** Substitua as credenciais do `praw` por suas próprias variáveis de ambiente (não faça commit de chaves reais).

---

## 🌐 Portabilidade de Caminhos (DB)

Para todos os scripts Python, recomenda-se padronizar o caminho do DB com **variável de ambiente** `CRYPTO_DB`. Adicione no **topo** de cada script:

```python
# --- DB path portability ---
import os
from pathlib import Path

DB_PATH = os.environ.get("CRYPTO_DB")
if not DB_PATH:
    ROOT = Path(__file__).resolve().parent
    DB_DIR = ROOT / "GIT" / "DB"
    DB_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = str(DB_DIR / "cripto.sqlite")

print(f"[CONFIG] usando DB: {DB_PATH}")
# --------------------------------
```

E **substitua** qualquer `sqlite3.connect("...")` por `sqlite3.connect(DB_PATH)`.

No Docker/Compose já injetamos `CRYPTO_DB=/app/GIT/DB/cripto.sqlite`.

---

## ▶️ Execução

### 1) Modo local (venv)
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# inicia todos os serviços
python launcher.py start     # ou --detach para background
python launcher.py status
python launcher.py stop
```

### 2) Docker / docker-compose
```bash
docker-compose up --build          # foreground
# ou
docker-compose up -d --build       # background
```

Acesse: <http://localhost:8501>.

---

## ⚙️ Variáveis Importantes

- `CRYPTO_DB` — caminho para o SQLite (default: `GIT/DB/cripto.sqlite`)  
- `STREAMLIT_SERVER_PORT` — porta do Streamlit (default 8501, via compose)  
- **Credenciais PRAW** — use env vars próprias, não commite segredos

---

## 🧪 Smoke Tests (rápidos)

- Coleta: `python btc_tempo_real_v4.py` (verifique linhas inseridas em `btc_realtime`).  
- Preditor 1m: `python btc_predictor_v4.py` (check `btc_predictions`).  
- Preditor 30m: `python btc_predictor_30min_v4.py` (check `btc_predictions_h30`).  
- Retreinamento: `python btc_retrain_v4.py` (check `btc_model_meta`).  
- Dashboard: `streamlit run streamlit_app_v4.py` (ou serviço no compose).

---

## 🧰 Troubleshooting

- **`OperationalError: database is locked`**  
  - Use `timeout` em `sqlite3.connect(DB_PATH, timeout=30)`; evite longas transações.  
  - Prefira `INSERT OR REPLACE` com batch menor; feche conexões rapidamente (`with` context).

- **Faltam pacotes / import errors**  
  - `pip install -r requirements.txt` (verifique extras: `pyarrow`, `praw`, `textblob`, `yfinance`).

- **Fuso horário / datas fora do esperado**  
  - Garanta conversão UTC → `America/Sao_Paulo` no coletor.  
  - No dashboard, o alinhamento usa `.dt.floor('min')`.

- **Previsões não aparecem**  
  - Verifique se há dados suficientes (`MIN_HISTORY`).  
  - Veja logs em `logs/*.err.log` / `logs/*.out.log`.

---

## 📄 Licença

MIT — Veja [LICENSE](LICENSE).
