import ccxt
import pandas as pd
import sqlite3
import os
import time
from datetime import timezone, timedelta
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

# === CONFIGURAÇÕES ===
symbol = 'BTC/USDT'
ativo = symbol.split('/')[0].lower()
tabela = f"{ativo}_realtime"
# base_path removed (using DB_PATH)
# db_path removed (using DB_PATH)
# === CONEXÃO BINANCE ===
binance = ccxt.binance()

# === CRIA ÍNDICE ÚNICO NO BANCO (SE AINDA NÃO EXISTIR) ===
def criar_indice_unico():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{tabela}_datetime ON {tabela}(Datetime)")

# === REMOVE DUPLICATAS EXISTENTES ===
def remover_duplicatas():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {tabela}", conn, parse_dates=["Datetime"])
        df = df.drop_duplicates(subset="Datetime")
        conn.execute(f"DELETE FROM {tabela}")
        df.to_sql(tabela, conn, if_exists="append", index=False)
    print(f"Duplicatas removidas de {tabela}")

# === CAPTURA E INSERE NOVO PREÇO SE AINDA NÃO EXISTIR ===
def atualizar_preco():
    ticker = binance.fetch_ticker(symbol)
    br_tz = timezone(timedelta(hours=-3))
    timestamp = pd.to_datetime(ticker['timestamp'], unit='ms').tz_localize('UTC').tz_convert(br_tz).replace(tzinfo=None)
    preco = ticker['last']
    volume = ticker['baseVolume']
    abertura = ticker['open']
    variacao_pct = ((preco - abertura) / abertura) * 100 if abertura else 0
    dados = pd.DataFrame([{'Datetime': timestamp, 'Symbol': symbol, 'Price': preco, 'Open': abertura, 'Volume': volume, 'Return_%': variacao_pct}])
    with sqlite3.connect(DB_PATH) as conn:
        try:
            dados.to_sql(tabela, conn, if_exists='append', index=False)
            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{tabela}_datetime ON {tabela}(Datetime)")  # <-- garante novamente
            print(f"{timestamp} | {symbol} | Preço: {preco:.2f} | Variação: {variacao_pct:.2f}%")
        except sqlite3.IntegrityError:
            print(f"[ignorado] {timestamp} já existe no banco.")

# === EXECUÇÃO ===
if __name__ == "__main__":
    criar_indice_unico()
    remover_duplicatas()
    while True:
        try:
            atualizar_preco()
            time.sleep(60)
        except Exception as e:
            print("Erro:", e)
            time.sleep(30)