#!/usr/bin/env python3
"""
launcher.py — lançador genérico cross-platform para os scripts do repo
Cria GIT/DB/cripto.sqlite (se não existir) e diretórios necessários.
Use:
  python launcher.py start [--detach]
  python launcher.py stop
  python launcher.py status
"""

from __future__ import annotations
import os, sys, subprocess, signal, time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
GIT_DB_DIR = ROOT / "GIT" / "DB"
DB_FILE = GIT_DB_DIR / "cripto.sqlite"
LOG_DIR = ROOT / "logs"
PID_DIR = ROOT / ".pids"

# scripts to run: (label, script_filename, [optional special command constructor])
SCRIPTS = [
    ("coletor_realtime", "btc_tempo_real_v4.py"),
    ("predictor_1m", "btc_predictor_v4.py"),
    ("predictor_30m", "btc_predictor_30min_v4.py"),
    ("retrain_offline", "btc_retrain_v4.py"),
    # streamlit is usually run with 'streamlit run' — but launcher can start it too if desired
    ("streamlit_app", "streamlit_app_v4.py"),
]

def ensure_dirs_and_db():
    GIT_DB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)
    # touch sqlite file if not exists
    if not DB_FILE.exists():
        open(DB_FILE, "a").close()
        print(f"[INIT] criado DB vazio: {DB_FILE}")

def read_pid(name: str) -> Optional[int]:
    f = PID_DIR / f"{name}.pid"
    if not f.exists(): return None
    try:
        return int(f.read_text().strip())
    except Exception:
        return None

def write_pid(name: str, pid: int):
    (PID_DIR / f"{name}.pid").write_text(str(pid))

def remove_pid(name: str):
    p = PID_DIR / f"{name}.pid"
    try:
        if p.exists(): p.unlink()
    except Exception:
        pass

def is_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            os.kill(pid, 0)
        else:
            os.kill(pid, 0)
        return True
    except Exception:
        return False

def start_script(label: str, filename: str, detach: bool):
    path = (ROOT / filename).resolve()
    if not path.exists():
        print(f"[WARN] script não encontrado: {path}")
        return
    py = sys.executable
    cmd = [py, str(path)]
    # streamlit prefer 'streamlit run' — but keep generic
    if label == "streamlit_app":
        # if streamlit is installed in env, prefer module run
        cmd = [py, "-m", "streamlit", "run", str(path), "--server.headless=true"]

    out_log = open(LOG_DIR / f"{label}.out.log", "a", encoding="utf-8", buffering=1)
    err_log = open(LOG_DIR / f"{label}.err.log", "a", encoding="utf-8", buffering=1)
    print(f"[START] {label}: {' '.join(cmd)} (logs: {out_log.name})")
    if os.name == "nt":
        p = subprocess.Popen(cmd, stdout=out_log, stderr=err_log, cwd=str(ROOT), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        if detach:
            p = subprocess.Popen(cmd, stdout=out_log, stderr=err_log, cwd=str(ROOT), preexec_fn=os.setsid)
        else:
            p = subprocess.Popen(cmd, stdout=out_log, stderr=err_log, cwd=str(ROOT))
    write_pid(label, p.pid)
    time.sleep(0.1)
    return p.pid

def stop_script(label: str):
    pid = read_pid(label)
    if not pid:
        print(f"[STOP] {label}: nenhum PID registrado.")
        return
    try:
        print(f"[STOP] tentando finalizar {label} pid={pid}")
        if os.name == "nt":
            subprocess.run(["taskkill","/PID",str(pid),"/T","/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                os.kill(pid, signal.SIGTERM)
    except Exception as e:
        print("Erro ao parar:", e)
    remove_pid(label)
    print(f"[STOP] {label}: finalizado (pid file removido)")

def status():
    print("Status dos serviços:")
    for label, fname in SCRIPTS:
        pid = read_pid(label)
        running = is_running(pid)
        # sempre mostrar como string (evita ValueError do ':s')
        pid_str = str(pid) if pid is not None else "-"
        print(f" - {label:<18} | script={fname:<25} | pid={pid_str:>6} | running={running}")

def start_all(detach: bool):
    ensure_dirs_and_db()
    # set CRYPTO_DB env so child processes see it
    os.environ.setdefault("CRYPTO_DB", str(DB_FILE))
    for label, fname in SCRIPTS:
        existing = read_pid(label)
        if existing and is_running(existing):
            print(f"[SKIP] {label} já em execução (pid={existing})")
            continue
        start_script(label, fname, detach)

def stop_all():
    for label, _ in SCRIPTS:
        stop_script(label)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start","stop","status"])
    parser.add_argument("--detach", action="store_true", help="start detached/background")
    args = parser.parse_args()
    if args.action == "start":
        start_all(detach=args.detach)
    elif args.action == "stop":
        stop_all()
    else:
        status()
