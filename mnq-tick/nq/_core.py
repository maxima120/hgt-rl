"""
core.py -- supervisor daemon. Owns no data path.

  ROUTER socket on PORT_BASE (50120). Each SC study connects and sends an ascii
  handshake "ID,TAG" (e.g. "1,mnq-TICK-9-12am_3s"). Core validates, spawns one
  worker process per ID, replies with the worker's ports. SC then talks directly
  to the worker -- core never sees a bar.

  Ports are deterministic from ID (ID >= 1):
      worker PULL = PORT_BASE + ID           (SC pushes bars here)
      worker PUSH = PORT_BASE + 100 + ID     (SC pulls results here)

  Lifecycle: systemd/cron starts the service before the session and stops it at
  the end. Core does not schedule anything. A re-handshake for a live ID means SC
  restarted its feed -> kill and respawn (workers have no reset; a fresh process
  IS the reset).

  Logs: LOG_DIR/core_<SERVICE_START>.log ; workers get the same SERVICE_START so
  one run's files group together.
"""

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
import zmq

# ---------------------------------------------------------------- CONFIG
PORT_BASE = 50120
MODEL_DIR = "stage-5"
WORKER_SCRIPT = "_worker.py"
PYTHON = sys.executable
LOG_DIR = "./logs"
POLL_MS = 500                       # handshake poll / reap interval
# ----------------------------------------------------------------

SERVICE_START = datetime.now().strftime("%Y-%m-%d_%H-%M")
WORKERS = {}                        # id -> dict(proc, tag, started)
RUNNING = True


def make_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"core_{SERVICE_START}.log")
    fh = open(path, "a", buffering=1)

    def log(msg):
        line = f"{datetime.now().isoformat(timespec='microseconds')}  {msg}"
        fh.write(line + "\n")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    return log, path


def parse_handshake(text):
    """'ID,TAG' -> (int, str). Raises ValueError with a reason."""

    parts = text.strip().split(",")

    if len(parts) != 2:
        raise ValueError(f"expected 'ID,TAG', got {text!r}")
    
    sid, tag = parts[0].strip(), parts[1].strip()
    
    wid = int(sid)

    return wid, tag


def model_path(tag):
    return os.path.join(MODEL_DIR, f"model_{tag}.joblib")


def validate(wid, tag):
    p = model_path(tag)
    if not os.path.exists(p):
        raise ValueError(f"no model file for tag {tag!r}: {p}")
    return p


def spawn(wid, tag, log):
    argv = [PYTHON, WORKER_SCRIPT, str(wid), tag, SERVICE_START]
    proc = subprocess.Popen(argv) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    WORKERS[wid] = {"proc": proc, "tag": tag, "started": time.time()}
    log(f"SPAWN id={wid} tag={tag} pid={proc.pid} pull={PORT_BASE + wid} push={PORT_BASE + 100 + wid}")
    return proc


def kill(wid, log, why=""):
    w = WORKERS.pop(wid, None)
    if not w:
        return
    proc = w["proc"]
    if proc.poll() is None:
        log(f"KILL id={wid} tag={w['tag']} pid={proc.pid} {why}")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            log(f"KILL id={wid} pid={proc.pid} -- SIGKILL after timeout")


def reap(log):
    for wid in list(WORKERS.keys()):
        w = WORKERS[wid]
        rc = w["proc"].poll()
        if rc is not None:
            up = time.time() - w["started"]
            log(f"EXIT id={wid} tag={w['tag']} pid={w['proc'].pid} rc={rc} "
                f"uptime={up:.1f}s -- see worker log")
            WORKERS.pop(wid, None)


def handle(text, log):
    """Returns the ascii reply for one handshake."""
    try:
        log(f"ZMQ RCV: '{text}'")        
        wid, tag = parse_handshake(text)
        validate(wid, tag)
    except ValueError as e:
        log(f"REJECT {text!r}: {e}")
        return f"ERR {e}"

    if wid in WORKERS and WORKERS[wid]["proc"].poll() is None:
        kill(wid, log, why="(re-handshake: SC restarted its feed)")
    spawn(wid, tag, log)
    # TODO - test it - time.sleep(0.5)
    return f"OK {wid} {tag} pull={PORT_BASE + wid} push={PORT_BASE + 100 + wid}"


def shutdown(log):
    global RUNNING
    RUNNING = False
    for wid in list(WORKERS.keys()):
        kill(wid, log, why="(service shutdown)")


def main():
    log, log_path = make_logger()
    log("=" * 66)
    log(f"CORE up   pid={os.getpid()}   service_start={SERVICE_START}")
    log(f"router    tcp://*:{PORT_BASE}   handshake 'ID,TAG'")
    log(f"ports     worker PULL {PORT_BASE}+ID   worker PUSH {PORT_BASE + 100}+ID")
    log(f"models    {MODEL_DIR}/model_<TAG>.joblib")
    log(f"log       {log_path}")
    log("=" * 66)

    def on_sig(signum, _frame):
        log(f"SIGNAL {signum} -- shutting down")
        shutdown(log)
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    # === ZMQ SETUP STUB ===============================================
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind(f"tcp://*:{PORT_BASE}")
    poller = zmq.Poller(); poller.register(router, zmq.POLLIN)
    
    def poll_handshake(timeout_ms):
        """-> (identity, text) or (None, None) on timeout."""
        socks = dict(poller.poll(timeout_ms))
        if router in socks:
            frames = router.recv_multipart()      # [identity, (empty), payload]
            identity, payload = frames[0], frames[-1]
            return identity, payload.decode("ascii", "replace")
        return None, None
    
    def reply(identity, text):
        router.send_multipart([identity, b"", text.encode("ascii")])
    # ==================================================================
    
    #def poll_handshake(_timeout_ms):
    #    raise NotImplementedError("wire up ZMQ ROUTER")

    #def reply(_identity, _text):
    #    raise NotImplementedError("wire up ZMQ ROUTER")

    while RUNNING:
        identity, text = poll_handshake(POLL_MS)
        if identity is not None:
            log(f"HANDSHAKE {text!r}")
            r = handle(text, log)
            reply(identity, r)
            log(f"REPLY {r}")
        reap(log)

    log("CORE down")


if __name__ == "__main__":
    main()