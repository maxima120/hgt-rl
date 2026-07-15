"""
worker.py -- single-study live scorer. One process per SC study, one session,
one continuous stream from bar 0 until the service stops it. No lifecycle logic:
a fresh process IS the reset; there is no mid-session reset (a 9-12 model cannot
be re-warmed at 10:02). Time-agnostic: assumes the stream starts at the model's
session start (SC chart limited to that start time).

Wire IN  <i{3+n}f>: bar_idx, rawOpen, rawLast, JMA, <stream cols in manifest order>
Wire OUT <iffff>  : bar_idx (echoed), p, p_cal, sticky, risk_consumed

bar_idx is a CORRECTNESS ASSERTION, not a control signal:
  - first bar: recorded.
  - every later bar must equal last + 1. Any gap => the gapless-grid contract is
    broken, model state (z-scores, event decay) is unrecoverable => LOG + EXIT.
  - echoed back so SC draws the marker on the exact bar scored.

Modes: "replay" (golden gate vs research parquet) | "serve" (ZMQ, stubbed).

RULES
  W2.1  Ingesting CLOSED bar k (SC bar_idx) emits the hazard of the FORMING bar
        t = local_k + 1, where local_k counts from the first bar this process saw
        (0-based). All state honors info <= k (features lag >= 1).
  W2.2  Contract derived from the bundle's baked manifest (frame, session, stream
        set); worker-generated feature names asserted == bundle["feature_names"].
  W2.3  Event detection = Stage-0 signfill/pivots: sign of diff with zero-carry;
        event on nonzero sign flip; opposing = polarity * leg_dir; opposing==0
        dropped. SELF = JMA flips (resets segment/age/gauge, NOT session state).
  W2.4  z: per-channel Welford read-before-update over bars 0..k-1, z=0 while
        count < ZWARM. Warm bars feed the stats but are not scored.
  W2.5  Warm masks mirror training: forming t < warmup_bars -> no score;
        warmup_bars <= t < value_warm -> value block 0; beyond -> full vector.
  W2.6  isotonic applied as np.interp over exported breakpoints (asserted vs
        sklearn at startup). Features cast float32 before predict.
  W2.7  Derived display signals (no new model claims):
        sticky        = max p_cal over the last STICKY_BARS emissions.
        risk_consumed = 1 - prod(1 - p_cal) over the current segment; resets on
                        a SELF event. A 0->1 "risk budget burned" gauge.
"""

import os
import sys
import struct
import hashlib
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import load_manifest, _expanding_z, _welford_check   # shared with stage_5

# ---------------------------------------------------------------- CONFIG
MODE = "replay"                        # "replay" | "serve"
SC_TAG = "NOTICK-9-12am_3s"            # single label; identifies the fork end-to-end
MODEL_PATH = f"../stage-5/model_{SC_TAG}.joblib"
PRED_PATH = f"../stage-5/pred_{SC_TAG}.pqt"     # replay reference only
REPLAY_FROM = "2026-01-01"             # None = full history

WORKER_ID = 1                          # set by core when spawned; port math below
PORT_BASE = 50120
LOG_DIR = "./logs"
SERVICE_START = None                   # 'YYYY-MM-DD_HH-MM', passed by core; None = self

STICKY_BARS = 3
# data paths (source/raw) come from the model's baked manifest, not set here
# ----------------------------------------------------------------

TAUS = (2.0, 6.0, 18.0)
ALPHAS = tuple(np.exp(-1.0 / t) for t in TAUS)
VALUE_LAGS = (1, 2)
ZWARM = 20
TOD_BIN_MIN = 30

OUT_MSG = struct.Struct("<iffff")      # bar_idx, p, p_cal, sticky, risk_consumed


# ---------------------------------------------------------------- logging
def make_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = SERVICE_START or pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(LOG_DIR, f"worker_{WORKER_ID}_{SC_TAG}_{stamp}.log")
    fh = open(path, "a", buffering=1)

    def log(msg):
        line = f"{pd.Timestamp.now().isoformat()}  {msg}"
        fh.write(line + "\n")
        print(line)
    return log, path


# ---------------------------------------------------------------- contract
def load_contract(model_path):
    b = joblib.load(model_path)
    man = b["manifest"]                                   # baked by stage_5
    start = pd.Timestamp(man["session_start"])
    end = pd.Timestamp(man["session_end"])
    smin = start.hour * 60 + start.minute
    c = {
        "bundle": b,
        "frame": int(man["frame_seconds"]),
        "session_start": man["session_start"],
        "session_end": man["session_end"],
        "warmup_bars": int(man["warmup_bars"]),
        "value_warm": ZWARM + max(VALUE_LAGS),
        "n_tod": int(np.ceil(((end.hour * 60 + end.minute) - smin) / TOD_BIN_MIN)),
        "stream_names": [s["name"] for s in man["streams"]],
        "stream_cols": [s["column"] for s in man["streams"]],
        "self": man["self_stream"],
        "source_file": man["source_file"],
        "raw_file": man["raw_file"],
    }
    c["classes"] = [(s, k) for s in c["stream_names"] for k in ("opp", "conf")] \
                   + [(c["self"], "all")]
    c["n_floats"] = 3 + len(c["stream_cols"])            # rawOpen,rawLast,JMA + streams
    c["in_msg"] = struct.Struct(f"<i{c['n_floats']}f")   # leading int = bar_idx
    c["n_value"] = len(c["stream_cols"]) * 2 + 3         # signed+mag per stream, +leg_amp,body*2
    return c


def worker_feature_names(c):
    names = []
    for (s, k) in c["classes"]:
        names.append(f"{s}|{k}|bsince")
        names += [f"{s}|{k}|ewm{t:g}" for t in TAUS]
    names += ["age", "tod"]
    base = ([f"z_{col}_signed" for col in c["stream_cols"]]
            + [f"z_{col}_mag" for col in c["stream_cols"]]
            + ["z_leg_amp", "z_body_raw_signed", "z_body_raw_mag"])
    names += base
    for L in VALUE_LAGS:
        names += [f"{nm}_lag{L}" for nm in base]
    return names


# ---------------------------------------------------------------- primitives
class Welford:                                                             # W2.4
    __slots__ = ("k", "mean", "M2")

    def __init__(self):
        self.k, self.mean, self.M2 = 0, 0.0, 0.0

    def z_then_update(self, x):
        z = 0.0
        if self.k >= ZWARM and self.k > 1:
            var = self.M2 / (self.k - 1)
            if var > 0.0:
                z = (x - self.mean) / np.sqrt(var)
        self.k += 1
        d = x - self.mean
        self.mean += d / self.k
        self.M2 += d * (x - self.mean)
        return z


class SignTrack:                                                           # W2.3
    __slots__ = ("prev", "sign")

    def __init__(self):
        self.prev, self.sign = None, 0

    def update(self, x):
        if self.prev is None:
            self.prev = x
            return False, 0
        d = x - self.prev
        self.prev = x
        s = 1 if d > 0 else (-1 if d < 0 else 0)
        if s == 0:
            return False, 0
        if self.sign != 0 and s != self.sign:
            pol = self.sign
            self.sign = s
            return True, pol
        self.sign = s
        return False, 0


class Engine:
    """Streaming state for one session. ingest(bar_idx, rawOpen, rawLast, jma,
    *osc) -> (bar_idx, feats|None). Raises GapError on bar_idx discontinuity."""

    class GapError(Exception):
        pass

    def __init__(self, c):
        self.c = c
        self.k = -1                                       # local 0-based bar counter
        self.prev_bar_idx = None                          # SC bar_idx contiguity
        self.jma_track = SignTrack()
        self.osc_tracks = [SignTrack() for _ in c["stream_cols"]]
        self.last_event = {cl: None for cl in c["classes"]}
        self.ewm = {cl: [0.0] * len(TAUS) for cl in c["classes"]}
        self.last_self = None
        self.leg_start_jma = None
        nb = c["n_value"]
        self.wf = [Welford() for _ in range(nb)]
        self.ring = [np.zeros(nb), np.zeros(nb), np.zeros(nb)]
        self.last_self_event = False

    def ingest(self, bar_idx, rawOpen, rawLast, jma, *osc):
        c = self.c
        if self.prev_bar_idx is not None and bar_idx != self.prev_bar_idx + 1:
            raise Engine.GapError(
                f"bar_idx gap: got {bar_idx}, expected {self.prev_bar_idx + 1}")
        self.prev_bar_idx = bar_idx
        self.k += 1
        k = self.k

        self_event, _ = self.jma_track.update(jma)                        # W2.3
        self.last_self_event = self_event
        leg_dir = self.jma_track.sign
        ind = {cl: 0 for cl in c["classes"]}
        if self_event:
            ind[(c["self"], "all")] = 1
            self.last_event[(c["self"], "all")] = k
            self.last_self = k
            self.leg_start_jma = jma
        if self.leg_start_jma is None:
            self.leg_start_jma = jma

        for i, name in enumerate(c["stream_names"]):
            ev, pol = self.osc_tracks[i].update(osc[i])
            if ev and leg_dir != 0:
                cl = (name, "opp" if pol * leg_dir == 1 else "conf")
                ind[cl] = 1
                self.last_event[cl] = k

        body = rawLast - rawOpen
        ch = np.array([v * leg_dir for v in osc]
                      + [abs(v) for v in osc]
                      + [abs(jma - self.leg_start_jma),
                         body * leg_dir, abs(body)], dtype=np.float64)
        z = np.array([self.wf[i].z_then_update(ch[i]) for i in range(len(ch))])
        self.ring = [z, self.ring[0], self.ring[1]]                        # W2.4

        for cl in c["classes"]:
            e = self.ewm[cl]
            i = ind[cl]
            for j, a in enumerate(ALPHAS):
                e[j] = a * (e[j] + i)

        t = k + 1                                                          # W2.1
        if t < c["warmup_bars"]:                                           # W2.5
            return bar_idx, None
        nb = len(z)
        feats = np.empty(len(c["classes"]) * 4 + 2 + nb * 3, dtype=np.float64)
        col = 0
        for cl in c["classes"]:
            le = self.last_event[cl]
            feats[col] = (t - le) if le is not None else (t + 1)
            col += 1
            for j in range(len(TAUS)):
                feats[col] = self.ewm[cl][j]
                col += 1
        feats[col] = (t - self.last_self) if self.last_self is not None else (t + 1)
        col += 1
        feats[col] = min(t * c["frame"] // (TOD_BIN_MIN * 60), c["n_tod"] - 1)
        col += 1
        if t < c["value_warm"]:                                            # W2.5
            feats[col:] = 0.0
        else:
            feats[col:col + nb] = self.ring[0]
            feats[col + nb:col + 2 * nb] = self.ring[1]
            feats[col + 2 * nb:col + 3 * nb] = self.ring[2]
        return bar_idx, feats


# ---------------------------------------------------------------- model host
class Scorer:
    def __init__(self, contract, log):
        c = contract
        b = c["bundle"]
        self.c = c
        self.booster = b["booster"]
        self.best_it = self.booster.best_iteration
        names = worker_feature_names(c)                                    # W2.2
        assert names == list(b["feature_names"]), (
            "feature contract mismatch:\n" + "\n".join(
                f"{i}: {a} != {x}" for i, (a, x)
                in enumerate(zip(names, b["feature_names"])) if a != x))
        iso = b["iso"]
        self.iso_x = np.asarray(iso.X_thresholds_, np.float64)
        self.iso_y = np.asarray(iso.y_thresholds_, np.float64)
        probe = np.linspace(0, 1, 4097)
        assert np.max(np.abs(np.interp(probe, self.iso_x, self.iso_y)
                             - iso.predict(probe))) < 1e-12               # W2.6
        self.n_feat = len(names)
        self.names_hash = hashlib.sha1("\n".join(names).encode()).hexdigest()[:12]
        self._banner(log)

    def _banner(self, log):
        c = self.c
        b = c["bundle"]
        log("=" * 60)
        log(f"WORKER {WORKER_ID}  tag={SC_TAG}")
        log(f"model      : {MODEL_PATH}")
        log(f"file mtime : {pd.Timestamp(os.path.getmtime(MODEL_PATH), unit='s')}")
        log(f"model tag  : {b.get('tag')}  valid_from={b.get('valid_from')}  "
            f"train_end={b.get('train_end')}")
        log(f"frame      : {c['frame']}s  session {c['session_start']}-{c['session_end']}"
            f"  warmup {c['warmup_bars']}  value_warm {c['value_warm']}")
        log(f"streams    : {c['stream_names']} (+ {c['self']})")
        log(f"features   : {self.n_feat}  hash {self.names_hash}")
        log(f"wire IN    : <i{c['n_floats']}f> bar_idx, rawOpen, rawLast, JMA, "
            + ", ".join(c["stream_cols"]))
        log(f"wire OUT   : <iffff> bar_idx, p, p_cal, sticky, risk_consumed")
        log(f"ports      : pull {PORT_BASE + 100 + WORKER_ID}  "
            f"push {PORT_BASE + WORKER_ID}")
        log("=" * 60)

    def predict(self, X):
        p = self.booster.predict(np.asarray(X, dtype=np.float32),
                                 num_iteration=self.best_it)
        return p, np.interp(p, self.iso_x, self.iso_y)


# ---------------------------------------------------------------- derived signals
class DerivedSignals:
    """sticky (rolling max of p_cal) + risk_consumed gauge (W2.7)."""

    def __init__(self, sticky_bars):
        self.ring = []
        self.sticky_bars = sticky_bars
        self.surv = 1.0

    def update(self, p_cal, self_event):
        if self_event:
            self.surv = 1.0
        self.surv *= (1.0 - p_cal)
        self.ring.append(p_cal)
        if len(self.ring) > self.sticky_bars:
            self.ring.pop(0)
        return float(max(self.ring)), float(1.0 - self.surv)


# ---------------------------------------------------------------- replay gate
def load_research_frame(c):
    lo = pd.Timestamp(c["session_start"]).time()
    hi = pd.Timestamp(c["session_end"]).time()
    src = pd.read_parquet('../' + c["source_file"])
    src = src[(src["timestamp"].dt.time >= lo) & (src["timestamp"].dt.time < hi)]
    raw1 = pd.read_parquet('../' + c["raw_file"])
    raw1 = raw1[(raw1["timestamp"].dt.time >= lo) & (raw1["timestamp"].dt.time < hi)]
    raw1 = raw1.rename(columns={"Open": "rawOpen", "High": "rawHigh",
                                "Low": "rawLow", "Last": "rawLast"})
    src = src.merge(raw1[["timestamp", "rawOpen", "rawLast"]],
                    on="timestamp", how="left")
    assert src[["rawOpen", "rawLast"]].notna().all().all()
    return src.sort_values("timestamp").reset_index(drop=True)


def replay(scorer, log):
    c = scorer.c
    src = load_research_frame(c)
    if REPLAY_FROM is not None:
        src = src[src["timestamp"] >= REPLAY_FROM]
    ref = pd.read_parquet(PRED_PATH)[["timestamp", "p", "p_cal"]]

    cols = ["rawOpen", "rawLast", "JMA"] + c["stream_cols"]
    out_rows, out_ts, out_self = [], [], []
    for day, g in src.groupby(src["timestamp"].dt.date, sort=True):
        eng = Engine(c)
        V = g[cols].to_numpy(np.float64)
        ts = g["timestamp"].to_numpy()
        n = len(g)
        for i in range(n):
            _, feats = eng.ingest(i, *V[i])              # bar_idx = i, contiguous by build
            if feats is not None and i + 1 < n:
                out_rows.append(feats)
                out_ts.append(ts[i + 1])                 # forming bar's timestamp
                out_self.append(eng.last_self_event)
    X = np.vstack(out_rows)
    p, p_cal = scorer.predict(X)
    live = pd.DataFrame({"timestamp": pd.to_datetime(np.array(out_ts)),
                         "p_live": p.astype(np.float32),
                         "pcal_live": p_cal.astype(np.float32),
                         "self_ev": np.array(out_self)})

    ds_sticky = np.zeros(len(live), np.float32)
    ds_gauge = np.zeros(len(live), np.float32)
    ds = DerivedSignals(STICKY_BARS)
    prev_date = None
    for i in range(len(live)):
        d = live["timestamp"].iloc[i].date()
        if d != prev_date:
            ds = DerivedSignals(STICKY_BARS)
            prev_date = d
        ds_sticky[i], ds_gauge[i] = ds.update(float(live["pcal_live"].iloc[i]),
                                              bool(live["self_ev"].iloc[i]))
    live["sticky"] = ds_sticky
    live["risk_consumed"] = ds_gauge

    m = live.merge(ref, on="timestamp", how="inner")
    dp = np.abs(m["p_live"] - m["p"])
    dc = np.abs(m["pcal_live"] - m["p_cal"])
    log(f"replay rows compared: {len(m)}  (live {len(live)}, ref {len(ref)})")
    log(f"max |dp|     = {dp.max():.3e}   n>1e-6: {int((dp > 1e-6).sum())}")
    log(f"max |dp_cal| = {dc.max():.3e}   n>1e-6: {int((dc > 1e-6).sum())}")
    log("GATE: " + ("PASS" if dp.max() <= 1e-6 and dc.max() <= 1e-6 else "FAIL"))
    return live, m


# ---------------------------------------------------------------- serve loop (ZMQ stubbed)
def serve(scorer, log):
    c = scorer.c
    # === ZMQ SETUP STUB ===============================================
    # import zmq
    # ctx = zmq.Context()
    # pull = ctx.socket(zmq.PULL); pull.bind(f"tcp://*:{PORT_BASE + 100 + WORKER_ID}")
    # push = ctx.socket(zmq.PUSH); push.bind(f"tcp://*:{PORT_BASE + WORKER_ID}")
    # recv = lambda: pull.recv()
    # send = lambda b: push.send(b)
    # ==================================================================
    def recv():
        raise NotImplementedError("wire up ZMQ pull")
    def send(_b):
        raise NotImplementedError("wire up ZMQ push")

    log(f"serving  pull {PORT_BASE + 100 + WORKER_ID}  push {PORT_BASE + WORKER_ID}")
    eng = Engine(c)
    ds = DerivedSignals(STICKY_BARS)
    while True:
        raw = recv()
        msg = c["in_msg"].unpack(raw)
        bar_idx, vals = msg[0], msg[1:]
        try:
            _, feats = eng.ingest(bar_idx, *vals)
        except Engine.GapError as e:
            log(f"FATAL {e} -- exiting (grid contract broken, state unrecoverable)")
            sys.exit(2)
        if feats is None:
            send(OUT_MSG.pack(bar_idx, -1.0, -1.0, 0.0, 0.0))
            continue
        p, p_cal = scorer.predict(feats.reshape(1, -1))
        pc = float(p_cal[0])
        sticky, gauge = ds.update(pc, eng.last_self_event)
        send(OUT_MSG.pack(bar_idx, float(p[0]), pc, sticky, gauge))


# ---------------------------------------------------------------- usage
if __name__ == "__main__":
    log, log_path = make_logger()
    contract = load_contract(MODEL_PATH)
    scorer = Scorer(contract, log)
    if MODE == "replay":
        live, m = replay(scorer, log)
    else:
        serve(scorer, log)