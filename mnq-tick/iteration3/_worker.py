"""
worker.py -- single-study live scorer. One process per SC study, one session,
one continuous stream from bar 0 until the service stops it.

Spawned by core.py:   python worker.py <ID> <TAG> <SERVICE_START>
Standalone (test):    python worker.py 1 mnq-TICK-9-12am_3s

No lifecycle logic: a fresh process IS the reset. No mid-session reset (a 9-12
model cannot be re-warmed at 10:02). Time-agnostic: assumes the stream starts at
the model's session start (SC chart limited to that start time).

================================ WIRE IN =================================
struct <i{3+n}f>   n = len(manifest streams)
  0  bar_idx    int32    SC session bar index, 0-based, contiguous
  1  rawOpen    float32  RAW chart bar open   (raw OHLC file: Open)
  2  rawLast    float32  RAW chart bar close  (raw OHLC file: Last)
  3  JMA        float32  source file: JMA
  4+ <stream cols in manifest order>, e.g. jmaD1, jmaD2, tickJmaD1, tickJmaD2

  !! rawOpen/rawLast are RAW chart OHLC. The source file ALSO carries
     Open/High/Low/Last -- those are HEIKIN-ASHI and are NOT model inputs.
     Sending HA here yields a plausible-looking wrong answer.

================================ WIRE OUT ================================
struct <iffff>
  0  bar_idx        int32    echo, so SC draws on the exact bar scored
  1  p              float32  raw hazard
  2  p_cal          float32  calibrated hazard -- THE number (colour/thresholds)
  3  sticky         float32  max p_cal over last STICKY_BARS bars (anti-flicker)
  4  risk_consumed  float32  1 - prod(1-p_cal) over current segment (0->1 gauge)
  Warm bars (t < warmup_bars) emit p = p_cal = -1.0, sticky = gauge = 0.

bar_idx is a CORRECTNESS ASSERTION, not a control signal:
  - every bar must equal previous + 1. Any gap => gapless-grid contract broken,
    state (z-scores, event decay) unrecoverable => LOG + EXIT(2).
  - NOTE: this catches a skipped *bar_idx*, not a skipped *wall-clock slot*.
    SC numbers the bars it creates; a missing 3s slot keeps bar_idx contiguous
    while shifting the tod feature by one bin vs batch. Known, accepted
    (~0.05% of rows; tod is ~0.9% of gain).

RULES
  W3.1  Ingesting CLOSED bar k emits the hazard of the FORMING bar t = local_k+1,
        where local_k counts from the first bar this process saw. State honors
        info <= k (features lag >= 1).
  W3.2  Contract derived from the bundle's baked Stage-0 manifest (frame, session,
        stream set); worker-generated feature names asserted == the bundle's.
  W3.3  Events = Stage-0 signfill/pivots: sign of diff with zero-carry; event on
        nonzero sign flip; opposing = polarity * leg_dir; opposing==0 dropped.
        Events during warm bars are NOT recorded (Stage-0 S0.7 parity).
  W3.4  z uses RunningZ -- the streaming form of common._expanding_z, bit-identical
        by construction (same csum/csq order, same mean/var expressions). Welford
        is more stable but NOT bit-equal; near-zero z then differs by float32 ULPs
        and flips LightGBM branches.
  W3.5  Warm masks mirror training: t < warmup_bars -> no score; warmup_bars <= t
        < ZWARM+max(VALUE_LAGS) -> value block zeroed; beyond -> full vector.
  W3.6  isotonic applied as np.interp over exported breakpoints (asserted vs the
        sklearn object at startup). Features cast float32 before predict.
"""

import os
import sys
import struct
import hashlib
from datetime import datetime

import numpy as np
import joblib

# ---------------------------------------------------------------- ARGV / CONFIG
WORKER_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SC_TAG = sys.argv[2] if len(sys.argv) > 2 else "mnq-TICK-9-12am_3s"
SERVICE_START = sys.argv[3] if len(sys.argv) > 3 else datetime.now().strftime("%Y-%m-%d_%H-%M")

MODEL_DIR = "../stage-5"
MODEL_PATH = f"{MODEL_DIR}/model_{SC_TAG}.joblib"
LOG_DIR = "./logs"

PORT_BASE = 50120
PORT_PULL = PORT_BASE + WORKER_ID              # worker RECEIVES bars here
PORT_PUSH = PORT_BASE + 100 + WORKER_ID        # worker SENDS results here

STICKY_BARS = 3
DUMP_FEATURES = True                           # per-bar wire+feature CSV (verification)
# ----------------------------------------------------------------

# frozen architecture constants -- must match stage_5 / common
TAUS = (2.0, 6.0, 18.0)
ALPHAS = tuple(np.exp(-1.0 / t) for t in TAUS)
VALUE_LAGS = (1, 2)
ZWARM = 20
TOD_BIN_MIN = 30

OUT_MSG = struct.Struct("<iffff")


# ---------------------------------------------------------------- logging
def make_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"worker_{WORKER_ID}_{SC_TAG}_{SERVICE_START}.log")
    fh = open(path, "a", buffering=1)

    def log(msg):
        line = f"{datetime.now().isoformat(timespec='microseconds')}  {msg}"
        fh.write(line + "\n")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    return log, path


def _hhmm(s):
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


# ---------------------------------------------------------------- contract
def load_contract(model_path):
    b = joblib.load(model_path)
    man = b["manifest"]                                   # baked by stage_5
    smin = _hhmm(man["session_start"])
    emin = _hhmm(man["session_end"])
    c = {
        "bundle": b,
        "frame": int(man["frame_seconds"]),
        "session_start": man["session_start"],
        "session_end": man["session_end"],
        "warmup_bars": int(man["warmup_bars"]),
        "value_warm": ZWARM + max(VALUE_LAGS),
        "n_tod": int(np.ceil((emin - smin) / TOD_BIN_MIN)),
        "stream_names": [s["name"] for s in man["streams"]],
        "stream_cols": [s["column"] for s in man["streams"]],
        "self": man["self_stream"],
        "source_file": man["source_file"],
        "raw_file": man["raw_file"],
    }
    c["classes"] = [(s, k) for s in c["stream_names"] for k in ("opp", "conf")] \
                   + [(c["self"], "all")]
    c["wire_in_names"] = ["bar_idx", "rawOpen", "rawLast", "JMA"] + c["stream_cols"]
    c["n_floats"] = 3 + len(c["stream_cols"])
    c["in_msg"] = struct.Struct(f"<i{c['n_floats']}f")
    c["n_value"] = len(c["stream_cols"]) * 2 + 3          # signed+mag per stream, +amp,body*2
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
class RunningZ:                                                            # W3.4
    """Streaming form of common._expanding_z. Bit-identical by construction."""
    __slots__ = ("k", "csum", "csq")

    def __init__(self):
        self.k, self.csum, self.csq = 0, 0.0, 0.0

    def z_then_update(self, x):
        k = self.k
        mean = self.csum / k if k > 0 else 0.0
        var = (self.csq - k * mean * mean) / (k - 1) if k > 1 else 0.0
        std = np.sqrt(max(var, 0.0))
        z = (x - mean) / std if (k >= ZWARM and std > 0.0) else 0.0
        self.k = k + 1
        self.csum += x
        self.csq += x * x
        return z


class SignTrack:                                                           # W3.3
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


class GapError(Exception):
    pass


class Engine:
    """Streaming state for one session.
    ingest(bar_idx, rawOpen, rawLast, jma, *osc) -> (bar_idx, feats|None)."""

    def __init__(self, c):
        self.c = c
        self.k = -1                                       # local 0-based bar counter
        self.prev_bar_idx = None
        self.jma_track = SignTrack()
        self.osc_tracks = [SignTrack() for _ in c["stream_cols"]]
        self.last_event = {cl: None for cl in c["classes"]}
        self.ewm = {cl: [0.0] * len(TAUS) for cl in c["classes"]}
        self.last_self = None
        self.leg_start_jma = None
        nb = c["n_value"]
        self.wf = [RunningZ() for _ in range(nb)]
        self.ring = [np.zeros(nb), np.zeros(nb), np.zeros(nb)]
        self.last_self_event = False

    def ingest(self, bar_idx, rawOpen, rawLast, jma, *osc):
        c = self.c
        if self.prev_bar_idx is not None and bar_idx != self.prev_bar_idx + 1:
            raise GapError(f"bar_idx gap: got {bar_idx}, expected {self.prev_bar_idx + 1}")
        self.prev_bar_idx = bar_idx
        self.k += 1
        k = self.k
        warm = k < c["warmup_bars"]                                        # S0.7 parity

        self_event, _ = self.jma_track.update(jma)                         # W3.3
        self.last_self_event = self_event and not warm
        leg_dir = self.jma_track.sign
        ind = {cl: 0 for cl in c["classes"]}
        if self_event and not warm:
            ind[(c["self"], "all")] = 1
            self.last_event[(c["self"], "all")] = k
            self.last_self = k
            self.leg_start_jma = jma
        if self.leg_start_jma is None:
            self.leg_start_jma = jma

        for i, name in enumerate(c["stream_names"]):
            ev, pol = self.osc_tracks[i].update(osc[i])
            if ev and leg_dir != 0 and not warm:
                cl = (name, "opp" if pol * leg_dir == 1 else "conf")
                ind[cl] = 1
                self.last_event[cl] = k

        body = rawLast - rawOpen
        ch = np.array([v * leg_dir for v in osc]
                      + [abs(v) for v in osc]
                      + [abs(jma - self.leg_start_jma),
                         body * leg_dir, abs(body)], dtype=np.float64)
        z = np.array([self.wf[i].z_then_update(ch[i]) for i in range(len(ch))])
        self.ring = [z, self.ring[0], self.ring[1]]                        # W3.4

        for cl in c["classes"]:
            e = self.ewm[cl]
            i = ind[cl]
            for j, a in enumerate(ALPHAS):
                e[j] = a * (e[j] + i)

        t = k + 1                                                          # W3.1
        if t < c["warmup_bars"]:                                           # W3.5
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
        if t < c["value_warm"]:                                            # W3.5
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
        self.names = worker_feature_names(c)                               # W3.2
        assert self.names == list(b["feature_names"]), (
            "feature contract mismatch:\n" + "\n".join(
                f"{i}: {a} != {x}" for i, (a, x)
                in enumerate(zip(self.names, b["feature_names"])) if a != x))
        iso = b["iso"]
        self.iso_x = np.asarray(iso.X_thresholds_, np.float64)
        self.iso_y = np.asarray(iso.y_thresholds_, np.float64)
        probe = np.linspace(0, 1, 4097)
        assert np.max(np.abs(np.interp(probe, self.iso_x, self.iso_y)
                             - iso.predict(probe))) < 1e-12               # W3.6
        self.n_feat = len(self.names)
        self.names_hash = hashlib.sha1("\n".join(self.names).encode()).hexdigest()[:12]
        self.banner(log)

    def banner(self, log):
        c = self.c
        b = c["bundle"]
        mt = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        log("=" * 66)
        log(f"WORKER {WORKER_ID}   tag={SC_TAG}   pid={os.getpid()}")
        log(f"model      : {MODEL_PATH}")
        log(f"file mtime : {mt.isoformat(timespec='seconds')}")
        log(f"model tag  : {b.get('tag')}  valid_from={b.get('valid_from')}  "
            f"train_end={b.get('train_end')}")
        log(f"frame      : {c['frame']}s   session {c['session_start']}-{c['session_end']}"
            f"   warmup {c['warmup_bars']}   value_warm {c['value_warm']}   n_tod {c['n_tod']}")
        log(f"streams    : {c['stream_names']}  (+ {c['self']})")
        log(f"features   : {self.n_feat}   hash {self.names_hash}")
        log(f"trained on : source={c['source_file']}  raw={c['raw_file']}")
        log(f"wire IN    : <i{c['n_floats']}f> {c['in_msg'].size}B  "
            + ", ".join(c["wire_in_names"]))
        log(f"             rawOpen/rawLast = RAW chart OHLC, NOT Heikin-Ashi")
        log(f"wire OUT   : <iffff> {OUT_MSG.size}B  bar_idx, p, p_cal, sticky, risk_consumed")
        log(f"ports      : PULL {PORT_PULL} (bars in)   PUSH {PORT_PUSH} (results out)")
        log(f"dump       : {DUMP_FEATURES}")
        log("=" * 66)

    def predict(self, X):
        p = self.booster.predict(np.asarray(X, dtype=np.float32),
                                 num_iteration=self.best_it)
        return p, np.interp(p, self.iso_x, self.iso_y)


# ---------------------------------------------------------------- derived signals
class DerivedSignals:
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


# ---------------------------------------------------------------- feature dump
class FeatureDump:
    """Per-bar wire inputs + full feature vector -> CSV, for live-vs-batch diff."""

    def __init__(self, contract, names):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(
            LOG_DIR, f"features_{WORKER_ID}_{SC_TAG}_{SERVICE_START}.csv")
        new = not os.path.exists(self.path)
        self.fh = open(self.path, "a", buffering=1)
        if new:
            self.fh.write(",".join(contract["wire_in_names"] + ["t"] + names) + "\n")

    def write(self, wire, t, feats):
        row = [str(wire[0])] + [repr(float(v)) for v in wire[1:]] + [str(t)]
        row += [repr(float(v)) for v in feats]
        self.fh.write(",".join(row) + "\n")


# ---------------------------------------------------------------- serve
def serve(scorer, log):
    c = scorer.c

    # === ZMQ SETUP STUB ===============================================
    # import zmq
    # ctx  = zmq.Context()
    # pull = ctx.socket(zmq.PULL); pull.bind(f"tcp://*:{PORT_PULL}")
    # push = ctx.socket(zmq.PUSH); push.bind(f"tcp://*:{PORT_PUSH}")
    # recv = pull.recv
    # send = push.send
    # ==================================================================
    def recv():
        raise NotImplementedError("wire up ZMQ PULL")

    def send(_b):
        raise NotImplementedError("wire up ZMQ PUSH")

    log(f"serving   PULL {PORT_PULL}   PUSH {PORT_PUSH}")
    eng = Engine(c)
    ds = DerivedSignals(STICKY_BARS)
    dump = FeatureDump(c, scorer.names) if DUMP_FEATURES else None
    if dump:
        log(f"dump file : {dump.path}")
    n = 0

    while True:
        wire = c["in_msg"].unpack(recv())
        bar_idx, vals = wire[0], wire[1:]
        try:
            _, feats = eng.ingest(bar_idx, *vals)
        except GapError as e:
            log(f"FATAL {e} -- exiting (grid contract broken, state unrecoverable)")
            sys.exit(2)

        if feats is None:
            send(OUT_MSG.pack(bar_idx, -1.0, -1.0, 0.0, 0.0))
            n += 1
            continue

        p, p_cal = scorer.predict(feats.reshape(1, -1))
        pc = float(p_cal[0])
        sticky, gauge = ds.update(pc, eng.last_self_event)
        send(OUT_MSG.pack(bar_idx, float(p[0]), pc, sticky, gauge))
        if dump:
            dump.write(wire, eng.k + 1, feats)
        n += 1
        if n % 200 == 0:
            log(f"bars={n}  last bar_idx={bar_idx}  p_cal={pc:.6f}  "
                f"sticky={sticky:.6f}  gauge={gauge:.4f}")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    log, log_path = make_logger()
    try:
        contract = load_contract(MODEL_PATH)
        scorer = Scorer(contract, log)
        serve(scorer, log)
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        log(f"FATAL {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        sys.exit(1)