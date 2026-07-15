import os
import sys
import struct
import hashlib
import numpy as np
import pandas as pd
import joblib

LOG_DIR = "./logs"

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
class RunningZ:                                                            # W2.4
    """Streaming form of common._expanding_z -- bit-identical by construction:
    same csum/csq accumulation order, same mean/var expressions. Welford is more
    stable but NOT bit-equal, and near-zero z (catastrophic cancellation in
    x-mean) then differs by float32 ULPs, flipping LightGBM branches."""
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
        self.wf = [RunningZ() for _ in range(nb)]
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

        warm = k < c["warmup_bars"]  
        self_event, _ = self.jma_track.update(jma)                        # W2.3
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