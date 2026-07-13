"""
hazard_worker.py -- live scorer for the MNQ JMA leg-death hazard (5c2-raw contract).
Python / WSL. Modes: "replay" (golden gate vs research parquet) | "serve" (ZMQ).

RULES
  W.1  Timing: on ingesting CLOSED bar k, the worker emits the hazard of the
       FORMING bar t = k+1. All state honors info <= k, matching batch p_t
       (features lag >= 1). Emission is keyed by the forming bar's open
       timestamp = ts[k] + FRAME seconds.
  W.2  Feature order is generated locally and asserted equal to
       bundle["feature_names"] at startup. Contract: 71 features =
       36 grammar (9 classes x [bsince, ewm2, ewm6, ewm18]) + age + tod +
       33 values (11 z-channels x {t-1, t-2, t-3}).
  W.3  Event detection replicates Stage-0 signfill/pivots exactly: sign of
       diff with zero-carry; event on nonzero sign flip; polarity = old sign;
       opposing = polarity * leg_dir at the event bar; opposing==0 dropped.
       SELF stream = JMA leg flips (also resets age / leg_start).
  W.4  z-scores replicate S5b.3: per-channel Welford read-before-update
       (stats over session bars 0..k-1), reset at session start, z = 0 while
       count < ZWARM or std == 0. Warm bars 0..9 ARE included in the stats.
  W.5  Warm masks mirror batch: forming bar t < 10 -> no score (lamp WARM);
       10 <= t < ZWARM + 2 -> value block forced to 0 (rows trained that way);
       t >= 22 -> full vector.
  W.6  Features cast to float32 before predict (batch X was float32); isotonic
       applied as np.interp over exported breakpoints (== out_of_bounds="clip"),
       asserted against sklearn iso on a probe grid at startup.
  W.7  Session = calendar date of the naive-ET timestamp; state resets on date
       change; bars before 09:30 are dropped at the boundary.
  W.8  Wire (little-endian): IN  <q7f>: ts_us(open, naive ET), rawOpen,
       rawLast, JMA, jmaD1, jmaD2, tickJmaD1, tickJmaD2.
       OUT <qffii>: ts_us(forming bar open), p, p_cal, lamp, dwell.
       lamp: 0 WARM, 1 GREEN, 2 AMBER, 3 RED. dwell = consecutive RED bars.
  W.9  Golden gate: replay the research dataframe through this state machine,
       compare p / p_cal to the tagged prediction parquet on timestamp;
       accept at max |dp| <= 1e-6. Run before any live use.
"""

import struct
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------- CONFIG
MODE = "replay"                       # "replay" | "serve"
FRAME = 6
TAG = "5c2raw_01-2025"

MODEL_PATH = f"data/stage-5/lgbm5b_model_{TAG}_{FRAME}s.joblib"
PRED_PATH = f"data/stage-5/pred_lgbm5b_{TAG}_{FRAME}s.parquet"     # replay reference
RAW_PATH = "data/mnq-tick-oscillator-6sec.pqt"
RAW_OHLC_FILE = "data/mnq-ohlc-raw-6sec.pqt"
REPLAY_FROM = "2026-01-01"            # replay window start (None = full history)

ZMQ_IN = "tcp://*:5555"               # SC PUSH -> worker PULL
ZMQ_OUT = "tcp://*:5556"              # worker PUSH -> SC PULL

GREEN_MAX = 0.0017                    # bottom-half cut, 01-2025 decile table
RED_MIN = 0.418                       # top-decile cut, 01-2025 decile table
# ----------------------------------------------------------------

WARMUP_BARS = 10
ZWARM = 20
VALUE_WARM = ZWARM + 2                # ZWARM + max lag
TAUS = (2.0, 6.0, 18.0)
ALPHAS = tuple(np.exp(-1.0 / t) for t in TAUS)
TOD_START_MIN = 570
TOD_BIN_MIN = 30
N_TOD = 13

CLASSES = [("MNQ_D1", "opp"), ("MNQ_D1", "conf"),
           ("MNQ_D2", "opp"), ("MNQ_D2", "conf"),
           ("TICK_D1", "opp"), ("TICK_D1", "conf"),
           ("TICK_D2", "opp"), ("TICK_D2", "conf"),
           ("MNQ_JMA_SELF", "all")]
OSC_STREAMS = [("MNQ_D1", "jmaD1"), ("MNQ_D2", "jmaD2"),
               ("TICK_D1", "tickJmaD1"), ("TICK_D2", "tickJmaD2")]
VALUE_BASE = ["z_jmaD1_signed", "z_jmaD2_signed", "z_tickJmaD1_signed",
              "z_tickJmaD2_signed", "z_jmaD1_mag", "z_jmaD2_mag",
              "z_tickJmaD1_mag", "z_tickJmaD2_mag", "z_leg_amp",
              "z_body_raw_signed", "z_body_raw_mag"]

IN_MSG = struct.Struct("<q7f")
OUT_MSG = struct.Struct("<qffii")


def worker_feature_names():
    names = []
    for (s, c) in CLASSES:
        names.append(f"{s}|{c}|bsince")
        names += [f"{s}|{c}|ewm{t:g}" for t in TAUS]
    names += ["age", "tod"]
    names += VALUE_BASE
    names += [f"{n}_lag1" for n in VALUE_BASE]
    names += [f"{n}_lag2" for n in VALUE_BASE]
    return names


# ---------------------------------------------------------------- state primitives

class Welford:                                                             # W.4
    __slots__ = ("k", "mean", "M2")

    def __init__(self):
        self.k = 0
        self.mean = 0.0
        self.M2 = 0.0

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


class SignTrack:                                                           # W.3
    __slots__ = ("prev", "sign")

    def __init__(self):
        self.prev = None
        self.sign = 0

    def update(self, x):
        """Returns (event, polarity). Zero diffs carry the sign, no event."""
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
    """One session's online state. ingest(bar_k) -> (t_next, feats|None)."""

    def __init__(self):
        self.k = -1                                   # local index of last ingested bar
        self.jma_track = SignTrack()
        self.osc_tracks = {name: SignTrack() for name, _ in OSC_STREAMS}
        self.last_event = {c: None for c in CLASSES}  # local bar of last event
        self.ewm = {c: [0.0] * len(TAUS) for c in CLASSES}
        self.last_self = None
        self.leg_start_jma = None
        self.wf = [Welford() for _ in VALUE_BASE]
        self.ring = [np.zeros(len(VALUE_BASE)),       # z at k, k-1, k-2
                     np.zeros(len(VALUE_BASE)),
                     np.zeros(len(VALUE_BASE))]

    def ingest(self, rawOpen, rawLast, jma, d1, d2, td1, td2, ts_us):
        self.k += 1
        k = self.k

        # 1. leg / SELF                                                    W.3
        self_event, _ = self.jma_track.update(jma)
        leg_dir = self.jma_track.sign
        ind = {c: 0 for c in CLASSES}
        if self_event:
            ind[("MNQ_JMA_SELF", "all")] = 1
            self.last_event[("MNQ_JMA_SELF", "all")] = k
            self.last_self = k
            self.leg_start_jma = jma
        if self.leg_start_jma is None:
            self.leg_start_jma = jma

        # 2. oscillator events
        vals = {"jmaD1": d1, "jmaD2": d2, "tickJmaD1": td1, "tickJmaD2": td2}
        for name, col in OSC_STREAMS:
            ev, pol = self.osc_tracks[name].update(vals[col])
            if ev and leg_dir != 0:                    # opposing==0 dropped
                cls = (name, "opp" if pol * leg_dir == 1 else "conf")
                ind[cls] = 1
                self.last_event[cls] = k

        # 3. raw channel values at k, signed confirming-positive
        body = rawLast - rawOpen
        ch = np.array([d1 * leg_dir, d2 * leg_dir, td1 * leg_dir, td2 * leg_dir,
                       abs(d1), abs(d2), abs(td1), abs(td2),
                       abs(jma - self.leg_start_jma),
                       body * leg_dir, abs(body)], dtype=np.float64)

        # 4. z read-then-update, push ring                                 W.4
        z = np.array([self.wf[i].z_then_update(ch[i]) for i in range(len(ch))])
        self.ring = [z, self.ring[0], self.ring[1]]

        # 5. grammar recursions -> state now reflects events <= k
        for c in CLASSES:
            e = self.ewm[c]
            i = ind[c]
            for j, a in enumerate(ALPHAS):
                e[j] = a * (e[j] + i)

        # 6. features for forming bar t = k+1                              W.1
        t = k + 1
        if t < WARMUP_BARS:                                                # W.5
            return t, None
        feats = np.empty(38 + 33, dtype=np.float64)
        col = 0
        for c in CLASSES:
            le = self.last_event[c]
            feats[col] = (t - le) if le is not None else (t + 1)
            col += 1
            for j in range(len(TAUS)):
                feats[col] = self.ewm[c][j]
                col += 1
        feats[col] = (t - self.last_self) if self.last_self is not None else (t + 1)
        col += 1
        ts_next = ts_us + FRAME * 1_000_000
        mins = (ts_next // 60_000_000) % 1440 - TOD_START_MIN
        feats[col] = min(max(mins // TOD_BIN_MIN, 0), N_TOD - 1)
        col += 1
        if t < VALUE_WARM:                                                 # W.5
            feats[col:] = 0.0
        else:
            feats[col:col + 11] = self.ring[0]
            feats[col + 11:col + 22] = self.ring[1]
            feats[col + 22:col + 33] = self.ring[2]
        return t, feats


# ---------------------------------------------------------------- model host

class Scorer:
    def __init__(self, model_path):
        b = joblib.load(model_path)
        self.booster = b["booster"]
        self.best_it = self.booster.best_iteration
        names = worker_feature_names()                                     # W.2
        assert names == list(b["feature_names"]), (
            "feature contract mismatch:\n"
            + "\n".join(f"{i}: {a} != {c}" for i, (a, c)
                        in enumerate(zip(names, b["feature_names"])) if a != c))
        iso = b["iso"]
        self.iso_x = np.asarray(iso.X_thresholds_, dtype=np.float64)       # W.6
        self.iso_y = np.asarray(iso.y_thresholds_, dtype=np.float64)
        probe = np.linspace(0, 1, 4097)
        assert np.max(np.abs(np.interp(probe, self.iso_x, self.iso_y)
                             - iso.predict(probe))) < 1e-12

    def predict(self, X):
        p = self.booster.predict(np.asarray(X, dtype=np.float32),
                                 num_iteration=self.best_it)
        return p, np.interp(p, self.iso_x, self.iso_y)


def lamp_of(p_cal):
    if p_cal < GREEN_MAX:
        return 1
    if p_cal >= RED_MIN:
        return 3
    return 2


# ---------------------------------------------------------------- replay gate

def load_research_frame():
    raw = pd.read_parquet(RAW_PATH)
    raw = raw[raw["timestamp"].dt.time >= pd.Timestamp("09:30").time()]
    ohlc = pd.read_parquet(RAW_OHLC_FILE)
    ohlc = ohlc[ohlc["timestamp"].dt.time >= pd.Timestamp("09:30").time()]
    ohlc = ohlc.rename(columns={"Open": "rawOpen", "High": "rawHigh",
                                "Low": "rawLow", "Last": "rawLast"})
    raw = raw.merge(ohlc, on="timestamp", how="left")
    assert raw[["rawOpen", "rawLast"]].notna().all().all()
    return raw.sort_values("timestamp").reset_index(drop=True)


def replay(scorer):                                                        # W.9
    raw = load_research_frame()
    if REPLAY_FROM is not None:
        raw = raw[raw["timestamp"] >= REPLAY_FROM]
    ref = pd.read_parquet(PRED_PATH)[["timestamp", "p", "p_cal"]]

    out_ts, out_rows = [], []
    cols = ["rawOpen", "rawLast", "JMA", "jmaD1", "jmaD2",
            "tickJmaD1", "tickJmaD2"]
    for day, g in raw.groupby(raw["timestamp"].dt.date, sort=True):
        eng = Engine()
        ts_us = g["timestamp"].astype("int64").to_numpy() // 1000
        V = g[cols].to_numpy(np.float64)
        n = len(g)
        rows, tss = [], []
        for i in range(n):
            t, feats = eng.ingest(*V[i], ts_us[i])
            if feats is not None and t < n:            # forming bar must exist
                rows.append(feats)
                tss.append(ts_us[i] + FRAME * 1_000_000)
        if rows:
            out_rows.append(np.vstack(rows))
            out_ts.append(np.asarray(tss))

    X = np.concatenate(out_rows)
    ts = pd.to_datetime(np.concatenate(out_ts), unit="us")
    p, p_cal = scorer.predict(X)
    live = pd.DataFrame({"timestamp": ts,
                         "p_live": p.astype(np.float32),
                         "pcal_live": p_cal.astype(np.float32)})
    m = live.merge(ref, on="timestamp", how="inner")
    dp = np.abs(m["p_live"] - m["p"])
    dc = np.abs(m["pcal_live"] - m["p_cal"])
    print(f"replay rows compared: {len(m)}  (live {len(live)}, ref {len(ref)})")
    print(f"max |dp|     = {dp.max():.3e}   n>1e-6: {(dp > 1e-6).sum()}")
    print(f"max |dp_cal| = {dc.max():.3e}   n>1e-6: {(dc > 1e-6).sum()}")
    print("GATE:", "PASS" if dp.max() <= 1e-6 and dc.max() <= 1e-6 else "FAIL")
    return m


# ---------------------------------------------------------------- serve loop

def serve(scorer):
    import zmq
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.bind(ZMQ_IN)
    push = ctx.socket(zmq.PUSH)
    push.bind(ZMQ_OUT)
    print(f"hazard worker up  in={ZMQ_IN}  out={ZMQ_OUT}  tag={TAG}")

    eng = Engine()
    cur_day = None
    dwell = 0
    while True:
        ts_us, rawOpen, rawLast, jma, d1, d2, td1, td2 = IN_MSG.unpack(pull.recv())
        ts = pd.Timestamp(ts_us, unit="us")
        if ts.time() < pd.Timestamp("09:30").time():                       # W.7
            continue
        if ts.date() != cur_day:
            cur_day = ts.date()
            eng = Engine()
            dwell = 0
        t, feats = eng.ingest(rawOpen, rawLast, jma, d1, d2, td1, td2, ts_us)
        ts_next = ts_us + FRAME * 1_000_000
        if feats is None:
            dwell = 0
            push.send(OUT_MSG.pack(ts_next, -1.0, -1.0, 0, 0))
            continue
        p, p_cal = scorer.predict(feats.reshape(1, -1))
        lamp = lamp_of(float(p_cal[0]))
        dwell = dwell + 1 if lamp == 3 else 0
        push.send(OUT_MSG.pack(ts_next, float(p[0]), float(p_cal[0]), lamp, dwell))


# ---------------------------------------------------------------- usage

if __name__ == "__main__":
    scorer = Scorer(MODEL_PATH)
    if MODE == "replay":
        m = replay(scorer)
    else:
        serve(scorer)