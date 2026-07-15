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
SC_TAG = "mnq-TICK-9-12am"             # single label; identifies the fork end-to-end

MODEL_PATH = f"stage-5/model_{SC_TAG}.joblib"
PRED_PATH = f"stage-5/pred_{SC_TAG}.pqt"     # replay reference only
REPLAY_FROM = "2026-01-01"             # None = full history

WORKER_ID = 1                          # set by core when spawned; port math below
LOG_DIR = "./logs"

STICKY_BARS = 3

# data paths (source/raw) come from the model's baked manifest, not set here
# ----------------------------------------------------------------

TAUS = (2.0, 6.0, 18.0)
ALPHAS = tuple(np.exp(-1.0 / t) for t in TAUS)
VALUE_LAGS = (1, 2)
ZWARM = 20
TOD_BIN_MIN = 30

OUT_MSG = struct.Struct("<iffff")      # bar_idx, p, p_cal, sticky, risk_consumed

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



# ---------------------------------------------------------------- usage
if __name__ == "__main__":
    log, log_path = make_logger()
    contract = load_contract(MODEL_PATH)
    scorer = Scorer(contract, log)

    live, m = replay(scorer, log)

    m["date"] = m.timestamp.dt.date
    m["k"] = m.groupby("date").cumcount()          # 0 = first scored bar (t=10)
    bad = m[(m.p_live - m.p).abs() > 1e-6]
    print(bad.date.nunique(), "of", m.date.nunique(), "sessions affected")
    print(bad.k.describe())
    print(bad.groupby("k").size().head(30))
