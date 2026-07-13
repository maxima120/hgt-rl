import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.signal import lfilter
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

STREAMS = ["MNQ_D1", "MNQ_D2", "TICK_D1", "TICK_D2"]
TOD_START_MIN = 570      # first session minute (9:30)
TOD_BIN_MIN = 30
N_TOD = 13               # bins covering the session 6.5hrs / 30 min

def _make_classes():
    cls = []
    for s in STREAMS:
        cls.append((s, "opp"))
        cls.append((s, "conf"))
    cls.append(("MNQ_JMA_SELF", "all"))
    return cls

def _bins_from_edges(edges):
    bins, lo = [], 1
    for e in edges:
        bins.append((lo, int(e)))
        lo = int(e) + 1
    return bins

def _age_bins_from_edges(edges):
    b = _bins_from_edges(edges)
    b.append((int(edges[-1]) + 1, 10 ** 9))
    return b

class Featurizer:
    def __init__(self, bars, events,
                 bin_edges=(1, 2, 3, 5, 8, 13, 21, 34),
                 age_edges=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89)):
        self.bin_edges = list(bin_edges)
        self.age_edges = list(age_edges)
        self.bins = _bins_from_edges(bin_edges)
        self.age_bins = _age_bins_from_edges(age_edges)
        self.classes = _make_classes()
        self.feature_names = (
            [f"{s}|{c}|lag{lo}_{hi}" for (s, c) in self.classes for (lo, hi) in self.bins]
            + [f"age|{lo}_{hi}" for (lo, hi) in self.age_bins]
            + [f"tod|{k}" for k in range(N_TOD)]
        )
        self.n_feat = len(self.feature_names)

        ev = events[events["stream"].isin([c[0] for c in self.classes])]
        ev = ev[~((ev["stream"] != "MNQ_JMA_SELF") & (ev["opposing"] == 0))]   # S2.6
        evg = dict(tuple(ev.groupby("date")))

        self.sessions = []
        bars = bars.sort_values("bar_index").reset_index(drop=True)
        for sess, g in bars.groupby("date", sort=True):
            n = len(g)
            first = int(g["bar_index"].iloc[0])
            tgt = g["is_target"].to_numpy()
            warm = g["warm"].to_numpy()
            ts = pd.DatetimeIndex(g["timestamp"])
            mins = ts.hour.to_numpy() * 60 + ts.minute.to_numpy() - TOD_START_MIN   # S2.10
            tod = np.clip(mins // TOD_BIN_MIN, 0, N_TOD - 1).astype(np.int16)
            lt_incl = np.maximum.accumulate(np.where(tgt, np.arange(n), -1))
            P = {}
            e = evg.get(sess)
            for (s, c) in self.classes:
                if e is None:
                    loc = np.empty(0, np.int64)
                else:
                    if s == "MNQ_JMA_SELF":
                        sub = e[e["stream"] == s]
                    else:
                        want = 1 if c == "opp" else -1
                        sub = e[(e["stream"] == s) & (e["opposing"] == want)]
                    loc = sub["event_bar"].to_numpy() - first
                ind = np.zeros(n, np.int32)
                ind[loc] = 1
                P[(s, c)] = np.concatenate(([0], np.cumsum(ind)))              # P[i] = count pos < i
            self.sessions.append(dict(
                sess=sess, first=first, n=n, tgt=tgt, warm=warm, tod=tod,
                lt_incl=lt_incl, P=P,
                bar_index=g["bar_index"].to_numpy(),
                timestamp=g["timestamp"].to_numpy()))

    def _fill(self, S, t, out):
        n = S["n"]
        col = 0
        for c in self.classes:
            P = S["P"][c]
            for (lo, hi) in self.bins:                                         # S2.1, S2.2
                b = np.clip(t - lo + 1, 0, n)
                a = np.clip(t - hi, 0, n)
                out[:, col] = P[b] - P[a]
                col += 1
        lt = np.where(t > 0, S["lt_incl"][np.maximum(t - 1, 0)], -1)           # S2.3
        age = np.where(lt >= 0, t - lt, t + 1)
        for (lo, hi) in self.age_bins:
            out[:, col] = (age >= lo) & (age <= hi)
            col += 1
        tod = S["tod"][t]
        for k in range(N_TOD):
            out[:, col] = tod == k
            col += 1

    def _fill_frozen(self, S, q, h, out):
        """S2.5: features at t = q+h using only events/targets with pos <= q."""
        n = S["n"]
        t = q + h
        cap = q + 1
        col = 0
        for c in self.classes:
            P = S["P"][c]
            for (lo, hi) in self.bins:
                b = np.clip(np.minimum(t - lo + 1, cap), 0, n)
                a = np.clip(np.minimum(t - hi, cap), 0, n)
                out[:, col] = P[b] - P[a]
                col += 1
        lt = S["lt_incl"][q]
        age = np.where(lt >= 0, t - lt, t + 1)
        for (lo, hi) in self.age_bins:
            out[:, col] = (age >= lo) & (age <= hi)
            col += 1
        tod = S["tod"][np.minimum(t, n - 1)]
        for k in range(N_TOD):
            out[:, col] = tod == k
            col += 1

    def _selected(self, date_from, date_to):
        for S in self.sessions:
            d = str(S["sess"])
            if date_from is not None and d < date_from:
                continue
            if date_to is not None and d > date_to:
                continue
            yield S

    def build(self, date_from=None, date_to=None):
        sel = []
        total = 0
        for S in self._selected(date_from, date_to):
            t = np.nonzero(~S["warm"])[0]                                      # S2.7
            sel.append((S, t))
            total += len(t)
        X = np.zeros((total, self.n_feat), np.float32)
        y = np.zeros(total, np.int8)
        bar_index = np.zeros(total, np.int64)
        sess_id = np.zeros(total, np.int32)
        timestamp = np.zeros(total, "datetime64[ns]")
        ofs = 0
        for sid, (S, t) in enumerate(sel):
            m = len(t)
            self._fill(S, t, X[ofs:ofs + m])
            y[ofs:ofs + m] = S["tgt"][t]
            bar_index[ofs:ofs + m] = S["bar_index"][t]
            sess_id[ofs:ofs + m] = sid
            timestamp[ofs:ofs + m] = S["timestamp"][t]
            ofs += m
        meta = pd.DataFrame({"bar_index": bar_index, "timestamp": timestamp,
                             "sess_id": sess_id, "is_target": y.astype(bool)})
        return X, y, meta