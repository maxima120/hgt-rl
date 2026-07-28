# file-splitting: 
# +load_manifest is the natural first thing to lift into a shared pipeline_common.py — Stage 5, 
# -the worker (reading the baked-in copy), score_all_dates, 
#  and the monitor will all want the same derivation (_tod_start_min, _n_tod, _stream_cols). 
# Writing it once there kills the biggest copy-paste surface. 
# the z-primitives (_expanding_z, _welford_check) and the Featurizer are the three obvious shared modules;

import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.signal import lfilter
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

# ---------------------------------------------------------------- Featurizer

class Featurizer:
    def __init__(self, bars, events, manifest, 
                 TOD_BIN_MIN, 
                 TAUS,
                 bin_edges=(1, 2, 3, 5, 8, 13, 21, 34),
                 age_edges=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89)):
        
        self.manifest = manifest
        self.tod_start_min = manifest["_tod_start_min"]
        self.n_tod = manifest["_n_tod"]
        self.bin_edges = list(bin_edges)
        self.age_edges = list(age_edges)
        self.bins = self._bins_from_edges(bin_edges)
        self.age_bins = self._age_bins_from_edges(age_edges)

        # classes derived from manifest stream set (opp/conf per stream + SELF)
        self.classes = []
        for s in manifest["_stream_names"]:
            self.classes.append((s, "opp"))
            self.classes.append((s, "conf"))
        self.classes.append((manifest["_self"], "all"))

        self.grammar_names = []
        for (s, c) in self.classes:
            self.grammar_names.append(f"{s}|{c}|bsince")
            self.grammar_names += [f"{s}|{c}|ewm{t:g}" for t in TAUS]
        self.grammar_names += ["age", "tod"]

        ev = events[events["stream"].isin([c[0] for c in self.classes])]
        ev = ev[~((ev["stream"] != manifest["_self"]) & (ev["opposing"] == 0))]
        evg = dict(tuple(ev.groupby("date")))

        self.sessions = []
        bars = bars.sort_values("bar_index").reset_index(drop=True)
        for sess, g in bars.groupby("date", sort=True):
            n = len(g)
            first = int(g["bar_index"].iloc[0])
            tgt = g["is_target"].to_numpy()
            warm = g["warm"].to_numpy()
            ts = pd.DatetimeIndex(g["timestamp"])
            mins = ts.hour.to_numpy() * 60 + ts.minute.to_numpy() - self.tod_start_min
            tod = np.clip(mins // TOD_BIN_MIN, 0, self.n_tod - 1).astype(np.int16)
            lt_incl = np.maximum.accumulate(np.where(tgt, np.arange(n), -1))
            P = {}
            e = evg.get(sess)
            for (s, c) in self.classes:
                if e is None:
                    loc = np.empty(0, np.int64)
                else:
                    if s == manifest["_self"]:
                        sub = e[e["stream"] == s]
                    else:
                        want = 1 if c == "opp" else -1
                        sub = e[(e["stream"] == s) & (e["opposing"] == want)]
                    loc = sub["event_bar"].to_numpy() - first
                ind = np.zeros(n, np.int32)
                ind[loc] = 1
                P[(s, c)] = np.concatenate(([0], np.cumsum(ind)))
            self.sessions.append(dict(
                sess=sess, first=first, n=n, tgt=tgt, warm=warm, tod=tod,
                lt_incl=lt_incl, P=P,
                bar_index=g["bar_index"].to_numpy(),
                timestamp=g["timestamp"].to_numpy()))

        self.augment_featurizer(bars)

    # public method
    def _selected(self, date_from, date_to):
        for S in self.sessions:
            d = str(S["sess"])
            if date_from is not None and d < date_from:
                continue
            if date_to is not None and d > date_to:
                continue
            yield S

    def _bins_from_edges(self,edges):
        bins, lo = [], 1
        for e in edges:
            bins.append((lo, int(e)))
            lo = int(e) + 1
        return bins


    def _age_bins_from_edges(self,edges):
        b = self._bins_from_edges(edges)
        b.append((int(edges[-1]) + 1, 10 ** 9))
        return b

    def augment_featurizer(self, bars):
        """Attach leg_dir and leg_start JMA level per session (for value features)."""
        bg = dict(tuple(bars.sort_values("bar_index").groupby("date")))
        for S in self.sessions:
            g = bg[S["sess"]]
            S["leg_dir"] = g["jma_leg_dir"].to_numpy(np.float64)
            jma = g["jma"].to_numpy(np.float64)
            tgt = g["is_target"].to_numpy()
            starts = np.where(tgt)[0]
            leg_id = np.cumsum(tgt)
            start_idx = np.concatenate(([0], starts))[leg_id]
            S["leg_start_jma"] = jma[start_idx]

# ---------------------------------------------------------------------------

def load_manifest(path,TOD_BIN_MIN):
    with open(path) as f:
        man = json.load(f)
    start = pd.Timestamp(man["session_start"])
    end = pd.Timestamp(man["session_end"])
    man["_tod_start_min"] = start.hour * 60 + start.minute
    man["_session_min"] = (end.hour * 60 + end.minute) - man["_tod_start_min"]
    man["_n_tod"] = int(np.ceil(man["_session_min"] / TOD_BIN_MIN))
    man["_stream_names"] = [s["name"] for s in man["streams"]]              # non-self
    man["_stream_cols"] = [s["column"] for s in man["streams"]]             # source cols
    man["_self"] = man["self_stream"]
    return man


 # ---------------------------------------------------------------- z primitives
def _expanding_z(x, warm):
    n = len(x)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    csq = np.concatenate(([0.0], np.cumsum(x * x)))
    k = np.arange(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(k > 0, csum[:-1] / np.maximum(k, 1), 0.0)
        var = np.where(k > 1, (csq[:-1] - k * mean * mean) / np.maximum(k - 1, 1), 0.0)
        std = np.sqrt(np.maximum(var, 0.0))
        z = np.where((k >= warm) & (std > 0), (x - mean) / std, 0.0)
    return z


def _welford_check(x, warm):
    z_vec = _expanding_z(x, warm)
    k, mean, M2 = 0, 0.0, 0.0
    z_ref = np.zeros(len(x))
    for t in range(len(x)):
        if k >= warm and M2 > 0:
            z_ref[t] = (x[t] - mean) / np.sqrt(M2 / (k - 1))
        k += 1
        d = x[t] - mean
        mean += d / k
        M2 += d * (x[t] - mean)
    return float(np.max(np.abs(z_vec - z_ref)))


