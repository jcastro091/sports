#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build strong/shadow configs + explicit sport×market pair whitelists,
and write robust, pair-aware filter waterfalls.

Hardenings:
- Case-insensitive column resolution everywhere (incl. group frames)
- Derive odds_bin if missing; allow SKIP_ODDS_BIN=1
- Allow SKIP_PROBA=1 or auto-detect alternative proba columns
- Re-verify columns inside each group before filtering
"""

import os, json, sys, glob
from typing import List, Dict, Tuple, Set

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Please: pip install pandas openpyxl numpy")
    sys.exit(1)

# ---------- Thresholds ----------
MIN_AUC      = float(os.getenv("MIN_AUC", "0.65"))
MIN_N        = int(os.getenv("MIN_N", "25"))
MIN_PROBA    = float(os.getenv("MIN_PROBA", "0.61"))
MIN_BIN_WIN  = float(os.getenv("MIN_BIN_WIN", "0.52"))
MIN_BIN_N    = int(os.getenv("MIN_BIN_N", "60"))
TOP_K_SPORTS = int(os.getenv("TOP_K_SPORTS", "3"))

# Shadow config thresholds
SHADOW_MIN_PROBA = float(os.getenv("SHADOW_MIN_PROBA", "0.58"))
SHADOW_MAX_PROBA = float(os.getenv("SHADOW_MAX_PROBA", "0.61"))  # exclusive

# Pairs / toggles
PAIR_MIN_N = int(os.getenv("PAIR_MIN_N", "20"))
ENFORCE_PAIRS_IN_WATERFALL = os.getenv("ENFORCE_PAIRS_IN_WATERFALL", "1").strip() == "1"
SKIP_ODDS_BIN = os.getenv("SKIP_ODDS_BIN", "0").strip() == "1"
SKIP_PROBA    = os.getenv("SKIP_PROBA", "0").strip() == "1"
DEBUG_COLS    = os.getenv("DEBUG_COLS", "0").strip() == "1"

INCLUDE_SPORTS  = [s.strip() for s in os.getenv("INCLUDE_SPORTS", "").split(",") if s.strip()]
INCLUDE_MARKETS = [s.strip() for s in os.getenv("INCLUDE_MARKETS", "").split(",") if s.strip()]
EXCLUDE_SPORTS  = set([s.strip() for s in os.getenv("EXCLUDE_SPORTS", "").split(",") if s.strip()])
EXCLUDE_MARKETS = set([s.strip() for s in os.getenv("EXCLUDE_MARKETS", "").split(",") if s.strip()])

# ---------- Candidate roots ----------
USER = os.path.expanduser("~")
CANDIDATE_ROOTS = [
    os.path.join(USER, "Documents", "alpha_signal_engine", "data", "results", "models"),
    os.path.join(USER, "OneDrive", "Documents", "alpha_signal_engine", "data", "results", "models"),
    os.getcwd(),
]

# ---------- Filenames ----------
SPORT_BASE   = "segments_sport_test"
MARKET_BASE  = "segments_market_test"
ODDSBIN_CSV  = "descriptive_winrate_by_odds_bin.csv"
PREDICTIONS  = "predictions_latest.csv"

# ---------- Utilities ----------
# --- Auto-min_proba from predictions_latest.csv ---
def _roi_for_sel(win_rate, avg_odds):
    # $1 stakes: ROI ≈ win_rate*(avg_odds-1) - (1 - win_rate)
    return win_rate*(avg_odds - 1.0) - (1.0 - win_rate)

def auto_min_proba_from_predictions(preds_path: str, min_bets: int = 40,
                                    t_low: float = 0.52, t_high: float = 0.70, step: float = 0.01) -> float | None:
    import pandas as pd, numpy as np
    if not preds_path or not os.path.isfile(preds_path):
        print("[-] auto-tune: predictions file not found.")
        return None

    try:
        if preds_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(preds_path)
        else:
            df = pd.read_csv(preds_path)
    except Exception as e:
        print(f"[warn] Could not read {preds_path}: {e}")
        return


    # -------- column resolution (case-insensitive) --------
    def get_col(*names):
        m = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in m: 
                return m[n.lower()]
        return None

    prob_c = get_col("proba","prob","probability","pred_proba","predprob","p","p_win","win_prob","win_probability")
    # decimal odds first; else implied prob we convert to decimal
    odds_c = get_col("decimal_odds","odds","close_dec","curr_dec","open_dec")
    imp_c  = None
    if odds_c is None:
        imp_c = get_col("curr_imp","close_imp","open_imp","implied","consensus_close_imp","final_imp","market_imp")
        if imp_c is not None:
            df["_auto_decimal_odds_"] = 1.0 / pd.to_numeric(df[imp_c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            odds_c = "_auto_decimal_odds_"

    # actual / label aliases
    actual_c = get_col("actual","y","label","target","result","outcome","prediction result","prediction_result","won")

    print(f"[auto-tune] resolved columns → prob={prob_c}, odds={odds_c} (imp={imp_c or '—'}), actual={actual_c}")
    if not prob_c or not odds_c or not actual_c:
        print("[-] auto-tune: missing one or more required columns.")
        return None

    d = df[[prob_c, odds_c, actual_c]].copy()
    # normalize proba; handle 0-100 style
    d[prob_c] = pd.to_numeric(d[prob_c], errors="coerce")
    if d[prob_c].max(skipna=True) and float(d[prob_c].max()) > 1.5:
        d[prob_c] = d[prob_c] / 100.0

    d[odds_c] = pd.to_numeric(d[odds_c], errors="coerce")

    # map textual outcomes to 1/0; then coerce numeric
    if d[actual_c].dtype == object:
        d[actual_c] = d[actual_c].astype(str).str.strip().str.lower()
        d[actual_c] = d[actual_c].replace({
            "win":1,"won":1,"w":1,"true":1,"t":1,"1":1,
            "loss":0,"lose":0,"lost":0,"l":0,"false":0,"f":0,"0":0
        })
    d[actual_c] = pd.to_numeric(d[actual_c], errors="coerce")

    # drop bad rows and zeros/negatives
    before = len(d)
    d = d.dropna()
    d = d[(d[prob_c] > 0) & (d[prob_c] < 1) & (d[odds_c] > 1.0) & (d[actual_c].isin([0,1]))]
    print(f"[auto-tune] usable rows: {len(d)} / {before}")

    if d.empty:
        print("[-] auto-tune: no usable rows after cleaning.")
        return None

    thr = np.round(np.arange(t_low, t_high + 1e-9, step), 2)

    def _roi(win_rate, avg_odds):
        return win_rate*(avg_odds - 1.0) - (1.0 - win_rate)

    best = None
    # first try the requested min_bets
    for t in thr:
        sel = d[d[prob_c] >= t]
        if len(sel) < min_bets:
            continue
        hit = sel[actual_c].mean()
        avg = sel[odds_c].mean()
        cand = dict(t=float(t), n=int(len(sel)), hit=float(hit), roi=float(_roi(hit, avg)))
        if (best is None) or (cand["roi"] > best["roi"]) or (cand["roi"] == best["roi"] and cand["n"] > best["n"]):
            best = cand

    # if nothing met min_bets, relax and try again with 10 bets
    if best is None and len(d) >= 10:
        print(f"[auto-tune] relaxing min_bets from {min_bets} → 10")
        for t in thr:
            sel = d[d[prob_c] >= t]
            if len(sel) < 10:
                continue
            hit = sel[actual_c].mean()
            avg = sel[odds_c].mean()
            cand = dict(t=float(t), n=int(len(sel)), hit=float(hit), roi=float(_roi(hit, avg)))
            if (best is None) or (cand["roi"] > best["roi"]) or (cand["roi"] == best["roi"] and cand["n"] > best["n"]):
                best = cand

    if best is None:
        print("[-] auto-tune: no threshold met the minimum sample size.")
        return None

    print(f"📌 Auto-selected min_proba={best['t']:.2f} (roi={best['roi']:.3f}, hit={best['hit']:.3f}, n={best['n']})")
    return float(best["t"])


def get_col(df: pd.DataFrame, *names_lower: str) -> str | None:
    m = {c.lower(): c for c in df.columns}
    for nm in names_lower:
        if nm in m:
            return m[nm]
    return None

def find_segment_file(root: str, base: str) -> str | None:
    patterns = [
        os.path.join(root, base + ".xlsx"),
        os.path.join(root, base + ".csv"),
        os.path.join(root, base + "*.*"),
    ]
    for p in patterns:
        for m in glob.glob(p):
            if os.path.isfile(m) and (m.lower().endswith(".xlsx") or m.lower().endswith(".csv")):
                return m
    return None

def find_any(root_list: List[str], filename: str) -> str | None:
    for root in root_list:
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            return path
    for root in root_list:
        matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if matches:
            return matches[0]
    return None

def read_segments(path: str) -> List[Dict]:
    if not path:
        return []
    df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    n_col   = cols.get("n") or cols.get("count") or cols.get("samples")
    auc_col = cols.get("auc") or cols.get("roc_auc") or cols.get("auroc")
    sport_col  = cols.get("sport")
    market_col = cols.get("market")
    out = []
    for _, row in df.iterrows():
        try:
            out.append({
                "n": int(row[n_col]) if n_col else 0,
                "auc": float(row[auc_col]) if auc_col else 0.0,
                "Sport": str(row[sport_col]) if sport_col else "",
                "Market": str(row[market_col]) if market_col else "",
            })
        except Exception:
            continue
    return out

# ---------- Selection ----------
def pick_sports(rows: List[Dict]) -> List[str]:
    keep = {r["Sport"] for r in rows if r.get("Sport") and r.get("n",0) >= MIN_N and r.get("auc",0.0) >= MIN_AUC}
    keep |= set(INCLUDE_SPORTS)
    keep -= EXCLUDE_SPORTS
    return sorted(keep)

def rank_top_k_sports(rows: List[Dict], k: int) -> List[str]:
    cand = [r for r in rows if r.get("Sport") and r.get("n",0) >= MIN_N]
    cand.sort(key=lambda r: (r.get("auc",0.0), r.get("n",0)), reverse=True)
    ordered = [r["Sport"] for r in cand]
    ordered = [s for s in ordered if s not in EXCLUDE_SPORTS]
    for s in INCLUDE_SPORTS:
        if s not in ordered:
            ordered.insert(0, s)
    seen, top = set(), []
    for s in ordered:
        if s not in seen:
            seen.add(s)
            top.append(s)
        if len(top) >= k:
            break
    return top

def pick_markets(rows: List[Dict]) -> List[str]:
    keep = {r["Market"] for r in rows if r.get("Market") and r.get("n",0) >= MIN_N and r.get("auc",0.0) >= MIN_AUC}
    keep |= set(INCLUDE_MARKETS)
    keep -= EXCLUDE_MARKETS
    return sorted(keep)

DEFAULT_SPORTS = ["soccer_france_ligue_one", "americanfootball_ncaaf"]
DEFAULT_MARKETS = ["H2H Home", "H2H Away"]
DEFAULT_BINS    = ["Fav", "Balanced"]

def pick_odds_bins(oddsbin_path: str | None) -> List[str]:
    if not oddsbin_path or not os.path.isfile(oddsbin_path):
        return DEFAULT_BINS.copy()
    try:
        dfb = pd.read_csv(oddsbin_path)
        cols = {c.lower().strip(): c for c in dfb.columns}
        bin_col = cols.get("odds_bin") or list(dfb.columns)[0]
        n_col   = cols.get("n") or list(dfb.columns)[1]
        wr_col  = cols.get("win_rate") or list(dfb.columns)[2]
        dfb = dfb[[bin_col, n_col, wr_col]].copy()
        dfb.columns = ["odds_bin", "n", "win_rate"]
        dfb = dfb.sort_values("n", ascending=False)
        chosen = dfb[(dfb["n"] >= MIN_BIN_N) & (dfb["win_rate"] >= MIN_BIN_WIN)]["odds_bin"].tolist()
        chosen = [b for b in chosen if b.lower() != "longshot"]
        return sorted(chosen) or DEFAULT_BINS.copy()
    except Exception:
        return DEFAULT_BINS.copy()

# ---------- Pairs ----------
def cartesian_pairs(sports: List[str], markets: List[str]) -> List[Tuple[str,str]]:
    return [(s, m) for s in sports for m in markets]

def filter_pairs_by_predictions(pairs: List[Tuple[str,str]], preds_path: str | None, min_n: int) -> List[Tuple[str,str]]:
    if not preds_path or not os.path.isfile(preds_path):
        return pairs
    try:
        df = pd.read_csv(preds_path)
    except Exception:
        return pairs
    sport_c  = get_col(df, "sport", "Sport")
    market_c = get_col(df, "market", "Market")
    if not sport_c or not market_c:
        return pairs
    gp = df.groupby([df[sport_c].astype(str), df[market_c].astype(str)]).size().reset_index(name="n")
    have = {(r[sport_c], r[market_c]) for _, r in gp.iterrows() if int(r["n"]) >= min_n}
    pruned = [p for p in pairs if p in have]
    return pruned or pairs

def write_pairs_json(path: str, pairs: List[Tuple[str,str]]) -> None:
    with open(path, "w") as f:
        json.dump([{"sport": s, "market": m} for (s,m) in pairs], f, indent=2)

# ---------- Odds-bin derivation ----------
def derive_odds_bin_inplace(df: pd.DataFrame) -> pd.DataFrame:
    if get_col(df, "odds_bin"):
        return df
    am_c = get_col(df, "american_odds")
    dec_c = get_col(df, "decimal_odds")
    if am_c:
        ao = pd.to_numeric(df[am_c], errors="coerce").fillna(0.0)
        bins = np.where(ao <= -200, "HeavyFav",
                np.where(ao <= -120, "Fav",
                np.where(ao < 120, "Balanced",
                np.where(ao < 250, "Dog", "Longshot"))))
        df["odds_bin"] = bins
        return df
    if dec_c:
        dec = pd.to_numeric(df[dec_c], errors="coerce").fillna(1.0)
        ao = np.where(dec >= 2.0, 100*(dec-1), -100/(dec-1 + 1e-9))
        bins = np.where(ao <= -200, "HeavyFav",
                np.where(ao <= -120, "Fav",
                np.where(ao < 120, "Balanced",
                np.where(ao < 250, "Dog", "Longshot"))))
        df["odds_bin"] = bins
        return df
    df["odds_bin"] = "Balanced"
    return df

# ---------- Proba resolver ----------
PROBA_ALIASES = ("proba","prob","probability","pred_proba","predprob","p","p_win","win_prob","win_probability")
def ensure_proba_col(df: pd.DataFrame) -> str | None:
    return get_col(df, *PROBA_ALIASES)

# ---------- Reports ----------
def run_filter_report(cfg: Dict, preds_path: str | None, suffix: str = "", proba_max: float | None = None, allow_pairs: Set[Tuple[str,str]] | None = None) -> None:
    if not preds_path or not os.path.isfile(preds_path):
        print("ℹ️ predictions_latest.csv not found; skipping filter-through report.")
        return
    try:
        df = pd.read_csv(preds_path)
    except Exception as e:
        print(f"[warn] Could not read {preds_path}: {e}")
        return

    # Global ensure/resolve
    if not get_col(df, "odds_bin"):
        df = derive_odds_bin_inplace(df)
    sport_c   = get_col(df, "sport", "Sport")
    market_c  = get_col(df, "market", "Market")
    oddsbin_c = get_col(df, "odds_bin")
    proba_c   = ensure_proba_col(df)

    if DEBUG_COLS:
        print(f"[debug] resolved columns: sport={sport_c} market={market_c} odds_bin={oddsbin_c} proba={proba_c}")

    # Required fields
    if not sport_c or not market_c:
        print("[warn] predictions missing sport/market; skipping filter report.")
        return

    if not SKIP_PROBA and not proba_c:
        print("[warn] predictions missing probability column; set SKIP_PROBA=1 to bypass. Skipping filter report.")
        return

    # Config
    s_keep = set(cfg.get("sports", []))
    m_keep = set(cfg.get("markets", []))
    b_keep = set(cfg.get("odds_bin", []))
    thr = float(cfg.get("min_proba", MIN_PROBA))
    pair_keep = set(allow_pairs or [])

    # Waterfall
    steps = []
    def add_step(name, n): steps.append({"step": name, "count": int(n)})

    df0 = df.copy()
    add_step("start_total", len(df0))

    df1 = df0[df0[sport_c].isin(s_keep)] if s_keep else df0
    add_step("after_sport", len(df1))

    df2 = df1[df1[market_c].isin(m_keep)] if m_keep else df1
    add_step("after_market", len(df2))

    if pair_keep:
        df2 = df2[[(sp, mk) in pair_keep for sp, mk in zip(df2[sport_c].astype(str), df2[market_c].astype(str))]]
        add_step("after_pairs", len(df2))

    if SKIP_ODDS_BIN or not b_keep:
        df3 = df2
    else:
        if not oddsbin_c:
            df2 = derive_odds_bin_inplace(df2.copy())
            oddsbin_c = get_col(df2, "odds_bin")
        df3 = df2[df2[oddsbin_c].isin(b_keep)] if oddsbin_c else df2
    add_step("after_odds_bin", len(df3))

    if SKIP_PROBA or not proba_c:
        df4 = df3
    else:
        df4 = df3[df3[proba_c] >= thr] if proba_max is None else df3[(df3[proba_c] >= thr) & (df3[proba_c] < proba_max)]
    add_step("after_proba", len(df4))

    # Save summary + survivors
    report = pd.DataFrame(steps)
    report_path = f"strong_filter_counts{suffix}.csv"
    report.to_csv(report_path, index=False)
    print(f"Strong filter waterfall (saved to {report_path}):")
    print(report.to_string(index=False))

    out_path = f"predictions_strong_candidates{suffix}.csv"
    df4.to_csv(out_path, index=False)
    print(f"📝 Saved filtered candidates → {out_path}")

    # ---------- Robust per-group waterfalls ----------
    def waterfall_counts(group_df: pd.DataFrame) -> pd.DataFrame:
        # Re-resolve/ensure inside each group, then filter stepwise
        spc  = get_col(group_df, "sport", "Sport")  or sport_c
        mkc  = get_col(group_df, "market", "Market") or market_c
        obc  = get_col(group_df, "odds_bin")  # may be None
        prc  = ensure_proba_col(group_df)     # may be None

        rows = []
        n0 = len(group_df); rows.append(("start_total", n0))

        g1 = group_df[group_df[spc].isin(s_keep)] if s_keep else group_df
        rows.append(("after_sport", len(g1)))

        g2 = g1[g1[mkc].isin(m_keep)] if m_keep else g1
        rows.append(("after_market", len(g2)))

        if pair_keep:
            g2 = g2[[(sp, mk) in pair_keep for sp, mk in zip(g2[spc].astype(str), g2[mkc].astype(str))]]
            rows.append(("after_pairs", len(g2)))

        # ----- ODDS BIN (derive if missing) -----
        if SKIP_ODDS_BIN or not b_keep:
            g3 = g2
        else:
            # derive odds_bin if the column is missing for this group
            if obc is None or obc not in g2.columns:
                tmp = derive_odds_bin_inplace(g2.copy())
                obc = get_col(tmp, "odds_bin")
                g3 = tmp[tmp[obc].isin(b_keep)] if obc else g2
            else:
                g3 = g2[g2[obc].isin(b_keep)]
        rows.append(("after_odds_bin", len(g3)))

            
        # ----- PROBA (skip if absent or disabled) -----
        if SKIP_PROBA or prc is None or prc not in g3.columns:
            g4 = g3
        else:
            g4 = g3[g3[prc] >= thr] if proba_max is None else g3[(g3[prc] >= thr) & (g3[prc] < proba_max)]
            
       
        rows.append(("after_proba", len(g4)))

        return pd.DataFrame(rows, columns=["step","count"])



    sport_frames = []
    for sp, g in df.groupby(df[sport_c].astype(str)):
        wf = waterfall_counts(g.copy())
        wf.insert(0, "sport", sp)
        sport_frames.append(wf)
    by_sport = pd.concat(sport_frames, ignore_index=True)
    by_sport_path = f"strong_filter_by_sport{suffix}.csv"
    by_sport.to_csv(by_sport_path, index=False)

    market_frames = []
    for mk, g in df.groupby(df[market_c].astype(str)):
        wf = waterfall_counts(g.copy())
        wf.insert(0, "market", mk)
        market_frames.append(wf)
    by_market = pd.concat(market_frames, ignore_index=True)
    by_market_path = f"strong_filter_by_market{suffix}.csv"
    by_market.to_csv(by_market_path, index=False)

    cuts = report["count"].shift(1) - report["count"]
    report["cut"] = cuts.fillna(0).astype(int)
    chokepoint = report.iloc[1:].sort_values("cut", ascending=False).head(1)
    if not chokepoint.empty:
        step = chokepoint.iloc[0]["step"]
        cutn = int(chokepoint.iloc[0]["cut"])
        print(f"Biggest choke overall: {step} (−{cutn} rows)")
        print(f"📂 Saved per-sport → {by_sport_path}, per-market → {by_market_path}")

# ---------- Main ----------
def main():
    # Locate inputs
    sport_path = market_path = None
    for root in CANDIDATE_ROOTS:
        if not os.path.isdir(root):
            continue
        sport_path  = sport_path  or find_segment_file(root, SPORT_BASE)
        market_path = market_path or find_segment_file(root, MARKET_BASE)
    oddsbin_path = find_any(CANDIDATE_ROOTS, ODDSBIN_CSV)
    preds_path   = find_any(CANDIDATE_ROOTS, PREDICTIONS)
    print(f"🔎 predictions_latest.csv: {preds_path or 'NOT FOUND'}")


    sport_rows = read_segments(sport_path)
    market_rows = read_segments(market_path)

    if sport_path:
        print(f"✅ Using sport segments:  {sport_path}")
    else:
        print("⚠️ Sport segments not found; using defaults/fallbacks.")

    if market_path:
        print(f"✅ Using market segments: {market_path}")
    else:
        print("⚠️ Market segments not found; using defaults/fallbacks.")

    # Build REAL (cash) config
    sports_hard = pick_sports(sport_rows)
    if not sports_hard:
        sports_hard = rank_top_k_sports(sport_rows, TOP_K_SPORTS) or DEFAULT_SPORTS.copy()
    else:
        filtered_rows = [r for r in sport_rows if r.get("Sport") in sports_hard]
        sports_hard = rank_top_k_sports(filtered_rows, TOP_K_SPORTS) or sports_hard

    markets = pick_markets(market_rows) or DEFAULT_MARKETS.copy()
    bins    = pick_odds_bins(oddsbin_path)

    cfg_real = {
        "sports": sports_hard,
        "markets": markets,
        "odds_bin": bins,
        "min_proba": round(MIN_PROBA, 2),
    }
    
    # Auto-tune min_proba from latest predictions (if present)
    auto_thr = auto_min_proba_from_predictions(preds_path, min_bets=int(os.getenv("AUTO_MIN_BETS", "40")))
    if auto_thr is not None:
        cfg_real["min_proba"] = round(auto_thr, 2)
        
    else:
        print(f"ℹ️ Auto-tune did not adjust min_proba; using {cfg_real['min_proba']:.2f}. "
              f"(preds_path={preds_path or 'not found'})")

    
    
    
    
    with open("strong_config.json", "w") as f:
        json.dump(cfg_real, f, indent=2)
    print("✅ Wrote strong_config.json:")
    print(json.dumps(cfg_real, indent=2))

    # Build SHADOW (paper)
    sports_shadow = rank_top_k_sports(sport_rows, max(TOP_K_SPORTS, 3)) or sports_hard
    bins_shadow   = list({*bins, "Dog"})  # never Longshot
    cfg_shadow = {
        "sports": sports_shadow,
        "markets": markets,
        "odds_bin": bins_shadow,
        "min_proba": round(SHADOW_MIN_PROBA, 2),
        "max_proba": round(SHADOW_MAX_PROBA, 2),
    }
    with open("shadow_config.json", "w") as f:
        json.dump(cfg_shadow, f, indent=2)
    print("📝 Wrote shadow_config.json:")
    print(json.dumps(cfg_shadow, indent=2))

    # ---------- Build & persist PAIRS ----------
    pairs_real = cartesian_pairs(sports_hard, markets)
    pairs_shadow = cartesian_pairs(sports_shadow, markets)
    pairs_real = filter_pairs_by_predictions(pairs_real, preds_path, PAIR_MIN_N)
    pairs_shadow = filter_pairs_by_predictions(pairs_shadow, preds_path, PAIR_MIN_N)
    write_pairs_json("strong_pairs.json", pairs_real)
    write_pairs_json("shadow_pairs.json", pairs_shadow)
    print(f"🔗 Wrote strong_pairs.json ({len(pairs_real)} pairs) and shadow_pairs.json ({len(pairs_shadow)} pairs)")

    # ---------- Pair-aware waterfalls ----------
    allow_pairs_real   = set(pairs_real)   if ENFORCE_PAIRS_IN_WATERFALL else None
    allow_pairs_shadow = set(pairs_shadow) if ENFORCE_PAIRS_IN_WATERFALL else None
    run_filter_report(cfg_real, preds_path, suffix="", proba_max=None, allow_pairs=allow_pairs_real)
    run_filter_report(cfg_shadow, preds_path, suffix="_shadow", proba_max=SHADOW_MAX_PROBA, allow_pairs=allow_pairs_shadow)

if __name__ == "__main__":
    main()
