"""Brain-muscle divergence per block.

    python -m eeg_fatigue.analyse data/fatigue/P03_session_*.npz
    python -m eeg_fatigue.analyse data/fatigue/P0*_session_*.npz        several people + group line
    python -m eeg_fatigue.analyse FILE --measure mrcp                    motor potential instead of beta drop

Steps
  1. EMG onsets = trigger (each squeeze), matched to squeeze cues
  2. EEG cut around each onset (-2 to +3 s); noisy epochs dropped
  3. Per block (~20 squeezes): EEG effort (C3 beta power drop during the squeeze,
     or motor potential) and EMG click peak, averaged
  4. Ratio = (EEG effort / block 1) / (EMG peak / block 1); 95% bootstrap interval

Verdict: "rises" if the last fatigue block's ratio >= 1.10 and its interval is above 1;
"drops after rest" if the recovery ratio is below the last fatigue ratio and its interval
does not overlap that ratio's point estimate.

Outputs: research/eeg_fatigue_results/<participant>/{blocks.csv, summary.json, figure.png}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import signals as S
from .recording import Recording, load


def per_squeeze(rec: Recording, measure: str):
    """Onsets, EMG peaks, median frequencies, EEG trials and their time grid."""
    env = S.emg_envelope(rec.emg)
    cues = rec.marker_times("cue_squeeze")
    onsets, thr = S.detect_onsets(env, rec.emg_t, cues if len(cues) else None)
    peaks = S.emg_peaks(env, rec.emg_t, onsets)
    mdf = S.emg_median_freq(S.emg_highpass(rec.emg), rec.emg_t, onsets)
    ch = C.MOTOR_CHANNEL[rec.meta.get("hand", C.HAND)]
    ci = rec.eeg_ch.index(ch)
    if measure == "mrcp":
        ep, grid, valid = S.mrcp_epochs(rec.eeg, rec.eeg_t, onsets)
    else:
        band = C.BETA if measure == "beta_erd" else C.MU
        ep, grid, valid = S.power_epochs(rec.eeg, rec.eeg_t, onsets, band)
    clean = valid & S.artifact_mask(rec.eeg, rec.eeg_t, onsets)
    return {"onsets": onsets, "peaks": peaks, "mdf": mdf, "trials": ep[:, :, ci], "grid": grid,
            "clean": clean, "threshold": thr, "channel": ch, "n_cues": len(cues)}


def assign_blocks(rec: Recording, onsets: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    blocks = rec.blocks()
    if not blocks:                                   # no markers: groups of 20, long gap = rest
        idx = np.arange(len(onsets)) // C.SQUEEZES_PER_BLOCK
        blocks = [{"name": f"G{i + 1}", "kind": "fatigue", "start": onsets[idx == i].min(),
                   "end": onsets[idx == i].max()} for i in np.unique(idx)]
        blocks[0]["kind"] = "baseline"
        gaps = np.diff([b["start"] for b in blocks])
        for i, g in enumerate(gaps):
            if g > 60 + C.SQUEEZES_PER_BLOCK * (C.SQUEEZE_S + C.RELAX_S):
                for b in blocks[i + 1:]:
                    b["kind"] = "recovery"
        return idx, blocks
    lab = np.full(len(onsets), -1)
    for bi, b in enumerate(blocks):
        lab[(onsets >= b["start"]) & (onsets <= b["end"] + 1.0)] = bi
    return lab, blocks


def analyse(rec: Recording, measure: str = C.DEFAULT_MEASURE, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> dict:
    sq = per_squeeze(rec, measure)
    lab, blocks = assign_blocks(rec, sq["onsets"])
    grid = sq["grid"]
    rng = np.random.default_rng(seed)

    use = [i for i, b in enumerate(blocks) if b["kind"] in ("baseline", "fatigue", "recovery", "sanity")]
    rows, curves, boots = [], {}, {}
    for bi in use:
        b = blocks[bi]
        m = (lab == bi) & sq["clean"] & np.isfinite(sq["peaks"])
        tr, pk, md = sq["trials"][m], sq["peaks"][m], sq["mdf"][m]
        n_all = int((lab == bi).sum())
        row = {"block": b["name"], "kind": b["kind"], "squeezes": n_all, "clean": int(m.sum()),
               "eeg_effort": np.nan, "emg_peak": np.nan, "emg_mdf_hz": np.nan}
        if m.sum() >= 5:
            row["eeg_effort"] = S.effort_from_trials(tr, grid, measure)
            row["emg_peak"] = float(np.mean(pk))
            row["emg_mdf_hz"] = float(np.nanmean(md))
            if measure != "mrcp":
                curves[b["name"]] = S.erd_curve(tr, grid)
            else:
                curves[b["name"]] = np.nanmean(tr, axis=0)
            bs_e, bs_p = [], []
            for _ in range(n_boot):
                k = rng.integers(0, len(tr), len(tr))
                bs_e.append(S.effort_from_trials(tr[k], grid, measure))
                bs_p.append(pk[k].mean())
            boots[b["name"]] = (np.array(bs_e), np.array(bs_p))
            lo, hi = np.percentile(bs_e, [2.5, 97.5])
            row["eeg_effort_lo"], row["eeg_effort_hi"] = float(lo), float(hi)
        rows.append(row)
    df = pd.DataFrame(rows)

    ref = df.iloc[0] if len(df) else None
    notes = []
    if ref is None or not np.isfinite(ref["eeg_effort"]):
        notes.append("block 1 has too few clean squeezes")
    elif ref["eeg_effort"] <= 0 or ref.get("eeg_effort_lo", -1) <= 0:
        notes.append(f"no clear EEG response in block 1 (effort {ref['eeg_effort']:.1f}, interval includes 0): "
                     "ratio is not meaningful")
    for col in ("effort_norm", "effort_norm_lo", "effort_norm_hi", "emg_norm", "ratio", "ratio_lo", "ratio_hi"):
        df[col] = np.nan
    if ref is not None and np.isfinite(ref["eeg_effort"]) and ref["eeg_effort"] != 0:
        e1b, p1b = boots[ref["block"]]
        for i, r in df.iterrows():
            if r["block"] not in boots:
                continue
            eb, pb = boots[r["block"]]
            df.loc[i, "effort_norm"] = r["eeg_effort"] / ref["eeg_effort"]
            df.loc[i, "emg_norm"] = r["emg_peak"] / ref["emg_peak"]
            df.loc[i, "ratio"] = df.loc[i, "effort_norm"] / df.loc[i, "emg_norm"]
            en = eb / e1b if i else eb / eb.mean()
            rb = en / ((pb / p1b) if i else (pb / pb.mean()))
            df.loc[i, "ratio_lo"], df.loc[i, "ratio_hi"] = np.percentile(rb, [2.5, 97.5])
            df.loc[i, "effort_norm_lo"], df.loc[i, "effort_norm_hi"] = np.percentile(en, [2.5, 97.5])

    verdict = verdict_for(df, notes)
    visible = [r["block"] for _, r in df.iterrows() if r.get("eeg_effort_lo", -1) > 0]
    return {"participant": rec.meta.get("participant", "?"), "measure": measure, "channel": sq["channel"],
            "squeeze_cues": sq["n_cues"], "onsets_found": int(len(sq["onsets"])),
            "clean_epochs": int(sq["clean"].sum()), "blocks": df, "curves": curves, "grid": grid,
            "eeg_response_visible_in": visible, "verdict": verdict, "notes": notes,
            "simulated": bool(rec.meta.get("simulated", False))}


def verdict_for(df: pd.DataFrame, notes: list) -> dict:
    fat = df[df["kind"].isin(["baseline", "fatigue"])].dropna(subset=["ratio"])
    rec = df[df["kind"] == "recovery"].dropna(subset=["ratio"])
    out = {"rises": None, "drops_after_rest": None, "text": ""}
    if notes or len(fat) < 2:
        out["text"] = "inconclusive: " + ("; ".join(notes) or "not enough blocks")
        return out
    last = fat.iloc[-1]
    rho = pd.Series(range(len(fat))).corr(fat["ratio"].reset_index(drop=True), method="spearman")
    out["trend_rho"] = float(rho)
    out["rises"] = bool(last["ratio"] >= C.RISE_MIN and last["ratio_lo"] > 1.0)
    if len(rec):
        r = rec.iloc[0]
        out["drops_after_rest"] = bool(r["ratio"] < last["ratio"] and r["ratio_hi"] < last["ratio"])
    # the ratio also rises if only the muscle weakens, so check brain effort on its own
    out["brain_effort_rises"] = bool(last["effort_norm"] >= C.RISE_MIN and last["effort_norm_lo"] > 1.0)
    rises = "RISES" if out["rises"] else "does not clearly rise"
    drops = {True: "and DROPS after rest", False: "and does not clearly drop after rest",
             None: "(no recovery block)"}[out["drops_after_rest"]]
    if out["rises"] and out["brain_effort_rises"]:
        kind = "brain effort up while muscle output down: central component"
    elif out["rises"]:
        kind = "muscle output fell but brain effort did not clearly rise: mostly muscle fatigue"
    else:
        kind = "no divergence"
    out["interpretation"] = kind
    out["text"] = (f"ratio {rises} with fatigue (last fatigue block {last['ratio']:.2f}, "
                   f"95% {last['ratio_lo']:.2f}-{last['ratio_hi']:.2f}) {drops}"
                   + (f" (recovery {rec.iloc[0]['ratio']:.2f})" if len(rec) else "")
                   + f". Brain effort alone x{last['effort_norm']:.2f} "
                   f"({last['effort_norm_lo']:.2f}-{last['effort_norm_hi']:.2f}) -> {kind}")
    return out


def figure(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df, grid = res["blocks"], res["grid"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    names = list(res["curves"])
    show = [n for n in (names[:1] + [r["block"] for _, r in df[df.kind == "fatigue"].tail(1).iterrows()]
                        + [r["block"] for _, r in df[df.kind == "recovery"].iterrows()]) if n in res["curves"]]
    colors = {"baseline": "#4C72B0", "fatigue": "#C44E52", "recovery": "#55A868", "sanity": "#4C72B0"}
    kinds = dict(zip(df["block"], df["kind"]))
    for n in dict.fromkeys(show):
        ax[0].plot(grid, res["curves"][n], label=f"{n} ({kinds[n]})", color=colors.get(kinds[n], "k"), lw=1.8)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].axvspan(0, C.SQUEEZE_S, color="0.9", zorder=0)
    ax[0].axhline(0, color="0.5", lw=0.6)
    lab = "power change (%)" if res["measure"] != "mrcp" else "µV"
    ax[0].set(title=f"{res['channel']} {res['measure'].replace('_', ' ')} around squeeze onset",
              xlabel="time from EMG onset (s)", ylabel=lab)
    ax[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(df))
    ax[1].plot(x, df["effort_norm"], "o-", color="#8172B2", label="EEG effort")
    ax[1].plot(x, df["emg_norm"], "s-", color="#CCB974", label="EMG peak")
    ax[1].axhline(1, color="0.5", lw=0.6)
    ax[1].set(title="each part, relative to block 1", xticks=x, xticklabels=df["block"])
    ax[1].legend(frameon=False, fontsize=8)

    err = np.vstack([df["ratio"] - df["ratio_lo"], df["ratio_hi"] - df["ratio"]])
    ax[2].errorbar(x, df["ratio"], yerr=np.nan_to_num(err), fmt="o-", color="k", capsize=3)
    ax[2].axhline(1, color="0.5", lw=0.6)
    if (df["kind"] == "recovery").any():
        r0 = int(np.flatnonzero(df["kind"].values == "recovery")[0])
        ax[2].axvspan(r0 - 0.5, r0 - 0.5 + 0.08, color="#55A868", alpha=0.6)
        ax[2].text(r0 - 0.45, ax[2].get_ylim()[1], " 3 min rest", va="top", fontsize=8, color="#55A868")
    ax[2].set(title="brain / muscle ratio (95% bootstrap)", xticks=x, xticklabels=df["block"])
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    sim = "  [SIMULATED]" if res["simulated"] else ""
    fig.suptitle(f"{res['participant']}{sim}: {res['verdict']['text']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_report(res: dict):
    df = res["blocks"]
    print(f"\n=== {res['participant']}{'  [SIMULATED]' if res['simulated'] else ''} | {res['measure']} at {res['channel']} ===")
    print(f"squeeze cues {res['squeeze_cues']}, EMG onsets found {res['onsets_found']}, clean EEG epochs {res['clean_epochs']}")
    cols = ["block", "kind", "squeezes", "clean", "eeg_effort", "emg_peak", "emg_mdf_hz", "effort_norm", "emg_norm",
            "ratio", "ratio_lo", "ratio_hi"]
    print(df[cols].round(2).to_string(index=False))
    print(f"EEG response visible (interval above 0) in: {', '.join(res['eeg_response_visible_in']) or 'NO BLOCK'}")
    print(f"VERDICT: {res['verdict']['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--measure", choices=C.MEASURES, default=C.DEFAULT_MEASURE)
    ap.add_argument("--boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    group = []
    for f in a.files:
        rec = load(f)
        res = analyse(rec, a.measure, a.boot)
        print_report(res)
        out = C.RESULTS_DIR / f"{res['participant']}_{a.measure}"
        out.mkdir(parents=True, exist_ok=True)
        res["blocks"].to_csv(out / "blocks.csv", index=False)
        figure(res, out / "figure.png")
        summ = {k: v for k, v in res.items() if k not in ("blocks", "curves", "grid")}
        summ["file"] = str(f)
        (out / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
        print(f"saved {out}/")
        group.append(res)
    if len(group) > 1:
        real = [g for g in group if not g["simulated"]]
        both = [g["participant"] for g in real if g["verdict"]["rises"] and g["verdict"]["drops_after_rest"]]
        print(f"\nGROUP ({len(real)} real recordings): ratio rises AND drops after rest in {len(both)}: {both}")
        print("Level 1 DONE (>= 2 people)" if len(both) >= 2 else "Level 1 not met yet (needs >= 2 people)")


if __name__ == "__main__":
    main()
