"""Three presentation charts (fatigue / weakness / tremor) from the 02:26 fatigue session CSVs.
    python -m devtool.slide_charts data/devtool/sessions/<id>_fatigue   (stem, no suffix)"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})

def rows(p):
    return [r for r in csv.DictReader(open(p)) if r.get("band", "right") == "right"]

def frame(ax, title, sub):
    ax.set_title(title, loc="left", fontsize=15, color=INK, fontweight="bold", pad=28)
    ax.text(0, 1.03, sub, transform=ax.transAxes, color=INK2, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

def label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=11)
    ax.plot([x], [y], "o", ms=8, color=color, mec=SURF, mew=2, zorder=5)

stem, out = sys.argv[1], Path("docs/charts")

# 1 fatigue: squeeze RMS as % of the first five squeezes
f = rows(stem + "_features.csv")
t = np.array([float(r["t_s"]) for r in f]) / 60; rms = np.array([float(r["rms"]) for r in f])
pct = 100 * rms / rms[:5].mean(); phase = [r["phase"] for r in f]
fig, ax = plt.subplots(figsize=(9, 4.6))
for name, shade in (("fatigue", None), ("recovery", "#f0efec"), ("post", None)):
    m = np.array([p == name for p in phase])
    if not m.any(): continue
    ax.plot(t[m], pct[m], "o", ms=4, color=BLUE, alpha=0.35, mew=0)
    k = 7; sm = np.convolve(pct[m], np.ones(k) / k, mode="valid")
    ax.plot(t[m][k // 2: k // 2 + len(sm)], sm, color=BLUE, lw=2)
    ax.text(t[m].mean(), 112, {"fatigue": "squeeze every 2.5 s", "recovery": "rest", "post": "squeezing again"}[name],
            ha="center", color=INK2, fontsize=10)
ax.text(7.5, 112, "3 min rest", ha="center", color=INK2, fontsize=10)
ax.set_ylim(0, 120); ax.set_xlabel("minutes"); ax.set_ylabel("squeeze strength (% of start)")
frame(ax, "Fatigue (real recording): the signal halved, the clicks didn't",
      "Decoder caught 168 of 168 cued squeezes, 9 false clicks over the session")
fig.tight_layout(); fig.savefig(out / "1_fatigue.png", dpi=160); plt.close(fig)

# 2 weakness
w = rows(stem + "_weakness.csv")
s = [100 * float(r["strength"]) for r in w]
fig, ax = plt.subplots(figsize=(9, 4.6))
for key, name, col in (("lda", "standard decoder", ORANGE),
                       ("lda adaptive", "+ rescale to user's strength (prototype)", BLUE),
                       ("recalibrated", "simple threshold, recalibrated", AQUA)):
    y = [float(r[f"{key}_caught_pct"]) for r in w]
    ax.plot(s, y, color=col, lw=2)
    ax.plot(s, y, "o", ms=8, color=col, mec=SURF, mew=2)
lab = {"lda": (35, "standard decoder", ORANGE), "lda adaptive": (18, "+ rescale to user's strength", BLUE),
       "recalibrated": (12, "simple threshold, recalibrated", AQUA)}
for key, (x, text, col) in lab.items():
    y = next(float(r[f"{key}_caught_pct"]) for r in w if abs(100 * float(r["strength"]) - x) < 1e-6)
    ax.annotate(text, (x, y), xytext=(8, -14 if key == "lda" else 6), textcoords="offset points", color=INK, fontsize=11)
ax.set_xlim(105, -32); ax.set_ylim(0, 108); ax.set_xticks([100, 80, 60, 40, 20, 0])
ax.set_xlabel("simulated strength (% of Dylan's)"); ax.set_ylabel("squeezes caught (%)")
frame(ax, "Weakness (simulated): rescaling holds up to a quarter strength",
      "Muscle signal shrunk, real sensor noise kept. False clicks stayed under 3 per minute.")
fig.tight_layout(); fig.savefig(out / "2_weakness.png", dpi=160); plt.close(fig)

# 3 tremor
tr = rows(stem + "_tremor.csv")
fig, ax = plt.subplots(figsize=(9, 4.6))
for hold, col, text in ((30, ORANGE, "30 ms hold (current default)"), (200, BLUE, "200 ms hold")):
    rs = [r for r in tr if int(r["hold_ms"]) == hold]
    x = [100 * float(r["tremor"]) for r in rs]; y = [float(r["false_per_min"]) for r in rs]
    ax.plot(x, y, color=col, lw=2); ax.plot(x, y, "o", ms=8, color=col, mec=SURF, mew=2)
    i = int(np.argmax(y)) if hold == 30 else len(y) - 1
    ax.annotate(text, (x[i], y[i]), xytext=(8, 4), textcoords="offset points", color=INK, fontsize=11)
ax.set_xlabel("simulated tremor (% of squeeze size)"); ax.set_ylabel("false clicks per minute")
ax.set_xlim(-3, 95)
frame(ax, "Tremor (simulated): a 200 ms hold stops tremor turning into clicks",
      "Cost: clicks land ~170 ms later. At 75% tremor: 63% caught, 5 false clicks/min.")
fig.tight_layout(); fig.savefig(out / "3_tremor.png", dpi=160); plt.close(fig)
print("ok")

# 4 live rescaling (today's model, live detector code)
import json
j = json.load(open(out / "live_rescale.json"))
fig, ax = plt.subplots(figsize=(9, 4.6))
for key, col, text in (("standard", ORANGE, "live decoder as before"), ("rescale", BLUE, "live decoder + --rescale")):
    y = j[key]["caught"]
    ax.plot(j["strength"], y, color=col, lw=2); ax.plot(j["strength"], y, "o", ms=8, color=col, mec=SURF, mew=2)
    i = 2 if key == "standard" else 4
    ax.annotate(text, (j["strength"][i], y[i]), xytext=(8, -16 if key == "standard" else 8), textcoords="offset points", color=INK, fontsize=11)
ax.set_xlim(105, -12); ax.set_ylim(0, 108); ax.set_xticks([100, 80, 60, 40, 20, 0])
ax.set_xlabel("simulated strength (% of Dylan's)"); ax.set_ylabel("squeezes caught (%)")
frame(ax, "In the live mouse now: clicks still work at 18% strength",
      "Same recording and scoring. False clicks with --rescale: 0.3 to 0.8 per minute.")
fig.tight_layout(); fig.savefig(out / "4_live_rescale.png", dpi=160); plt.close(fig)
