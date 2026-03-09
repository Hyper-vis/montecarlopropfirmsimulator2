"""
regime_validator.py
-------------------
Three-dimensional regime classification:
  Axis 1 — VIX LEVEL    (5 buckets: Calm / Low / Elevated / High / Crisis)
  Axis 2 — VIX DIRECTION (Expanding / Stable / Contracting)
  Axis 3 — ATR RANGE    (Compressed / Normal / Expanded / Extreme)

ATR is calculated from the instrument you actually trade (default ES=F).
It answers: regardless of what VIX says, what did the market ACTUALLY do?

Saves:
  regime_vix_bands.png       — VIX + direction + ATR 3-panel chart
  regime_price_shaded.png    — SPY shaded by level x direction
  regime_duration_hist.png   — streak lengths per level bucket
  regime_heatmap.png         — level x direction day-count heatmap
  regime_atr_heatmap.png     — level x ATR day-count heatmap
  regime_atr_dist.png        — ATR distribution + bucket thresholds
"""

import os
import sys
# Force unbuffered stdout so output isn't silently swallowed in PowerShell pipes
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration — adjust and re-run to test different cuts
# ---------------------------------------------------------------------------

# VIX level buckets: (low_bound, high_bound, name, color)
VIX_LEVELS = [
    (0,    12,   "Calm",     "#27ae60"),   # very low vol, strong trend
    (12,   17,   "Low",      "#2ecc71"),   # normal bull market
    (17,   22,   "Elevated", "#f39c12"),   # uncertainty creeping in
    (22,   28,   "High",     "#e67e22"),   # significant stress
    (28,   999,  "Crisis",   "#e74c3c"),   # crash / extreme event
]

# VIX direction: rolling N-day change in VIX level
DIRECTION_WINDOW   = 5      # days to measure VIX change over
EXPANDING_THRESH   = +1.5   # VIX rising faster than this → Expanding
CONTRACTING_THRESH = -1.5   # VIX falling faster than this → Contracting
# Between those → Stable

# ATR instrument — the contract you actually trade
# ES=F = E-mini S&P 500, YM=F = Dow, NQ=F = Nasdaq
ATR_TICKER  = "ES=F"
ATR_WINDOW  = 14    # standard 14-day ATR

# ATR buckets are derived from the ACTUAL distribution after data loads
# (quartile-based: Q1 / Q2 / Q3 as thresholds → 4 equal-sized buckets)
# You can override by setting ATR_OVERRIDE_THRESHOLDS = [p25, p50, p75]
# e.g. ATR_OVERRIDE_THRESHOLDS = [30, 50, 75]
ATR_OVERRIDE_THRESHOLDS = None   # None = compute from data

START_DATE   = "2018-01-01"
END_DATE     = "2026-01-01"
PRICE_TICKER = "SPY"

# Known events annotated for visual ground-truth verification
EVENTS = {
    "2018-12-24": "Q4 2018\ncorrection",
    "2020-03-16": "COVID\ncrash",
    "2020-11-09": "Vaccine\nrally",
    "2022-01-24": "Rate hike\nselloff",
    "2022-10-13": "CPI pivot\nbottom",
    "2023-03-13": "SVB\ncrisis",
    "2024-08-05": "Yen carry\nunwind",
}

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

print("Fetching VIX, price, and ATR data from Yahoo Finance...")
raw_vix   = yf.download("^VIX",       start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
raw_price = yf.download(PRICE_TICKER, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
raw_atr   = yf.download(ATR_TICKER,   start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)

vix   = raw_vix["Close"].squeeze().dropna()
price = raw_price["Close"].squeeze().dropna()

# True Range = max(H-L, |H-PrevC|, |L-PrevC|)
atr_high  = raw_atr["High"].squeeze()
atr_low   = raw_atr["Low"].squeeze()
atr_close = raw_atr["Close"].squeeze()
tr = pd.concat([
    atr_high - atr_low,
    (atr_high - atr_close.shift(1)).abs(),
    (atr_low  - atr_close.shift(1)).abs(),
], axis=1).max(axis=1)
atr_series = tr.rolling(ATR_WINDOW).mean().dropna()

common = vix.index.intersection(price.index).intersection(atr_series.index)
vix    = vix.loc[common]
price  = price.loc[common]
atr_series = atr_series.loc[common]
print(f"  Loaded {len(vix)} trading days  ({START_DATE} -> {END_DATE})")
print(f"  ATR ticker: {ATR_TICKER}  |  window: {ATR_WINDOW} days")
print(f"  ATR range: {atr_series.min():.1f} – {atr_series.max():.1f} pts  "
      f"(mean {atr_series.mean():.1f} pts)\n")

# ---------------------------------------------------------------------------
# Build ATR buckets from percentiles (or override)
# ---------------------------------------------------------------------------

if ATR_OVERRIDE_THRESHOLDS is not None:
    p25, p50, p75 = ATR_OVERRIDE_THRESHOLDS
else:
    p25 = float(np.percentile(atr_series, 25))
    p50 = float(np.percentile(atr_series, 50))
    p75 = float(np.percentile(atr_series, 75))

ATR_BUCKETS = [
    (0,    p25,  "Compressed",  "#3498db"),   # bottom quartile — tight range
    (p25,  p50,  "Normal",      "#2ecc71"),   # Q1–Q2 — typical day
    (p50,  p75,  "Expanded",    "#f39c12"),   # Q2–Q3 — above-average range
    (p75,  9999, "Extreme",     "#e74c3c"),   # top quartile — high-range / crisis
]
atr_names = [x[2] for x in ATR_BUCKETS]

print(f"  ATR bucket thresholds (quartile-derived):")
print(f"    Compressed  : 0 – {p25:.1f} pts  (bottom 25% of days)")
print(f"    Normal      : {p25:.1f} – {p50:.1f} pts  (25th–50th percentile)")
print(f"    Expanded    : {p50:.1f} – {p75:.1f} pts  (50th–75th percentile)")
print(f"    Extreme     : {p75:.1f}+ pts         (top 25% of days)")
print(f"  Override via ATR_OVERRIDE_THRESHOLDS if you prefer fixed points.\n")

# ---------------------------------------------------------------------------
# Level labeling
# ---------------------------------------------------------------------------

def label_level(v: float) -> str:
    for lo, hi, name, _ in VIX_LEVELS:
        if lo <= v < hi:
            return name
    return VIX_LEVELS[-1][2]

def level_color(name: str) -> str:
    for _, _, n, c in VIX_LEVELS:
        if n == name:
            return c
    return "#aaaaaa"

level_names  = [x[2] for x in VIX_LEVELS]
level_labels = vix.apply(label_level)
level_labels.name = "level"

# ---------------------------------------------------------------------------
# Direction labeling
# ---------------------------------------------------------------------------

vix_change = vix.diff(DIRECTION_WINDOW)

def label_direction(delta) -> str:
    if pd.isna(delta):
        return "Stable"
    if delta > EXPANDING_THRESH:
        return "Expanding"
    elif delta < CONTRACTING_THRESH:
        return "Contracting"
    return "Stable"

dir_labels = vix_change.apply(label_direction)
dir_labels.name = "direction"

dir_order = ["Expanding", "Stable", "Contracting"]

# ---------------------------------------------------------------------------
# ATR labeling
# ---------------------------------------------------------------------------

def label_atr(v: float) -> str:
    for lo, hi, name, _ in ATR_BUCKETS:
        if lo <= v < hi:
            return name
    return ATR_BUCKETS[-1][2]

def atr_color(name: str) -> str:
    for _, _, n, c in ATR_BUCKETS:
        if n == name:
            return c
    return "#aaaaaa"

atr_labels = atr_series.apply(label_atr)
atr_labels.name = "atr_bucket"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str):
    bar = "=" * 62
    print(f"\n{bar}\n  {title}\n{bar}")


def compute_streaks(labels: pd.Series) -> pd.DataFrame:
    records = []
    current = labels.iloc[0]
    start   = labels.index[0]
    count   = 1
    for date, lbl in labels.iloc[1:].items():
        if lbl == current:
            count += 1
        else:
            records.append({"regime": current, "start": start, "days": count})
            current = lbl
            start   = date
            count   = 1
    records.append({"regime": current, "start": start, "days": count})
    return pd.DataFrame(records)


def transition_matrix(labels: pd.Series, ordered: list) -> pd.DataFrame:
    mat = pd.DataFrame(0, index=ordered, columns=ordered)
    for i in range(len(labels) - 1):
        fr = labels.iloc[i]
        to = labels.iloc[i + 1]
        if fr in mat.index and to in mat.columns:
            mat.loc[fr, to] += 1
    row_sums = mat.sum(axis=1)
    return mat.div(row_sums.where(row_sums > 0, 1), axis=0)


def shade_axis(ax, index, labels, color_fn, alpha=0.22):
    prev_date  = index[0]
    prev_label = labels.iloc[0]
    for i in range(1, len(labels)):
        lbl = labels.iloc[i]
        if lbl != prev_label or i == len(labels) - 1:
            ax.axvspan(prev_date, index[i],
                       color=color_fn(prev_label), alpha=alpha, linewidth=0)
            prev_date  = index[i]
            prev_label = lbl


def annotate_events(ax, index, y_series):
    for date_str, label in EVENTS.items():
        ts  = pd.Timestamp(date_str)
        idx = index.searchsorted(ts, side="left")
        if idx >= len(index):
            continue
        closest = index[idx]
        ax.annotate(label, xy=(closest, y_series.loc[closest]),
                    xytext=(0, 18), textcoords="offset points",
                    fontsize=6.5, ha="center",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.7),
                    color="#111111")

# ---------------------------------------------------------------------------
# Print: level distribution
# ---------------------------------------------------------------------------

total = len(level_labels)

print_section("VIX Level Distribution  (5 buckets)")
print(f"  {'Bucket':<12} {'VIX range':<14} {'Days':>6}  {'%':>6}   Bar")
print("  " + "-" * 56)
for lo, hi, name, color in VIX_LEVELS:
    n      = (level_labels == name).sum()
    pct    = n / total * 100
    hi_str = str(hi) if hi < 999 else "inf"
    bar    = "X" * int(pct / 2)
    print(f"  {name:<12} {lo}-{hi_str:<10}  {n:>6}  {pct:>5.1f}%   {bar}")

# ---------------------------------------------------------------------------
# Print: direction distribution
# ---------------------------------------------------------------------------

print_section(f"VIX Direction Distribution  "
              f"({DIRECTION_WINDOW}-day change, thresh +/-{EXPANDING_THRESH})")
for d in dir_order:
    n   = (dir_labels == d).sum()
    pct = n / total * 100
    bar = "X" * int(pct / 2)
    print(f"  {d:<14} {n:>6} days  {pct:>5.1f}%   {bar}")

# ---------------------------------------------------------------------------
# Print: 2D cross-tab
# ---------------------------------------------------------------------------

print_section("2D Regime Breakdown  (% of total days)  rows=level  cols=direction")
col_w  = 14
header = f"  {'':13}" + "".join(f"{d:>{col_w}}" for d in dir_order)
print(header)
print("  " + "-" * (13 + col_w * 3))
for name in level_names:
    row = f"  {name:<13}"
    for d in dir_order:
        mask = (level_labels == name) & (dir_labels == d)
        pct  = mask.sum() / total * 100
        row += f"{pct:>{col_w}.1f}%"
    print(row)

# ---------------------------------------------------------------------------
# Print: ATR distribution
# ---------------------------------------------------------------------------

print_section(f"ATR Range Distribution  ({ATR_TICKER}, {ATR_WINDOW}-day ATR, in points)")
print(f"  {'Bucket':<14} {'Range (pts)':<16} {'Days':>6}  {'%':>6}   Bar")
print("  " + "-" * 58)
for lo, hi, name, color in ATR_BUCKETS:
    n      = (atr_labels == name).sum()
    pct    = n / total * 100
    hi_str = str(hi) if hi < 9999 else "inf"
    bar    = "X" * int(pct / 2)
    print(f"  {name:<14} {lo}-{hi_str:<13}  {n:>6}  {pct:>5.1f}%   {bar}")

# ---------------------------------------------------------------------------
# Print: level x ATR cross-tab
# ---------------------------------------------------------------------------

print_section("Level x ATR Heatmap  (% of days)  rows=VIX level  cols=ATR bucket")
col_w = 14
header = f"  {'':13}" + "".join(f"{d:>{col_w}}" for d in atr_names)
print(header)
print("  " + "-" * (13 + col_w * len(atr_names)))
for lv in level_names:
    row = f"  {lv:<13}"
    for ar in atr_names:
        mask = (level_labels == lv) & (atr_labels == ar)
        pct  = mask.sum() / total * 100
        row += f"{pct:>{col_w}.1f}%"
    print(row)
print()
print("  Key insight: these combinations tell you the ACTUAL trading environment.")
print("  e.g. Elevated VIX + Expanded ATR vs Elevated VIX + Compressed ATR")
print("  are completely different days even though VIX looks the same.")

# ---------------------------------------------------------------------------
# Print: streak stats
# ---------------------------------------------------------------------------

streaks = compute_streaks(level_labels)
print_section("Level Streak Statistics  (consecutive days in same bucket)")
print(f"  {'Bucket':<14} {'Mean':>8} {'Median':>8} {'Max':>6} {'Streaks':>9}")
print("  " + "-" * 50)
for name in level_names:
    sub = streaks[streaks["regime"] == name]["days"]
    if len(sub) == 0:
        continue
    print(f"  {name:<14} {sub.mean():>7.1f}d {sub.median():>7.0f}d "
          f"{sub.max():>5.0f}d {len(sub):>9}")

# ---------------------------------------------------------------------------
# Print: transition matrix
# ---------------------------------------------------------------------------

tm = transition_matrix(level_labels, level_names)
print_section("Level Transition Matrix  (prob of moving to next bucket tomorrow)")
col_w  = 13
header = f"  {'':14}" + "".join(f"{c:>{col_w}}" for c in level_names)
print(header)
print("  " + "-" * (14 + col_w * len(level_names)))
for rn in level_names:
    line = f"  {rn:<14}"
    for cn in level_names:
        val  = tm.loc[rn, cn] if rn in tm.index else 0.0
        line += f"{val:>{col_w}.3f}"
    print(line)
print("\n  High diagonal = regime is sticky.  Low diagonal = regime flips fast.")

# ---------------------------------------------------------------------------
# Plot 1 — VIX + level shading + direction panel
# ---------------------------------------------------------------------------

fig1, (ax1a, ax1b, ax1c) = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1, 1]})

# Panel 1 — VIX level
shade_axis(ax1a, vix.index, level_labels, level_color, alpha=0.22)
ax1a.plot(vix.index, vix.values, color="black", linewidth=0.9, label="VIX")
for lo, hi, name, color in VIX_LEVELS:
    if lo > 0:
        ax1a.axhline(lo, color=color, linestyle="--", linewidth=0.9, alpha=0.8,
                     label=f"{name} floor ({lo})")
annotate_events(ax1a, vix.index, vix)
patches = [mpatches.Patch(color=level_color(n), alpha=0.55, label=n)
           for n in level_names]
h, _ = ax1a.get_legend_handles_labels()
ax1a.legend(handles=h[:1] + patches, fontsize=8, loc="upper right", ncol=3)
ax1a.set_title("VIX Level  (5-bucket)", fontsize=11)
ax1a.set_ylabel("VIX", fontsize=10)
ax1a.grid(alpha=0.2)

# Panel 2 — VIX direction
dir_num = dir_labels.map({"Expanding": 1, "Stable": 0, "Contracting": -1})
ax1b.fill_between(vix.index, dir_num, 0,
                  where=(dir_num > 0), color="#e74c3c", alpha=0.50, label="Expanding")
ax1b.fill_between(vix.index, dir_num, 0,
                  where=(dir_num < 0), color="#2ecc71", alpha=0.50, label="Contracting")
ax1b.axhline(0, color="grey", linewidth=0.8)
ax1b.set_yticks([-1, 0, 1])
ax1b.set_yticklabels(["Contract", "Stable", "Expand"], fontsize=8)
ax1b.set_title(
    f"VIX Direction  ({DIRECTION_WINDOW}-day change, thresh +/-{EXPANDING_THRESH})",
    fontsize=10)
ax1b.legend(fontsize=8, loc="upper right")
ax1b.grid(alpha=0.2)

# Panel 3 — ATR
ax1c.plot(atr_series.index, atr_series.values, color="#2980b9",
          linewidth=0.9, label=f"{ATR_TICKER} ATR-{ATR_WINDOW}")
for lo, hi, name, color in ATR_BUCKETS:
    if lo > 0:
        ax1c.axhline(lo, color=color, linestyle="--", linewidth=0.9, alpha=0.8,
                     label=f"{name} floor ({lo} pts)")
# Shade ATR panels
shade_axis(ax1c, atr_series.index, atr_labels, atr_color, alpha=0.18)
atr_patches = [mpatches.Patch(color=atr_color(n), alpha=0.55, label=n)
               for n in atr_names]
h2, _ = ax1c.get_legend_handles_labels()
ax1c.legend(handles=h2[:1] + atr_patches, fontsize=7, loc="upper right", ncol=4)
ax1c.set_title(f"{ATR_TICKER} ATR-{ATR_WINDOW}  (instrument range in points)",
               fontsize=10)
ax1c.set_ylabel("ATR (pts)", fontsize=9)
ax1c.grid(alpha=0.2)

plt.suptitle("Three-Dimensional Regime Signal: VIX Level  |  VIX Direction  |  ATR Range",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("regime_vix_bands.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\nSaved: regime_vix_bands.png")

# ---------------------------------------------------------------------------
# Plot 2 — SPY shaded by level (color) x direction (opacity)
# ---------------------------------------------------------------------------

COMBINED_ALPHA = {"Expanding": 0.38, "Stable": 0.20, "Contracting": 0.10}

fig2, ax2 = plt.subplots(figsize=(15, 5))
prev_date = price.index[0]
prev_lv   = level_labels.iloc[0]
prev_dr   = dir_labels.iloc[0]
for i in range(1, len(price)):
    lv = level_labels.iloc[i]
    dr = dir_labels.iloc[i]
    if lv != prev_lv or dr != prev_dr or i == len(price) - 1:
        ax2.axvspan(prev_date, price.index[i],
                    color=level_color(prev_lv),
                    alpha=COMBINED_ALPHA.get(prev_dr, 0.20),
                    linewidth=0)
        prev_date = price.index[i]
        prev_lv   = lv
        prev_dr   = dr

ax2.plot(price.index, price.values, color="black", linewidth=0.9)
annotate_events(ax2, price.index, price)

level_patches = [mpatches.Patch(color=level_color(n), alpha=0.55, label=n)
                 for n in level_names]
dir_patches = [
    mpatches.Patch(facecolor="grey", alpha=0.38, label="+ Expanding (darker)"),
    mpatches.Patch(facecolor="grey", alpha=0.20, label="+ Stable"),
    mpatches.Patch(facecolor="grey", alpha=0.10, label="+ Contracting (lighter)"),
]
ax2.legend(handles=level_patches + dir_patches, fontsize=8,
           loc="upper left", ncol=4)
ax2.set_title(
    f"{PRICE_TICKER} — Color = VIX Level  |  Opacity = VIX Direction  "
    f"(darker = vol rising into that level)",
    fontsize=12)
ax2.set_ylabel(f"{PRICE_TICKER} ($)", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax2.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("regime_price_shaded.png", dpi=150)
plt.close(fig2)
print("Saved: regime_price_shaded.png")

# ---------------------------------------------------------------------------
# Plot 3 — Streak duration histograms per level bucket
# ---------------------------------------------------------------------------

fig3, axes3 = plt.subplots(1, len(VIX_LEVELS), figsize=(16, 4))
for idx, (lo, hi, name, color) in enumerate(VIX_LEVELS):
    ax  = axes3[idx]
    sub = streaks[streaks["regime"] == name]["days"]
    if len(sub) == 0:
        ax.set_visible(False)
        continue
    ax.hist(sub, bins=max(8, len(sub) // 3), color=color,
            edgecolor="white", linewidth=0.5)
    ax.axvline(sub.mean(),   color="black", linestyle="--", linewidth=1.1,
               label=f"Mean {sub.mean():.1f}d")
    ax.axvline(sub.median(), color="grey",  linestyle=":",  linewidth=1.1,
               label=f"Med {sub.median():.0f}d")
    hi_str = str(hi) if hi < 999 else "inf"
    ax.set_title(f"{name}\nVIX {lo}-{hi_str}", fontsize=10)
    ax.set_xlabel("Consecutive Days", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

plt.suptitle("How Long Does Each VIX Level Regime Typically Last?",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("regime_duration_hist.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved: regime_duration_hist.png")

# ---------------------------------------------------------------------------
# Plot 4 — 2D heatmap: level x direction
# ---------------------------------------------------------------------------

heat_data = np.zeros((len(level_names), len(dir_order)))
for i, lv in enumerate(level_names):
    for j, dr in enumerate(dir_order):
        heat_data[i, j] = ((level_labels == lv) & (dir_labels == dr)).sum()
heat_pct = heat_data / total * 100

fig4, ax4 = plt.subplots(figsize=(9, 5))
im = ax4.imshow(heat_pct, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax4, label="% of total trading days")
ax4.set_xticks(range(len(dir_order)))
ax4.set_yticks(range(len(level_names)))
ax4.set_xticklabels(dir_order, fontsize=10)
ax4.set_yticklabels(level_names, fontsize=10)
for i in range(len(level_names)):
    for j in range(len(dir_order)):
        txt_color = "white" if heat_pct[i, j] > 14 else "black"
        ax4.text(j, i,
                 f"{heat_pct[i,j]:.1f}%\n({int(heat_data[i,j])}d)",
                 ha="center", va="center", fontsize=9, color=txt_color)
ax4.set_xlabel("VIX Direction", fontsize=11)
ax4.set_ylabel("VIX Level", fontsize=11)
ax4.set_title(
    "2D Regime Heatmap: Level x Direction\n"
    "Each cell = % of trading days in that combined state",
    fontsize=12)
plt.tight_layout()
plt.savefig("regime_heatmap.png", dpi=150)
plt.close(fig4)
print("Saved: regime_heatmap.png")

# ---------------------------------------------------------------------------
# Plot 5 — Level x ATR heatmap
# ---------------------------------------------------------------------------

heat_atr = np.zeros((len(level_names), len(atr_names)))
for i, lv in enumerate(level_names):
    for j, ar in enumerate(atr_names):
        heat_atr[i, j] = ((level_labels == lv) & (atr_labels == ar)).sum()
heat_atr_pct = heat_atr / total * 100

fig5, ax5 = plt.subplots(figsize=(10, 5))
im5 = ax5.imshow(heat_atr_pct, cmap="YlOrRd", aspect="auto")
plt.colorbar(im5, ax=ax5, label="% of total trading days")
ax5.set_xticks(range(len(atr_names)))
ax5.set_yticks(range(len(level_names)))
ax5.set_xticklabels(atr_names, fontsize=10)
ax5.set_yticklabels(level_names, fontsize=10)
for i in range(len(level_names)):
    for j in range(len(atr_names)):
        txt_color = "white" if heat_atr_pct[i, j] > 14 else "black"
        ax5.text(j, i,
                 f"{heat_atr_pct[i,j]:.1f}%\n({int(heat_atr[i,j])}d)",
                 ha="center", va="center", fontsize=9, color=txt_color)
ax5.set_xlabel(f"ATR Bucket  ({ATR_TICKER}, pts)", fontsize=11)
ax5.set_ylabel("VIX Level", fontsize=11)
ax5.set_title(
    f"VIX Level x ATR Range Heatmap\n"
    f"Same VIX level, very different actual market behaviour",
    fontsize=12)
plt.tight_layout()
plt.savefig("regime_atr_heatmap.png", dpi=150)
plt.close(fig5)
print("Saved: regime_atr_heatmap.png")

# ---------------------------------------------------------------------------
# Plot 6 — ATR distribution with bucket thresholds
# ---------------------------------------------------------------------------

fig6, ax6 = plt.subplots(figsize=(10, 4))
ax6.hist(atr_series.values, bins=60, color="#2980b9",
         edgecolor="white", linewidth=0.4, label=f"{ATR_TICKER} ATR-{ATR_WINDOW}")
for lo, hi, name, color in ATR_BUCKETS:
    if lo > 0:
        ax6.axvline(lo, color=color, linestyle="--", linewidth=1.4,
                    label=f"{name} ({lo} pts)")
ax6.axvline(atr_series.mean(),   color="black", linestyle="-",
            linewidth=1.2, label=f"Mean {atr_series.mean():.1f} pts")
ax6.axvline(atr_series.median(), color="grey",  linestyle=":",
            linewidth=1.2, label=f"Median {atr_series.median():.1f} pts")
ax6.set_title(f"{ATR_TICKER} ATR-{ATR_WINDOW} Distribution with Bucket Thresholds",
              fontsize=12)
ax6.set_xlabel("ATR (points)", fontsize=10)
ax6.set_ylabel("Frequency", fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("regime_atr_dist.png", dpi=150)
plt.close(fig6)
print("Saved: regime_atr_dist.png")

print("\nDone. Review all 6 charts.")
print("To adjust thresholds: change VIX_LEVELS, ATR_BUCKETS,")
print("DIRECTION_WINDOW, or EXPANDING_THRESH at the top of this file.")
