"""
Football Player ROI — Visualization Suite
==========================================
Produces three presentation-ready charts:
  1. Scatter Plot  – Market Value vs G+A per 90 (underpriced assets highlighted)
  2. Correlation Heatmap – Age, Minutes Played, Market Value
  3. Horizontal Bar Chart – Top 10 players by ROI Score

Depends on: football_roi_analysis.py (imported as a module to reuse all
data-generation and calculation logic without duplication).

Run:
    python football_roi_charts.py
Outputs:
    chart_1_scatter.png
    chart_2_heatmap.png
    chart_3_bar.png
    football_roi_dashboard.png   ← all three combined
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import sys, os

# ── Import the analysis module we built in Step 1 ───────────────────────────
sys.path.insert(0, "/mnt/user-data/outputs")
from football_roi_analysis import (
    generate_dataset,
    add_ga_per_90,
    add_roi_score,
    get_underpriced_assets,
)

# ── Global style ─────────────────────────────────────────────────────────────
# A clean, dark theme that reads well in business presentations.
plt.rcParams.update({
    "figure.facecolor"  : "#0f1117",
    "axes.facecolor"    : "#1a1d27",
    "axes.edgecolor"    : "#2e3147",
    "axes.labelcolor"   : "#c8ccd8",
    "axes.titlecolor"   : "#ffffff",
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "xtick.color"       : "#8b8fa8",
    "ytick.color"       : "#8b8fa8",
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "grid.color"        : "#2e3147",
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.6,
    "legend.facecolor"  : "#1a1d27",
    "legend.edgecolor"  : "#2e3147",
    "legend.labelcolor" : "#c8ccd8",
    "text.color"        : "#c8ccd8",
    "font.family"       : "DejaVu Sans",
})

# ── Palette tokens ───────────────────────────────────────────────────────────
C_NORMAL     = "#3d5a8a"   # muted blue  – regular players
C_UNDERPRICED = "#00e5a0"  # neon green  – underpriced assets
C_ACCENT     = "#f4a532"   # amber       – bar chart fill
C_HIGHLIGHT  = "#ff5e7e"   # coral red   – correlation accent
C_GRID_BG    = "#1a1d27"


# =============================================================================
# DATA PIPELINE  (re-uses Step-1 functions)
# =============================================================================

def build_dataframe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rebuild the full dataset and the top-10 underpriced subset.
    Returns (full_df, underpriced_df).
    """
    df = generate_dataset(n=100)
    df = add_ga_per_90(df)
    df = add_roi_score(df)
    top10 = get_underpriced_assets(df, top_n=10, max_value_pct=0.40)

    # Flag column used for colour-splitting in the scatter plot
    df["Is_Underpriced"] = df["Name"].isin(top10["Name"])
    return df, top10


# =============================================================================
# CHART 1 – Scatter: Market Value vs G+A per 90
# =============================================================================

def chart_scatter(ax: plt.Axes, df: pd.DataFrame, top10: pd.DataFrame) -> None:
    """
    Scatter plot that visually separates underpriced assets from the rest.

    Visual logic
    ------------
    * Two layers: regular players (blue, small, semi-transparent) drawn first
      so the underpriced assets (neon green, larger, opaque) sit on top.
    * Annotations for the top-10 names avoid overplotting by using a small
      offset and a subtle arrow.
    * A trend line (1st-degree polyfit) shows the expected positive correlation
      between value and output — underpriced assets sitting *above* it are the
      interesting outliers.
    """
    normal     = df[~df["Is_Underpriced"]]
    underpriced = df[df["Is_Underpriced"]]

    # Regular players
    ax.scatter(
        normal["Market_Value_M"], normal["GA_per_90"],
        color=C_NORMAL, s=55, alpha=0.55, zorder=2, label="Regular Players"
    )
    # Underpriced assets
    ax.scatter(
        underpriced["Market_Value_M"], underpriced["GA_per_90"],
        color=C_UNDERPRICED, s=110, alpha=0.95, zorder=3,
        edgecolors="#ffffff", linewidths=0.6, label="Underpriced Assets"
    )

    # Trend line — polyfit degree 1 over the full dataset
    # This gives context: assets above the line outperform their price tier.
    z   = np.polyfit(df["Market_Value_M"], df["GA_per_90"], 1)
    p   = np.poly1d(z)
    xr  = np.linspace(df["Market_Value_M"].min(), df["Market_Value_M"].max(), 200)
    ax.plot(xr, p(xr), color="#ffffff", linewidth=1.1, linestyle="--",
            alpha=0.35, zorder=1, label="Trend (OLS fit)")

    # Annotate underpriced players
    for _, row in underpriced.iterrows():
        short_name = row["Name"].split()[-1]   # surname only to save space
        ax.annotate(
            short_name,
            xy=(row["Market_Value_M"], row["GA_per_90"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, color=C_UNDERPRICED, alpha=0.9,
        )

    ax.set_title("Market Value vs G+A per 90  —  Underpriced Assets Highlighted")
    ax.set_xlabel("Market Value (€M)")
    ax.set_ylabel("G+A per 90 Minutes")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)


# =============================================================================
# CHART 2 – Correlation Heatmap
# =============================================================================

def chart_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Pearson correlation matrix for Age, Minutes Played, Market Value,
    G+A per 90, and ROI Score.

    Statistical note
    ----------------
    Pearson r measures *linear* association between –1 and +1.
    A high positive r between Minutes and Market Value would confirm that
    clubs pay more for consistently available players.
    A low or negative r between Market Value and ROI_Score exposes
    market inefficiency — expensive players don't always deliver better ROI.
    """
    cols   = ["Age", "Minutes_Played", "Market_Value_M", "GA_per_90", "ROI_Score"]
    labels = ["Age", "Minutes\nPlayed", "Market\nValue (€M)", "G+A\nper 90", "ROI\nScore"]
    corr   = df[cols].corr()

    # Custom diverging palette anchored at 0 for clear positive/negative split
    cmap = sns.diverging_palette(220, 20, s=80, l=45, as_cmap=True)

    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1, vmax=1,
        linewidths=1.5,
        linecolor="#0f1117",
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        square=True,
    )
    ax.set_xticklabels(labels, rotation=0,  fontsize=9)
    ax.set_yticklabels(labels, rotation=0,  fontsize=9)
    ax.set_title("Correlation Matrix  —  Key Player Metrics")

    # Style the colour-bar text
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#c8ccd8")
    cbar.ax.tick_params(colors="#8b8fa8")


# =============================================================================
# CHART 3 – Horizontal Bar: Top 10 by ROI Score
# =============================================================================

def chart_bar(ax: plt.Axes, top10: pd.DataFrame) -> None:
    """
    Horizontal bar chart ranking the top-10 underpriced assets by ROI Score.

    Design choices
    --------------
    * Horizontal orientation fits long player names without rotation.
    * Bars are sorted ascending so the best player appears at the top
      (natural reading order in horizontal charts).
    * Value labels printed at bar ends for quick reading in presentations.
    * Colour gradient (alpha ramp) adds visual hierarchy without distraction.
    """
    sorted_df = top10.sort_values("ROI_Score", ascending=True).reset_index(drop=True)

    n      = len(sorted_df)
    alphas = np.linspace(0.45, 1.0, n)   # lower-ranked bars are more transparent
    colors = [(*plt.matplotlib.colors.to_rgb(C_ACCENT), a) for a in alphas]

    bars = ax.barh(
        sorted_df["Name"],
        sorted_df["ROI_Score"],
        color=colors,
        edgecolor="#0f1117",
        linewidth=0.5,
        height=0.65,
    )

    # Inline value labels
    for bar, val in zip(bars, sorted_df["ROI_Score"]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left",
            fontsize=8.5, color="#f4a532", fontweight="bold",
        )

    # Secondary annotation: market value badge
    for i, row in sorted_df.iterrows():
        ax.text(
            0.002,
            i,
            f"€{row['Market_Value_M']}M",
            va="center", ha="left",
            fontsize=7.5, color="#0f1117", fontweight="bold", alpha=0.85,
        )

    ax.set_title("Top 10 Underpriced Assets  —  Ranked by ROI Score")
    ax.set_xlabel("ROI Score  (G+A per 90 / Market Value €M)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, axis="x")
    ax.set_xlim(0, sorted_df["ROI_Score"].max() * 1.22)


# =============================================================================
# DASHBOARD ASSEMBLY
# =============================================================================

def build_dashboard(df: pd.DataFrame, top10: pd.DataFrame,
                    out_dir: str = "/mnt/user-data/outputs") -> None:
    """
    Arrange all three charts in a single 16×14 figure suitable for
    a slide deck or PDF export.
    """
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0f1117")

    # Layout: scatter (top, full width), heatmap + bar (bottom, side by side)
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)
    ax1 = fig.add_subplot(gs[0, :])   # row 0, both columns
    ax2 = fig.add_subplot(gs[1, 0])   # row 1, left
    ax3 = fig.add_subplot(gs[1, 1])   # row 1, right

    chart_scatter(ax1, df, top10)
    chart_heatmap(ax2, df)
    chart_bar(ax3, top10)

    fig.suptitle(
        "Football Transfer Market  —  ROI & Value Analysis",
        fontsize=18, fontweight="bold", color="#ffffff", y=0.975,
    )

    # ── Save individual charts ───────────────────────────────────────────────
    def _save_single(chart_fn, data_args, filename):
        f, a = plt.subplots(figsize=(10, 6))
        f.patch.set_facecolor("#0f1117")
        chart_fn(a, *data_args)
        f.tight_layout()
        path = os.path.join(out_dir, filename)
        f.savefig(path, dpi=150, bbox_inches="tight", facecolor=f.get_facecolor())
        plt.close(f)
        print(f"  ✓  {filename}")

    _save_single(chart_scatter, (df, top10), "chart_1_scatter.png")
    _save_single(chart_heatmap, (df,),       "chart_2_heatmap.png")
    _save_single(chart_bar,     (top10,),    "chart_3_bar.png")

    # ── Save combined dashboard ──────────────────────────────────────────────
    dash_path = os.path.join(out_dir, "football_roi_dashboard.png")
    fig.savefig(dash_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  football_roi_dashboard.png")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 55)
    print("  Football ROI Visualization Suite")
    print("=" * 55)

    print("\n[1] Building dataset …")
    df, top10 = build_dataframe()

    print("[2] Rendering charts …")
    build_dashboard(df, top10)

    print("\n[✓] All charts saved to /mnt/user-data/outputs/")


if __name__ == "__main__":
    main()
