"""
Football Player ROI Analysis
==============================
Generates a synthetic dataset of 100 football players and calculates
ROI scores to identify underpriced assets using G+A per 90 normalization.
"""

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)


# =============================================================================
# 1. DATASET GENERATION
# =============================================================================

def generate_player_names(n: int) -> list[str]:
    """
    Build plausible-looking synthetic player names by randomly combining
    first names and surnames drawn from a small pool.
    """
    first_names = [
        "Luca", "Marco", "Carlos", "João", "Ahmed", "Karim", "Diego",
        "Nils", "Takumi", "Sven", "Mateo", "Yusuf", "Ali", "James",
        "Pierre", "Andres", "Ivan", "Omar", "Finn", "Ryo",
    ]
    last_names = [
        "Silva", "Müller", "García", "Rossi", "Nkosi", "Diallo", "Ferreira",
        "Santos", "Lindqvist", "Okafor", "Martínez", "Balogun", "Eriksen",
        "Nakamura", "Bogdanov", "Kowalski", "Tremblay", "Hussain", "Park", "Alves",
    ]
    # Sample with replacement so we can always produce n names
    firsts = rng.choice(first_names, size=n, replace=True)
    lasts  = rng.choice(last_names,  size=n, replace=True)
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]


def generate_dataset(n: int = 100) -> pd.DataFrame:
    """
    Simulate a realistic football player dataset.

    Statistical rationale
    ---------------------
    * Age          – Uniform[18, 35]: covers youth to veteran range.
    * Minutes      – Uniform[500, 3420]: 500 min is a squad-rotation floor;
                     3420 is a full 38-game season × 90 min.
    * Goals/Assists – Poisson(λ) where λ scales with minutes so that players
                      who play more have proportionally more opportunities.
                      Poisson is the canonical distribution for count data
                      bounded below at 0.
    * Market Value  – Log-Normal(μ=2.3, σ=0.8) in millions.
                      Log-Normal is standard for financial / salary data because
                      it naturally skews right (a few superstars command huge fees)
                      while preventing negative values.
    """
    ages    = rng.integers(18, 36, size=n)                  # Uniform discrete
    minutes = rng.uniform(500, 3420, size=n)                 # Continuous uniform

    # λ for goals/assists grows with minutes to reflect opportunity
    goal_lambda   = (minutes / 90) * 0.25   # avg ~0.25 goals per 90
    assist_lambda = (minutes / 90) * 0.20   # avg ~0.20 assists per 90

    goals   = rng.poisson(lam=goal_lambda)
    assists = rng.poisson(lam=assist_lambda)

    # Log-Normal market value: exp(μ + σ·Z), Z ~ N(0,1)
    market_value = np.exp(rng.normal(loc=2.3, scale=0.8, size=n)).round(2)

    df = pd.DataFrame({
        "Name"          : generate_player_names(n),
        "Age"           : ages,
        "Goals"         : goals,
        "Assists"        : assists,
        "Minutes_Played": minutes.round(0).astype(int),
        "Market_Value_M": market_value,       # in millions (€)
    })
    return df


# =============================================================================
# 2. PERFORMANCE NORMALISATION – G+A per 90
# =============================================================================

def add_ga_per_90(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise goal contributions to a per-90-minute rate.

    Formula:
        G+A per 90 = (Goals + Assists) / Minutes_Played × 90

    Why normalise?
    --------------
    Raw counts are confounded by playing time.  A player with 10 G+A in
    500 minutes is far more productive than one with 12 G+A in 3 000 minutes.
    Dividing by 90-min blocks converts counts to a rate, making players with
    very different workloads directly comparable – the same logic behind
    'per-game' stats in basketball or ERA in baseball.

    A tiny epsilon guard prevents division-by-zero for hypothetical 0-minute
    entries (none expected here, but good defensive practice).
    """
    epsilon = 1e-6
    df = df.copy()
    df["GA_per_90"] = (
        (df["Goals"] + df["Assists"]) / (df["Minutes_Played"] + epsilon) * 90
    ).round(4)
    return df


# =============================================================================
# 3. ROI SCORE
# =============================================================================

def add_roi_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple transfer-market ROI score.

    Formula:
        ROI_Score = G+A per 90 / Market_Value_M

    Interpretation
    --------------
    Higher score → more attacking output per million euros spent.
    This is a first-order efficiency metric: we are asking
    "how many goal contributions per 90 min does each €1M buy?"

    Limitations (acknowledged for academic completeness)
    ----------------------------------------------------
    * Ignores defensive contributions, dribbles, progressive passes, etc.
    * Market value may reflect factors beyond on-pitch output (age, potential,
      commercial appeal).  A more robust model would regress value on multiple
      features and compute residuals as the "mispricing" signal.
    * Nevertheless, G+A per 90 / Value is a clean, interpretable proxy that
      is widely used in introductory sports analytics.
    """
    epsilon = 1e-6
    df = df.copy()
    df["ROI_Score"] = (df["GA_per_90"] / (df["Market_Value_M"] + epsilon)).round(6)
    return df


# =============================================================================
# 4. IDENTIFY TOP-10 UNDERPRICED ASSETS
# =============================================================================

def get_underpriced_assets(df: pd.DataFrame,
                            top_n: int = 10,
                            max_value_pct: float = 0.40) -> pd.DataFrame:
    """
    Filter for players who are both high-ROI *and* low market value.

    Strategy
    --------
    1. ROI_Score percentile cut  – keep players above the median ROI (top 50%).
       This ensures we're selecting genuinely productive players, not just cheap
       ones who hardly contribute.

    2. Market_Value percentile cut – keep players below the `max_value_pct`
       percentile of market value (default: bottom 40 %).
       This enforces the "affordable" constraint: a stellar ROI on a €100 M
       player is useless for a mid-table club.

    3. Sort by ROI_Score descending and return the top `top_n` rows.

    The combined filter operationalises the classic value-investing idea:
    "cheap relative to intrinsic worth" – here proxied by performance output.
    """
    roi_threshold   = df["ROI_Score"].quantile(0.50)          # above median ROI
    value_threshold = df["Market_Value_M"].quantile(max_value_pct)  # low value

    mask = (df["ROI_Score"] >= roi_threshold) & (df["Market_Value_M"] <= value_threshold)

    underpriced = (
        df[mask]
        .sort_values("ROI_Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    underpriced.index += 1   # 1-based ranking for readability
    return underpriced


# =============================================================================
# 5. SUMMARY STATISTICS HELPER
# =============================================================================

def print_summary(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics for the key derived metrics.

    Mean, std, min/max give a quick sanity-check that the synthetic
    distributions look plausible (e.g. no negative values, reasonable ranges).
    """
    cols = ["Goals", "Assists", "Minutes_Played", "Market_Value_M",
            "GA_per_90", "ROI_Score"]
    print("\n── Full-Dataset Descriptive Statistics ─────────────────────────────")
    print(df[cols].describe().round(4).to_string())


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    print("=" * 65)
    print("  Football Player ROI Analysis  |  Synthetic Dataset  ")
    print("=" * 65)

    # Step 1 – Generate
    players = generate_dataset(n=100)
    print(f"\n[1] Dataset generated: {len(players)} players")

    # Step 2 – Normalise performance
    players = add_ga_per_90(players)
    print("[2] G+A per 90 calculated")

    # Step 3 – ROI score
    players = add_roi_score(players)
    print("[3] ROI Score calculated")

    # Step 4 – Underpriced assets
    top10 = get_underpriced_assets(players, top_n=10, max_value_pct=0.40)
    print("\n── Top 10 Underpriced Assets ────────────────────────────────────────")
    display_cols = ["Name", "Age", "Goals", "Assists",
                    "Minutes_Played", "Market_Value_M", "GA_per_90", "ROI_Score"]
    print(top10[display_cols].to_string())

    # Step 5 – Summary stats
    print_summary(players)

    # Persist results
    out_path = "football_roi_results.csv"
    players.to_csv(out_path, index=False)
    print(f"\n[✓] Full dataset saved to '{out_path}'")


if __name__ == "__main__":
    main()
