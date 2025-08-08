# app.py
# Streamlit scaffold for an NBA shot-chart + xFG% app
# ---------------------------------------------------
# Runs with: streamlit run app.py
#
# This scaffold:
# - Loads shots.csv if present, else generates a minimal fake dataset so it runs immediately
# - Normalizes coordinates to a single half-court (simple reflect/origin assumptions)
# - Engineers basic features (distance, angle, three/corner flags, shot-clock bins)
# - Shows: shot frequency plot, xFG% surface (placeholder), summary metrics, best/worst zones table
#
# IMPORTANT TODOs (search for "# TODO:"):
# - Replace the placeholder xFG% estimator with your trained model (LogReg/GBDT/etc.)
# - Adjust coordinate normalization to match your data’s origin/units
# - Swap fake data with your real shots.csv schema if needed

import os
import io
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use headless backend for Streamlit Cloud
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="NBA Shot Quality (xFG%) Explorer",
    layout="wide",
    page_icon="🏀",
)

# ----------------------------
# Constants (court geometry)
# ----------------------------
COURT_WIDTH_FT = 50.0
HALF_COURT_LENGTH_FT = 47.0
RIM_RADIUS_FT = 0.75
FT_LINE_Y = 19.0
FT_RADIUS_FT = 6.0
THREE_ARC_RADIUS_FT = 23.75  # NBA arc radius (23'9")
CORNER_LINE_DIST_FT = 22.0   # NBA corners (22')
CORNER_LINE_LENGTH_FT = 14.0 # straight segment up to 14 ft
HALF_WIDTH_X = 25.0

# ----------------------------
# Helper: draw simple half-court
# ----------------------------
def draw_half_court(ax):
    # Rim (circle)
    rim = plt.Circle((0, 0), RIM_RADIUS_FT, fill=False)
    ax.add_artist(rim)
    # Backboard (approx)
    ax.plot([-3, 3], [-0.75, -0.75], linewidth=1)
    # Paint
    ax.plot([-8, -8], [0, 19], linewidth=1)
    ax.plot([8, 8], [0, 19], linewidth=1)
    ax.plot([-8, 8], [19, 19], linewidth=1)
    # Free throw circle
    ft_circle = plt.Circle((0, FT_LINE_Y), FT_RADIUS_FT, fill=False)
    ax.add_artist(ft_circle)
    # Three-point arc
    theta = np.linspace(-np.radians(68), np.radians(68), 200)
    arc_x = THREE_ARC_RADIUS_FT * np.sin(theta)
    arc_y = THREE_ARC_RADIUS_FT * np.cos(theta)
    ax.plot(arc_x, arc_y, linewidth=1)
    # Corners
    ax.plot([22, 22], [0, 14], linewidth=1)
    ax.plot([-22, -22], [0, 14], linewidth=1)
    # Outer boundaries
    ax.plot([-HALF_WIDTH_X, HALF_WIDTH_X], [0, 0], linewidth=1)
    ax.plot([-HALF_WIDTH_X, -HALF_WIDTH_X], [0, HALF_COURT_LENGTH_FT], linewidth=1)
    ax.plot([HALF_WIDTH_X, HALF_WIDTH_X], [0, HALF_COURT_LENGTH_FT], linewidth=1)
    ax.set_xlim(-HALF_WIDTH_X, HALF_WIDTH_X)
    ax.set_ylim(-1, HALF_COURT_LENGTH_FT + 1)
    ax.set_aspect("equal")
    ax.axis("off")

# ----------------------------
# Data Loading & Generation
# ----------------------------
def generate_fake_data(n=2000, players=("Player A", "Player B", "Player C"), seasons=("2023-24", "2024-25")):
    """Generate a small synthetic dataset that "looks" like NBA shots, in feet."""
    rng = np.random.default_rng(42)

    # Random clusters: rim, corners, above the break
    x1 = rng.normal(0, 4, size=n // 2)                                # paint cluster
    y1 = np.abs(rng.gamma(2.0, 4.0, size=n // 2))                     # 0..~30
    x2 = rng.choice([-22, 22], size=n // 3) + rng.normal(0, 1.0, size=n // 3)  # corners
    y2 = rng.uniform(0, 14, size=n // 3)
    x3 = rng.normal(0, 8, size=n - (n // 2 + n // 3))                 # above the break
    y3 = rng.uniform(14, 25, size=n - (n // 2 + n // 3))

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    y = np.clip(y, 0, HALF_COURT_LENGTH_FT)

    # Simple probability model: distance decay + slight bonus for corners
    dist = np.sqrt(x**2 + y**2)
    base_prob = 0.72 * np.exp(-(dist - 2.5) / 14.0)
    corner_bonus = (np.abs(x) >= 21.0) & (y <= 14.0)
    p = np.clip(base_prob + corner_bonus * 0.05, 0.05, 0.85)
    made = (rng.random(size=n) < p).astype(int)

    df = pd.DataFrame({
        "player": rng.choice(players, size=n, replace=True),
        "season": rng.choice(seasons, size=n, replace=True),
        "x": x,                 # feet; hoop at (0,0); baseline at y=0
        "y": y,
        "made": made,
        "period": rng.integers(1, 5, size=n),
        "shot_clock": rng.uniform(0, 24, size=n)
    })

    # Add optional columns with fallbacks
    df["is_three"] = ((dist >= THREE_ARC_RADIUS_FT) | ((np.abs(x) >= 22) & (y <= 14))).astype(int)
    df["is_corner"] = ((np.abs(x) >= 22) & (y <= 14)).astype(int)

    return df

def load_data():
    """
    Load shots.csv if present; else generate a fake dataset.
    Expected columns (fallbacks applied if missing):
    player, season, x, y (feet), is_three, is_corner, made (0/1), period, shot_clock
    """
    csv_path = Path("shots.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            st.warning(f"Failed to read shots.csv ({e}). Generating fake data instead.")
            df = generate_fake_data()
    else:
        df = generate_fake_data()

    # Normalize column names to lower-case
    df.columns = [c.strip().lower() for c in df.columns]

    # Provide fallbacks if required columns are missing
    for col, default in [
        ("player", "Unknown"),
        ("season", "Unknown"),
        ("x", np.nan),
        ("y", np.nan),
        ("made", np.nan),
        ("period", np.nan),
        ("shot_clock", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default

    # Optional columns
    if "is_three" not in df.columns:
        df["is_three"] = 0
    if "is_corner" not in df.columns:
        df["is_corner"] = 0

    return df

# ----------------------------
# Normalization & Features
# ----------------------------
def normalize_coords_to_half_court(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure single half-court and consistent origin.
    Assumptions:
      - Coordinates are in FEET.
      - Basket (hoop) at (0, 0).
      - Baseline at y=0 and we keep only y >= 0 (reflect if needed).
    # TODO: Adjust this normalization if your data uses a different origin or units.
    """
    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Reflect any negative y to attacking half
    df["y"] = df["y"].abs()

    # Clip to court bounds for plotting robustness
    df["x"] = df["x"].clip(-HALF_WIDTH_X, HALF_WIDTH_X)
    df["y"] = df["y"].clip(0, HALF_COURT_LENGTH_FT)

    return df

def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds distance/angle and fills is_three/is_corner if missing.
    Also bins shot_clock.
    """
    df = df.copy()
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    df["shot_distance_ft"] = np.sqrt(x**2 + y**2)
    df["shot_angle_deg"] = np.degrees(np.arctan2(x, y + 1e-9))

    # Fill three/corner logic if not present or constant
    if ("is_three" not in df.columns) or (df["is_three"].nunique() <= 1):
        is_corner_region = (np.abs(x) >= 22) & (y <= 14)
        is_three_arc = (df["shot_distance_ft"] >= THREE_ARC_RADIUS_FT)
        df["is_three"] = ((is_three_arc | is_corner_region) | ((np.abs(x) >= 22) & (y <= 14))).astype(int)
    if ("is_corner" not in df.columns) or (df["is_corner"].nunique() <= 1):
        df["is_corner"] = ((np.abs(x) >= 22) & (y <= 14)).astype(int)

    # Shot clock bins
    sc = pd.to_numeric(df.get("shot_clock", np.nan), errors="coerce")
    bins = [-1, 4, 7, 14, 24]
    labels = ["0-4", "4-7", "7-14", "14-24"]
    df["shot_clock_bin"] = pd.cut(sc, bins=bins, labels=labels)

    # Placeholder xFG% estimator per-shot (distance-based decay + corner bonus)
    # TODO: Replace this with your trained model's predicted probabilities
    base = 0.72 * np.exp(-(df["shot_distance_ft"] - 2.5) / 14.0)
    corner_bonus = df["is_corner"].astype(int) * 0.05
    df["xfg_placeholder"] = np.clip(base + corner_bonus, 0.05, 0.90)

    return df

# ----------------------------
# Summary & Zone Table
# ----------------------------
def compute_summary(df: pd.DataFrame) -> dict:
    """
    Returns attempts, actual FG%, average xFG% (placeholder), and shot-making (Actual − xFG).
    """
    df = df.copy()
    made = pd.to_numeric(df.get("made", np.nan), errors="coerce")
    xfg = pd.to_numeric(df.get("xfg_placeholder", np.nan), errors="coerce")
    attempts = int(made.notna().sum())
    actual_fg = float(made.mean()) if attempts > 0 else float("nan")
    avg_xfg = float(xfg.mean()) if xfg.notna().sum() > 0 else float("nan")
    shot_making = actual_fg - avg_xfg if (not math.isnan(actual_fg) and not math.isnan(avg_xfg)) else float("nan")
    return {
        "attempts": attempts,
        "actual_fg": actual_fg,
        "avg_xfg": avg_xfg,
        "shot_making": shot_making
    }

def zone_table(df: pd.DataFrame, xbins=20, ybins=20, min_fga=5) -> pd.DataFrame:
    """
    Aggregates by (x, y) bins to compute FG%, xFG%, and (FG − xFG) per zone.
    """
    if df.empty:
        return pd.DataFrame(columns=["x_bin", "y_bin", "FGA", "FG%", "xFG%", "FG%-xFG%"])

    df = df.copy()
    # Use plotting bins across court extents
    x_edges = np.linspace(-HALF_WIDTH_X, HALF_WIDTH_X, xbins + 1)
    y_edges = np.linspace(0, HALF_COURT_LENGTH_FT, ybins + 1)
    df["x_bin"] = pd.cut(df["x"], bins=x_edges, include_lowest=True)
    df["y_bin"] = pd.cut(df["y"], bins=y_edges, include_lowest=True)

    grouped = df.groupby(["x_bin", "y_bin"])
    out = grouped.apply(lambda g: pd.Series({
        "FGA": int(g["made"].notna().sum()),
        "FG%": float(pd.to_numeric(g["made"], errors="coerce").mean()),
        "xFG%": float(pd.to_numeric(g["xfg_placeholder"], errors="coerce").mean())
    })).reset_index()

    out = out[out["FGA"] >= min_fga].copy()
    out["FG%-xFG%"] = out["FG%"] - out["xFG%"]
    out = out.sort_values("FG%-xFG%", ascending=False)
    return out

# ----------------------------
# Plotting: Shot Chart & xFG Surface
# ----------------------------
def plot_shot_chart(df: pd.DataFrame, kind="hexbin"):
    """
    Matplotlib shot chart (hexbin or scatter) over a half-court.
    kind: "hexbin" or "scatter"
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    draw_half_court(ax)

    if df.empty:
        ax.set_title("No shots to display")
        return fig

    if kind == "hexbin":
        hb = ax.hexbin(df["x"], df["y"], gridsize=30, extent=(-HALF_WIDTH_X, HALF_WIDTH_X, 0, HALF_COURT_LENGTH_FT))
    else:  # scatter
        ax.scatter(df["x"], df["y"], s=8, alpha=0.6)

    ax.set_title("Shot Frequency")
    return fig

def plot_xfg_surface(df: pd.DataFrame):
    """
    Placeholder xFG% surface: bin then average xFG per cell to mimic a heatmap.
    # TODO: Replace the averaging with your trained model's predictions over a grid.
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    draw_half_court(ax)

    if df.empty or "xfg_placeholder" not in df.columns:
        ax.set_title("xFG% Heatmap (placeholder)")
        return fig

    # Bin to a grid and average xfg
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    ps = df["xfg_placeholder"].to_numpy()

    H_sum, xedges, yedges = np.histogram2d(xs, ys, bins=[50, 47],
                                           range=[[-HALF_WIDTH_X, HALF_WIDTH_X], [0, HALF_COURT_LENGTH_FT]],
                                           weights=ps)
    H_cnt, _, _ = np.histogram2d(xs, ys, bins=[50, 47],
                                 range=[[-HALF_WIDTH_X, HALF_WIDTH_X], [0, HALF_COURT_LENGTH_FT]])
    with np.errstate(invalid="ignore"):
        H_avg = H_sum / np.maximum(H_cnt, 1e-9)

    X, Y = np.meshgrid(xedges, yedges, indexing="ij")
    im = ax.pcolormesh(X, Y, H_avg, shading="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("xFG% Heatmap (placeholder)")
    return fig

# ----------------------------
# UI: Sidebar & Main Layout
# ----------------------------
def sidebar_filters(df: pd.DataFrame):
    players = ["All"] + sorted([p for p in df["player"].dropna().unique().tolist()])
    seasons = ["All"] + sorted([s for s in df["season"].dropna().unique().tolist()])
    st.sidebar.header("Filters")
    sel_player = st.sidebar.selectbox("Player", players, index=0)
    sel_season = st.sidebar.selectbox("Season", seasons, index=0)
    chart_kind = st.sidebar.selectbox("Shot Chart Style", ["hexbin", "scatter"], index=0)
    return sel_player, sel_season, chart_kind

def apply_filters(df: pd.DataFrame, player: str, season: str) -> pd.DataFrame:
    df2 = df.copy()
    if player != "All":
        df2 = df2[df2["player"] == player]
    if season != "All":
        df2 = df2[df2["season"] == season]
    return df2

# ----------------------------
# App Entry
# ----------------------------
def main():
    st.title("NBA Shot Quality (xFG%) Explorer")
    st.caption("Shot frequency, xFG% heatmap, summary metrics, and best/worst zones.")

    # Load & prepare data
    df = load_data()
    df = normalize_coords_to_half_court(df)
    df = engineer_basic_features(df)

    # Sidebar filters
    sel_player, sel_season, chart_kind = sidebar_filters(df)
    df_view = apply_filters(df, sel_player, sel_season)

    # Main layout: two columns
    left, right = st.columns(2, gap="large")

    with left:
        fig1 = plot_shot_chart(df_view, kind=chart_kind)
        st.pyplot(fig1, use_container_width=True)

    with right:
        fig2 = plot_xfg_surface(df_view)
        st.pyplot(fig2, use_container_width=True)

    # Summary panel
    st.markdown("---")
    st.subheader("Summary")
    summary = compute_summary(df_view)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Attempts", f"{summary['attempts']}")
    m2.metric("Actual FG%", f"{summary['actual_fg']:.3f}" if not math.isnan(summary["actual_fg"]) else "—")
    m3.metric("Avg xFG%", f"{summary['avg_xfg']:.3f}" if not math.isnan(summary["avg_xfg"]) else "—")
    sm_val = summary["shot_making"]
    m4.metric("Shot-Making (FG% − xFG%)", f"{sm_val:+.3f}" if not math.isnan(sm_val) else "—")

    # Best/Worst zones
    st.markdown("---")
    st.subheader("Best / Worst Zones by Shot-Making (FG% − xFG%)")
    zones = zone_table(df_view, xbins=18, ybins=16, min_fga=5)
    if zones.empty:
        st.info("Not enough data to compute zone table.")
    else:
        top_k = zones.head(5).copy()
        bot_k = zones.tail(5).copy()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Best Zones (over-performing vs xFG%)**")
            st.dataframe(top_k.reset_index(drop=True))
        with c2:
            st.markdown("**Worst Zones (under-performing vs xFG%)**")
            st.dataframe(bot_k.reset_index(drop=True))

    # Footer instructions
    st.markdown("---")
    st.markdown("### Deployment Notes (Streamlit Cloud)")
    st.markdown()