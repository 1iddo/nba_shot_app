import os
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NBA Shot App (Debug)", layout="wide")
st.write("App booted…")  # heartbeat

# ---- helpers ----
def generate_fake_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "player": rng.choice(["LeBron James","Stephen Curry","Kevin Durant"], n),
        "season": rng.choice(["2023-24","2024-25"], n),
        "x": rng.uniform(-25, 25, n),
        "y": rng.uniform(0, 47, n),
        "made": rng.integers(0, 2, n),
        "is_three": rng.integers(0, 2, n),
        "is_corner": rng.integers(0, 2, n),
        "period": rng.integers(1, 5, n),
        "shot_clock": rng.uniform(0, 24, n),
    })

@st.cache_data(show_spinner=True)
def load_data():
    csv_path = Path("shots.csv")
    if csv_path.exists():
        st.success("Found shots.csv — loading it.")
        try:
            # Read header first to avoid dtype issues
            cols = pd.read_csv(csv_path, nrows=0).columns.str.lower().tolist()
            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = [c.lower() for c in df.columns]
        except Exception as e:
            st.error(f"Error reading shots.csv: {e}")
            df = generate_fake_data()
    else:
        st.warning("shots.csv NOT found — using generated sample data.")
        df = generate_fake_data()
    # ensure expected columns exist
    for col, default in [
        ("player","Unknown"), ("season","Unknown"), ("x",np.nan), ("y",np.nan), ("made",0),
        ("is_three",0), ("is_corner",0), ("period",1), ("shot_clock",np.nan),
    ]:
        if col not in df.columns:
            df[col] = default
    return df

# ---- show environment + data preview (always runs) ----
st.caption(f"Working dir: {os.getcwd()}")
try:
    st.caption(f"Repo files: {sorted(os.listdir('.'))[:25]}")
except Exception as ls_e:
    st.caption(f"Could not list files: {ls_e}")

df = load_data()
st.caption(f"Rows: {len(df):,} | Columns: {list(df.columns)}")
st.dataframe(df.head(10), use_container_width=True)

# ---- do the rest inside a try so errors show on page ----
def app_body(data: pd.DataFrame):
    # Sidebar filters
    players = ["All"] + sorted(data["player"].dropna().unique().tolist())
    seasons = ["All"] + sorted(data["season"].dropna().unique().tolist())
    sel_player = st.sidebar.selectbox("Player", players, index=0)
    sel_season = st.sidebar.selectbox("Season", seasons, index=0)

    filt = data.copy()
    if sel_player != "All":
        filt = filt[filt["player"] == sel_player]
    if sel_season != "All":
        filt = filt[filt["season"] == sel_season]

    # Simple metrics (no model yet)
    attempts = len(filt)
    fg_pct = float(filt["made"].mean())*100 if attempts else 0.0
    xfg_pct = 45.0  # TODO: replace with your model’s mean prediction
    st.write({"Attempts": attempts, "Actual FG%": round(fg_pct,2), "Avg xFG% (placeholder)": xfg_pct,
              "Shot-making": round(fg_pct - xfg_pct, 2)})

    # Plots (guarded)
    import matplotlib
    matplotlib.use("Agg")   # headless backend
    import matplotlib.pyplot as plt

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5,4))
        sc = ax.scatter(filt["x"], filt["y"], c=filt["made"], alpha=0.6)
        ax.set_title("Shot Chart (debug)")
        ax.set_xlim(-25, 25); ax.set_ylim(0, 47)
        st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5,4))
        hb = ax.hexbin(filt["x"], filt["y"], gridsize=20)
        fig.colorbar(hb, ax=ax)
        ax.set_title("xFG% surface (placeholder)")
        st.pyplot(fig, clear_figure=True)

try:
    app_body(df)
except Exception as e:
    st.error("App body crashed — details below.")
    st.exception(e)
    st.code(traceback.format_exc())
