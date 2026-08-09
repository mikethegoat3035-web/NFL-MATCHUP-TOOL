"""
streamlit_app.py
NFL Matchup Tool - main UI. Scans a week's slate, shows every prop with
mu-based inputs, and lets you type in a line per row to get live edge/p_over,
same workflow as the MLB tool's adjustable Best Edges table.
"""

import streamlit as st
import pandas as pd
import numpy as np
from nfl_model_combined import scan_full_slate_nfl, rescore_quality_mu_row_nfl, backtest_week

st.set_page_config(page_title="NFL Matchup Tool", layout="wide")
st.title("NFL Matchup Tool")
st.caption("Scan a week's slate, then type in lines per row to get live edge/probability.")

# -----------------------------------------------------------------------
# Season / week selection
# -----------------------------------------------------------------------
mode = st.radio(
    "Mode",
    ["Scan (adjustable lines)", "Backtest (compare mu vs actual results)"],
    horizontal=True,
    help="Backtest mode only works for a week that's already been played - "
         "it shows what the model would have projected vs what actually happened, "
         "no line needed.",
)

col1, col2 = st.columns(2)
with col1:
    season = st.number_input("Season", min_value=2020, max_value=2030, value=2025, step=1)
with col2:
    week = st.number_input("Week", min_value=1, max_value=18, value=10, step=1)

if "slate_df" not in st.session_state:
    st.session_state.slate_df = None
if "backtest_mode" not in st.session_state:
    st.session_state.backtest_mode = False

button_label = "Run backtest" if mode.startswith("Backtest") else "Scan full slate"
if st.button(button_label, type="primary"):
    with st.spinner(f"Pulling and scoring Week {week}, {season}..."):
        try:
            if mode.startswith("Backtest"):
                st.session_state.slate_df = backtest_week(season, week)
                st.session_state.backtest_mode = True
            else:
                st.session_state.slate_df = scan_full_slate_nfl(season, week)
                st.session_state.backtest_mode = False
            st.success(f"Loaded {len(st.session_state.slate_df)} prop rows.")
        except Exception as e:
            st.error(f"{'Backtest' if mode.startswith('Backtest') else 'Scan'} failed: {e}")
            st.session_state.slate_df = None

# -----------------------------------------------------------------------
# Filters + editable table
# -----------------------------------------------------------------------
if st.session_state.slate_df is not None and not st.session_state.slate_df.empty:
    df = st.session_state.slate_df.copy()

    st.subheader("Filters")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        prop_types = ["All"] + sorted(df["prop_type"].dropna().unique().tolist())
        prop_filter = st.selectbox("Prop type", prop_types)
    with fcol2:
        positions = ["All"] + sorted(df["position"].dropna().unique().tolist())
        position_filter = st.selectbox("Position", positions)
    with fcol3:
        if not st.session_state.backtest_mode:
            min_edge_filter = st.slider("Minimum edge (after entering lines)", 0.0, 1.0, 0.0, 0.05)
        else:
            min_edge_filter = 0.0

    filtered = df.copy()
    if prop_filter != "All":
        filtered = filtered[filtered["prop_type"] == prop_filter]
    if position_filter != "All":
        filtered = filtered[filtered["position"] == position_filter]

    if st.session_state.backtest_mode:
        # -----------------------------------------------------------
        # BACKTEST DISPLAY: mu vs actual, sorted by biggest miss
        # -----------------------------------------------------------
        st.subheader(f"Backtest — Week {week}, {season}: projected mu vs actual result")
        st.caption(
            "mu was computed using only weeks before this one - actual is what "
            "really happened. Sorted by biggest miss first."
        )
        display_cols = ["player_display_name", "team", "position", "prop_type",
                         "mu", "sigma", "actual", "miss", "abs_miss"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].sort_values("abs_miss", ascending=False, na_position="last"),
            use_container_width=True,
        )

        valid = filtered.dropna(subset=["mu", "actual"])
        if not valid.empty:
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Mean absolute miss", round(valid["abs_miss"].mean(), 1))
            with mcol2:
                st.metric("Mean miss (bias)", round(valid["miss"].mean(), 1))
            with mcol3:
                st.metric("Rows with a real comparison", len(valid))

    else:
        # -----------------------------------------------------------
        # SCAN DISPLAY: adjustable lines, live edge/p_over
        # -----------------------------------------------------------
        st.subheader("Slate - enter a line per row to compute edge/probability")
        st.caption(
            "Type a value in the 'line' column for any prop you want scored. "
            "edge/p_over recompute automatically once you enter a line."
        )

        edited = st.data_editor(
            filtered,
            column_config={
                "line": st.column_config.NumberColumn("line", help="Enter the book/DFS line for this prop"),
            },
            disabled=[c for c in filtered.columns if c not in ("line",)],
            num_rows="fixed",
            use_container_width=True,
            key="slate_editor",
        )

        results = []
        for _, row in edited.iterrows():
            mu = row.get("mu")
            line = row.get("line")
            sigma = row.get("sigma")
            if pd.notna(line) and pd.notna(mu) and pd.notna(sigma):
                scored = rescore_quality_mu_row_nfl(mu, line, sigma)
                results.append({**row.to_dict(), **scored})
            else:
                results.append({**row.to_dict(), "p_over": np.nan, "edge": np.nan})

        scored_df = pd.DataFrame(results)
        if min_edge_filter > 0:
            scored_df = scored_df[scored_df["edge"].fillna(0) >= min_edge_filter]

        display_cols = ["player_display_name", "team", "position", "prop_type",
                         "mu", "sigma", "line", "p_over", "edge"]
        display_cols = [c for c in display_cols if c in scored_df.columns]
        st.dataframe(scored_df[display_cols].sort_values("edge", ascending=False, na_position="last"),
                     use_container_width=True)

else:
    st.info("Click the button above to load this week's props.")
