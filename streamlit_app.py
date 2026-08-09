"""
streamlit_app.py
NFL Matchup Tool - main UI. Scans a week's slate, shows every prop with
mu-based inputs, and lets you type in a line per row to get live edge/p_over,
same workflow as the MLB tool's adjustable Best Edges table.
"""

import streamlit as st
import pandas as pd
import numpy as np
from nfl_model_combined import scan_full_slate_nfl, rescore_quality_mu_row_nfl

st.set_page_config(page_title="NFL Matchup Tool", layout="wide")
st.title("NFL Matchup Tool")
st.caption("Scan a week's slate, then type in lines per row to get live edge/probability.")

# -----------------------------------------------------------------------
# Season / week selection
# -----------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    season = st.number_input("Season", min_value=2020, max_value=2030, value=2026, step=1)
with col2:
    week = st.number_input("Week", min_value=1, max_value=18, value=1, step=1)

if "slate_df" not in st.session_state:
    st.session_state.slate_df = None

if st.button("Scan full slate", type="primary"):
    with st.spinner(f"Pulling and scoring Week {week}, {season}..."):
        try:
            st.session_state.slate_df = scan_full_slate_nfl(season, week)
            st.success(f"Loaded {len(st.session_state.slate_df)} prop rows.")
        except Exception as e:
            st.error(f"Scan failed: {e}")
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
        min_edge_filter = st.slider("Minimum edge (after entering lines)", 0.0, 1.0, 0.0, 0.05)

    filtered = df.copy()
    if prop_filter != "All":
        filtered = filtered[filtered["prop_type"] == prop_filter]
    if position_filter != "All":
        filtered = filtered[filtered["position"] == position_filter]

    st.subheader("Slate - enter a line per row to compute edge/probability")
    st.caption(
        "Type a value in the 'line' column for any prop you want scored. "
        "mu is derived from the other columns for now - edge/p_over recompute "
        "automatically once you enter a line and sigma is available."
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

    # Recompute edge/p_over live wherever a line was entered. Every row now
    # has a real mu (see calc_prop_mu / fantasy lookback-average in the
    # backend) and sigma, so this is a straight rescore per row - no more
    # special-casing by prop_type needed.
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
    st.info("Click 'Scan full slate' to load this week's props.")
