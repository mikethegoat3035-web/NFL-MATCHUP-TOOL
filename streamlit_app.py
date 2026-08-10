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
from draft_rankings import build_yahoo_style_rankings, detect_risers

st.set_page_config(page_title="NFL Matchup Tool", layout="wide")
st.title("NFL Matchup Tool")
st.caption("Scan a week's slate, then type in lines per row to get live edge/probability.")

# -----------------------------------------------------------------------
# Season / week selection
# -----------------------------------------------------------------------
mode = st.radio(
    "Mode",
    ["Scan (adjustable lines)", "Backtest (compare mu vs actual results)", "Draft Rankings"],
    horizontal=True,
    help="Backtest mode only works for a week that's already been played. "
         "Draft Rankings builds a full season projection/ranking for your "
         "league format, using last season's data as the projection basis.",
)

col1, col2 = st.columns(2)
with col1:
    season = st.number_input("Season", min_value=2020, max_value=2030, value=2025, step=1)
with col2:
    if mode == "Draft Rankings":
        st.caption("Draft Rankings uses the prior completed season as the projection basis - "
                   "week isn't used in this mode.")
        week = None
    else:
        week = st.number_input("Week", min_value=1, max_value=18, value=10, step=1)

if "slate_df" not in st.session_state:
    st.session_state.slate_df = None
if "backtest_mode" not in st.session_state:
    st.session_state.backtest_mode = False
if "draft_rankings_df" not in st.session_state:
    st.session_state.draft_rankings_df = None

if mode == "Draft Rankings":
    if st.button("Build Draft Rankings", type="primary"):
        with st.spinner(f"Building season projections and rankings for {season}..."):
            try:
                rankings = build_yahoo_style_rankings(season)
                rankings = detect_risers(rankings, season)
                st.session_state.draft_rankings_df = rankings
                st.success(f"Ranked {len(rankings)} players.")
            except Exception as e:
                st.error(f"Draft rankings failed: {e}")
                st.session_state.draft_rankings_df = None
else:
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
# DRAFT RANKINGS DISPLAY
# -----------------------------------------------------------------------
if mode == "Draft Rankings":
    if st.session_state.draft_rankings_df is not None and not st.session_state.draft_rankings_df.empty:
        rankings = st.session_state.draft_rankings_df.copy()

        st.subheader("Draft Rankings — 6-team, full PPR, 1QB/2RB/2WR/1TE/3FLEX/1DEF/1K/6BN")
        st.caption(
            "Ranked by Value Over Replacement (VOR) for your specific 6-team, "
            "1QB/2RB/2WR/1TE/3FLEX/1DEF/1K/6BN full-PPR league - not generic "
            "industry rankings. our_rank_delta shows how much higher we rank a "
            "player than FantasyPros consensus (ecr) - a big positive number "
            "means a potential riser the public hasn't caught up to yet."
        )

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            positions_available = ["All"] + sorted(rankings["position"].dropna().unique().tolist())
            pos_filter = st.selectbox("Position", positions_available, key="draft_pos_filter")
        with dcol2:
            sort_by = st.selectbox("Sort by", ["Overall Rank (VOR)", "Biggest Risers (our_rank_delta)"], key="draft_sort")

        display_rankings = rankings.copy()
        if pos_filter != "All":
            display_rankings = display_rankings[display_rankings["position"] == pos_filter]

        if sort_by == "Biggest Risers (our_rank_delta)":
            display_rankings = display_rankings.sort_values("our_rank_delta", ascending=False, na_position="last")
        else:
            display_rankings = display_rankings.sort_values("overall_rank", ascending=True)

        display_cols = [
            "overall_rank", "player", "pos_rank_label", "team", "bye",
            "season_proj_points", "vor", "ppg_prior", "games_played_prior",
            "fantasypros_ecr", "our_rank_delta",
        ]
        display_cols = [c for c in display_cols if c in display_rankings.columns]
        st.dataframe(display_rankings[display_cols], use_container_width=True)
    else:
        st.info("Click 'Build Draft Rankings' to generate your league-specific board.")

# -----------------------------------------------------------------------
# Filters + editable table (Scan / Backtest modes)
# -----------------------------------------------------------------------
elif st.session_state.slate_df is not None and not st.session_state.slate_df.empty:
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
        # BACKTEST DISPLAY: only significant surprises among real starters
        # -----------------------------------------------------------
        st.subheader(f"Backtest — Week {week}, {season}: significant surprises")
        st.caption(
            "Only starters who actually played are shown, and only the games "
            "where mu was meaningfully off (a line near mu would've been "
            "mispriced) - close matches are filtered out automatically. "
            "Sorted biggest surprise first."
        )

        display_cols = ["player_display_name", "team", "position", "prop_type", "mu", "actual", "miss"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        backtest_sorted = filtered[display_cols + ["match_ratio"]].sort_values(
            "match_ratio", ascending=False, na_position="last"
        ) if "match_ratio" in filtered.columns else filtered[display_cols]
        display_only = backtest_sorted[display_cols]

        # Normalize color scale to the ACTUAL range of match_ratio present in
        # this filtered result set, not a fixed 0-3.0 scale - since results
        # are now pre-filtered to match_ratio >= 2.0, a fixed scale calibrated
        # for the old 0-3.0 range compressed everything into the dim tail end
        # (that was the bug: every row looked uniformly dark because the
        # "bright" part of the old scale had already been filtered out).
        valid_ratios = backtest_sorted["match_ratio"].dropna() if "match_ratio" in backtest_sorted.columns else pd.Series(dtype=float)
        ratio_min = valid_ratios.min() if not valid_ratios.empty else 0
        ratio_max = valid_ratios.max() if not valid_ratios.empty else 1
        ratio_range = max(ratio_max - ratio_min, 0.001)  # avoid divide-by-zero

        def _row_color(row):
            ratio = backtest_sorted.loc[row.name, "match_ratio"] if "match_ratio" in backtest_sorted.columns else np.nan
            if pd.isna(ratio):
                return [""] * len(row)
            # smallest surviving ratio (least extreme, but still past the
            # threshold) = brightest; largest surviving ratio (most extreme
            # outlier) = fades to background
            intensity = max(0, 1 - ((ratio - ratio_min) / ratio_range))
            return [f"background-color: rgba(0, 140, 0, {intensity:.2f})"] * len(row)

        styled_backtest = display_only.style.apply(_row_color, axis=1)
        st.dataframe(styled_backtest, use_container_width=True)
        st.caption("Brighter green = closer to the significance threshold. Fading toward the dark background = the most extreme, rarest surprises.")

        valid = filtered.dropna(subset=["mu", "actual"])
        if not valid.empty:
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Mean absolute miss", round(valid["abs_miss"].mean(), 1))
            with mcol2:
                st.metric("Mean miss (bias)", round(valid["miss"].mean(), 1))
            with mcol3:
                st.metric("Significant surprises found", len(valid))

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
        scan_sorted = scored_df[display_cols].sort_values("edge", ascending=False, na_position="last")
        # Color-coded like the MLB tool: brighter/darker green = stronger edge/p_over.
        styled_scan = scan_sorted.style.background_gradient(
            subset=[c for c in ["edge", "p_over"] if c in scan_sorted.columns], cmap="Greens"
        )
        st.dataframe(styled_scan, use_container_width=True)

elif mode != "Draft Rankings":
    st.info("Click the button above to load this week's props.")
