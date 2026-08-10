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
from draft_rankings import (
    build_yahoo_style_rankings, detect_risers, build_league_settings,
    build_snake_draft_targets, compute_blended_rankings, build_draft_rankings_backtest,
)

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
    if mode == "Draft Rankings":
        season = st.number_input(
            "Draft season", min_value=2020, max_value=2030, value=2026, step=1,
            help="This is the season you're drafting FOR. Projections are built from "
                 "the completed prior season's per-game rates (season-1), applied to "
                 "the CURRENT roster for this season - so 2026 rankings use 2025 stats "
                 "but 2026 rosters (reflecting trades/signings like Etienne to NO).",
        )
    else:
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
    st.subheader("League Settings")
    lcol1, lcol2, lcol3, lcol4 = st.columns(4)
    with lcol1:
        num_teams = st.number_input("Number of teams", min_value=2, max_value=20, value=6, step=1)
        draft_position = st.number_input("Your draft position", min_value=1, max_value=num_teams, value=3, step=1)
    with lcol2:
        ppr_label = st.selectbox("Scoring", ["Full PPR", "Half PPR", "Standard (no PPR)"])
        ppr_value = {"Full PPR": 1.0, "Half PPR": 0.5, "Standard (no PPR)": 0.0}[ppr_label]
    with lcol3:
        qb_slots = st.number_input("QB slots", min_value=0, max_value=3, value=1, step=1)
        rb_slots = st.number_input("RB slots", min_value=0, max_value=5, value=2, step=1)
        wr_slots = st.number_input("WR slots", min_value=0, max_value=5, value=2, step=1)
        te_slots = st.number_input("TE slots", min_value=0, max_value=3, value=1, step=1)
    with lcol4:
        flex_slots = st.number_input("FLEX slots", min_value=0, max_value=6, value=3, step=1)
        def_slots = st.number_input("DEF slots", min_value=0, max_value=2, value=1, step=1)
        k_slots = st.number_input("K slots", min_value=0, max_value=2, value=1, step=1)
        bench_slots = st.number_input("Bench slots", min_value=0, max_value=15, value=6, step=1)

    league_settings = build_league_settings(
        num_teams=num_teams, draft_position=draft_position,
        qb=qb_slots, rb=rb_slots, wr=wr_slots, te=te_slots, flex=flex_slots,
        def_=def_slots, k=k_slots, bench=bench_slots, ppr_value=ppr_value,
    )

    if st.button("Build Draft Rankings", type="primary"):
        with st.spinner(f"Building season projections and rankings for {season}..."):
            try:
                rankings = build_yahoo_style_rankings(season, league_settings)
                rankings = detect_risers(rankings, season)
                st.session_state.draft_rankings_df = rankings
                st.session_state.league_settings = league_settings
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
        current_settings = st.session_state.get("league_settings", league_settings)

        view = st.radio(
            "View",
            ["Full Rankings", "Round-by-Round Targets (snake draft)", "Test Projection Accuracy (backtest)"],
            horizontal=True,
        )

        if view == "Test Projection Accuracy (backtest)":
            st.subheader("How accurate is the stats-only projection historically?")
            st.caption(
                "Tests ONLY the pure stats-based projection (last season's rate stats "
                "projected forward) against a real completed season - NOT the blended "
                "FantasyPros portion, since there's no free historical archive of past "
                "FantasyPros rankings to test that part against. This tells you how much "
                "to trust the stats side of the blend, not the blend itself."
            )
            test_season = st.number_input(
                "Test season (projects using test_season-1 stats, compares to real test_season results)",
                min_value=2021, max_value=2025, value=2025, step=1,
            )
            if st.button("Run projection backtest"):
                with st.spinner(f"Building {test_season} projections and comparing to real results..."):
                    try:
                        backtest_results = build_draft_rankings_backtest(test_season, current_settings)
                        st.session_state.draft_backtest_df = backtest_results
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
                        st.session_state.draft_backtest_df = None

            if st.session_state.get("draft_backtest_df") is not None and not st.session_state.draft_backtest_df.empty:
                bt = st.session_state.draft_backtest_df
                valid_bt = bt.dropna(subset=["actual_season_points"])
                if not valid_bt.empty:
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        st.metric("Mean absolute miss (season pts)", round(valid_bt["projection_miss"].abs().mean(), 1))
                    with bcol2:
                        st.metric("Players compared", len(valid_bt))
                bt_display = bt[[
                    "player", "position", "team", "season_proj_points",
                    "actual_season_points", "actual_games_played", "projection_miss",
                ]]
                styled_bt = bt_display.style.background_gradient(
                    subset=["projection_miss"], cmap="RdYlGn_r"
                )
                st.dataframe(styled_bt, use_container_width=True)

        elif view == "Round-by-Round Targets (snake draft)":
            st.subheader(f"Your picks — drafting #{current_settings['draft_position']} in a {current_settings['num_teams']}-team snake draft")
            st.caption(
                "FIXED: this now uses the SAME blended ranking (our stats + FantasyPros "
                "consensus) as the Full Rankings view, instead of pure stats-only VOR. "
                "Previously this bypassed the blend entirely, which could show a real "
                "riser (e.g. a receiver whose role just expanded because a teammate left "
                "in free agency) going far too late. Assumes every other team also drafts "
                "off the blended ranking each pick - a reasonable planning baseline, not "
                "a guarantee of your real draft."
            )
            snake_blend_weight = st.slider(
                "Blend weight: pure stats ← → public consensus", 0.0, 1.0, 0.5, 0.1, key="snake_blend_weight",
            )
            rankings_for_snake = compute_blended_rankings(rankings, our_weight=snake_blend_weight)
            with st.spinner("Simulating snake draft..."):
                targets = build_snake_draft_targets(
                    rankings_for_snake, current_settings,
                    sort_column="blended_score", sort_ascending=True,
                )
            if not targets.empty:
                for round_num in sorted(targets["round"].unique()):
                    round_targets = targets[targets["round"] == round_num]
                    pick_num = round_targets["your_overall_pick"].iloc[0]
                    st.markdown(f"**Round {round_num} (overall pick #{pick_num})**")
                    round_display = round_targets[["player", "position", "team", "season_proj_points", "vor"]]
                    styled_round = round_display.style.background_gradient(subset=["vor"], cmap="Greens")
                    st.dataframe(styled_round, use_container_width=True, hide_index=True)
            else:
                st.info("No targets generated - check league settings.")

        else:
            st.subheader("Draft Rankings — Yahoo-style board")
            st.caption(
                "blended_rank combines our pure stats-based rank (last season's rate "
                "stats only) with FantasyPros public consensus, since public rankings "
                "DO account for things ours can't see - new coordinators, scheme fits, "
                "offseason situational buzz. Adjust the slider to weight one side more."
            )

            blend_weight = st.slider(
                "Blend weight: pure stats ← → public consensus", 0.0, 1.0, 0.5, 0.1,
                help="0.0 = pure FantasyPros consensus, 1.0 = pure our stats-only "
                     "ranking, 0.5 = even blend (recommended default).",
            )
            rankings = compute_blended_rankings(rankings, our_weight=blend_weight)

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                positions_available = ["All"] + sorted(rankings["position"].dropna().unique().tolist())
                pos_filter = st.selectbox("Position", positions_available, key="draft_pos_filter")
            with dcol2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Blended Rank (recommended)", "Pure Stats (VOR)", "Biggest Risers (our_rank_delta)"],
                    key="draft_sort",
                )

            display_rankings = rankings.copy()
            if pos_filter != "All":
                display_rankings = display_rankings[display_rankings["position"] == pos_filter]

            if sort_by == "Biggest Risers (our_rank_delta)":
                display_rankings = display_rankings.sort_values("our_rank_delta", ascending=False, na_position="last")
            elif sort_by == "Pure Stats (VOR)":
                display_rankings = display_rankings.sort_values("overall_rank", ascending=True)
            else:
                display_rankings = display_rankings.sort_values("blended_rank", ascending=True)

            display_cols = [
                "blended_rank", "overall_rank", "player", "pos_rank_label", "team", "bye",
                "season_proj_points", "vor", "ppg_prior", "games_played_prior", "is_rookie_projection",
                "shares_backfield_with", "fantasypros_ecr", "our_rank_delta",
            ]
            display_cols = [c for c in display_cols if c in display_rankings.columns]
            display_final = display_rankings[display_cols]

            # Color-coded like the MLB/Scan tool: brighter green = better VOR
            # (stronger stats-based value), so you can see at a glance which
            # rows are backed by strong underlying production regardless of
            # where blended_rank puts them.
            if "vor" in display_final.columns:
                styled_rankings = display_final.style.background_gradient(subset=["vor"], cmap="Greens")
                st.dataframe(styled_rankings, use_container_width=True)
            else:
                st.dataframe(display_final, use_container_width=True)
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

        base_display_cols = ["player_display_name", "team", "position", "prop_type",
                              "mu", "sigma", "opponent", "opp_dominant_coverage",
                              "opp_dominant_coverage_pct", "opp_num_elevated_coverages",
                              "opp_man_pct", "opp_zone_pct",
                              "quality_score", "line", "p_over", "edge"]
        # Include the full individual coverage-type breakdown AND every
        # advanced metric grade (QB/WR/TE/RB own performance grades, plus
        # opponent defense grades) - column names vary by what's actually
        # available for a given player/week, so these are picked up
        # dynamically rather than hardcoded.
        cov_breakdown_cols = sorted([c for c in scored_df.columns if c.startswith("opp_cov_")])
        grade_cols = sorted([c for c in scored_df.columns if c.endswith("_grade")])
        display_cols = base_display_cols + grade_cols + cov_breakdown_cols
        display_cols = [c for c in display_cols if c in scored_df.columns]
        scan_sorted = scored_df[display_cols].sort_values("edge", ascending=False, na_position="last")
        # Color-coded: edge/p_over use the MLB tool's green scale. Every
        # coverage % column and every advanced metric grade (0-100 scale,
        # already normalized so higher = always better/more notable) is
        # ALSO color-coded the same way - brighter green = higher grade.
        gradient_cols = [c for c in (["edge", "p_over", "opp_man_pct", "opp_zone_pct",
                                       "opp_dominant_coverage_pct", "quality_score"]
                                      + grade_cols + cov_breakdown_cols)
                          if c in scan_sorted.columns]
        styled_scan = scan_sorted.style.background_gradient(subset=gradient_cols, cmap="Greens")
        st.dataframe(styled_scan, use_container_width=True)
        if "opp_dominant_coverage" in scan_sorted.columns:
            st.caption(
                "opp_cov_* columns show the opponent defense's FULL coverage breakdown "
                "(e.g. Cover 1 19%, Cover 2 17.5%, etc.). *_grade columns are 0-100 "
                "percentile grades against this season's league-wide distribution - "
                "player grades (EPA, target share, separation, etc.) and opponent "
                "defense grades (pass/run EPA allowed, pressure rate) are both included. "
                "Defense 'allowed' grades are inverted so high = good defense, consistent "
                "with every other grade. mu itself is adjusted using each player's real "
                "man/zone efficiency split (see mu_before_coverage_adj to compare)."
            )

elif mode != "Draft Rankings":
    st.info("Click the button above to load this week's props.")
