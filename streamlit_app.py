"""
streamlit_app.py
NFL Matchup Tool - main UI. Scans a week's slate, shows every prop with
mu-based inputs, and lets you type in a line per row to get live edge/p_over,
same workflow as the MLB tool's adjustable Best Edges table.
"""

import streamlit as st
import pandas as pd
import numpy as np
from nfl_model_combined import (
    scan_full_slate_nfl, rescore_quality_mu_row_nfl, backtest_week, build_season_accuracy_report,
    diagnose_participation_data,
)
from draft_rankings import (
    build_yahoo_style_rankings, detect_risers, build_league_settings,
    build_snake_draft_targets, compute_blended_rankings, build_draft_rankings_backtest,
)

st.set_page_config(page_title="Dallas Cowboys Matchup Tool", layout="wide", page_icon="⭐")

# -----------------------------------------------------------------------
# COWBOYS THEME - navy/silver/white color scheme + navy star accents.
# This only restyles chrome (header, buttons, tabs, dataframe accents) -
# the scanner itself still covers every NFL team/matchup; it doesn't
# change what data is pulled or how anything is scored. No team logos/
# wordmarks are used (would be copyrighted team IP), just the color
# palette and a plain unicode star for the visual accent.
# -----------------------------------------------------------------------
COWBOYS_NAVY = "#041E42"
COWBOYS_SILVER = "#869397"
COWBOYS_WHITE = "#FFFFFF"
COWBOYS_ACCENT_BLUE = "#7F9695"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {COWBOYS_WHITE};
    }}
    [data-testid="stHeader"] {{
        background-color: {COWBOYS_NAVY};
    }}
    h1, h2, h3 {{
        color: {COWBOYS_NAVY} !important;
    }}
    .stRadio > label, .stNumberInput > label, .stSelectbox > label {{
        color: {COWBOYS_NAVY} !important;
        font-weight: 600;
    }}
    div.stButton > button {{
        background-color: {COWBOYS_NAVY};
        color: {COWBOYS_WHITE};
        border: 1px solid {COWBOYS_SILVER};
        border-radius: 6px;
        font-weight: 600;
    }}
    div.stButton > button:hover {{
        background-color: {COWBOYS_SILVER};
        color: {COWBOYS_NAVY};
        border: 1px solid {COWBOYS_NAVY};
    }}
    [data-testid="stMetricValue"] {{
        color: {COWBOYS_NAVY};
    }}
    .cowboys-banner {{
        background: linear-gradient(90deg, {COWBOYS_NAVY} 0%, {COWBOYS_SILVER} 100%);
        padding: 14px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
    }}
    .cowboys-banner h1 {{
        color: {COWBOYS_WHITE} !important;
        margin: 0;
        font-size: 28px;
    }}
    .cowboys-banner p {{
        color: {COWBOYS_WHITE} !important;
        margin: 2px 0 0 0;
        opacity: 0.9;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="cowboys-banner"><h1>\u2605 Dallas Cowboys Matchup Tool</h1>'
    '<p>Scan a week\'s slate across the league, then type in lines per row for live edge/probability.</p></div>',
    unsafe_allow_html=True,
)

# DEPLOY VERSION MARKER - bump this string on every file delivered, so a
# glance at the app tells you in 5 seconds whether a new deploy actually
# took effect, instead of waiting through a full readiness-report run to
# find out indirectly. If this doesn't match what was just sent, the
# deploy didn't land - no need to test anything further until it does.
DEPLOY_VERSION = "v13-coverage-root-cause-fix-2026-08-13"
st.caption(f"🔧 Deploy check: `{DEPLOY_VERSION}` — if this doesn't match what was just sent to you, the deploy hasn't taken effect yet.")

# -----------------------------------------------------------------------
# Season / week selection
# -----------------------------------------------------------------------

with st.expander("🔍 Debug: Inspect real participation data (coverage adjustment diagnostic)"):
    st.caption(
        "The coverage mu-adjustment has stayed at a 0% fire rate through two separate "
        "fixes with no change either time - this shows the REAL raw data instead of "
        "guessing a third time. Pick a season/week and run it."
    )
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        diag_season = st.number_input("Diagnostic season", min_value=2020, max_value=2030, value=2025, step=1, key="diag_season")
    with dcol2:
        diag_week = st.number_input("Diagnostic week", min_value=1, max_value=18, value=8, step=1, key="diag_week")
    if st.button("Run diagnostic", key="run_diag_btn"):
        with st.spinner("Pulling real participation data and checking..."):
            try:
                diag = diagnose_participation_data(int(diag_season), int(diag_week))
                st.session_state.diag_result = diag
            except Exception as e:
                st.error(f"Diagnostic failed: {e}")
    if "diag_result" in st.session_state:
        diag = st.session_state.diag_result
        st.write("**Does `defense_man_zone_type` exist as a column at all?**", diag.get("has_defense_man_zone_type"))
        if diag.get("has_defense_man_zone_type"):
            st.write("**Real values found (including how much is missing/NaN):**")
            st.json(diag.get("defense_man_zone_type_value_counts"))
        st.write("**Sample player used:**", diag.get("sample_gsis_id_used"))
        st.write("**Total merged rows for that player:**", diag.get("sample_player_total_merged_rows"))
        st.write("**Of those, how many have a non-null defense_man_zone_type:**", diag.get("sample_player_non_null_man_zone_rows"))
        if "sample_player_man_zone_values_seen" in diag:
            st.write("**Real values seen for that specific player's rows:**")
            st.json(diag["sample_player_man_zone_values_seen"])
        with st.expander("Full raw diagnostic output"):
            st.json(diag)

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
if "season_report" not in st.session_state:
    st.session_state.season_report = None
if "show_season_report" not in st.session_state:
    st.session_state.show_season_report = False

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
    if mode.startswith("Backtest"):
        btn_col1, btn_col2 = st.columns(2)
    else:
        btn_col1 = st.container()
        btn_col2 = None

    with btn_col1:
        if st.button(button_label, type="primary"):
            with st.spinner(f"Pulling and scoring Week {week}, {season}..."):
                try:
                    if mode.startswith("Backtest"):
                        st.session_state.slate_df = backtest_week(season, week)
                        st.session_state.backtest_mode = True
                        st.session_state.show_season_report = False
                    else:
                        st.session_state.slate_df = scan_full_slate_nfl(season, week)
                        st.session_state.backtest_mode = False
                        st.session_state.show_season_report = False
                    st.success(f"Loaded {len(st.session_state.slate_df)} prop rows.")
                except Exception as e:
                    st.error(f"{'Backtest' if mode.startswith('Backtest') else 'Scan'} failed: {e}")
                    st.session_state.slate_df = None

    if btn_col2 is not None:
        with btn_col2:
            st.caption(
                "Runs a range of weeks, not necessarily the whole season - start small "
                "(e.g. a 4-6 week range) to confirm it works within Streamlit Cloud's free-"
                "tier memory limit before attempting the full season in one run."
            )
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                report_start_week = st.number_input(
                    "Report start week", min_value=2, max_value=18, value=2, step=1,
                    help="Week 1 is skipped automatically - there's no prior-week history to project from yet.",
                )
            with rcol2:
                report_end_week = st.number_input(
                    "Report end week", min_value=2, max_value=18, value=6, step=1,
                )
            if st.button("Run Readiness Report for this week range", type="secondary"):
                if report_end_week < report_start_week:
                    st.error("End week must be >= start week.")
                else:
                    weeks_to_run = list(range(report_start_week, report_end_week + 1))
                    with st.spinner(f"Scoring weeks {report_start_week}-{report_end_week} of {season} against real results..."):
                        try:
                            st.session_state.season_report = build_season_accuracy_report(season, weeks=weeks_to_run)
                            st.session_state.backtest_mode = True
                            st.session_state.show_season_report = True
                            n_rows = len(st.session_state.season_report["raw"])
                            st.success(f"Scored {n_rows} rows across weeks {report_start_week}-{report_end_week} of {season}.")
                        except Exception as e:
                            st.error(f"Season readiness report failed: {e}")
                            st.session_state.season_report = None

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
                st.dataframe(styled_bt, width='stretch')

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
                    round_display = round_targets[["player", "position", "team", "season_proj_points",
                                                     "vor", "blended_rank", "overall_rank", "fantasypros_ecr"]]
                    styled_round = round_display.style.background_gradient(subset=["vor"], cmap="Greens")
                    st.dataframe(styled_round, width='stretch', hide_index=True)
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
                st.dataframe(styled_rankings, width='stretch')
            else:
                st.dataframe(display_final, width='stretch')
    else:
        st.info("Click 'Build Draft Rankings' to generate your league-specific board.")

# -----------------------------------------------------------------------
# SEASON READINESS REPORT DISPLAY
# -----------------------------------------------------------------------
elif st.session_state.show_season_report and st.session_state.season_report is not None:
    report = st.session_state.season_report
    raw = report["raw"]

    if raw.empty:
        st.warning("No scoreable rows came back for this season - check that the season has completed weeks with real player_stats data.")
    else:
        st.subheader(f"Season Readiness Report — {season}")
        st.caption(
            "Every starter row across every completed week, mu vs real result - not just "
            "the biggest surprises. This is the pre-season calibration check: is "
            "quality_score actually predictive, are the coverage/box mu adjustments "
            "moving mu the right direction more than a coinflip, and is accuracy uneven "
            "across any prop type or position. There's no free historical NFL prop-line "
            "archive, so edge/lean itself can't be backtested against a real market line "
            "- these are the checks that ARE possible without one."
        )

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.metric("Total scored rows", len(raw))
        with rcol2:
            st.metric("Mean absolute miss (all rows)", round(raw["abs_miss"].mean(), 1))
        with rcol3:
            adj_acc = report["adjustment_direction_accuracy"]
            st.metric(
                "mu-adjustment direction accuracy",
                f"{adj_acc:.1%}" if pd.notna(adj_acc) else "n/a",
                help="Of rows where the coverage/box adjustment actually moved mu, the % "
                     "of the time that move was toward the real result. Should clear 50% "
                     "by a real margin - if it doesn't, the adjustment isn't adding signal "
                     "as currently weighted.",
            )

        st.markdown("**Accuracy by prop type** — is any specific prop systematically worse?")
        st.dataframe(
            report["by_prop_type"].style.background_gradient(subset=["mean_abs_miss"], cmap="RdYlGn_r"),
            width='stretch',
        )

        st.markdown("**Accuracy by position**")
        st.dataframe(
            report["by_position"].style.background_gradient(subset=["mean_abs_miss"], cmap="RdYlGn_r"),
            width='stretch',
        )

        if not report["by_quality_tier"].empty:
            st.markdown(
                "**Is quality_score actually predictive?** Higher tiers should show "
                "tighter/more favorable misses than lower tiers - if they don't, "
                "quality_score isn't earning its keep as currently weighted."
            )
            st.dataframe(
                report["by_quality_tier"].style.background_gradient(
                    subset=["mean_abs_miss", "mean_match_ratio"], cmap="RdYlGn_r"
                ),
                width='stretch',
            )
            st.caption(
                "Color both columns for a reason: mean_abs_miss alone can mislead - it's "
                "naturally bigger for high-volume players regardless of tier, so a tier full "
                "of bell-cow RBs can look green without actually being more accurate. "
                "mean_match_ratio (miss scaled to that player's own normal variance) is the "
                "real apples-to-apples check - if IT climbs (gets worse/redder) as the tier "
                "goes up, quality_score is overconfident at the top even if raw miss looks fine."
            )

        if not report["role_verification_check"].empty:
            st.markdown("**Does the role-verification trend signal add real accuracy?**")
            st.dataframe(report["role_verification_check"], width='stretch')

        with st.expander("Every scored row (raw)"):
            st.dataframe(raw, width='stretch')

        st.markdown("---")
        st.markdown("**Filtered check** — set your own floor and see if it actually tightens the miss")
        st.caption(
            "games_sampled here means weeks of real history behind that row, scaled for a "
            "17-game season - not the same '10' MLB used for a 162-game season. edge isn't "
            "filterable here since there's no real line in a backtest row to compute it from."
        )
        fcol_q, fcol_g = st.columns(2)
        with fcol_q:
            min_quality_check = st.slider("Minimum quality_score", 0, 100, 70, 5, key="readiness_quality_filter")
        with fcol_g:
            min_games_check = st.slider("Minimum games_sampled", 0, 17, 3, 1, key="readiness_games_filter")

        check_cols = [c for c in ["player_display_name", "team", "position", "prop_type", "week",
                                   "mu", "actual", "miss", "abs_miss", "sigma", "match_ratio",
                                   "quality_score", "games_sampled"] if c in raw.columns]
        filtered_check = raw[
            (raw["quality_score"].fillna(0) >= min_quality_check)
            & (raw["games_sampled"].fillna(0) >= min_games_check)
        ][check_cols].sort_values("quality_score", ascending=False, na_position="last")

        if filtered_check.empty:
            st.info("No rows clear that floor - try lowering it.")
        else:
            fchk1, fchk2 = st.columns(2)
            with fchk1:
                st.metric(f"Rows at quality_score>={min_quality_check}", len(filtered_check))
            with fchk2:
                st.metric("Mean absolute miss (this subset)", round(filtered_check["abs_miss"].mean(), 1))
            st.caption(
                "Compare this mean absolute miss to the overall mean absolute miss above - if "
                f"this quality_score>={min_quality_check} subset isn't meaningfully tighter than "
                "the all-rows number, quality_score isn't separating good matchups from bad ones "
                "yet at this threshold."
            )
            styled_check = filtered_check.style.background_gradient(subset=["abs_miss"], cmap="RdYlGn_r")
            st.dataframe(styled_check, width='stretch')

# -----------------------------------------------------------------------
# Filters + editable table (Scan / Backtest modes)
# -----------------------------------------------------------------------
elif st.session_state.slate_df is not None and not st.session_state.slate_df.empty:
    df = st.session_state.slate_df.copy()

    # -----------------------------------------------------------------------
    # GAME-BY-GAME PICKER - lets you pick a single matchup (like a scoreboard)
    # and see just that game's props, ranked by quality instead of only
    # filtering by prop_type/position across the whole week's slate.
    # -----------------------------------------------------------------------
    if "selected_game" not in st.session_state:
        st.session_state.selected_game = "All Games"

    if "matchup" in df.columns:
        available_games = sorted([g for g in df["matchup"].dropna().unique().tolist()])
        if available_games:
            st.subheader("Games this week")
            st.caption("Pick a matchup to see just its props, ranked best-quality first - or leave 'All Games' selected to filter the whole week's slate like before.")

            game_options = ["All Games"] + available_games
            # Buttons in a row, mirroring a scoreboard-style pick - the
            # currently-selected game is shown as the primary (highlighted) button.
            n_cols = min(len(game_options), 4)
            game_cols = st.columns(n_cols)
            for i, g in enumerate(game_options):
                with game_cols[i % n_cols]:
                    is_selected = (st.session_state.selected_game == g)
                    if st.button(g, key=f"game_btn_{g}", type=("primary" if is_selected else "secondary"), width='stretch'):
                        st.session_state.selected_game = g
                        st.rerun()

    if st.session_state.selected_game != "All Games" and "matchup" in df.columns:
        df = df[df["matchup"] == st.session_state.selected_game]
        st.info(f"Showing {st.session_state.selected_game} only - sorted by quality_score (best matchups first).")

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
        st.dataframe(styled_backtest, width='stretch')
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

        if week is not None and week <= 3:
            st.warning(
                f"Week {week}: most players are still leaning on prior-season fallback data "
                "(mu needs 2 real current-season games, sigma needs 3). Check the "
                "data_confidence column per row before trusting a number - don't assume "
                "everyone has switched over to real 2026 data yet."
            )
        elif week is not None and week == 4:
            st.info(
                "Week 4 (~end of September): most returning players should now be on real "
                "current-season data for mu and sigma. Time to start trusting 2026 numbers "
                "over last season's - but still check data_confidence for traded players "
                "and rookies, who may need longer."
            )

        edited = st.data_editor(
            filtered,
            column_config={
                "line": st.column_config.NumberColumn("line", help="Enter the book/DFS line for this prop"),
            },
            disabled=[c for c in filtered.columns if c not in ("line",)],
            num_rows="fixed",
            width='stretch',
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

        base_display_cols = ["player_display_name", "team", "matchup", "position", "prop_type",
                              "mu", "sigma", "data_confidence", "games_sampled_current",
                              "opponent", "opp_dominant_coverage",
                              "opp_dominant_coverage_pct", "opp_num_elevated_coverages",
                              "opp_man_pct", "opp_zone_pct",
                              "opp_box_stack_pct", "opp_box_elevated",
                              "playaction_exploit_strength", "playaction_used_coverage_specific_data",
                              "personnel_exploit_strength", "dominant_personnel",
                              "quality_score", "grade_matchup_strength",
                              "role_verification_score", "role_trend_ratio",
                              "line", "p_over", "edge"]
        # Include the full individual coverage-type breakdown AND every
        # advanced metric grade (QB/WR/TE/RB own performance grades, plus
        # opponent defense grades) - column names vary by what's actually
        # available for a given player/week, so these are picked up
        # dynamically rather than hardcoded.
        cov_breakdown_cols = sorted([c for c in scored_df.columns if c.startswith("opp_cov_")])
        grade_cols = sorted([c for c in scored_df.columns if c.endswith("_grade")])
        display_cols = base_display_cols + grade_cols + cov_breakdown_cols
        display_cols = [c for c in display_cols if c in scored_df.columns]
        # Default sort is edge (once lines are entered) across the whole
        # week's slate - but when a single game is selected, no line has
        # necessarily been entered yet for THIS specific game's props, so
        # sort by quality_score instead (best matchups first), matching
        # what was actually asked for: pick a game, see its best-quality
        # plays ranked by how the model grades them, not by an as-yet-
        # unentered edge number.
        if st.session_state.get("selected_game", "All Games") != "All Games" and "quality_score" in scored_df.columns:
            scan_sorted = scored_df[display_cols].sort_values("quality_score", ascending=False, na_position="last")
        else:
            scan_sorted = scored_df[display_cols].sort_values("edge", ascending=False, na_position="last")
        # Color-coded: edge/p_over use the MLB tool's green scale. Every
        # coverage/box % column and every advanced metric grade (0-100
        # scale, already normalized so higher = always better/more
        # notable) is ALSO color-coded the same way - brighter green =
        # higher grade. grade_matchup_strength/role_verification_score are
        # 0-1 scale (not 0-100 like the raw grades) but still "higher =
        # better", so the same gradient direction applies.
        gradient_cols = [c for c in (["edge", "p_over", "opp_man_pct", "opp_zone_pct",
                                       "opp_dominant_coverage_pct", "opp_box_stack_pct",
                                       "playaction_exploit_strength", "personnel_exploit_strength",
                                       "quality_score", "grade_matchup_strength",
                                       "role_verification_score"]
                                      + grade_cols + cov_breakdown_cols)
                          if c in scan_sorted.columns]
        styled_scan = scan_sorted.style.background_gradient(subset=gradient_cols, cmap="Greens")
        st.dataframe(styled_scan, width='stretch')
        if "opp_dominant_coverage" in scan_sorted.columns:
            st.caption(
                "opp_cov_* columns show the opponent defense's FULL coverage breakdown "
                "(e.g. Cover 1 19%, Cover 2 17.5%, etc.); opp_box_stack_pct is the run-game "
                "equivalent (share of plays with 7+ in the box). playaction_exploit_strength "
                "combines whether this QB runs play-action often AND performs well in it with "
                "whether the opponent is specifically vulnerable to play-action in their "
                "dominant coverage (falls back to their overall PA-allowed number if that "
                "coverage lacks a PA-specific sample - see playaction_used_coverage_specific_data). "
                "*_grade columns are 0-100 percentile grades against this season's league-wide "
                "distribution - player grades (EPA, target share, separation, pressure faced, "
                "PROE, etc.) and opponent defense grades (pass/run EPA allowed, pressure rate, "
                "play-action allowed) are both included. Defense 'allowed'/'faced' grades are "
                "inverted so high = good defense/QB, consistent with every other grade. "
                "quality_score blends THREE signals: the structural coverage/box/play-action "
                "tendency exploit shown above, grade_matchup_strength "
                "(this player's own skill grades vs the opponent's allowed grades, tailored "
                "per prop type), and role_verification_score (whether this player's recent "
                "real usage backs up their season-long role - see role_trend_ratio). mu "
                "itself is separately adjusted using each player's real man/zone or "
                "light-box/stacked-box efficiency split (see mu_before_coverage_adj / "
                "mu_before_box_adj to compare)."
            )

elif mode != "Draft Rankings":
    st.info("Click the button above to load this week's props.")
