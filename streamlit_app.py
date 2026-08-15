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
    diagnose_participation_data, get_player_matchup_explanation, diagnose_injuries_data,
    diagnose_alignment_data, pull_player_stats, pull_schedules,
    build_coverage_crossref_game_log, diagnose_player_stats_for_game_log,
)
from draft_rankings import (
    build_yahoo_style_rankings, detect_risers, build_league_settings,
    build_snake_draft_targets, compute_blended_rankings, build_draft_rankings_backtest,
)
from coverage_matchup import (
    load_full_dataset, get_matchup, TEAM_ABBREV_TO_FULL, COVERAGE_FIELDS,
    _same_team, check_alignment_fit, ALIGNMENT_RTE_COLUMNS, ALIGNMENTS,
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
    /* Body text (markdown, write, caption) was defaulting to white in some
       browser/theme combos, making report content invisible against the
       forced-white app background - only headers (h1-h3, forced navy above)
       were visible. This forces readable dark text everywhere EXCEPT
       buttons, which stay white-on-navy via the more specific button rule
       above (higher CSS specificity wins regardless of rule order). */
    .stApp, .stApp p, .stApp li, .stApp span,
    .stMarkdown, [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {{
        color: {COWBOYS_NAVY} !important;
    }}
    /* Coverage Matchup - StatRankings CoverageIQ-style cards */
    .cov-card {{
        background: {COWBOYS_WHITE};
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .cov-card-header {{
        font-size: 19px;
        font-weight: 700;
        color: {COWBOYS_NAVY};
        margin-bottom: 2px;
    }}
    .cov-card-usage {{
        font-size: 13px;
        color: #5a6b7a;
        margin-bottom: 6px;
    }}
    .cov-z-badge {{
        display: inline-block;
        padding: 1px 9px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: 700;
        color: {COWBOYS_WHITE};
        background: {COWBOYS_NAVY};
        margin-left: 6px;
    }}
    .cov-fit-warning {{
        font-size: 12px;
        color: #b02a37;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    .cov-grid {{
        display: flex;
        gap: 28px;
        margin-top: 10px;
    }}
    .cov-align-block {{
        background: #fafafa;
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
    }}
    .cov-align-header {{
        font-weight: 700;
        color: {COWBOYS_NAVY};
        font-size: 13px;
        margin-bottom: 4px;
    }}
    .cov-col {{
        flex: 1;
        min-width: 0;
    }}
    .cov-col-title {{
        font-weight: 700;
        color: {COWBOYS_NAVY};
        font-size: 14px;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 2px solid {COWBOYS_NAVY};
    }}
    .cov-thin-flag {{
        font-size: 11px;
        font-weight: 700;
        color: {COWBOYS_WHITE};
        background: #b02a37;
        padding: 1px 6px;
        border-radius: 8px;
        margin-left: 6px;
    }}
    .cov-no-data {{
        font-size: 13px;
        color: #8a8a8a;
        font-style: italic;
    }}
    .stat-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 13px;
    }}
    .stat-label {{
        color: #444;
        font-weight: 600;
    }}
    .stat-value {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .tier-badge {{
        padding: 2px 9px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        color: {COWBOYS_WHITE};
        white-space: nowrap;
    }}
    .tier-elite {{ background: #1b7a3d; }}
    .tier-above-avg {{ background: #66bb6a; }}
    .tier-average {{ background: #8a8a8a; }}
    .tier-below-avg {{ background: #ef8c1e; }}
    .tier-poor {{ background: #c0392b; }}
    .cov-more {{
        margin-top: 6px;
    }}
    .cov-more summary {{
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
        color: #5a6b7a;
        list-style: none;
    }}
    .cov-more summary::-webkit-details-marker {{
        display: none;
    }}
    .cov-more summary::before {{
        content: "▸ ";
    }}
    .cov-more[open] summary::before {{
        content: "▾ ";
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
DEPLOY_VERSION = "v25-alignment-diagnostic-2026-08-13"
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

with st.expander("🔍 Debug: Inspect real injury-report data (for the planned injury/active-status check)"):
    st.caption(
        "Real columns for nflreadpy's injury data have never been checked against live data - "
        "this shows exactly what's actually there before anything gets built to read specific "
        "column names from it, same approach that found the real coverage-type bug earlier."
    )
    icol1, icol2 = st.columns(2)
    with icol1:
        inj_season = st.number_input("Diagnostic season", min_value=2020, max_value=2030, value=2025, step=1, key="inj_season")
    with icol2:
        inj_week = st.number_input("Diagnostic week", min_value=1, max_value=18, value=8, step=1, key="inj_week")
    if st.button("Run injuries diagnostic", key="run_inj_diag_btn"):
        with st.spinner("Pulling real injury-report data and checking..."):
            try:
                inj_diag = diagnose_injuries_data(int(inj_season), int(inj_week))
                st.session_state.inj_diag_result = inj_diag
            except Exception as e:
                st.error(f"Diagnostic failed: {e}")
    if "inj_diag_result" in st.session_state:
        inj_diag = st.session_state.inj_diag_result
        if "error" in inj_diag:
            st.error(inj_diag["error"])
        else:
            st.write("**Real columns pull_injuries() returns:**", inj_diag.get("columns"))
            st.write("**Total rows pulled:**", inj_diag.get("n_rows_total"))
            st.write("**Columns whose name looks like an injury status field:**", inj_diag.get("status_like_columns_found"))
            st.write("**Columns whose name looks like a player-ID field:**", inj_diag.get("id_like_columns_found"))
            for key in inj_diag:
                if key.startswith("value_counts__"):
                    st.write(f"**Real values in `{key.replace('value_counts__', '')}`:**")
                    st.json(inj_diag[key])
            st.write("**Real seasons present:**", inj_diag.get("real_seasons_present"))
            st.write("**Real weeks present:**", inj_diag.get("real_weeks_present"))
            with st.expander("Raw sample rows (unfiltered)"):
                st.json(inj_diag.get("sample_rows"))
            with st.expander("Full raw diagnostic output"):
                st.json(inj_diag)

with st.expander("🔍 Debug: Check for real receiver alignment data (wide/slot/backfield/inline)"):
    st.caption(
        "There's a confirmed 'route' column (route TYPE - slant/go/screen/etc.), which is NOT "
        "the same thing as pre-snap ALIGNMENT (wide/slot/backfield/inline). This checks every "
        "real column in both data sources for anything alignment-related before building "
        "anything on a guess."
    )
    acol1, acol2 = st.columns(2)
    with acol1:
        align_season = st.number_input("Diagnostic season", min_value=2020, max_value=2030, value=2025, step=1, key="align_season")
    with acol2:
        align_week = st.number_input("Diagnostic week", min_value=1, max_value=18, value=8, step=1, key="align_week")
    if st.button("Run alignment diagnostic", key="run_align_diag_btn"):
        with st.spinner("Checking real data for alignment fields..."):
            try:
                align_diag = diagnose_alignment_data(int(align_season), int(align_week))
                st.session_state.align_diag_result = align_diag
            except Exception as e:
                st.error(f"Diagnostic failed: {e}")
    if "align_diag_result" in st.session_state:
        align_diag = st.session_state.align_diag_result
        st.write("**Alignment-sounding columns found in participation data:**", align_diag.get("participation_alignment_like_columns"))
        st.write("**Alignment-sounding columns found in FTN charting data:**", align_diag.get("ftn_alignment_like_columns"))
        if "route_value_counts" in align_diag:
            st.write("**Real values in `route` (route TYPE, not alignment):**")
            st.json(align_diag["route_value_counts"])
        if "n_offense_backfield_value_counts" in align_diag:
            st.write("**Real values in `n_offense_backfield` (a COUNT, not per-player alignment):**")
            st.json(align_diag["n_offense_backfield_value_counts"])
        for key in align_diag:
            if key.startswith("participation_value_counts__") or key.startswith("ftn_value_counts__"):
                st.write(f"**Real values in `{key.split('__', 1)[1]}`:**")
                st.json(align_diag[key])
        with st.expander("Full raw diagnostic output (every real column name from both sources)"):
            st.json(align_diag)

mode = st.radio(
    "Mode",
    ["Scan (adjustable lines)", "Backtest (compare mu vs actual results)", "Draft Rankings",
     "Coverage Matchup (premium data)"],
    horizontal=True,
    help="Backtest mode only works for a week that's already been played. "
         "Draft Rankings builds a full season projection/ranking for your "
         "league format, using last season's data as the projection basis. "
         "Coverage Matchup uses the manually-collected FantasyPoints premium "
         "dataset (Cover 0-6 shell-level splits) - separate from the free "
         "nflreadpy pipeline the other three modes run on.",
)

if mode != "Coverage Matchup (premium data)":
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
if "coverage_bundle" not in st.session_state:
    st.session_state.coverage_bundle = None
if "coverage_data_dir" not in st.session_state:
    st.session_state.coverage_data_dir = None

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
elif mode == "Coverage Matchup (premium data)":
    pass  # own section, rendered below alongside the Draft Rankings display block
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

    # -------------------------------------------------------------------
    # BEST QUALITY MATCHUPS - built directly into the scan itself now
    # (was a separate mode, folded in per feedback: no reason to force a
    # mode switch just to see the curated best-quality view). Shown for
    # BOTH Scan and Backtest (a backtest row has the same gsis_id/prop_type/
    # team/opponent/week columns needed for the same real breakdown - a
    # played week is actually a great way to sanity-check this feature
    # works, no different underlying data than a live scan). Filters by a
    # MINIMUM quality_score (not just top-N) so "keep options limited but
    # quality high" is a real floor, not just a count.
    # -------------------------------------------------------------------
    if True:
        bm_df = df[df["prop_type"].isin(["pass_yards", "rec_yards"])].dropna(subset=["quality_score"])
        if not bm_df.empty:
            st.subheader("🏆 Best Quality Matchups")
            if st.session_state.backtest_mode:
                st.caption("Backtest data - real per-coverage breakdown for an already-played week, "
                           "same underlying function as a live Scan.")
            bmf1, bmf2 = st.columns(2)
            with bmf1:
                bm_min_quality = st.slider("Minimum quality_score", 0, 100, 75, 5, key="bm_min_quality")
            with bmf2:
                bm_prop_filter = st.selectbox("Prop type", ["Both", "pass_yards", "rec_yards"], key="bm_prop_filter")

            bm_filtered = bm_df[bm_df["quality_score"] >= bm_min_quality]
            if bm_prop_filter != "Both":
                bm_filtered = bm_filtered[bm_filtered["prop_type"] == bm_prop_filter]
            bm_filtered = bm_filtered.sort_values("quality_score", ascending=False)

            if bm_filtered.empty:
                st.caption(f"No rows clear quality_score >= {bm_min_quality} - lower the floor to see more.")
            else:
                summary_cols = [c for c in ["player_display_name", "team", "matchup", "position", "prop_type",
                                             "mu", "quality_score", "opponent"] if c in bm_filtered.columns]
                st.dataframe(
                    bm_filtered[summary_cols].style.background_gradient(subset=["quality_score"], cmap="Greens"),
                    width='stretch', hide_index=True,
                )

                st.markdown("**Why did the model pick one of these?**")
                use_full_season_toggle = st.checkbox(
                    "Use full season for this breakdown (more real sample volume)", value=True,
                    key="bm_full_season",
                    help="ON (default): uses every real play from the whole season for the "
                         "coverage/efficiency breakdown below - this is a validation view, not "
                         "the live betting mu itself, so more real volume gives a fuller picture "
                         "with no leakage concern. OFF: shows only the same before-this-week "
                         "data mu itself actually used.",
                )
                player_options = [
                    f"{r['player_display_name']} ({r['prop_type']}, quality={r['quality_score']:.0f})"
                    for _, r in bm_filtered.iterrows()
                ]
                picked = st.selectbox("Pick one to see the breakdown", player_options, key="bm_picked_player")
                picked_row = bm_filtered.iloc[player_options.index(picked)]

                with st.spinner("Pulling this player's real per-coverage sample..."):
                    try:
                        explanation = get_player_matchup_explanation(
                            picked_row["gsis_id"], picked_row["prop_type"], picked_row["team"],
                            picked_row["opponent"], int(season), int(week),
                            use_full_season=use_full_season_toggle,
                        )
                    except Exception as e:
                        explanation = None
                        st.error(f"Couldn't pull the detailed breakdown: {e}")

                if explanation is not None:
                    coverage_mix = explanation.get("coverage_mix", {})
                    player_sample = explanation.get("player_coverage_sample", {})

                    if coverage_mix:
                        # BUGFIX: the Altair version of this chart broke in
                        # production (a Python typing/schema compatibility
                        # error deep inside altair's own import, unrelated
                        # to anything in this file - it fails before our
                        # code even runs). Reverted to a combination of
                        # things ALREADY confirmed working in this exact
                        # deployment all session: matplotlib (the original
                        # pie chart worked fine) + a styled, sortable
                        # dataframe (background_gradient has worked
                        # reliably in every other table all session) -
                        # zero new dependency risk, same "tied together"
                        # goal achieved a different way.
                        rows = []
                        for cov_type, usage_pct in coverage_mix.items():
                            sample = player_sample.get(cov_type)
                            rows.append({
                                "coverage_type": cov_type,
                                "defense_usage_pct": round(usage_pct * 100, 1),
                                "player_ypp_here": sample["ypp"] if sample else None,
                                "real_plays_sampled": sample["n_plays"] if sample else 0,
                                "sample_status": "✅ Reliable" if sample else "⬜ No sample yet",
                            })
                        detail_df = pd.DataFrame(rows).sort_values("defense_usage_pct", ascending=False)

                        dcol1, dcol2 = st.columns([1, 1])
                        with dcol1:
                            st.markdown(f"**{picked_row['opponent']}'s real coverage mix**")
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.pie(coverage_mix.values(), labels=coverage_mix.keys(), autopct="%1.0f%%",
                                   textprops={"fontsize": 8})
                            ax.set_title(f"{picked_row['opponent']} coverage mix", fontsize=9)
                            st.pyplot(fig)
                        with dcol2:
                            st.markdown(
                                f"**Tied directly to {picked_row['player_display_name']}'s real "
                                "efficiency in each one:**"
                            )
                            styled_detail = detail_df.style.background_gradient(
                                subset=["defense_usage_pct"], cmap="Blues"
                            ).background_gradient(
                                subset=["player_ypp_here"], cmap="Greens"
                            )
                            st.dataframe(styled_detail, width='stretch', hide_index=True)
                        st.caption(
                            "✅ Reliable = 8+ real plays behind that coverage's yards/play number, used in "
                            "the weighting. ⬜ No sample yet = the defense runs it, but there's not enough "
                            "real sample yet for this player against it - excluded rather than guessed at."
                        )
                    else:
                        st.caption("No real coverage-mix data available for this defense yet.")

                    no_sample = explanation.get("coverage_types_no_sample", [])
                    if no_sample:
                        st.warning(
                            f"**{picked_row['opponent']} also runs real snaps of {', '.join(no_sample)}, but "
                            f"{picked_row['player_display_name']} doesn't have a reliable sample against "
                            f"{'that' if len(no_sample) == 1 else 'those'} yet — excluded from the weighting "
                            "rather than guessed at."
                        )
                    else:
                        st.success(f"{picked_row['player_display_name']} has a reliable real sample against every "
                                   f"coverage {picked_row['opponent']} meaningfully uses.")

                    st.caption(
                        f"Overall real yards/play across every coverage this season: "
                        f"{explanation.get('player_overall_ypp', 'n/a')} — "
                        "the baseline the coverage-specific numbers above are compared against."
                    )

                    grade_cols_to_show = [c for c in picked_row.index if c.endswith("_grade") and pd.notna(picked_row[c])]
                    if grade_cols_to_show:
                        st.markdown("**Underlying skill grades (0-100 percentile vs the league this season)**")
                        grade_table = pd.DataFrame({
                            "metric": grade_cols_to_show,
                            "grade": [picked_row[c] for c in grade_cols_to_show],
                        }).sort_values("grade", ascending=False)
                        st.dataframe(grade_table, width='stretch', hide_index=True)

            st.markdown("---")
            st.caption("Full scan results (every prop, every position) below.")

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

elif mode == "Coverage Matchup (premium data)":
    st.subheader("Coverage Matchup — Premium FantasyPoints Data")
    st.caption(
        "Season-aggregate Cover 0/1/2/2-Man/3/4/6 splits from the manually-collected "
        "FantasyPoints Data Suite export (separate from the free nflreadpy pipeline the "
        "other three modes run on). Flags each opponent's REAL statistically-outlier "
        "coverage tendencies (z-score vs league, not raw rank), then shows the player's "
        "own history plus what that defense specifically allows against that coverage."
    )

    data_dir = st.text_input(
        "Coverage data folder (relative path in the repo)",
        value=st.session_state.coverage_data_dir or "coverage_data",
        help="Folder containing all 70 CSVs using the established naming convention "
             "(OFF_COVG_.csv, DEF_COVG__.csv, VS_COVER_<N>.csv, def_allowed_cover<N>.csv, "
             "<alignment>_vs_cover<N>.csv, def_<alignment>_cover<N>.csv).",
    )

    load_col1, load_col2 = st.columns([1, 3])
    with load_col1:
        if st.button("Load coverage dataset", type="primary"):
            with st.spinner("Loading coverage matrix + all coverage/alignment files..."):
                try:
                    st.session_state.coverage_bundle = load_full_dataset(data_dir=data_dir)
                    st.session_state.coverage_data_dir = data_dir
                    n_missing = len(st.session_state.coverage_bundle.missing)
                    if n_missing:
                        st.warning(f"Loaded with {n_missing} file(s) missing - see details below.")
                    else:
                        st.success("Loaded all 70 files - dataset complete.")
                except Exception as e:
                    st.error(f"Failed to load coverage dataset: {e}")
                    st.session_state.coverage_bundle = None
    with load_col2:
        if st.session_state.coverage_bundle is not None and st.session_state.coverage_bundle.missing:
            with st.expander(f"{len(st.session_state.coverage_bundle.missing)} file(s) not found - gaps handled gracefully, but listed here"):
                st.write(st.session_state.coverage_bundle.missing)

    bundle = st.session_state.coverage_bundle
    if bundle is None:
        st.info("Load the dataset above before building a matchup report.")
    else:
        st.divider()
        team_names_sorted = sorted(bundle.def_coverage.keys()) or sorted(set(TEAM_ABBREV_TO_FULL.values()))

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            player_name = st.text_input("Player name (exact, as it appears in the export)", value="")
            player_team = st.text_input("Player's own team (optional - abbrev or full name, blocks same-team matchups)", value="")
        with mcol2:
            position = st.selectbox("Position", ["QB", "WR", "TE", "RB"])
            alignment = None
            use_auto_weight = False
            if position != "QB":
                align_mode = st.radio(
                    "Alignment weighting", ["Auto-weight by real usage (recommended)", "Manual - pick one alignment"],
                    help="Every receiver row already carries this player's real season-long "
                         "route% split across Wide/Slot/Inline/Backfield (WIDE RTE %, SLOT RTE %, "
                         "etc.). Auto-weight blends the quality score across every alignment they "
                         "actually play, weighted by that real split - a receiver who's 70% Slot "
                         "won't get graded off his thin 10%-of-routes Wide numbers.",
                )
                if align_mode.startswith("Auto"):
                    use_auto_weight = True
                else:
                    alignment = st.selectbox("Alignment", ["wide", "slot", "inline", "backfield"])
        with mcol3:
            opponent_team = st.selectbox("Opponent (defense)", team_names_sorted)
            top_n_rank = st.number_input(
                "Include coverages ranked in the top N by usage rate (vs all 32 teams)",
                min_value=1, max_value=32, value=10, step=1,
                help="Old behavior only showed a coverage if it was a statistical z-score "
                     "outlier (z>=1.0). This instead includes ANY coverage this defense runs "
                     "often enough to rank in the top N of all 32 teams for that specific "
                     "coverage type - broader, so real heavy-usage tendencies aren't skipped "
                     "just because they weren't a statistical outlier.",
            )


        def _top_n_coverage_fields(bundle, opp_profile, top_n):
            """Shared ranking logic: for each of the 7 coverage fields, where
            does this team rank (1=highest) among all 32 teams' usage rate?
            Returns (field, z_score, rank) for every field where rank<=top_n."""
            all_profiles = list(bundle.def_coverage.values())
            included = []
            for field in COVERAGE_FIELDS:
                ranked = sorted(all_profiles, key=lambda p: p.rates.get(field, 0.0), reverse=True)
                rank = next((i + 1 for i, p in enumerate(ranked) if p.team_name == opp_profile.team_name), None)
                if rank is not None and rank <= top_n:
                    included.append((field, opp_profile.z_scores.get(field, 0.0), rank))
            return included

        def _find_cross_reference_teams(bundle, included_fields, exclude_team, top_n, min_match=2):
            """Which OTHER teams also rank in the top N for at least
            min_match of this opponent's own top-N coverage fields - i.e.
            real games against teams with a similarly heavy lean on the
            same coverage shell(s), used as the best available proxy for
            'games where this player likely saw similar coverage looks'
            since no source tracks real per-play coverage calls."""
            field_names = [f for f, _, _ in included_fields]
            if not field_names:
                return []
            all_profiles = list(bundle.def_coverage.values())
            matches = []
            for field in field_names:
                ranked = sorted(all_profiles, key=lambda p: p.rates.get(field, 0.0), reverse=True)
                top_teams = {p.team_name for p in ranked[:top_n]}
                matches.append(top_teams)
            match_counts = {}
            for team_set in matches:
                for t in team_set:
                    if t == exclude_team:
                        continue
                    match_counts[t] = match_counts.get(t, 0) + 1
            threshold = min(min_match, len(field_names))
            return sorted([t for t, c in match_counts.items() if c >= threshold])

        TIER_WEIGHTS = {"Elite": 100, "Above Avg": 75, "Average": 50, "Below Avg": 25, "Poor": 0}

        def _quality_score(tiers: dict) -> float:
            """0-100 composite: average of every tiered stat's weight. Same
            weighting for coverage cards and game-log cards so the two are
            visually/numerically comparable."""
            vals = [TIER_WEIGHTS[t] for t in tiers.values() if t in TIER_WEIGHTS]
            return round(sum(vals) / len(vals), 1) if vals else None

        # Prop-decision stats only - the ones that actually separate "best
        # prop." TD and longest catch are NOT included: TD isn't a confirmed
        # column anywhere, and longest catch needs per-play pbp data not yet
        # wired (see diagnose_player_stats_for_game_log). Real crucial-stat
        # set here, not a guess: TGT=opportunity, REC=realized volume,
        # YDS=production.
        PROP_STAT_MAP = {"targets": "TGT", "receptions": "REC", "rec_yards": "YDS"}
        GAME_LOG_PROP_MAP = {"targets": "targets", "receptions": "receptions", "rec_yards": "receiving_yards"}
        PROP_LABELS = {"targets": "Targets", "receptions": "Receptions", "rec_yards": "Receiving Yards"}

        def _actual_best_prop(tiers: dict):
            """Same argmax-with-ties logic as _predict_best_prop, applied to a
            REAL game's actual tiers (already computed by
            build_coverage_crossref_game_log against the player's own season
            distribution)."""
            valid = {prop: TIER_WEIGHTS[tiers[col]] for prop, col in GAME_LOG_PROP_MAP.items()
                     if col in tiers and tiers[col] in TIER_WEIGHTS}
            if not valid:
                return None, []
            best = max(valid, key=valid.get)
            ties = [p for p, s in valid.items() if p != best and (valid[best] - s) <= 10]
            return best, ties

        def _predict_best_prop(bundle, player_name, position, opponent_team_full,
                                 alignment=None, weights=None, top_n=10):
            """For each candidate prop, blends its tier-weight across every
            included top-N coverage (and across alignments if weights is
            given - same real-usage blending as the quality score). Returns
            {prop: score or None} plus the argmax and any close ties (within
            10 points - shown as genuine toss-ups, not a false single pick)."""
            opp_profile = bundle.def_coverage.get(opponent_team_full)
            if opp_profile is None:
                return None
            included = _top_n_coverage_fields(bundle, opp_profile, top_n)
            if not included:
                return None

            scores = {}
            for prop, stat_col in PROP_STAT_MAP.items():
                weighted_vals = []
                for field, z, rank in included:
                    if weights:  # auto-weight across real alignments
                        for align, w in weights.items():
                            row = bundle.receiver_by_alignment.get(align, {}).get(field, {}).get(player_name)
                            if row is not None:
                                tier = row.get("_tiers", {}).get(stat_col)
                                if tier in TIER_WEIGHTS:
                                    weighted_vals.append((w, TIER_WEIGHTS[tier]))
                    else:
                        source = bundle.qb_vs_coverage if position.upper() == "QB" else bundle.receiver_by_alignment.get(alignment, {})
                        row = source.get(field, {}).get(player_name)
                        if row is not None:
                            tier = row.get("_tiers", {}).get(stat_col)
                            if tier in TIER_WEIGHTS:
                                weighted_vals.append((1.0, TIER_WEIGHTS[tier]))
                total_w = sum(w for w, _ in weighted_vals)
                scores[prop] = round(sum(w * s for w, s in weighted_vals) / total_w, 1) if total_w else None

            valid = {p: s for p, s in scores.items() if s is not None}
            if not valid:
                return {"scores": scores, "best": None, "ties": []}
            best = max(valid, key=valid.get)
            ties = [p for p, s in valid.items() if p != best and (valid[best] - s) <= 10]
            return {"scores": scores, "best": best, "ties": ties}

        def _score_badge_class(score):
            if score is None:
                return "tier-average"
            if score >= 80:
                return "tier-elite"
            if score >= 60:
                return "tier-above-avg"
            if score >= 40:
                return "tier-average"
            if score >= 20:
                return "tier-below-avg"
            return "tier-poor"

        def _get_real_alignment_weights(bundle, player_name):
            """Real season-long alignment split for this player, read straight
            off any row we can find for them (WIDE/SLOT/INLINE/BACK RTE % -
            already present on every receiver row regardless of which
            alignment file it came from, confirmed earlier this session).
            Returns {alignment: weight 0-1}, normalized to sum to 1, only for
            alignments with real usage (>0). None if the player isn't found
            in ANY alignment file at all."""
            raw = {}
            for align in ALIGNMENTS:
                for cov_data in bundle.receiver_by_alignment.get(align, {}).values():
                    row = cov_data.get(player_name)
                    if row is not None:
                        for a in ALIGNMENTS:
                            pct = check_alignment_fit(row, a)
                            if pct is not None:
                                raw[a] = pct
                        break
                if raw:
                    break
            if not raw:
                return None
            total = sum(raw.values())
            if not total:
                return None
            return {a: v / total for a, v in raw.items() if v > 0}

        def _build_full_coverage_report(bundle, player_name, position, opponent_team,
                                          player_team=None, alignment=None, top_n=10,
                                          use_auto_weight=False):
            """Same shape as get_matchup()'s output, but includes every coverage
            where the opponent ranks in the top N of all 32 teams by usage rate
            for that specific coverage - not just statistical z-score outliers.
            Shows ALL tiered stat columns per row instead of a curated subset.

            use_auto_weight=True (WR/TE/RB only): instead of one manually-
            picked alignment, blends the quality score + stat rows across
            EVERY alignment the player really plays, weighted by their real
            RTE % split (see _get_real_alignment_weights). Alignments with no
            data for a given coverage are dropped and the remaining weights
            re-normalized, rather than silently zero-filling."""
            opp_profile = bundle.def_coverage.get(opponent_team)
            if opp_profile is None:
                return [{"error": f"'{opponent_team}' not found in loaded team coverage data."}], [], None
            if player_team and _same_team(player_team, opp_profile.team_name):
                return [{"error": f"{player_name} plays for {opp_profile.team_name} - "
                                   f"cannot build a matchup report against his own team."}], [], opp_profile

            included = _top_n_coverage_fields(bundle, opp_profile, top_n)

            if not included:
                return [{"note": f"{opp_profile.team_name} has no coverage ranking in the top "
                                  f"{top_n} of all 32 teams - no coverage edge to flag at this threshold."}], [], opp_profile

            if position.upper() == "QB":
                own_data, def_data = bundle.qb_vs_coverage, bundle.def_allowed_to_qb
                weights = None
            elif use_auto_weight:
                weights = _get_real_alignment_weights(bundle, player_name)
                if weights is None:
                    return [{"error": f"'{player_name}' not found in any alignment file - "
                                       f"can't compute a real alignment split."}], [], opp_profile
                own_data = def_data = None  # per-alignment lookups happen in the loop below
            else:
                if alignment is None or alignment.lower() not in bundle.receiver_by_alignment:
                    return [{"error": f"alignment is required for position '{position}' "
                                       f"(one of: wide, slot, inline, backfield)."}], [], opp_profile
                alignment = alignment.lower()
                own_data = bundle.receiver_by_alignment[alignment]
                def_data = bundle.def_allowed_by_alignment[alignment]
                weights = None

            report = []
            for field, z, rank in included:
                entry = {
                    "coverage": field.replace(" %", ""),
                    "opponent_usage_pct": opp_profile.rates.get(field, 0.0),
                    "opponent_z_score": round(z, 2),
                    "opponent_rank": rank,
                }

                if use_auto_weight and position.upper() != "QB":
                    entry["auto_weighted"] = True
                    breakdown = []
                    weighted_scores = []
                    for align in ALIGNMENTS:
                        w = weights.get(align, 0.0)
                        if w <= 0:
                            continue
                        a_own = bundle.receiver_by_alignment.get(align, {}).get(field, {}).get(player_name)
                        a_def = bundle.def_allowed_by_alignment.get(align, {}).get(field, {}).get(opp_profile.team_name)
                        a_qs = _quality_score(a_own.get("_tiers", {})) if a_own is not None else None
                        if a_qs is not None:
                            weighted_scores.append((w, a_qs))
                        breakdown.append({
                            "alignment": align, "weight": w, "own_row": a_own, "defense_allows": a_def,
                            "quality_score": a_qs,
                            "confidence": ("thin_sample" if a_own and a_own.get("_thin_sample") else "solid") if a_own is not None else "no_data",
                            "defense_confidence": ("thin_sample" if a_def and a_def.get("_thin_sample") else "solid") if a_def is not None else "no_data",
                        })
                    breakdown.sort(key=lambda b: -b["weight"])
                    entry["alignment_breakdown"] = breakdown
                    total_w = sum(w for w, _ in weighted_scores)
                    entry["quality_score"] = (
                        round(sum(w * s for w, s in weighted_scores) / total_w, 1) if total_w else None
                    )
                else:
                    own_row = own_data.get(field, {}).get(player_name)
                    own_key = "qb_data" if position.upper() == "QB" else "receiver_data"
                    if own_row is None:
                        entry[own_key] = None
                        entry["confidence"] = "no_data"
                    else:
                        entry[own_key] = own_row
                        entry["confidence"] = "thin_sample" if own_row.get("_thin_sample") else "solid"
                        entry["quality_score"] = _quality_score(own_row.get("_tiers", {}))
                        if position.upper() != "QB":
                            fit = check_alignment_fit(own_row, alignment)
                            entry["alignment_fit_pct"] = fit
                            entry["alignment_fit_warning"] = (fit is not None and fit < 60)

                    def_row = def_data.get(field, {}).get(opp_profile.team_name)
                    entry["defense_allows"] = def_row
                    entry["defense_confidence"] = ("thin_sample" if def_row and def_row.get("_thin_sample")
                                                    else "solid" if def_row else "no_data")
                report.append(entry)
            return report, included, opp_profile

        gl_col1, gl_col2 = st.columns(2)
        with gl_col1:
            game_log_season = st.number_input(
                "Game log season (real weekly logs, free nflreadpy data)",
                min_value=2020, max_value=2030, value=2025, step=1,
            )
        with gl_col2:
            min_match_coverages = st.number_input(
                "Cross-reference: min matching top-N coverages with another team",
                min_value=1, max_value=7, value=2, step=1,
                help="A past opponent counts as a real cross-reference game if THEY "
                     "also ranked in the top-N of all 32 teams for at least this many "
                     "of the SAME coverage types this week's opponent leans on.",
            )

        if st.button("Get matchup report", type="primary"):
            if not player_name.strip():
                st.warning("Enter a player name first.")
            else:
                report, included, opp_profile = _build_full_coverage_report(
                    bundle, player_name.strip(), position, opponent_team,
                    player_team=player_team.strip() or None, alignment=alignment,
                    top_n=int(top_n_rank), use_auto_weight=use_auto_weight,
                )
                st.session_state["_coverage_report"] = report
                st.session_state["_coverage_report_ctx"] = (player_name.strip(), position, opponent_team, alignment)

                st.session_state["_crossref_game_log"] = None
                if opp_profile is not None and included:
                    cross_teams_full = _find_cross_reference_teams(
                        bundle, included, opp_profile.team_name,
                        top_n=int(top_n_rank), min_match=int(min_match_coverages),
                    )
                    full_to_abbrevs = {}
                    for abbr, full in TEAM_ABBREV_TO_FULL.items():
                        full_to_abbrevs.setdefault(full, set()).add(abbr)
                    cross_team_abbrevs = set()
                    for full_name in cross_teams_full:
                        cross_team_abbrevs |= full_to_abbrevs.get(full_name, set())

                    try:
                        pstats = pull_player_stats([int(game_log_season)])
                        sched = pull_schedules([int(game_log_season)])
                        matches = pstats[pstats["position"].astype(str).str.upper() == position.upper()]
                        name_col = "player_display_name" if "player_display_name" in pstats.columns else (
                            "player_name" if "player_name" in pstats.columns else None)
                        target_gsis = None
                        if name_col:
                            hit = matches[matches[name_col].astype(str).str.lower() == player_name.strip().lower()]
                            if not hit.empty:
                                target_gsis = hit.iloc[0]["gsis_id"]
                        if target_gsis is None or not cross_team_abbrevs:
                            st.session_state["_crossref_game_log"] = {
                                "error": (
                                    f"Couldn't match '{player_name}' to a real nflreadpy player record "
                                    f"for {game_log_season}" if target_gsis is None else
                                    "No cross-reference teams found at this top-N / min-match threshold."
                                )
                            }
                        else:
                            log = build_coverage_crossref_game_log(
                                target_gsis, position, cross_team_abbrevs, pstats, sched,
                                seasons=[int(game_log_season)],
                            )
                            st.session_state["_crossref_game_log"] = {
                                "log": log, "cross_teams": sorted(cross_team_abbrevs),
                            }
                    except Exception as e:
                        st.session_state["_crossref_game_log"] = {"error": f"Game log lookup failed: {e}"}

        report = st.session_state.get("_coverage_report")
        if report:
            p_name, p_pos, opp, align = st.session_state["_coverage_report_ctx"]

            if "error" in report[0]:
                st.error(report[0]["error"])
            elif "note" in report[0]:
                st.info(report[0]["note"])
            else:
                st.markdown(f"### {p_name} ({p_pos}{f' - {align}' if align else ''}) vs {opp}")
                st.caption(
                    "Showing every real column from the export, not a curated subset - "
                    "each stat's tier is computed against the actual distribution of "
                    "players/teams who faced that specific coverage."
                )
                own_key = "qb_data" if p_pos == "QB" else "receiver_data"
                own_vol_label = "ATT" if p_pos == "QB" else "TGT"

                TIER_CLASS = {
                    "Elite": "tier-elite", "Above Avg": "tier-above-avg",
                    "Average": "tier-average", "Below Avg": "tier-below-avg",
                    "Poor": "tier-poor",
                }

                # Curated to just what actually decides "which prop is the
                # play" - volume (TGT/REC), efficiency/reliability (CR %),
                # production (YDS), per-route quality (YPRR), and upside
                # (TD). Everything else the export has (RTE, aDOT, AY, TPRR,
                # RecYDS/G, YPT, YPR, YAC, YAC/REC, YACO, YACO/REC, I20,
                # EZTGT, etc.) is still there - see the expander below each
                # block instead of cluttering the default view.
                CURATED_STATS = {
                    "QB": ("CMP %", "YPA", "TD", "INT", "RATE"),
                    "WR": ("TGT", "REC", "CR %", "YDS", "YPRR", "TD"),
                    "TE": ("TGT", "REC", "CR %", "YDS", "YPRR", "TD"),
                    "RB": ("TGT", "REC", "CR %", "YDS", "YPRR", "TD"),
                }

                def _stat_rows_html(row, tiers_source, keys=None):
                    """Renders tiered stat columns. keys=None -> every real
                    column (full detail). keys=<tuple> -> just those, in
                    that order, skipping any not present in this row."""
                    fields = [k for k in keys if k in tiers_source] if keys else list(tiers_source.keys())
                    parts = []
                    for s in fields:
                        tier = tiers_source[s]
                        cls = TIER_CLASS.get(tier, "tier-average")
                        parts.append(
                            f'<div class="stat-row"><span class="stat-label">{s}</span>'
                            f'<span class="stat-value">{row.get(s)}'
                            f'<span class="tier-badge {cls}">{tier}</span></span></div>'
                        )
                    return "".join(parts)

                def _stat_block_html(row, tiers_source):
                    """Curated stats up front, everything else the export has
                    tucked behind a native <details> toggle - nothing hidden,
                    just not cluttering the default view."""
                    curated = CURATED_STATS.get(p_pos, ())
                    curated_html = _stat_rows_html(row, tiers_source, keys=curated)
                    remaining = [k for k in tiers_source if k not in curated]
                    if not remaining:
                        return curated_html
                    more_html = _stat_rows_html(row, tiers_source, keys=remaining)
                    return (
                        curated_html
                        + f'<details class="cov-more"><summary>+{len(remaining)} more stats</summary>{more_html}</details>'
                    )

                for entry in report:
                    z = entry["opponent_z_score"]
                    rank = entry.get("opponent_rank")
                    rank_badge = f'<span class="cov-z-badge">rank {rank} of 32</span>' if rank else ""
                    qs = entry.get("quality_score")
                    qs_badge = (
                        f'<span class="tier-badge {_score_badge_class(qs)}">Quality {qs:.0f}</span>'
                        if qs is not None else ""
                    )
                    fit_html = ""
                    if entry.get("alignment_fit_warning"):
                        fit_html = (
                            f'<div class="cov-fit-warning">⚠️ Only {entry["alignment_fit_pct"]:.0f}% '
                            f'of {p_name}\'s routes are {align} - this alignment split may not '
                            f'represent his usual usage.</div>'
                        )

                    if entry.get("auto_weighted"):
                        # One mini-block per alignment the player actually plays,
                        # each with its own real weight%, quality score, and stat
                        # rows - not a single averaged-away number.
                        blocks = []
                        for b in entry["alignment_breakdown"]:
                            b_qs = b["quality_score"]
                            b_qs_badge = (
                                f'<span class="tier-badge {_score_badge_class(b_qs)}">Quality {b_qs:.0f}</span>'
                                if b_qs is not None else ""
                            )
                            if b["own_row"] is None:
                                own_part = f'<div class="cov-no-data">no recorded {own_vol_label.lower()}s vs this coverage.</div>'
                            else:
                                thin = '<span class="cov-thin-flag">THIN SAMPLE</span>' if b["confidence"] == "thin_sample" else ""
                                own_part = (
                                    f'<div class="cov-col-title">{p_name} — {b["own_row"]["_att"]} {own_vol_label}{thin}</div>'
                                    + _stat_block_html(b["own_row"], b["own_row"].get("_tiers", {}))
                                )
                            if b["defense_allows"] is None:
                                def_part = f'<div class="cov-no-data">{opp}: no defense-allowed data for this alignment.</div>'
                            else:
                                thin = '<span class="cov-thin-flag">THIN SAMPLE</span>' if b["defense_confidence"] == "thin_sample" else ""
                                def_part = (
                                    f'<div class="cov-col-title">{opp} allows — {b["defense_allows"]["_att"]} {own_vol_label}{thin}</div>'
                                    + _stat_block_html(b["defense_allows"], b["defense_allows"].get("_tiers", {}))
                                )
                            blocks.append(
                                f'<div class="cov-align-block">'
                                f'<div class="cov-align-header">{b["alignment"].upper()} '
                                f'— {b["weight"]*100:.0f}% of real routes{b_qs_badge}</div>'
                                f'<div class="cov-grid">'
                                f'<div class="cov-col">{own_part}</div>'
                                f'<div class="cov-col">{def_part}</div>'
                                f'</div></div>'
                            )
                        grid_html = "".join(blocks)
                    else:
                        own_row = entry.get(own_key)
                        if own_row is None:
                            own_col_html = f'<div class="cov-no-data">{p_name}: no recorded {own_vol_label.lower()}s vs this coverage.</div>'
                        else:
                            thin = '<span class="cov-thin-flag">THIN SAMPLE</span>' if entry["confidence"] == "thin_sample" else ""
                            own_col_html = (
                                f'<div class="cov-col-title">{p_name} — {own_row["_att"]} {own_vol_label}{thin}</div>'
                                + _stat_block_html(own_row, own_row.get("_tiers", {}))
                            )

                        def_row = entry.get("defense_allows")
                        if def_row is None:
                            def_col_html = f'<div class="cov-no-data">{opp}: no defense-allowed data vs this coverage.</div>'
                        else:
                            thin = '<span class="cov-thin-flag">THIN SAMPLE</span>' if entry.get("defense_confidence") == "thin_sample" else ""
                            def_col_html = (
                                f'<div class="cov-col-title">{opp} allows — {def_row["_att"]} {own_vol_label}{thin}</div>'
                                + _stat_block_html(def_row, def_row.get("_tiers", {}))
                            )
                        grid_html = (
                            '<div class="cov-grid">'
                            f'<div class="cov-col">{own_col_html}</div>'
                            f'<div class="cov-col">{def_col_html}</div>'
                            '</div>'
                        )

                    # Built as ONE continuous line (no embedded newlines/indentation) -
                    # a multi-line indented f-string here previously broke rendering:
                    # when fit_html was empty, it left a whitespace-only line, which
                    # Markdown reads as "the HTML block just ended" - everything after
                    # that point then got swept up by Markdown's indented-code-block
                    # rule and printed as literal text instead of rendering as HTML.
                    card_html = (
                        '<div class="cov-card">'
                        f'<div class="cov-card-header">{entry["coverage"]}'
                        f'<span class="cov-z-badge">z={z:+.2f}</span>{rank_badge}{qs_badge}</div>'
                        f'<div class="cov-card-usage">{opp} runs this coverage at '
                        f'{entry["opponent_usage_pct"]:.1f}% of snaps</div>'
                        f'{fit_html}'
                        f'{grid_html}'
                        '</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                st.divider()
                st.markdown("### Best Prop Verdict")
                is_auto = bool(report[0].get("auto_weighted")) if report and isinstance(report[0], dict) else False
                verdict_weights = _get_real_alignment_weights(bundle, p_name) if (is_auto and p_pos != "QB") else None
                verdict_alignment = None if is_auto else align
                verdict = _predict_best_prop(
                    bundle, p_name, p_pos, opp, alignment=verdict_alignment,
                    weights=verdict_weights, top_n=int(top_n_rank),
                )
                if not verdict or verdict["best"] is None:
                    st.info("Not enough data across these coverages to determine a best prop.")
                else:
                    best = verdict["best"]
                    ties = verdict["ties"]
                    rows_html = "".join(
                        f'<div class="stat-row"><span class="stat-label">{PROP_LABELS[p]}</span>'
                        f'<span class="stat-value">{verdict["scores"][p] if verdict["scores"][p] is not None else "no data"}'
                        f'<span class="tier-badge {_score_badge_class(verdict["scores"][p])}">'
                        f'{"BEST" if p == best else ("TIE" if p in ties else "")}</span></span></div>'
                        for p in PROP_LABELS
                    )
                    tie_note = f" (essentially tied with {', '.join(PROP_LABELS[t] for t in ties)})" if ties else ""
                    verdict_html = (
                        '<div class="cov-card">'
                        f'<div class="cov-card-header">Best Prop: {PROP_LABELS[best]}{tie_note}</div>'
                        f'<div class="cov-card-usage">Based on {p_name}\'s season splits vs '
                        f'{opp}\'s top-{int(top_n_rank)} coverages</div>'
                        f'<div class="cov-col">{rows_html}</div>'
                        '</div>'
                    )
                    st.markdown(verdict_html, unsafe_allow_html=True)
                st.caption(
                    "TD props and longest catch aren't included yet - not confirmed columns "
                    "anywhere in the pipeline. Only targets/receptions/rec yards, which are."
                )

                st.divider()
                def _run_season_backtest(bundle, p_name, p_pos, verdict_alignment, verdict_weights,
                                           game_log_season, top_n):
                    """One full-season backtest run at a given top_n threshold. Pulled
                    out as its own function so both the single-run button and the
                    threshold sweep call the exact same logic - a sweep that secretly
                    used different code per threshold would make the comparison
                    meaningless."""
                    bt_pstats = pull_player_stats([int(game_log_season)])
                    bt_sched = pull_schedules([int(game_log_season)])
                    bt_matches = bt_pstats[bt_pstats["position"].astype(str).str.upper() == p_pos.upper()]
                    bt_name_col = "player_display_name" if "player_display_name" in bt_pstats.columns else (
                        "player_name" if "player_name" in bt_pstats.columns else None)
                    bt_gsis = None
                    if bt_name_col:
                        bt_hit = bt_matches[bt_matches[bt_name_col].astype(str).str.lower() == p_name.lower()]
                        if not bt_hit.empty:
                            bt_gsis = bt_hit.iloc[0]["gsis_id"]
                    if bt_gsis is None:
                        return {"error": f"Couldn't match '{p_name}' to a real nflreadpy player record for {game_log_season}."}

                    all_abbrevs = set(TEAM_ABBREV_TO_FULL.keys())
                    full_log = build_coverage_crossref_game_log(
                        bt_gsis, p_pos, all_abbrevs, bt_pstats, bt_sched,
                        seasons=[int(game_log_season)], max_games=25,
                    )
                    rows, strict_hits, generous_hits, graded = [], 0, 0, 0
                    for g in full_log:
                        opp_full = TEAM_ABBREV_TO_FULL.get(g["opponent"])
                        pred = (_predict_best_prop(bundle, p_name, p_pos, opp_full,
                                                    alignment=verdict_alignment, weights=verdict_weights,
                                                    top_n=top_n) if opp_full else None)
                        pred_best = pred["best"] if pred else None
                        actual_best, actual_ties = _actual_best_prop(g.get("tiers", {}))
                        result = "—"
                        if pred_best is not None and actual_best is not None:
                            graded += 1
                            if pred_best == actual_best:
                                strict_hits += 1
                                generous_hits += 1
                                result = "✅ Hit"
                            elif pred_best in actual_ties:
                                generous_hits += 1
                                result = "〰️ Tie"
                            else:
                                result = "❌ Miss"
                        rows.append({
                            "Week": g["week"], "Opponent": g["opponent"],
                            "Predicted Best": PROP_LABELS.get(pred_best, "no data"),
                            "Actual Best": PROP_LABELS.get(actual_best, "no data"),
                            "Result": result,
                        })
                    return {"rows": rows, "strict_hits": strict_hits, "generous_hits": generous_hits, "graded": graded}

                st.markdown("### Backtest This Method — Full Season")
                st.caption(
                    "Checks, for every real game this player played, whether the prop this "
                    "method would've picked (using the season coverage splits vs that week's "
                    "real opponent) actually was the best-performing prop that week (graded "
                    "against the player's own real season average). "
                    "⚠️ Real limitation, stated plainly: the coverage splits are season-aggregate, "
                    "so a prediction for week 4 technically has access to data through week 18 - "
                    "a real look-ahead bias. This tests whether the underlying signal correlates "
                    "at all, not a clean pre-game prediction."
                )
                bt_col1, bt_col2 = st.columns(2)
                with bt_col1:
                    run_single = st.button("Run season backtest (current top-N)", type="secondary")
                with bt_col2:
                    run_sweep = st.button("Sweep top-N thresholds (find the best N)", type="secondary")

                if run_single:
                    try:
                        st.session_state["_prop_backtest"] = _run_season_backtest(
                            bundle, p_name, p_pos, verdict_alignment, verdict_weights,
                            game_log_season, int(top_n_rank),
                        )
                    except Exception as e:
                        st.session_state["_prop_backtest"] = {"error": f"Backtest failed: {e}"}

                if run_sweep:
                    try:
                        candidates = [3, 5, 8, 10, 12, 15, 20, 25, 32]
                        sweep_rows = []
                        for n in candidates:
                            result = _run_season_backtest(
                                bundle, p_name, p_pos, verdict_alignment, verdict_weights,
                                game_log_season, n,
                            )
                            if result.get("error"):
                                st.session_state["_prop_sweep"] = {"error": result["error"]}
                                break
                            graded = result["graded"]
                            sweep_rows.append({
                                "Top-N": n,
                                "Strict Hit Rate": f"{result['strict_hits']}/{graded} ({result['strict_hits']/graded*100:.0f}%)" if graded else "no data",
                                "Including Near-Ties": f"{result['generous_hits']}/{graded} ({result['generous_hits']/graded*100:.0f}%)" if graded else "no data",
                                "_strict_pct": (result['strict_hits'] / graded * 100) if graded else -1,
                            })
                        else:
                            st.session_state["_prop_sweep"] = {"rows": sweep_rows}
                    except Exception as e:
                        st.session_state["_prop_sweep"] = {"error": f"Sweep failed: {e}"}

                sweep = st.session_state.get("_prop_sweep")
                if sweep:
                    if sweep.get("error"):
                        st.warning(sweep["error"])
                    else:
                        rows = sweep["rows"]
                        best_row = max(rows, key=lambda r: r["_strict_pct"]) if any(r["_strict_pct"] >= 0 for r in rows) else None
                        if best_row:
                            st.markdown(
                                f"**Best-performing threshold for {p_name} vs {game_log_season}: "
                                f"top-{best_row['Top-N']}** (strict hit rate {best_row['Strict Hit Rate']}) — "
                                f"current setting is top-{int(top_n_rank)}."
                            )
                        display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
                        st.dataframe(pd.DataFrame(display_rows), width='stretch')
                        st.caption(
                            "Read this as a rough signal, not a precise optimum - each threshold is graded "
                            "on the SAME small set of real games for one player/season, so differences of a "
                            "game or two easily swing the percentage. Worth re-running for a few different "
                            "players before locking in a threshold across the whole tool."
                        )

                bt = st.session_state.get("_prop_backtest")
                if bt:
                    if bt.get("error"):
                        st.warning(bt["error"])
                    elif bt["graded"] == 0:
                        st.info("No graded weeks - not enough real data to backtest this player/season.")
                    else:
                        sr = bt["strict_hits"] / bt["graded"] * 100
                        gr = bt["generous_hits"] / bt["graded"] * 100
                        st.markdown(
                            f"**Strict hit rate:** {bt['strict_hits']}/{bt['graded']} ({sr:.0f}%) "
                            f"&nbsp;&nbsp;|&nbsp;&nbsp; **Including near-ties:** {bt['generous_hits']}/{bt['graded']} ({gr:.0f}%)"
                        )
                        st.dataframe(pd.DataFrame(bt["rows"]), width='stretch')

                st.divider()
                st.markdown("### Real Weekly Game Log — Cross-Referenced Opponents")
                st.caption(
                    "This is an APPROXIMATION, not verified per-play coverage tracking - no free "
                    "or paid source tracks real per-play coverage calls. These are real weekly "
                    "game logs (nflreadpy) against teams that were ALSO top-N users of the same "
                    "coverage shell(s) as this week's opponent, used as the best available proxy "
                    "for 'games where this player likely saw a similar coverage look.'"
                )
                crossref = st.session_state.get("_crossref_game_log")
                if crossref is None:
                    st.info("No game log loaded - click 'Get matchup report' above to build it.")
                elif crossref.get("error"):
                    st.warning(crossref["error"])
                else:
                    log = crossref.get("log", [])
                    st.caption(f"Cross-reference teams at this threshold: {', '.join(crossref.get('cross_teams', [])) or 'none'}")
                    if not log:
                        st.info("No real games found against a cross-reference team at this threshold/season.")
                    for g in log:
                        tiers = g.get("tiers", {})
                        stats = g.get("stats", {})
                        gqs = _quality_score(tiers)
                        gqs_badge = (
                            f'<span class="tier-badge {_score_badge_class(gqs)}">Quality {gqs:.0f}</span>'
                            if gqs is not None else ""
                        )
                        rows_html = "".join(
                            f'<div class="stat-row"><span class="stat-label">{s}</span>'
                            f'<span class="stat-value">{stats.get(s)}'
                            f'<span class="tier-badge {TIER_CLASS.get(tiers.get(s), "tier-average")}">{tiers.get(s, "-")}</span>'
                            f'</span></div>'
                            for s in stats
                        )
                        note_html = (
                            f'<div class="cov-fit-warning">{g["sample_size_note"]}</div>'
                            if g.get("sample_size_note") else ""
                        )
                        game_card_html = (
                            '<div class="cov-card">'
                            f'<div class="cov-card-header">{g["season"]} Week {g["week"]} '
                            f'— {g["team"]} vs {g["opponent"]}{gqs_badge}</div>'
                            f'{note_html}'
                            f'<div class="cov-col">{rows_html}</div>'
                            '</div>'
                        )
                        st.markdown(game_card_html, unsafe_allow_html=True)

elif mode not in ("Draft Rankings", "Coverage Matchup (premium data)"):
    st.info("Click the button above to load this week's props.")
