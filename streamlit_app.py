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
    diagnose_alignment_data, pull_player_stats, pull_schedules, pull_pbp,
    build_coverage_crossref_game_log, diagnose_player_stats_for_game_log,
    build_longest_play_by_game,
)
from draft_rankings import (
    build_yahoo_style_rankings, detect_risers, build_league_settings,
    build_snake_draft_targets, compute_blended_rankings, build_draft_rankings_backtest,
)
from coverage_matchup import (
    load_full_dataset, get_matchup, TEAM_ABBREV_TO_FULL, COVERAGE_FIELDS,
    _same_team, check_alignment_fit, ALIGNMENT_RTE_COLUMNS, ALIGNMENTS,
)
from rb_matchup import (
    load_full_rb_dataset, get_rb_matchup, CONCEPT_FILES as RB_CONCEPT_FILES,
    CRUCIAL_RB_STATS,
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
    /* BUG FIX: the rule above (needed to fix white-on-white report text)
       was ALSO forcing navy text inside text inputs/text areas - but
       those widgets keep Streamlit's own dark input background, so navy
       text on a dark background was nearly unreadable (this is the
       "dark, hard to read" bug reported when typing multiple player
       names into a text area). Inputs get their own explicit light
       color here, scoped narrowly so it doesn't leak back into report
       text. -webkit-text-fill-color covers a real Chrome/Safari quirk
       where autofill/theme styling can override plain color. */
    input, textarea,
    .stTextArea textarea, .stTextInput input, .stNumberInput input,
    [data-testid="stTextAreaContainer"] textarea,
    [data-testid="stTextInputRootElement"] input {{
        color: {COWBOYS_WHITE} !important;
        -webkit-text-fill-color: {COWBOYS_WHITE} !important;
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
if "rb_bundle" not in st.session_state:
    st.session_state.rb_bundle = None
if "rb_data_dir" not in st.session_state:
    st.session_state.rb_data_dir = None
if "rb_player_dir" not in st.session_state:
    st.session_state.rb_player_dir = None
if "rb_def_dir" not in st.session_state:
    st.session_state.rb_def_dir = None

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

        def _find_extreme_low_usage_fields(bundle, opp_profile, position, alignment, weights, already_fields):
            """A pure usage-rate cutoff misses this real case: a defense that
            rarely runs Cover 4 (doesn't crack the top-N) but is genuinely
            elite or terrible against this player's alignment the rare times
            it does. Checks every coverage NOT already included via top-N,
            and adds it (rank=None, flagged low_usage_extreme) if the real
            defense-allowed data for this player's alignment(s) is Elite or
            Poor tier AND not itself a thin sample - an extreme reading on a
            genuinely unreliable small sample isn't worth surfacing, but a
            real extreme on a real sample is exactly the gap this closes."""
            extra = []
            for field in COVERAGE_FIELDS:
                if field in already_fields:
                    continue
                if position.upper() == "QB":
                    rows_to_check = [bundle.def_allowed_to_qb.get(field, {}).get(opp_profile.team_name)]
                elif weights:
                    rows_to_check = [
                        bundle.def_allowed_by_alignment.get(a, {}).get(field, {}).get(opp_profile.team_name)
                        for a, w in weights.items() if w > 0
                    ]
                else:
                    rows_to_check = [bundle.def_allowed_by_alignment.get(alignment, {}).get(field, {}).get(opp_profile.team_name)]

                for row in rows_to_check:
                    if row is None or row.get("_thin_sample"):
                        continue
                    qs = _quality_score(row.get("_tiers", {}), position=None)
                    if qs is not None and (qs >= 80 or qs <= 20):
                        extra.append((field, opp_profile.z_scores.get(field, 0.0), None))
                        break
            return extra

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

        # Stats that actually decide prop quality get 2x weight in the score;
        # everything else (RTE%, AY, TPRR, RecYDS/G, YPT, YPR, YAC, I20,
        # EZTGT, FP/DB, FP/OPP, FP/G, etc.) still counts, just at 1x -
        # included, not silently dropped, but no longer diluting the crucial
        # signal down to the same weight as everything else. aDOT and the
        # YAC splits are in here even though they're not in the default
        # curated DISPLAY set (CURATED_STATS below) - display and scoring
        # weight are two separate decisions.
        CRUCIAL_QUALITY_STATS = {
            "QB": {"CMP %", "YPA", "TD", "INT", "RATE", "CPOE", "ATT", "CMP", "YDS",
                   "SACK %", "PRESS %", "Deep Throw %", "ACC %", "ADJ CMP %", "1Read %",
                   "CHK %", "ANY/A", "EZATT"},
            "WR": {"TGT", "REC", "CR %", "YDS", "YPRR", "TD", "aDOT", "YAC/REC", "YACO/REC",
                   "TPRR", "YAC", "YPT", "YPR", "TGT %", "AY Share", "1READ %", "CTGT %",
                   "CC %", "DRP %", "YPTOE", "EZTGT", "i20", "EZTD"},
            "TE": {"TGT", "REC", "CR %", "YDS", "YPRR", "TD", "aDOT", "YAC/REC", "YACO/REC",
                   "TPRR", "YAC", "YPT", "YPR", "TGT %", "AY Share", "1READ %", "CTGT %",
                   "CC %", "DRP %", "YPTOE", "EZTGT", "i20", "EZTD"},
            "RB": {"TGT", "REC", "CR %", "YDS", "YPRR", "TD", "aDOT", "YAC/REC", "YACO/REC",
                   "TPRR", "YAC", "YPT", "YPR", "TGT %", "AY Share", "1READ %", "CTGT %",
                   "CC %", "DRP %", "YPTOE", "EZTGT", "i20", "EZTD"},
            # Separate key for RB RUSHING (run-concept section) - real bug
            # this fixes: "RB" above is the receiving-side set (RB as a
            # pass-catcher in the coverage/alignment section), which uses
            # completely different column names (TGT/REC/CR%) than the
            # rushing concept data (ATT/YPC/Success%/etc). Both used to
            # share the same "RB" key by coincidence of the dict having a
            # single per-position entry - meaning the rushing Quality
            # Score was only ever double-weighting YDS and TD (the only
            # two names that happen to overlap), with every other real
            # rushing crucial stat sitting at flat 1x weight the whole
            # time. Reuses rb_matchup.py's CRUCIAL_RB_STATS directly so
            # there's exactly one real list to maintain, not two that can
            # drift out of sync with each other.
            "RB_RUSH": CRUCIAL_RB_STATS,
        }

        def _quality_score(tiers: dict, position: str = None, thin_sample: bool = False) -> float:
            """0-100 composite, weighted average of every tiered stat.
            Crucial stats (see CRUCIAL_QUALITY_STATS) count 2x; everything
            else counts 1x - real signal isn't averaged away by ~45 minor
            columns. position=None (used by the game-log cards, whose stat
            names don't match this map at all) falls back to a flat 1x
            average, unchanged from before.

            thin_sample=True shrinks the raw score halfway toward neutral
            (50) - a score built on 5 real targets shouldn't display with
            the same visual confidence as one built on 50. Same shrinkage
            philosophy already used elsewhere in this project's mu
            calculations (calc_prop_mu blends thin samples toward a league
            fallback rather than trusting them at full weight), applied
            here to the display score instead of a projection number."""
            crucial = CRUCIAL_QUALITY_STATS.get(position, set()) if position else set()
            weighted = [
                (2.0 if stat in crucial else 1.0, TIER_WEIGHTS[tier])
                for stat, tier in tiers.items() if tier in TIER_WEIGHTS
            ]
            if not weighted:
                return None
            total_w = sum(w for w, _ in weighted)
            raw = sum(w * s for w, s in weighted) / total_w
            if thin_sample:
                raw = raw * 0.5 + 50 * 0.5
            return round(raw, 1)

        # Prop-decision stats only - the ones that actually separate "best
        # prop." TD and longest catch are NOT included: TD isn't a confirmed
        # column anywhere, and longest catch needs per-play pbp data not yet
        # wired (see diagnose_player_stats_for_game_log). Real crucial-stat
        # set here, not a guess: TGT=opportunity, REC=realized volume,
        # YDS=production. receiving_td added on BOTH sides (TD is a real
        # confirmed column in the CSV predicted data - already tiered and
        # shown on every coverage card - and receiving_tds is a confirmed
        # real nflreadpy column, used elsewhere in this codebase for
        # fantasy point math) - safe to add directly, unlike longest_catch
        # below which only has an actual-side column and would corrupt
        # the backtest if treated the same way.
        PROP_STAT_MAP = {"targets": "TGT", "receptions": "REC", "rec_yards": "YDS", "receiving_td": "TD"}
        GAME_LOG_PROP_MAP = {"targets": "targets", "receptions": "receptions", "rec_yards": "receiving_yards",
                              "receiving_td": "receiving_tds"}
        PROP_LABELS = {"targets": "Targets", "receptions": "Receptions", "rec_yards": "Receiving Yards",
                        "receiving_td": "Receiving TD"}
        # longest_catch deliberately NOT in GAME_LOG_PROP_MAP/PROP_STAT_MAP -
        # the backtest compares predicted-best vs actual-best, and there's
        # no CSV column to PREDICT longest catch from at all. Adding it to
        # GAME_LOG_PROP_MAP would let it win "actual best" some weeks while
        # the predicted side could never pick it - every one of those weeks
        # would silently register as a false miss and drag the hit rate
        # down for no real reason. Kept in its own map instead, used only
        # by the real-line comparison below (which needs real mu/sigma,
        # not a predicted-vs-actual comparison).
        LINE_COMPARE_PROP_MAP = dict(GAME_LOG_PROP_MAP, longest_catch="longest_play")
        LINE_COMPARE_PROP_LABELS = dict(PROP_LABELS, longest_catch="Longest Catch")

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
            already_fields = {f for f, _, _ in included}
            included = included + _find_extreme_low_usage_fields(
                bundle, opp_profile, position, alignment, weights, already_fields,
            )
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

        def _apply_best_signal_adjustment(mu, quality_score, is_thin, max_pct=0.12):
            """Only nudges mu when the premium matchup signal is genuinely
            strong - not any signal, the BEST ones. Requires quality_score
            to be at least 25 points from neutral (50) - i.e. real Above
            Avg/Elite or Below Avg/Poor territory, not Average - AND not a
            thin sample. Anything softer than that leaves mu untouched
            rather than nudging on a mediocre or unreliable read.
            Adjustment scales linearly from 0% (at the 25-point threshold)
            up to max_pct (at the extremes, quality_score=0 or 100), so a
            marginal-but-qualifying signal moves mu less than a truly
            extreme one. Returns (adjusted_mu, pct_applied, applied: bool)."""
            if quality_score is None or is_thin:
                return mu, 0.0, False
            distance = quality_score - 50  # -50..+50
            if abs(distance) < 25:
                return mu, 0.0, False
            # scale from 25->0% up to 50->max_pct
            pct = max_pct * (abs(distance) - 25) / 25
            pct = pct if distance > 0 else -pct
            return round(mu * (1 + pct), 2), round(pct * 100, 1), True

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

        def _run_mu_source_comparison_backtest(bundle, p_name, p_pos, alignment, weights,
                                                  game_log_season, top_n, prop, line):
            """
            MLB-format walk-forward backtest, real fixed line, ONE real prop
            at a time - the direct test, not the "which of N props wins"
            relative ranking. Compares THREE ways of computing mu, head to
            head, same real games, same real line:

              raw: flat trailing average of the player's own real games -
              no matchup awareness at all. The control.

              adjusted: raw mu nudged by the SAME Premium Adjustment logic
              already used in Line Value (_predict_best_prop +
              _apply_best_signal_adjustment) - only when the predicted
              quality for THAT WEEK'S REAL opponent is genuinely strong.
              Already includes both "grade both sides" (crucial-weighted
              Quality Score) and "every heavy coverage, not just one"
              (top-N inclusion) - reused, not rebuilt.

              crossref: trailing average computed ONLY from real games
              against teams that graded similarly (same coverage-lean
              signature) to that week's real opponent - reuses
              _find_cross_reference_teams directly.

            No look-ahead in any of the three - only games strictly BEFORE
            the one being graded feed that game's mu. Honest simplification
            noted, not hidden: the adjusted-mu thin-sample check here is
            simpler than the Line Value tool's (doesn't re-derive the full
            report), so it's a slightly looser filter than that tool uses -
            worth tightening in a later pass if this proves out.
            """
            pstats = pull_player_stats([int(game_log_season)])
            sched = pull_schedules([int(game_log_season)])
            matches = pstats[pstats["position"].astype(str).str.upper() == p_pos.upper()]
            name_col = "player_display_name" if "player_display_name" in pstats.columns else (
                "player_name" if "player_name" in pstats.columns else None)
            gsis = None
            if name_col:
                hit = matches[matches[name_col].astype(str).str.lower() == p_name.lower()]
                if not hit.empty:
                    gsis = hit.iloc[0]["gsis_id"]
            if gsis is None:
                return {"error": f"Couldn't match '{p_name}' to a real nflreadpy record for {game_log_season}."}

            stat_col = GAME_LOG_PROP_MAP.get(prop)
            if stat_col is None:
                return {"error": f"'{prop}' isn't a real receiving prop this tool tracks."}

            all_abbrevs = set(TEAM_ABBREV_TO_FULL.keys())
            full_log = build_coverage_crossref_game_log(
                gsis, p_pos, all_abbrevs, pstats, sched, seasons=[int(game_log_season)], max_games=25,
            )
            # full_log is sorted most-recent-first (see build_coverage_crossref_game_log) -
            # "games strictly before" game i means every game AFTER it in this list.
            full_to_abbrevs = {}
            for abbr, full in TEAM_ABBREV_TO_FULL.items():
                full_to_abbrevs.setdefault(full, set()).add(abbr)

            rows = []
            for i, g in enumerate(full_log):
                prior_games = full_log[i + 1:]
                prior_vals = [pg["stats"].get(stat_col) for pg in prior_games if stat_col in pg.get("stats", {})]
                prior_vals = [v for v in prior_vals if v is not None]
                if len(prior_vals) < 3:
                    continue
                raw_mu = sum(prior_vals) / len(prior_vals)

                opp_full = TEAM_ABBREV_TO_FULL.get(g["opponent"])
                adjusted_mu = raw_mu
                q_score = None
                if opp_full and prop in PROP_STAT_MAP:
                    pred = _predict_best_prop(bundle, p_name, p_pos, opp_full,
                                                alignment=alignment, weights=weights, top_n=top_n)
                    q_score = pred["scores"].get(prop) if pred else None
                    adjusted_mu, _, _ = _apply_best_signal_adjustment(raw_mu, q_score, is_thin=False)

                crossref_mu = None
                cr_sample_size = 0
                if opp_full:
                    opp_profile = bundle.def_coverage.get(opp_full)
                    if opp_profile:
                        included = _top_n_coverage_fields(bundle, opp_profile, top_n)
                        cross_teams_full = _find_cross_reference_teams(bundle, included, opp_full, top_n, min_match=2)
                        cross_abbrevs = set()
                        for full_name in cross_teams_full:
                            cross_abbrevs |= full_to_abbrevs.get(full_name, set())
                        cr_vals = [pg["stats"].get(stat_col) for pg in prior_games
                                   if stat_col in pg.get("stats", {}) and pg["opponent"] in cross_abbrevs]
                        cr_vals = [v for v in cr_vals if v is not None]
                        cr_sample_size = len(cr_vals)
                        if cr_sample_size >= 3:
                            crossref_mu = sum(cr_vals) / cr_sample_size

                actual_value = g["stats"].get(stat_col)
                if actual_value is None:
                    continue

                row = {"Week": g["week"], "Opponent": g["opponent"], "Actual": actual_value,
                       "Quality Score": round(q_score, 1) if q_score is not None else "no data",
                       "CrossRef Sample": cr_sample_size}
                for label, mu_val in [("Raw", raw_mu), ("Adjusted", adjusted_mu), ("CrossRef", crossref_mu)]:
                    if mu_val is None:
                        row[f"{label} mu"] = "no data"
                        row[f"{label} Hit"] = None
                        continue
                    predicted_over = mu_val > line
                    actual_over = actual_value > line
                    row[f"{label} mu"] = round(mu_val, 2)
                    row[f"{label} Hit"] = predicted_over == actual_over

                # The real distinction the user is drawing: mu-vs-line is a
                # STATISTICAL trend (does his trailing average sit above or
                # below this number), Quality Score is a MATCHUP GRADE (is
                # this a favorable or unfavorable coverage/scheme fit for
                # him specifically) - two genuinely different signals that
                # were being blended into one "Adjusted mu" number without
                # ever showing whether they actually agree. Quality Lean:
                # >50 means the matchup grade itself leans favorable (more
                # OVER-supportive conditions), <50 leans unfavorable.
                # Mu Lean uses RAW mu specifically (the pure statistical
                # trend, unadjusted) so the comparison isn't circular -
                # comparing Quality against a mu that Quality already
                # nudged would trivially "agree" by construction.
                mu_lean = "OVER" if raw_mu > line else "UNDER"
                if q_score is None:
                    quality_lean = "no data"
                    agreement = "N/A"
                else:
                    quality_lean = "Favorable" if q_score > 50 else "Unfavorable"
                    mu_wants_over = mu_lean == "OVER"
                    quality_wants_over = quality_lean == "Favorable"
                    agreement = "Agree" if mu_wants_over == quality_wants_over else "Conflict"
                row["Mu Lean"] = mu_lean
                row["Quality Lean"] = quality_lean
                row["Agreement"] = agreement
                rows.append(row)

            return {"rows": rows}

        def _run_season_backtest(bundle, p_name, p_pos, verdict_alignment, verdict_weights,
                                   game_log_season, top_n):
            """One full-season backtest run at a given top_n threshold, for
            WR/TE/RB receiving props (targets/receptions/rec_yards/
            receiving_td). Moved to this outer scope (was originally nested
            inside the single-matchup report block) so it's usable WITHOUT
            requiring a player name/report to exist first - needed for the
            standalone league-wide scan below, which doesn't ask for a name
            at all. Pulled out as its own function so the single-run
            button, the threshold sweep, and the league-wide scan all call
            the exact same logic - having any of them secretly use
            different code would make the comparisons meaningless."""
            bt_pstats = pull_player_stats([int(game_log_season)])
            bt_sched = pull_schedules([int(game_log_season)])
            bt_pbp = pull_pbp([int(game_log_season)])
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
                seasons=[int(game_log_season)], max_games=25, pbp_df=bt_pbp,
            )
            rows, strict_hits, generous_hits, graded = [], 0, 0, 0
            for g in full_log:
                opp_full = TEAM_ABBREV_TO_FULL.get(g["opponent"])
                pred = (_predict_best_prop(bundle, p_name, p_pos, opp_full,
                                            alignment=verdict_alignment, weights=verdict_weights,
                                            top_n=top_n) if opp_full else None)
                pred_best = pred["best"] if pred else None
                # How strong was THIS week's prediction, not just which prop won -
                # a prop that barely edged out the others (score near 50, or tied
                # with another prop) is a much weaker call than one that scored
                # 85+ standalone. Kept per-week so the auto-scan below can filter
                # to genuinely strong weeks instead of grading every call equally.
                pred_quality = pred["scores"].get(pred_best) if (pred and pred_best) else None
                pred_has_tie = bool(pred["ties"]) if pred else False
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
                    "_pred_quality": pred_quality, "_pred_has_tie": pred_has_tie,
                })
            return {"rows": rows, "strict_hits": strict_hits, "generous_hits": generous_hits, "graded": graded}

        # ---- QB equivalent of everything above - QB has genuinely different
        # props (Pass Attempts/Completions/Yards/TD, not Targets/Receptions),
        # confirmed real on both the CSV predicted side (ATT/CMP/YDS/TD) and
        # the nflreadpy actual side (attempts/completions/passing_yards/
        # passing_tds) during the QB Line Value work earlier. QB has no
        # alignment concept (no Wide/Slot/Inline/Backfield), so this is
        # simpler than the receiving version - no weights/alignment params.
        QB_BT_PREDICT_MAP = {"pass_attempts": "ATT", "pass_completions": "CMP",
                               "pass_yards": "YDS", "pass_td": "TD"}
        QB_BT_ACTUAL_MAP = {"pass_attempts": "attempts", "pass_completions": "completions",
                              "pass_yards": "passing_yards", "pass_td": "passing_tds"}
        QB_BT_LABELS = {"pass_attempts": "Pass Attempts", "pass_completions": "Pass Completions",
                          "pass_yards": "Pass Yards", "pass_td": "Pass TD"}

        def _predict_best_qb_prop(bundle, qb_name, opponent_team_full, top_n):
            """QB version of _predict_best_prop - same real logic (top-N +
            extreme-low-usage coverage inclusion, weighted blend of own tier
            across included coverages), just against QB's own real prop set
            instead of receiving props."""
            opp_profile = bundle.def_coverage.get(opponent_team_full)
            if opp_profile is None:
                return None
            included = _top_n_coverage_fields(bundle, opp_profile, top_n)
            already_fields = {f for f, _, _ in included}
            included = included + _find_extreme_low_usage_fields(
                bundle, opp_profile, "QB", None, None, already_fields,
            )
            if not included:
                return None
            scores = {}
            for prop, stat_col in QB_BT_PREDICT_MAP.items():
                weighted_vals = []
                for field, z, rank in included:
                    row = bundle.qb_vs_coverage.get(field, {}).get(qb_name)
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

        def _actual_best_qb_prop(tiers: dict):
            valid = {p: TIER_WEIGHTS[tiers[c]] for p, c in QB_BT_ACTUAL_MAP.items()
                     if c in tiers and tiers[c] in TIER_WEIGHTS}
            if not valid:
                return None, []
            best = max(valid, key=valid.get)
            ties = [p for p, s in valid.items() if p != best and (valid[best] - s) <= 10]
            return best, ties

        def _run_qb_season_backtest(bundle, qb_name, game_log_season, top_n):
            """QB version of _run_season_backtest - same real per-week
            grading logic, QB's own prop set and no alignment concept."""
            bt_pstats = pull_player_stats([int(game_log_season)])
            bt_sched = pull_schedules([int(game_log_season)])
            bt_matches = bt_pstats[bt_pstats["position"].astype(str).str.upper() == "QB"]
            bt_name_col = "player_display_name" if "player_display_name" in bt_pstats.columns else (
                "player_name" if "player_name" in bt_pstats.columns else None)
            bt_gsis = None
            if bt_name_col:
                bt_hit = bt_matches[bt_matches[bt_name_col].astype(str).str.lower() == qb_name.lower()]
                if not bt_hit.empty:
                    bt_gsis = bt_hit.iloc[0]["gsis_id"]
            if bt_gsis is None:
                return {"error": f"Couldn't match '{qb_name}' to a real nflreadpy player record for {game_log_season}."}

            all_abbrevs = set(TEAM_ABBREV_TO_FULL.keys())
            full_log = build_coverage_crossref_game_log(
                bt_gsis, "QB", all_abbrevs, bt_pstats, bt_sched,
                seasons=[int(game_log_season)], max_games=25,
            )
            rows, strict_hits, generous_hits, graded = [], 0, 0, 0
            for g in full_log:
                opp_full = TEAM_ABBREV_TO_FULL.get(g["opponent"])
                pred = _predict_best_qb_prop(bundle, qb_name, opp_full, top_n) if opp_full else None
                pred_best = pred["best"] if pred else None
                pred_quality = pred["scores"].get(pred_best) if (pred and pred_best) else None
                pred_has_tie = bool(pred["ties"]) if pred else False
                actual_best, actual_ties = _actual_best_qb_prop(g.get("tiers", {}))
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
                    "Predicted Best": QB_BT_LABELS.get(pred_best, "no data"),
                    "Actual Best": QB_BT_LABELS.get(actual_best, "no data"),
                    "Result": result,
                    "_pred_quality": pred_quality, "_pred_has_tie": pred_has_tie,
                })
            return {"rows": rows, "strict_hits": strict_hits, "generous_hits": generous_hits, "graded": graded}

        # ---- RB rushing equivalent - genuinely different props again
        # (Rush Attempts/Rush Yards, from the SEPARATE rb_matchup.py concept
        # data, not the coverage bundle at all). Relocated here from deep
        # inside the single-matchup RB report flow - same reason
        # _run_season_backtest was relocated earlier: the League-Wide scan
        # needs these callable WITHOUT a report/player already built first.
        # Real gap this closes: the League-Wide scan previously routed RB
        # through the WR/TE receiving-prop pipeline (targets/receptions/
        # rec_yards/receiving_td) - correct for RBs as pass-catchers, but
        # it meant Rush Attempts/Rush Yards were NEVER actually tested for
        # RBs in that scan at all, despite the RB rushing backtest already
        # existing as its own separate single-player tool.
        RB_PROP_LABELS = {"rush_attempts": "Rush Attempts", "rush_yards": "Rush Yards"}
        RB_PROP_STAT_MAP = {"rush_attempts": "ATT", "rush_yards": "YDS"}
        RB_GAME_LOG_PROP_MAP = {"rush_attempts": "carries", "rush_yards": "rushing_yards"}

        def _predict_best_rb_prop(rb_bundle, rb_name, opponent_team_full):
            """Blends BOTH the RB's own tier AND the opponent's
            defense-allowed tier per concept (50/50 when both exist),
            weighted by his real attempt share per concept - identical
            logic to the single-matchup version, just callable standalone."""
            rb_own_atts = {
                c: (rb_bundle.rb_vs_concept.get(c, {}).get(rb_name, {}) or {}).get("_att", 0)
                for c in RB_CONCEPT_FILES
            }
            total_att = sum(rb_own_atts.values())
            weights = {c: (a / total_att) for c, a in rb_own_atts.items() if total_att and a > 0}
            if not weights:
                return None
            scores = {}
            for prop, stat_col in RB_PROP_STAT_MAP.items():
                weighted_vals = []
                for concept, w in weights.items():
                    own_row = rb_bundle.rb_vs_concept.get(concept, {}).get(rb_name)
                    own_tier = own_row.get("_tiers", {}).get(stat_col) if own_row else None
                    def_row = (rb_bundle.def_allowed.get(concept, {}).get(opponent_team_full)
                               if opponent_team_full else None)
                    def_tier = def_row.get("_tiers", {}).get(stat_col) if def_row else None
                    own_w = TIER_WEIGHTS.get(own_tier)
                    def_w = TIER_WEIGHTS.get(def_tier)
                    if own_w is None and def_w is None:
                        continue
                    blended = (0.5 * own_w + 0.5 * def_w) if (own_w is not None and def_w is not None) \
                        else (own_w if own_w is not None else def_w)
                    weighted_vals.append((w, blended))
                total_w = sum(w for w, _ in weighted_vals)
                scores[prop] = round(sum(w * s for w, s in weighted_vals) / total_w, 1) if total_w else None
            valid = {p: s for p, s in scores.items() if s is not None}
            if not valid:
                return {"scores": scores, "best": None, "ties": []}
            best = max(valid, key=valid.get)
            ties = [p for p, s in valid.items() if p != best and (valid[best] - s) <= 10]
            return {"scores": scores, "best": best, "ties": ties}

        def _actual_best_rb_prop(tiers):
            valid = {p: TIER_WEIGHTS[tiers[c]] for p, c in RB_GAME_LOG_PROP_MAP.items()
                     if c in tiers and tiers[c] in TIER_WEIGHTS}
            if not valid:
                return None, []
            best = max(valid, key=valid.get)
            ties = [p for p, s in valid.items() if p != best and (valid[best] - s) <= 10]
            return best, ties

        def _run_rb_season_backtest(rb_bundle, rb_name, game_log_season):
            """RB rushing version of _run_season_backtest/_run_qb_season_backtest -
            same real per-week grading logic, RB's own rushing-concept props."""
            rbt_pstats = pull_player_stats([int(game_log_season)])
            rbt_sched = pull_schedules([int(game_log_season)])
            rbt_matches = rbt_pstats[rbt_pstats["position"].astype(str).str.upper() == "RB"]
            rbt_name_col = "player_display_name" if "player_display_name" in rbt_pstats.columns else (
                "player_name" if "player_name" in rbt_pstats.columns else None)
            rbt_gsis = None
            if rbt_name_col:
                rbt_hit = rbt_matches[rbt_matches[rbt_name_col].astype(str).str.lower() == rb_name.lower()]
                if not rbt_hit.empty:
                    rbt_gsis = rbt_hit.iloc[0]["gsis_id"]
            if rbt_gsis is None:
                return {"error": f"Couldn't match '{rb_name}' to a real nflreadpy player record for {game_log_season}."}

            all_abbrevs = set(TEAM_ABBREV_TO_FULL.keys())
            rbt_log = build_coverage_crossref_game_log(
                rbt_gsis, "RB", all_abbrevs, rbt_pstats, rbt_sched,
                seasons=[int(game_log_season)], max_games=25,
            )
            rows, strict_hits, generous_hits, graded = [], 0, 0, 0
            for g in rbt_log:
                g_opp_full = TEAM_ABBREV_TO_FULL.get(g["opponent"])
                pred = _predict_best_rb_prop(rb_bundle, rb_name, g_opp_full) if g_opp_full else None
                pred_best = pred["best"] if pred else None
                pred_quality = pred["scores"].get(pred_best) if (pred and pred_best) else None
                pred_has_tie = bool(pred["ties"]) if pred else False
                actual_best, actual_ties = _actual_best_rb_prop(g.get("tiers", {}))
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
                    "Predicted Best": RB_PROP_LABELS.get(pred_best, "no data"),
                    "Actual Best": RB_PROP_LABELS.get(actual_best, "no data"),
                    "Result": result,
                    "_pred_quality": pred_quality, "_pred_has_tie": pred_has_tie,
                })
            return {"rows": rows, "strict_hits": strict_hits, "generous_hits": generous_hits, "graded": graded}

        st.divider()
        st.divider()
        st.markdown("### Mu Comparison Backtest — Raw vs Adjusted vs Cross-Reference")
        st.caption(
            "The MLB-format direct test: does mu beat a REAL fixed line, one real prop at a "
            "time - not 'which of several props wins.' Three mu sources compared head to head, "
            "same real games, same real line: Raw (his own trailing average, no matchup "
            "awareness), Adjusted (nudged by the coverage/alignment Quality Score - same "
            "Premium Adjustment logic as Line Value, already grades both sides and every heavy "
            "coverage a defense leans on, not just one), and CrossRef (his real games only "
            "against teams that graded similarly to this week's real opponent). No look-ahead - "
            "only real games strictly before the one being graded feed any of the three."
        )
        mucomp_col1, mucomp_col2, mucomp_col3, mucomp_col4 = st.columns(4)
        with mucomp_col1:
            mucomp_pos = st.selectbox("Position", ["WR", "TE", "RB"], key="mucomp_pos",
                                       help="Receiving props only for now (QB/rushing versions "
                                            "follow this same pattern next).")
        with mucomp_col2:
            mucomp_prop = st.selectbox(
                "Prop to test", ["targets", "receptions", "rec_yards", "receiving_td"],
                key="mucomp_prop")
        with mucomp_col3:
            mucomp_line = st.number_input("Real fixed line to test against", min_value=0.0,
                                          value=5.5, step=0.5, key="mucomp_line")
        with mucomp_col4:
            mucomp_season = st.number_input("Season", min_value=2020, max_value=2030, value=2025,
                                            step=1, key="mucomp_season")

        mucomp_name = st.text_input("Player name (exact, as it appears in the export)",
                                     value="", key="mucomp_name")
        mucomp_top_n = st.number_input(
            "Top-N coverage threshold (for the Adjusted/CrossRef mu sources)",
            min_value=1, max_value=32, value=10, step=1, key="mucomp_top_n")

        if st.button("Run mu comparison backtest", type="primary", key="mucomp_run_btn"):
            try:
                mc_weights = _get_real_alignment_weights(bundle, mucomp_name) if mucomp_pos != "QB" else None
                mc_result = _run_mu_source_comparison_backtest(
                    bundle, mucomp_name, mucomp_pos, None, mc_weights,
                    int(mucomp_season), int(mucomp_top_n), mucomp_prop, float(mucomp_line),
                )
                st.session_state["_mucomp_result"] = mc_result
            except Exception as e:
                st.session_state["_mucomp_result"] = {"error": f"Backtest failed: {e}"}

        mc = st.session_state.get("_mucomp_result")
        if mc:
            if mc.get("error"):
                st.warning(mc["error"])
            elif not mc.get("rows"):
                st.info("No graded games came back - try a lower minimum or a different player.")
            else:
                mc_df = pd.DataFrame(mc["rows"])
                for label in ["Raw", "Adjusted", "CrossRef"]:
                    hit_col = f"{label} Hit"
                    if hit_col in mc_df.columns:
                        valid = mc_df[hit_col].dropna()
                        if len(valid):
                            pct = valid.mean() * 100
                            st.markdown(f"**{label}: {int(valid.sum())}/{len(valid)} ({pct:.0f}%)**")

                # Does the two-signal agreement itself predict better than
                # either signal alone? Split real-hit-rate by whether the
                # statistical trend (Raw mu vs line) and the matchup grade
                # (Quality Score) actually pointed the same direction that
                # week, using "Raw Hit" as the real outcome check either way.
                if "Agreement" in mc_df.columns and "Raw Hit" in mc_df.columns:
                    agree_rows = mc_df[(mc_df["Agreement"] == "Agree") & mc_df["Raw Hit"].notna()]
                    conflict_rows = mc_df[(mc_df["Agreement"] == "Conflict") & mc_df["Raw Hit"].notna()]
                    if len(agree_rows) or len(conflict_rows):
                        st.markdown("**When the two signals agree vs conflict:**")
                        if len(agree_rows):
                            a_pct = agree_rows["Raw Hit"].mean() * 100
                            st.markdown(f"- Agree ({len(agree_rows)} weeks): "
                                        f"{int(agree_rows['Raw Hit'].sum())}/{len(agree_rows)} ({a_pct:.0f}%)")
                        if len(conflict_rows):
                            c_pct = conflict_rows["Raw Hit"].mean() * 100
                            st.markdown(f"- Conflict ({len(conflict_rows)} weeks): "
                                        f"{int(conflict_rows['Raw Hit'].sum())}/{len(conflict_rows)} ({c_pct:.0f}%)")
                        st.caption(
                            "If 'Agree' weeks hit noticeably more often than 'Conflict' weeks, that's "
                            "real proof the matchup grade adds something beyond the raw trend alone - "
                            "the actual question you're asking. If they're close, the grade isn't "
                            "adding much on top of what mu already knew."
                        )

                st.dataframe(mc_df, width='stretch')
                st.caption(
                    "Mu Lean = which way RAW mu (unadjusted) points vs the line. Quality Lean = "
                    "which way the matchup grade alone points, independent of mu. Agreement shows "
                    "whether those two genuinely different signals point the same way that week - "
                    "'Agree' is a real convergence of two independent reads, not the same signal "
                    "counted twice. Read Raw as the honest baseline - if Adjusted and CrossRef "
                    "don't clear it by a real margin, the matchup data isn't adding value for THIS "
                    "prop/player yet, and that's a real finding worth knowing, not a failure to hide."
                )

        st.divider()
        st.markdown("### League-Wide Scan — All Positions, Best Matchups Only")
        st.caption(
            "No name, no position to pick - scans QB, WR, TE, and RB all at once, finds every "
            "real player at each position with enough games to be worth grading, and only "
            "counts the BEST possible calls (score ≥ 70, no toss-ups) unless you uncheck that "
            "below. Results are grouped by prop across ALL positions together - Targets, "
            "Receptions, Rec Yards, Receiving TD, Pass Attempts, Pass Completions, Pass Yards, "
            "Pass TD all in one table. Same look-ahead-bias caveat as every other backtest here: "
            "season-aggregate splits, not a clean pre-game test."
        )
        awide_col1, awide_col2, awide_col3 = st.columns(3)
        with awide_col1:
            awide_season = st.number_input(
                "Season", min_value=2020, max_value=2030, value=2025, step=1, key="awide_season",
            )
        with awide_col2:
            awide_top_n = st.number_input(
                "Top-N coverage threshold", min_value=1, max_value=32, value=10, step=1, key="awide_top_n",
            )
        with awide_col3:
            awide_min_games = st.number_input(
                "Min real games per player", min_value=1, max_value=18, value=8, step=1, key="awide_min_games",
            )
        awide_max_col1, awide_max_col2 = st.columns(2)
        with awide_max_col1:
            awide_max_per_pos = st.number_input(
                "Max players to scan PER position (higher = more complete, slower)",
                min_value=5, max_value=150, value=30, step=5, key="awide_max_per_pos",
            )
        with awide_max_col2:
            awide_high_conf = st.checkbox(
                "Only count the BEST possible calls (score ≥ 70, no toss-ups)",
                value=True, key="awide_high_conf",
            )

        if st.button("Scan ALL positions, full season", type="primary"):
            try:
                with st.spinner("Scanning QB, WR, TE, and RB - this takes a while, real work happening..."):
                    aw_pstats_cache = pull_player_stats([int(awide_season)])
                    aw_name_col = "player_display_name" if "player_display_name" in aw_pstats_cache.columns else (
                        "player_name" if "player_name" in aw_pstats_cache.columns else None)
                    prop_totals = {}
                    per_pos_summary = []
                    per_player_rows = []
                    total_strict = total_generous = total_graded = 0
                    total_weeks_seen = total_weeks_kept = 0
                    errors = []

                    for scan_pos in ["QB", "WR", "TE", "RB"]:
                        if not aw_name_col:
                            break
                        pos_matches = aw_pstats_cache[aw_pstats_cache["position"].astype(str).str.upper() == scan_pos]
                        games_per_player = pos_matches.groupby(aw_name_col)["week"].nunique()
                        eligible = games_per_player[games_per_player >= int(awide_min_games)].sort_values(ascending=False)
                        names_to_scan = list(eligible.index[:int(awide_max_per_pos)])

                        pos_strict = pos_generous = pos_graded = 0
                        for nm in names_to_scan:
                            try:
                                if scan_pos == "QB":
                                    results_list = [_run_qb_season_backtest(bundle, nm, awide_season, int(awide_top_n))]
                                elif scan_pos == "RB":
                                    # Real fix: RB used to ONLY get the WR/TE
                                    # receiving-prop test (targets/receptions/
                                    # rec_yards/receiving_td) - correct for RBs
                                    # as pass-catchers, but Rush Attempts/Rush
                                    # Yards were NEVER tested at all. RBs are
                                    # genuinely both, so both get tested and
                                    # merged - not one replacing the other.
                                    results_list = []
                                    nm_weights = _get_real_alignment_weights(bundle, nm)
                                    rec_result = _run_season_backtest(
                                        bundle, nm, scan_pos, None, nm_weights, awide_season, int(awide_top_n),
                                    )
                                    if not rec_result.get("error"):
                                        results_list.append(rec_result)
                                    rb_bundle_live = st.session_state.get("rb_bundle")
                                    if rb_bundle_live is not None:
                                        rush_result = _run_rb_season_backtest(rb_bundle_live, nm, awide_season)
                                        if not rush_result.get("error"):
                                            results_list.append(rush_result)
                                    elif not results_list:
                                        errors.append(f"RB {nm}: rushing skipped - RB run-concept dataset "
                                                       f"not loaded (load it in the RB Run Concept Matchup "
                                                       f"section above, then re-run this scan for rushing props).")
                                else:
                                    nm_weights = _get_real_alignment_weights(bundle, nm)
                                    results_list = [_run_season_backtest(
                                        bundle, nm, scan_pos, None, nm_weights, awide_season, int(awide_top_n),
                                    )]
                            except Exception as e:
                                errors.append(f"{scan_pos} {nm}: {e}")
                                continue

                            kept_rows = []
                            for result in results_list:
                                if result.get("error"):
                                    continue
                                for wk_row in result["rows"]:
                                    if wk_row["Predicted Best"] == "no data":
                                        continue
                                    total_weeks_seen += 1
                                    if awide_high_conf:
                                        q = wk_row.get("_pred_quality")
                                        if q is None or q < 70 or wk_row.get("_pred_has_tie"):
                                            continue
                                    total_weeks_kept += 1
                                    kept_rows.append(wk_row)
                            if not kept_rows:
                                continue

                            p_strict = sum(1 for r in kept_rows if r["Result"] == "✅ Hit")
                            p_generous = sum(1 for r in kept_rows if r["Result"] in ("✅ Hit", "〰️ Tie"))
                            p_graded = len(kept_rows)
                            total_strict += p_strict
                            total_generous += p_generous
                            total_graded += p_graded
                            pos_strict += p_strict
                            pos_generous += p_generous
                            pos_graded += p_graded
                            per_player_rows.append({
                                "Position": scan_pos, "Player": nm, "Graded Games": p_graded,
                                "Strict Hit Rate": f"{p_strict}/{p_graded} ({p_strict/p_graded*100:.0f}%)",
                            })
                            for wk_row in kept_rows:
                                prop = wk_row["Predicted Best"]
                                prop_totals.setdefault(prop, [0, 0, 0])
                                if wk_row["Result"] == "✅ Hit":
                                    prop_totals[prop][0] += 1
                                    prop_totals[prop][1] += 1
                                elif wk_row["Result"] == "〰️ Tie":
                                    prop_totals[prop][1] += 1
                                prop_totals[prop][2] += 1

                        if pos_graded:
                            per_pos_summary.append({
                                "Position": scan_pos, "Players Graded": sum(1 for r in per_player_rows if r["Position"] == scan_pos),
                                "Graded Games": pos_graded,
                                "Strict Hit Rate": f"{pos_strict}/{pos_graded} ({pos_strict/pos_graded*100:.0f}%)",
                            })

                    prop_rows = []
                    for prop, (sh, gh, gr) in prop_totals.items():
                        if gr:
                            prop_rows.append({
                                "Predicted Prop": prop, "Times Predicted": gr,
                                "Strict Hit Rate": f"{sh}/{gr} ({sh/gr*100:.0f}%)",
                                "Including Near-Ties": f"{gh}/{gr} ({gh/gr*100:.0f}%)",
                            })
                    prop_rows.sort(key=lambda r: -r["Times Predicted"])

                    st.session_state["_all_pos_scan"] = {
                        "per_pos_summary": per_pos_summary, "per_player_rows": per_player_rows,
                        "prop_rows": prop_rows, "total_strict": total_strict,
                        "total_generous": total_generous, "total_graded": total_graded,
                        "errors": errors, "high_conf_only": awide_high_conf,
                        "weeks_seen": total_weeks_seen, "weeks_kept": total_weeks_kept,
                    }
            except Exception as e:
                st.session_state["_all_pos_scan"] = {"error": f"Scan failed: {e}"}

        awide = st.session_state.get("_all_pos_scan")
        if awide:
            if awide.get("error"):
                st.warning(awide["error"])
            elif awide["total_graded"] == 0:
                st.info("No graded games at this filter level - try unchecking 'best possible calls only', or lowering the minimum games filter.")
            else:
                awsr = awide["total_strict"] / awide["total_graded"] * 100
                awgr = awide["total_generous"] / awide["total_graded"] * 100
                filter_note = (
                    f" (filtered to {awide['weeks_kept']}/{awide['weeks_seen']} weeks that were genuinely strong calls)"
                    if awide.get("high_conf_only") else " (unfiltered - every prediction counted, weak or strong)"
                )
                st.markdown(
                    f"**Across all 4 positions, {awide['total_graded']} total real games — overall strict hit rate: "
                    f"{awide['total_strict']}/{awide['total_graded']} ({awsr:.0f}%)** &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"including near-ties: {awide['total_generous']}/{awide['total_graded']} ({awgr:.0f}%) "
                    f"&nbsp;&nbsp;(random baseline varies by position - ~33% for WR/TE/RB with 3 props, ~25% for QB with 4)"
                    f"{filter_note}"
                )
                st.markdown("**By prop — across every position, which one the method actually predicts best:**")
                st.dataframe(pd.DataFrame(awide["prop_rows"]), width='stretch')
                st.markdown("**By position:**")
                st.dataframe(pd.DataFrame(awide["per_pos_summary"]), width='stretch')
                with st.expander(f"Per-player breakdown ({len(awide['per_player_rows'])} players)"):
                    st.dataframe(pd.DataFrame(awide["per_player_rows"]), width='stretch')
                if awide["errors"]:
                    with st.expander(f"{len(awide['errors'])} skipped (name-match issues)"):
                        st.write(awide["errors"])

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

            included = _top_n_coverage_fields(bundle, opp_profile, top_n)
            already_fields = {f for f, _, _ in included}
            extreme_extra = _find_extreme_low_usage_fields(
                bundle, opp_profile, position, alignment, weights, already_fields,
            )
            extreme_field_set = {f for f, _, _ in extreme_extra}
            included = included + extreme_extra

            if not included:
                return [{"note": f"{opp_profile.team_name} has no coverage ranking in the top "
                                  f"{top_n} of all 32 teams, and no extreme low-usage tendency "
                                  f"either - no coverage edge to flag here."}], [], opp_profile

            report = []
            for field, z, rank in included:
                entry = {
                    "coverage": field.replace(" %", ""),
                    "opponent_usage_pct": opp_profile.rates.get(field, 0.0),
                    "opponent_z_score": round(z, 2),
                    "low_usage_extreme": field in extreme_field_set,
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
                        a_qs = _quality_score(a_own.get("_tiers", {}), position=position.upper(), thin_sample=a_own.get("_thin_sample", False)) if a_own is not None else None
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
                        entry["quality_score"] = _quality_score(own_row.get("_tiers", {}), position=position.upper(), thin_sample=own_row.get("_thin_sample", False))
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
                        pbp = pull_pbp([int(game_log_season)])
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
                                seasons=[int(game_log_season)], pbp_df=pbp,
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
                    "The default view shows the crucial prop-decision stats; every real "
                    "column from the export is still there under \"+more stats\" below each "
                    "block. The Quality score is computed from ALL of them, and the crucial "
                    "ones (TGT/REC/CR%/YDS/YPRR/TD, plus aDOT and the YAC splits) count double "
                    "so they aren't averaged down to the same weight as minor columns. A \"~\" "
                    "after a score means it's built on a THIN SAMPLE and has been pulled "
                    "halfway toward neutral (50) rather than shown at full confidence."
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
                    if entry.get("low_usage_extreme"):
                        rank_badge = '<span class="cov-thin-flag">LOW USAGE, EXTREME ALLOWED</span>'
                    else:
                        rank_badge = f'<span class="cov-z-badge">rank {rank} of 32</span>' if rank else ""
                    qs = entry.get("quality_score")
                    if entry.get("auto_weighted"):
                        qs_is_thin = any(
                            b["confidence"] == "thin_sample" for b in entry.get("alignment_breakdown", [])
                            if b["quality_score"] is not None
                        )
                    else:
                        qs_is_thin = entry.get("confidence") == "thin_sample"
                    qs_badge = (
                        f'<span class="tier-badge {_score_badge_class(qs)}">Quality {qs:.0f}{" ~" if qs_is_thin else ""}</span>'
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
                            b_thin = b["confidence"] == "thin_sample"
                            b_qs_badge = (
                                f'<span class="tier-badge {_score_badge_class(b_qs)}">Quality {b_qs:.0f}{" ~" if b_thin else ""}</span>'
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
                    "Longest catch isn't included in this verdict - no season-aggregate CSV "
                    "column to predict it from, only real per-game data (see the line "
                    "comparison below, where it IS available). Targets/receptions/rec yards/"
                    "receiving TD are all real on both the predicted and actual side."
                )

                st.markdown("### Line Value + Backtest Reliability")
                st.caption(
                    "One table: real mu/edge for whatever lines you enter, PLUS how reliable "
                    "this method has been historically for this player (run the backtest above "
                    "first to populate that column - optional, but the honest context for how "
                    "much to trust the edge shown). mu only gets adjusted by the premium coverage "
                    "data when the signal is genuinely strong - Above Avg/Elite or Below Avg/Poor "
                    "(25+ points from neutral) and NOT a thin sample. A merely Average matchup "
                    "grade leaves mu untouched rather than nudging on a mediocre read."
                )

                def _get_player_mu_sigma(bundle, gsis_id, position, stat_col, pstats, sched, season, pbp=None):
                    all_abbrevs = set(TEAM_ABBREV_TO_FULL.keys())
                    log = build_coverage_crossref_game_log(
                        gsis_id, position, all_abbrevs, pstats, sched, seasons=[season],
                        max_games=25, pbp_df=pbp,
                    )
                    vals = [g["stats"].get(stat_col) for g in log if stat_col in g.get("stats", {})]
                    vals = [v for v in vals if v is not None]
                    if len(vals) < 2:
                        return None, None, len(vals)
                    mu = float(np.mean(vals))
                    sigma = float(np.std(vals))
                    if sigma == 0:
                        sigma = max(mu * 0.15, 1.0)
                    return mu, sigma, len(vals)

                line_col1, line_col2, line_col3, line_col4, line_col5 = st.columns(5)
                with line_col1:
                    tgt_line = st.number_input("Targets line", min_value=0.0, value=0.0, step=0.5, key="tgt_line")
                with line_col2:
                    rec_line = st.number_input("Receptions line", min_value=0.0, value=0.0, step=0.5, key="rec_line")
                with line_col3:
                    yds_line = st.number_input("Rec Yards line", min_value=0.0, value=0.0, step=0.5, key="yds_line")
                with line_col4:
                    lng_line = st.number_input("Longest Catch line", min_value=0.0, value=0.0, step=0.5, key="lng_catch_line")
                with line_col5:
                    rtd_line = st.number_input("Receiving TD line", min_value=0.0, value=0.0, step=0.5, key="rtd_line")

                if p_pos == "QB":
                    st.caption(
                        "QB props: mu/sigma come from real game logs (all 4 confirmed real "
                        "nflreadpy columns). Premium Adj applies to all 4 props - Attempts, "
                        "Completions, Yards, and TD are all confirmed real columns on the CSV "
                        "side too, so every one can get the adjustment when the signal qualifies."
                    )
                    qb_line_col1, qb_line_col2, qb_line_col3, qb_line_col4 = st.columns(4)
                    with qb_line_col1:
                        patt_line = st.number_input("Pass Attempts line", min_value=0.0, value=0.0, step=0.5, key="patt_line")
                    with qb_line_col2:
                        pcmp_line = st.number_input("Pass Completions line", min_value=0.0, value=0.0, step=0.5, key="pcmp_line")
                    with qb_line_col3:
                        pyds_line = st.number_input("Pass Yards line", min_value=0.0, value=0.0, step=0.5, key="pyds_line")
                    with qb_line_col4:
                        ptd_line = st.number_input("Pass TD line", min_value=0.0, value=0.0, step=0.5, key="ptd_line")

                    QB_LINE_PROP_MAP = {"pass_attempts": "attempts", "pass_completions": "completions",
                                         "pass_yards": "passing_yards", "pass_td": "passing_tds"}
                    QB_LINE_PROP_LABELS = {"pass_attempts": "Pass Attempts", "pass_completions": "Pass Completions",
                                            "pass_yards": "Pass Yards", "pass_td": "Pass TD"}
                    # pass_completions now included - CMP confirmed as a real
                    # raw completions column (not just the CMP % rate this
                    # was originally built against), so it gets the same
                    # Premium Adjustment treatment as the other 3 props now.
                    QB_PREDICT_STAT_MAP = {"pass_attempts": "ATT", "pass_completions": "CMP",
                                            "pass_yards": "YDS", "pass_td": "TD"}

                    if st.button("Check QB value vs these lines", type="primary"):
                        try:
                            qbl_pstats = pull_player_stats([int(game_log_season)])
                            qbl_sched = pull_schedules([int(game_log_season)])
                            qbl_matches = qbl_pstats[qbl_pstats["position"].astype(str).str.upper() == "QB"]
                            qbl_name_col = "player_display_name" if "player_display_name" in qbl_pstats.columns else (
                                "player_name" if "player_name" in qbl_pstats.columns else None)
                            qbl_gsis = None
                            if qbl_name_col:
                                qbl_hit = qbl_matches[qbl_matches[qbl_name_col].astype(str).str.lower() == p_name.lower()]
                                if not qbl_hit.empty:
                                    qbl_gsis = qbl_hit.iloc[0]["gsis_id"]
                            if qbl_gsis is None:
                                st.session_state["_line_compare"] = {
                                    "error": f"Couldn't match '{p_name}' to a real nflreadpy record for {game_log_season}."}
                            else:
                                qb_lines = {"pass_attempts": patt_line, "pass_completions": pcmp_line,
                                            "pass_yards": pyds_line, "pass_td": ptd_line}
                                ubt_cached = st.session_state.get("_unified_backtest")
                                reliability = "run backtest above first"
                                if ubt_cached and not ubt_cached.get("error") and ubt_cached.get("total_graded"):
                                    match_row = next((r for r in ubt_cached["rows"] if r["Player"].lower() == p_name.lower()), None)
                                    if match_row:
                                        reliability = match_row["Strict Hit Rate"]
                                rows = []
                                for prop, line_val in qb_lines.items():
                                    if not line_val:
                                        continue
                                    stat_col = QB_LINE_PROP_MAP[prop]
                                    mu, sigma, n_games = _get_player_mu_sigma(
                                        bundle, qbl_gsis, "QB", stat_col, qbl_pstats, qbl_sched, int(game_log_season),
                                    )
                                    if mu is None:
                                        rows.append({"Prop": QB_LINE_PROP_LABELS[prop], "Line": line_val,
                                                     "mu": "no data", "Premium Adj": "-", "sigma": "-", "Games": n_games,
                                                     "p(Over)": "-", "Edge": "-", "Backtest Reliability": reliability})
                                        continue
                                    q_score = None
                                    is_thin_signal = True
                                    if prop in QB_PREDICT_STAT_MAP:
                                        stat_col_csv = QB_PREDICT_STAT_MAP[prop]
                                        weighted_vals = []
                                        for e in report:
                                            row = e.get(own_key)
                                            if row is None:
                                                continue
                                            tier = row.get("_tiers", {}).get(stat_col_csv)
                                            if tier in TIER_WEIGHTS:
                                                weighted_vals.append(TIER_WEIGHTS[tier])
                                            if not row.get("_thin_sample"):
                                                is_thin_signal = False
                                        if weighted_vals:
                                            q_score = sum(weighted_vals) / len(weighted_vals)
                                    adj_mu, adj_pct, applied = _apply_best_signal_adjustment(mu, q_score, is_thin_signal)
                                    scored = rescore_quality_mu_row_nfl(adj_mu, line_val, sigma)
                                    rows.append({
                                        "Prop": QB_LINE_PROP_LABELS[prop], "Line": line_val,
                                        "mu": round(mu, 1),
                                        "Premium Adj": f"{adj_pct:+.1f}%" if applied else "none (not extreme enough)",
                                        "sigma": round(sigma, 1), "Games": n_games,
                                        "p(Over)": scored["p_over"], "Edge": scored["edge"],
                                        "Backtest Reliability": reliability,
                                    })
                                st.session_state["_line_compare"] = {"rows": rows}
                        except Exception as e:
                            st.session_state["_line_compare"] = {"error": f"Line check failed: {e}"}

                elif st.button("Check value vs these lines", type="primary"):
                    try:
                        lc_pstats = pull_player_stats([int(game_log_season)])
                        lc_sched = pull_schedules([int(game_log_season)])
                        lc_pbp = pull_pbp([int(game_log_season)])
                        lc_matches = lc_pstats[lc_pstats["position"].astype(str).str.upper() == p_pos.upper()]
                        lc_name_col = "player_display_name" if "player_display_name" in lc_pstats.columns else (
                            "player_name" if "player_name" in lc_pstats.columns else None)
                        lc_gsis = None
                        if lc_name_col:
                            lc_hit = lc_matches[lc_matches[lc_name_col].astype(str).str.lower() == p_name.lower()]
                            if not lc_hit.empty:
                                lc_gsis = lc_hit.iloc[0]["gsis_id"]
                        if lc_gsis is None:
                            st.session_state["_line_compare"] = {
                                "error": f"Couldn't match '{p_name}' to a real nflreadpy record for {game_log_season}."}
                        else:
                            lines = {"targets": tgt_line, "receptions": rec_line, "rec_yards": yds_line,
                                     "longest_catch": lng_line, "receiving_td": rtd_line}
                            # Backtest reliability, if already run this session for this player -
                            # optional context column, doesn't block the line check if absent.
                            ubt_cached = st.session_state.get("_unified_backtest")
                            reliability = "run backtest above first"
                            if ubt_cached and not ubt_cached.get("error") and ubt_cached.get("total_graded"):
                                match_row = next((r for r in ubt_cached["rows"] if r["Player"].lower() == p_name.lower()), None)
                                if match_row:
                                    reliability = match_row["Strict Hit Rate"]

                            rows = []
                            for prop, line_val in lines.items():
                                if not line_val:
                                    continue
                                stat_col = LINE_COMPARE_PROP_MAP[prop]
                                mu, sigma, n_games = _get_player_mu_sigma(
                                    bundle, lc_gsis, p_pos, stat_col, lc_pstats, lc_sched,
                                    int(game_log_season), pbp=lc_pbp,
                                )
                                if mu is None:
                                    rows.append({"Prop": LINE_COMPARE_PROP_LABELS[prop], "Line": line_val,
                                                 "mu": "no data", "Premium Adj": "-", "sigma": "-", "Games": n_games,
                                                 "p(Over)": "-", "Edge": "-", "Backtest Reliability": reliability})
                                    continue
                                # Only the props with a real CSV column to grade (targets/
                                # receptions/rec_yards) get the premium adjustment -
                                # longest_catch has no season-aggregate signal to pull from
                                # (same reasoning as it being excluded from the verdict/backtest).
                                q_score = verdict["scores"].get(prop) if verdict and prop in PROP_STAT_MAP else None
                                # The blended verdict score looks precise even when EVERY
                                # contributing coverage entry is thin - only trust it as a
                                # "best possible" signal if at least ONE real solid-sample
                                # entry backs it. Handles BOTH entry shapes: manual/QB
                                # entries (own_key) and auto-weighted entries
                                # (alignment_breakdown) - auto-weight is the default mode,
                                # so skipping it here would've silently treated every
                                # default-mode matchup as thin and never adjusted at all.
                                has_solid_entry = False
                                for e in report:
                                    if e.get("auto_weighted"):
                                        if any(b.get("own_row") and b.get("confidence") != "thin_sample"
                                               for b in e.get("alignment_breakdown", [])):
                                            has_solid_entry = True
                                            break
                                    elif e.get(own_key) and not e[own_key].get("_thin_sample"):
                                        has_solid_entry = True
                                        break
                                is_thin_signal = not has_solid_entry
                                adj_mu, adj_pct, applied = _apply_best_signal_adjustment(mu, q_score, is_thin_signal)
                                scored = rescore_quality_mu_row_nfl(adj_mu, line_val, sigma)
                                rows.append({
                                    "Prop": LINE_COMPARE_PROP_LABELS[prop], "Line": line_val,
                                    "mu": round(mu, 1),
                                    "Premium Adj": f"{adj_pct:+.1f}%" if applied else "none (not extreme enough)",
                                    "sigma": round(sigma, 1), "Games": n_games,
                                    "p(Over)": scored["p_over"], "Edge": scored["edge"],
                                    "Backtest Reliability": reliability,
                                })
                            st.session_state["_line_compare"] = {"rows": rows}
                    except Exception as e:
                        st.session_state["_line_compare"] = {"error": f"Line check failed: {e}"}

                lc = st.session_state.get("_line_compare")
                if lc:
                    if lc.get("error"):
                        st.warning(lc["error"])
                    elif not lc.get("rows"):
                        st.info("Enter at least one line above, then click the button.")
                    else:
                        st.dataframe(pd.DataFrame(lc["rows"]), width='stretch')
                        st.caption(
                            "Edge is 0 (coinflip) to 1 (max conviction) - same scale as Scan mode. "
                            "'mu' is the player's raw real game-log average; 'Premium Adj' shows "
                            "whether the coverage matchup data was strong enough to nudge it (Edge/"
                            "p(Over) above are computed from the ADJUSTED number when a nudge "
                            "applied). 'Backtest Reliability' is this method's real historical hit "
                            "rate for this player, if you've run the backtest above."
                        )

                st.divider()
                st.markdown("### Backtest This Method — Full Season")
                st.caption(
                    "Checks, for every real game entered player(s) played, whether the prop this "
                    "method would've picked (using the season coverage splits vs that week's real "
                    "opponent) actually was the best-performing prop that week. Works for one "
                    "player or several - enter multiple names for an aggregate verdict instead of "
                    "trusting one player's noisy ~19-game sample. "
                    "⚠️ Real limitation, stated plainly: the coverage splits are season-aggregate, "
                    "so a prediction for week 4 technically has access to data through week 18 - "
                    "a real look-ahead bias. This tests whether the underlying signal correlates "
                    "at all, not a clean pre-game prediction. There are 3 real prop options "
                    "(targets/receptions/rec yards), so random guessing lands around 33% strict."
                )
                bt_names_raw = st.text_area(
                    "Player name(s), one per line (exact, as they appear in the export)",
                    value=p_name if p_name else "", height=80,
                )
                bt_sweep_check = st.checkbox(
                    "Also sweep top-N thresholds (aggregate across all entered players above)",
                    value=False,
                )

                if st.button("Run Backtest", type="primary"):
                    names = [n.strip() for n in bt_names_raw.splitlines() if n.strip()]
                    if not names:
                        st.warning("Enter at least one player name.")
                    else:
                        per_player_rows, per_player_detail, errors = [], {}, []
                        total_strict = total_generous = total_graded = 0
                        for nm in names:
                            nm_weights = _get_real_alignment_weights(bundle, nm) if (use_auto_weight and p_pos != "QB") else None
                            nm_alignment = None if use_auto_weight else align
                            try:
                                result = _run_season_backtest(
                                    bundle, nm, p_pos, nm_alignment, nm_weights,
                                    game_log_season, int(top_n_rank),
                                )
                            except Exception as e:
                                errors.append(f"{nm}: {e}")
                                continue
                            if result.get("error"):
                                errors.append(f"{nm}: {result['error']}")
                                continue
                            g = result["graded"]
                            total_strict += result["strict_hits"]
                            total_generous += result["generous_hits"]
                            total_graded += g
                            per_player_rows.append({
                                "Player": nm, "Graded Games": g,
                                "Strict Hit Rate": f"{result['strict_hits']}/{g} ({result['strict_hits']/g*100:.0f}%)" if g else "no data",
                                "Including Near-Ties": f"{result['generous_hits']}/{g} ({result['generous_hits']/g*100:.0f}%)" if g else "no data",
                            })
                            per_player_detail[nm] = result["rows"]

                        sweep_rows = None
                        if bt_sweep_check and names:
                            candidates = [3, 5, 8, 10, 12, 15, 20, 25, 32]
                            sweep_rows = []
                            for n in candidates:
                                agg_strict = agg_generous = agg_graded = 0
                                for nm in names:
                                    nm_weights = _get_real_alignment_weights(bundle, nm) if (use_auto_weight and p_pos != "QB") else None
                                    nm_alignment = None if use_auto_weight else align
                                    try:
                                        r = _run_season_backtest(bundle, nm, p_pos, nm_alignment, nm_weights, game_log_season, n)
                                    except Exception:
                                        continue
                                    if r.get("error"):
                                        continue
                                    agg_strict += r["strict_hits"]
                                    agg_generous += r["generous_hits"]
                                    agg_graded += r["graded"]
                                sweep_rows.append({
                                    "Top-N": n,
                                    "Strict Hit Rate": f"{agg_strict}/{agg_graded} ({agg_strict/agg_graded*100:.0f}%)" if agg_graded else "no data",
                                    "Including Near-Ties": f"{agg_generous}/{agg_graded} ({agg_generous/agg_graded*100:.0f}%)" if agg_graded else "no data",
                                    "_strict_pct": (agg_strict / agg_graded * 100) if agg_graded else -1,
                                })

                        st.session_state["_unified_backtest"] = {
                            "rows": per_player_rows, "detail": per_player_detail, "errors": errors,
                            "total_strict": total_strict, "total_generous": total_generous,
                            "total_graded": total_graded, "sweep_rows": sweep_rows,
                        }

                ubt = st.session_state.get("_unified_backtest")
                if ubt:
                    for e in ubt["errors"]:
                        st.warning(e)
                    if ubt["total_graded"] == 0:
                        st.info("No graded games across any tested player.")
                    else:
                        osr = ubt["total_strict"] / ubt["total_graded"] * 100
                        ogr = ubt["total_generous"] / ubt["total_graded"] * 100
                        st.markdown(
                            f"**Overall across {len(ubt['rows'])} player(s), {ubt['total_graded']} real "
                            f"games — strict hit rate: {ubt['total_strict']}/{ubt['total_graded']} ({osr:.0f}%)** "
                            f"&nbsp;&nbsp;|&nbsp;&nbsp; including near-ties: {ubt['total_generous']}/{ubt['total_graded']} "
                            f"({ogr:.0f}%) &nbsp;&nbsp;(random baseline: ~33%)"
                        )
                        st.dataframe(pd.DataFrame(ubt["rows"]), width='stretch')

                        if ubt["rows"]:
                            detail_pick = st.selectbox(
                                "Show week-by-week detail for:", [r["Player"] for r in ubt["rows"]],
                                key="bt_detail_pick",
                            )
                            if detail_pick and ubt["detail"].get(detail_pick):
                                with st.expander(f"Week-by-week — {detail_pick}"):
                                    detail_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                                                    for r in ubt["detail"][detail_pick]]
                                    st.dataframe(pd.DataFrame(detail_rows), width='stretch')

                    if ubt.get("sweep_rows"):
                        st.markdown("**Threshold sweep** (aggregate across all entered players):")
                        rows_s = ubt["sweep_rows"]
                        best_row = max(rows_s, key=lambda r: r["_strict_pct"]) if any(r["_strict_pct"] >= 0 for r in rows_s) else None
                        if best_row:
                            st.caption(
                                f"Best-performing threshold: top-{best_row['Top-N']} "
                                f"(strict hit rate {best_row['Strict Hit Rate']}) — current setting is top-{int(top_n_rank)}. "
                                f"Rough signal, not a precise optimum - small real sample, differences of a game or "
                                f"two easily swing the percentage."
                            )
                        display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_s]
                        st.dataframe(pd.DataFrame(display_rows), width='stretch')

                st.divider()
                st.markdown("### League-Wide Auto-Scan Backtest (no typing required)")
                st.caption(
                    "Automatically finds every real player at this position with enough games "
                    "to be worth grading (no names to type), runs the full-season backtest on "
                    "each one using their REAL weekly opponents. By default this only counts "
                    "the BEST possible calls - weeks where the predicted prop scored genuinely "
                    "strong (Above Avg/Elite territory) AND wasn't a toss-up with another prop - "
                    "not every prediction regardless of strength. Uncheck below to see the "
                    "unfiltered number for comparison. Same look-ahead-bias caveat as every "
                    "other backtest on this page: season-aggregate splits, not a clean pre-game "
                    "test."
                )
                scan_col1, scan_col2 = st.columns(2)
                with scan_col1:
                    scan_min_games = st.number_input(
                        "Minimum real games played this season (filters out thin/irrelevant players)",
                        min_value=1, max_value=18, value=8, step=1,
                    )
                with scan_col2:
                    scan_max_players = st.number_input(
                        "Max players to scan (higher = more complete, slower)",
                        min_value=5, max_value=150, value=40, step=5,
                    )
                scan_high_conf_only = st.checkbox(
                    "Only count the BEST possible calls (score ≥ 70, no toss-ups)", value=True,
                )

                if st.button("Scan full season — all players", type="primary"):
                    try:
                        with st.spinner(f"Scanning up to {int(scan_max_players)} real {p_pos} seasons - this can take a bit..."):
                            scan_pstats = pull_player_stats([int(game_log_season)])
                            scan_matches = scan_pstats[scan_pstats["position"].astype(str).str.upper() == p_pos.upper()]
                            scan_name_col = "player_display_name" if "player_display_name" in scan_pstats.columns else (
                                "player_name" if "player_name" in scan_pstats.columns else None)
                            if not scan_name_col:
                                st.session_state["_league_scan"] = {"error": "Couldn't find a player name column in the real data."}
                            else:
                                games_per_player = scan_matches.groupby(scan_name_col)["week"].nunique()
                                eligible = games_per_player[games_per_player >= int(scan_min_games)].sort_values(ascending=False)
                                names_to_scan = list(eligible.index[:int(scan_max_players)])

                                per_player_rows = []
                                prop_totals = {}  # prop label -> [strict_hits, generous_hits, graded]
                                total_strict = total_generous = total_graded = 0
                                total_weeks_seen = total_weeks_kept = 0
                                errors = []
                                for nm in names_to_scan:
                                    nm_weights = _get_real_alignment_weights(bundle, nm) if (use_auto_weight and p_pos != "QB") else None
                                    nm_alignment = None if use_auto_weight else align
                                    try:
                                        result = _run_season_backtest(
                                            bundle, nm, p_pos, nm_alignment, nm_weights,
                                            game_log_season, int(top_n_rank),
                                        )
                                    except Exception as e:
                                        errors.append(f"{nm}: {e}")
                                        continue
                                    if result.get("error"):
                                        continue

                                    # Filter to genuinely strong weeks only, unless the box
                                    # above is unchecked - a call that barely edged out the
                                    # others (score near 50, or tied) isn't a "best possible"
                                    # matchup, it's a coin flip the method happened to lean on.
                                    kept_rows = []
                                    for wk_row in result["rows"]:
                                        if wk_row["Predicted Best"] == "no data":
                                            continue
                                        total_weeks_seen += 1
                                        if scan_high_conf_only:
                                            q = wk_row.get("_pred_quality")
                                            if q is None or q < 70 or wk_row.get("_pred_has_tie"):
                                                continue
                                        total_weeks_kept += 1
                                        kept_rows.append(wk_row)

                                    if not kept_rows:
                                        continue

                                    p_strict = sum(1 for r in kept_rows if r["Result"] == "✅ Hit")
                                    p_generous = sum(1 for r in kept_rows if r["Result"] in ("✅ Hit", "〰️ Tie"))
                                    p_graded = len(kept_rows)
                                    total_strict += p_strict
                                    total_generous += p_generous
                                    total_graded += p_graded
                                    per_player_rows.append({
                                        "Player": nm, "Graded Games": p_graded,
                                        "Strict Hit Rate": f"{p_strict}/{p_graded} ({p_strict/p_graded*100:.0f}%)",
                                    })
                                    for wk_row in kept_rows:
                                        prop = wk_row["Predicted Best"]
                                        prop_totals.setdefault(prop, [0, 0, 0])
                                        if wk_row["Result"] == "✅ Hit":
                                            prop_totals[prop][0] += 1
                                            prop_totals[prop][1] += 1
                                        elif wk_row["Result"] == "〰️ Tie":
                                            prop_totals[prop][1] += 1
                                        prop_totals[prop][2] += 1

                                prop_rows = []
                                for prop, (sh, gh, gr) in prop_totals.items():
                                    if gr:
                                        prop_rows.append({
                                            "Predicted Prop": prop, "Times Predicted": gr,
                                            "Strict Hit Rate": f"{sh}/{gr} ({sh/gr*100:.0f}%)",
                                            "Including Near-Ties": f"{gh}/{gr} ({gh/gr*100:.0f}%)",
                                        })
                                prop_rows.sort(key=lambda r: -r["Times Predicted"])

                                st.session_state["_league_scan"] = {
                                    "players_scanned": len(names_to_scan),
                                    "players_graded": len(per_player_rows),
                                    "per_player_rows": per_player_rows,
                                    "prop_rows": prop_rows,
                                    "total_strict": total_strict, "total_generous": total_generous,
                                    "total_graded": total_graded, "errors": errors,
                                    "high_conf_only": scan_high_conf_only,
                                    "weeks_seen": total_weeks_seen, "weeks_kept": total_weeks_kept,
                                }
                    except Exception as e:
                        st.session_state["_league_scan"] = {"error": f"Scan failed: {e}"}

                lscan = st.session_state.get("_league_scan")
                if lscan:
                    if lscan.get("error"):
                        st.warning(lscan["error"])
                    elif lscan["total_graded"] == 0:
                        st.info("No graded games at this filter level - try unchecking 'best possible calls only', or lowering the minimum games filter.")
                    else:
                        lsr = lscan["total_strict"] / lscan["total_graded"] * 100
                        lgr = lscan["total_generous"] / lscan["total_graded"] * 100
                        filter_note = (
                            f" (filtered to {lscan['weeks_kept']}/{lscan['weeks_seen']} weeks that were genuinely strong calls)"
                            if lscan.get("high_conf_only") else " (unfiltered - every prediction counted, weak or strong)"
                        )
                        st.markdown(
                            f"**Scanned {lscan['players_scanned']} real {p_pos}s ({lscan['players_graded']} had "
                            f"gradable games), {lscan['total_graded']} total real games — overall strict hit rate: "
                            f"{lscan['total_strict']}/{lscan['total_graded']} ({lsr:.0f}%)** &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"including near-ties: {lscan['total_generous']}/{lscan['total_graded']} ({lgr:.0f}%) "
                            f"&nbsp;&nbsp;(random baseline: ~33%){filter_note}"
                        )
                        st.markdown("**By prop — which one the method actually predicts best:**")
                        st.dataframe(pd.DataFrame(lscan["prop_rows"]), width='stretch')
                        with st.expander(f"Per-player breakdown ({lscan['players_graded']} players)"):
                            st.dataframe(pd.DataFrame(lscan["per_player_rows"]), width='stretch')
                        if lscan["errors"]:
                            with st.expander(f"{len(lscan['errors'])} player(s) skipped (name-match issues)"):
                                st.write(lscan["errors"])

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

                if p_pos == "RB":
                    st.divider()
                    st.markdown("### RB Run Concept Matchup")
                    st.caption(
                        "Separate premium dataset (FantasyPoints run-concept exports), not the "
                        "coverage CSVs above. All 6 real concepts always show - unlike coverage, "
                        "run concept is called by the OFFENSE, so ranking defenses by how often "
                        "they FACE a concept doesn't mean anything; the defense-allowed Quality "
                        "Score on each card IS the signal here. No 'longest rush' column exists "
                        "in this data - same real gap as longest catch, not guessed at."
                    )

                    def _rb_stat_block_html(row, tiers_source):
                        """RB rushing has its own real crucial-stat set (ATT/
                        YDS/YPC/Success%/MTF/ATT/YACO/ATT/TD) - NOT the same
                        as CURATED_STATS above, which is keyed for the
                        coverage-side receiving columns (TGT/REC/CR%) that
                        don't exist in this rushing data at all."""
                        curated = tuple(CRUCIAL_RB_STATS)
                        curated_html = _stat_rows_html(row, tiers_source, keys=curated)
                        remaining = [k for k in tiers_source if k not in curated]
                        if not remaining:
                            return curated_html
                        more_html = _stat_rows_html(row, tiers_source, keys=remaining)
                        return (
                            curated_html
                            + f'<details class="cov-more"><summary>+{len(remaining)} more stats</summary>{more_html}</details>'
                        )

                    rb_folder_mode = st.radio(
                        "RB data folder layout",
                        ["Two folders (no renaming needed)", "One folder (DEF_ prefix)"],
                        horizontal=True,
                        help="Two folders: upload your player-side folder and defense-allowed "
                             "folder exactly as you already have them named, no renaming. One "
                             "folder: matches the coverage_data convention (DEF_ prefix on the "
                             "defense-side filenames, everything in one folder).",
                    )
                    if rb_folder_mode.startswith("Two"):
                        rb_dir_col1, rb_dir_col2 = st.columns(2)
                        with rb_dir_col1:
                            rb_player_dir_input = st.text_input(
                                "Player-side folder (as named in your repo)",
                                value=st.session_state.get("rb_player_dir", "") or "RUSH METRICS",
                            )
                        with rb_dir_col2:
                            rb_def_dir_input = st.text_input(
                                "Defense-allowed folder (as named in your repo)",
                                value=st.session_state.get("rb_def_dir", "") or "RUSH METRICS ALLOWED",
                            )
                        rb_data_dir_input = None
                    else:
                        rb_data_dir_input = st.text_input(
                            "RB data folder (one flat folder, defense files prefixed DEF_)",
                            value=st.session_state.rb_data_dir or "rb_data",
                            help="Player-side: INSIDE_ZONE.csv, OUTSIDE_ZONE.csv, MAN-DUO.csv, "
                                 "COUNTER.csv, POWER.csv, PULL_LEAD.csv. Defense-allowed: same 6 "
                                 "names with DEF_ in front (DEF_INSIDE_ZONE.csv, etc.).",
                        )
                        rb_player_dir_input = rb_def_dir_input = None

                    if st.button("Load RB run-concept dataset", type="primary"):
                        with st.spinner("Loading all 6 run concepts, both sides..."):
                            try:
                                if rb_folder_mode.startswith("Two"):
                                    st.session_state.rb_bundle = load_full_rb_dataset(
                                        player_dir=rb_player_dir_input, def_dir=rb_def_dir_input,
                                    )
                                    st.session_state.rb_player_dir = rb_player_dir_input
                                    st.session_state.rb_def_dir = rb_def_dir_input
                                else:
                                    st.session_state.rb_bundle = load_full_rb_dataset(data_dir=rb_data_dir_input)
                                    st.session_state.rb_data_dir = rb_data_dir_input
                                n_missing = len(st.session_state.rb_bundle.missing)
                                if n_missing:
                                    st.warning(f"Loaded with {n_missing} file(s) missing - see details below.")
                                else:
                                    st.success("Loaded all 12 files (6 concepts x 2 sides) - dataset complete.")
                            except Exception as e:
                                st.error(f"Failed to load RB dataset: {e}")
                                st.session_state.rb_bundle = None

                    rb_bundle = st.session_state.rb_bundle
                    if rb_bundle is not None and rb_bundle.missing:
                        with st.expander(f"{len(rb_bundle.missing)} file(s) not found"):
                            st.write(rb_bundle.missing)

                    if rb_bundle is None:
                        st.info("Load the RB dataset above to see the run-concept matchup.")
                    else:
                        rb_report = get_rb_matchup(rb_bundle, p_name, opp, rb_team_name=player_team.strip() or None)
                        if "error" in rb_report[0]:
                            st.error(rb_report[0]["error"])
                        elif "note" in rb_report[0]:
                            st.info(rb_report[0]["note"])
                        else:
                            for entry in rb_report:
                                own_row = entry["own_row"]
                                def_row = entry["defense_allows"]
                                own_qs = (_quality_score(own_row.get("_tiers", {}), position="RB_RUSH",
                                                          thin_sample=own_row.get("_thin_sample", False))
                                          if own_row is not None else None)
                                own_qs_badge = (
                                    f'<span class="tier-badge {_score_badge_class(own_qs)}">Own Quality {own_qs:.0f}'
                                    f'{" ~" if own_row and own_row.get("_thin_sample") else ""}</span>'
                                    if own_qs is not None else ""
                                )
                                def_qs = (_quality_score(def_row.get("_tiers", {}), position="RB_RUSH",
                                                          thin_sample=def_row.get("_thin_sample", False))
                                          if def_row is not None else None)
                                def_qs_badge = (
                                    f'<span class="tier-badge {_score_badge_class(def_qs)}">Def Allows {def_qs:.0f}'
                                    f'{" ~" if def_row and def_row.get("_thin_sample") else ""}</span>'
                                    if def_qs is not None else ""
                                )
                                # Per-CONCEPT prop lean - real gap this closes: Own/Def
                                # Quality tells you if the matchup is good HERE, but not
                                # which prop that good matchup actually favors. Blends
                                # his own tier + the defense's tier 50/50 (same formula
                                # as the whole-matchup verdict below), just scoped to
                                # THIS one concept instead of blended across all 6.
                                concept_prop_scores = {}
                                for lean_prop, lean_stat_col in {"rush_attempts": "ATT", "rush_yards": "YDS"}.items():
                                    lean_own_tier = own_row.get("_tiers", {}).get(lean_stat_col) if own_row else None
                                    lean_def_tier = def_row.get("_tiers", {}).get(lean_stat_col) if def_row else None
                                    lean_own_w = TIER_WEIGHTS.get(lean_own_tier)
                                    lean_def_w = TIER_WEIGHTS.get(lean_def_tier)
                                    if lean_own_w is None and lean_def_w is None:
                                        continue
                                    concept_prop_scores[lean_prop] = (
                                        (0.5 * lean_own_w + 0.5 * lean_def_w) if (lean_own_w is not None and lean_def_w is not None)
                                        else (lean_own_w if lean_own_w is not None else lean_def_w)
                                    )
                                lean_badge = ""
                                if concept_prop_scores:
                                    lean_best = max(concept_prop_scores, key=concept_prop_scores.get)
                                    lean_label = "Rush Attempts" if lean_best == "rush_attempts" else "Rush Yards"
                                    lean_ties = [p for p, s in concept_prop_scores.items()
                                                 if p != lean_best and (concept_prop_scores[lean_best] - s) <= 10]
                                    lean_badge = f'<span class="cov-z-badge">Lean: {lean_label}{" (toss-up)" if lean_ties else ""}</span>'
                                if own_row is None:
                                    own_col_html = f'<div class="cov-no-data">{p_name}: no recorded attempts on this concept.</div>'
                                else:
                                    thin = '<span class="cov-thin-flag">THIN SAMPLE</span>' if entry["own_confidence"] == "thin_sample" else ""
                                    own_col_html = (
                                        f'<div class="cov-col-title">{p_name} — {own_row["_att"]} ATT{thin}</div>'
                                        + _rb_stat_block_html(own_row, own_row.get("_tiers", {}))
                                    )
                                if def_row is None:
                                    def_col_html = f'<div class="cov-no-data">{opp}: no defense-allowed data on this concept.</div>'
                                else:
                                    thin = '<span class="cov-thin-flag">THIN SAMPLE</span>' if entry["defense_confidence"] == "thin_sample" else ""
                                    def_col_html = (
                                        f'<div class="cov-col-title">{opp} allows — {def_row["_att"]} ATT{thin}</div>'
                                        + _rb_stat_block_html(def_row, def_row.get("_tiers", {}))
                                    )
                                rb_card_html = (
                                    '<div class="cov-card">'
                                    f'<div class="cov-card-header">{entry["concept"]}{own_qs_badge}{def_qs_badge}{lean_badge}</div>'
                                    '<div class="cov-grid">'
                                    f'<div class="cov-col">{own_col_html}</div>'
                                    f'<div class="cov-col">{def_col_html}</div>'
                                    '</div></div>'
                                )
                                st.markdown(rb_card_html, unsafe_allow_html=True)

                            st.divider()
                            st.markdown("### Best Prop Verdict (Rush Attempts vs Rush Yards)")
                            RB_PROP_LABELS = {"rush_attempts": "Rush Attempts", "rush_yards": "Rush Yards"}
                            RB_PROP_STAT_MAP = {"rush_attempts": "ATT", "rush_yards": "YDS"}

                            def _predict_best_rb_prop(rb_bundle, rb_name, opponent_team_full):
                                """Blends BOTH the RB's own tier AND the opponent's
                                defense-allowed tier per concept (50/50 when both
                                exist) - NOT just the RB's own tendency. An
                                earlier version of this only used his own side,
                                which meant the verdict never actually changed
                                based on who he was facing - the exact flaw
                                already caught in the WR backtest (Jefferson
                                predicting "Targets" every single week). Weighted
                                by his real attempt share per concept either way."""
                                rb_own_atts = {
                                    c: (rb_bundle.rb_vs_concept.get(c, {}).get(rb_name, {}) or {}).get("_att", 0)
                                    for c in RB_CONCEPT_FILES
                                }
                                total_att = sum(rb_own_atts.values())
                                weights = {c: (a / total_att) for c, a in rb_own_atts.items() if total_att and a > 0}
                                if not weights:
                                    return None
                                scores = {}
                                for prop, stat_col in RB_PROP_STAT_MAP.items():
                                    weighted_vals = []
                                    for concept, w in weights.items():
                                        own_row = rb_bundle.rb_vs_concept.get(concept, {}).get(rb_name)
                                        own_tier = own_row.get("_tiers", {}).get(stat_col) if own_row else None
                                        def_row = (rb_bundle.def_allowed.get(concept, {}).get(opponent_team_full)
                                                   if opponent_team_full else None)
                                        def_tier = def_row.get("_tiers", {}).get(stat_col) if def_row else None
                                        own_w = TIER_WEIGHTS.get(own_tier)
                                        def_w = TIER_WEIGHTS.get(def_tier)
                                        if own_w is None and def_w is None:
                                            continue
                                        blended = (0.5 * own_w + 0.5 * def_w) if (own_w is not None and def_w is not None) \
                                            else (own_w if own_w is not None else def_w)
                                        weighted_vals.append((w, blended))
                                    total_w = sum(w for w, _ in weighted_vals)
                                    scores[prop] = round(sum(w * s for w, s in weighted_vals) / total_w, 1) if total_w else None
                                valid = {p: s for p, s in scores.items() if s is not None}
                                if not valid:
                                    return {"scores": scores, "best": None, "ties": []}
                                best = max(valid, key=valid.get)
                                ties = [p for p, s in valid.items() if p != best and (valid[best] - s) <= 10]
                                return {"scores": scores, "best": best, "ties": ties}

                            rb_verdict = _predict_best_rb_prop(rb_bundle, p_name, opp)
                            if rb_verdict is None or rb_verdict["best"] is None:
                                st.info("Not enough real attempt volume across these concepts to determine a best prop.")
                            else:
                                rb_scores = rb_verdict["scores"]
                                rb_best = rb_verdict["best"]
                                rb_ties = rb_verdict["ties"]
                                rb_tie_note = f" (essentially tied with {', '.join(RB_PROP_LABELS[t] for t in rb_ties)})" if rb_ties else ""
                                rb_verdict_rows_html = "".join(
                                    f'<div class="stat-row"><span class="stat-label">{RB_PROP_LABELS[p]}</span>'
                                    f'<span class="stat-value">{rb_scores[p] if rb_scores[p] is not None else "no data"}'
                                    f'<span class="tier-badge {_score_badge_class(rb_scores[p])}">'
                                    f'{"BEST" if p == rb_best else ("TIE" if p in rb_ties else "")}</span></span></div>'
                                    for p in RB_PROP_LABELS
                                )
                                rb_verdict_html = (
                                    '<div class="cov-card">'
                                    f'<div class="cov-card-header">Best Prop: {RB_PROP_LABELS[rb_best]}{rb_tie_note}</div>'
                                    f'<div class="cov-card-usage">Blended across all 6 real concepts (his own tier + '
                                    f'{opp}\'s defense-allowed tier, 50/50), weighted by {p_name}\'s real attempt '
                                    f'share in each concept</div>'
                                    f'<div class="cov-col">{rb_verdict_rows_html}</div>'
                                    '</div>'
                                )
                                st.markdown(rb_verdict_html, unsafe_allow_html=True)
                            st.caption(
                                "TD isn't included yet - not a confirmed column in the real "
                                "game-log pipeline. Longest rush IS now real (see the line "
                                "comparison below) but isn't part of THIS verdict - there's no "
                                "season-aggregate column for it in the concept CSVs to predict "
                                "from, only real per-game data to compare a line against."
                            )

                            st.divider()
                            st.markdown("### Line Value + Backtest Reliability (RB)")
                            st.caption(
                                "Same merged table as the WR side: real mu/edge for whatever lines "
                                "you enter, PLUS this method's real historical hit rate for this RB "
                                "(run the RB backtest below first to populate that column). mu only "
                                "gets adjusted by the run-concept data when the signal is genuinely "
                                "strong (25+ points from neutral, not a thin sample) - a merely "
                                "Average concept grade leaves mu untouched."
                            )
                            RB_LINE_PROP_MAP = {"rush_attempts": "carries", "rush_yards": "rushing_yards",
                                                 "longest_rush": "longest_play"}
                            RB_LINE_PROP_LABELS = {"rush_attempts": "Rush Attempts", "rush_yards": "Rush Yards",
                                                    "longest_rush": "Longest Rush"}
                            rb_line_col1, rb_line_col2, rb_line_col3 = st.columns(3)
                            with rb_line_col1:
                                rb_att_line = st.number_input("Rush Attempts line", min_value=0.0, value=0.0, step=0.5, key="rb_att_line")
                            with rb_line_col2:
                                rb_yds_line = st.number_input("Rush Yards line", min_value=0.0, value=0.0, step=0.5, key="rb_yds_line")
                            with rb_line_col3:
                                rb_lng_line = st.number_input("Longest Rush line", min_value=0.0, value=0.0, step=0.5, key="rb_lng_line")

                            if st.button("Check RB value vs these lines", type="primary"):
                                try:
                                    rbl_pstats = pull_player_stats([int(game_log_season)])
                                    rbl_sched = pull_schedules([int(game_log_season)])
                                    rbl_pbp = pull_pbp([int(game_log_season)])
                                    rbl_matches = rbl_pstats[rbl_pstats["position"].astype(str).str.upper() == "RB"]
                                    rbl_name_col = "player_display_name" if "player_display_name" in rbl_pstats.columns else (
                                        "player_name" if "player_name" in rbl_pstats.columns else None)
                                    rbl_gsis = None
                                    if rbl_name_col:
                                        rbl_hit = rbl_matches[rbl_matches[rbl_name_col].astype(str).str.lower() == p_name.lower()]
                                        if not rbl_hit.empty:
                                            rbl_gsis = rbl_hit.iloc[0]["gsis_id"]
                                    if rbl_gsis is None:
                                        st.session_state["_rb_line_compare"] = {
                                            "error": f"Couldn't match '{p_name}' to a real nflreadpy record for {game_log_season}."}
                                    else:
                                        rb_lines = {"rush_attempts": rb_att_line, "rush_yards": rb_yds_line,
                                                    "longest_rush": rb_lng_line}
                                        rb_ubt_cached = st.session_state.get("_rb_backtest")
                                        rb_reliability = "run RB backtest below first"
                                        if rb_ubt_cached and not rb_ubt_cached.get("error") and rb_ubt_cached.get("graded"):
                                            rb_sr = rb_ubt_cached["strict_hits"] / rb_ubt_cached["graded"] * 100
                                            rb_reliability = f"{rb_ubt_cached['strict_hits']}/{rb_ubt_cached['graded']} ({rb_sr:.0f}%)"
                                        rb_line_rows = []
                                        for prop, line_val in rb_lines.items():
                                            if not line_val:
                                                continue
                                            stat_col = RB_LINE_PROP_MAP[prop]
                                            mu, sigma, n_games = _get_player_mu_sigma(
                                                rb_bundle, rbl_gsis, "RB", stat_col, rbl_pstats, rbl_sched,
                                                int(game_log_season), pbp=rbl_pbp,
                                            )
                                            if mu is None:
                                                rb_line_rows.append({"Prop": RB_LINE_PROP_LABELS[prop], "Line": line_val,
                                                                      "mu": "no data", "Premium Adj": "-", "sigma": "-",
                                                                      "Games": n_games, "p(Over)": "-", "Edge": "-",
                                                                      "Backtest Reliability": rb_reliability})
                                                continue
                                            # longest_rush has no season-aggregate concept
                                            # column to grade from - same reasoning as WR's
                                            # longest_catch, no adjustment applies to it.
                                            rb_q_score = rb_verdict["scores"].get(prop) if rb_verdict and prop in RB_PROP_STAT_MAP else None
                                            rb_has_solid = any(
                                                e.get("own_row") and not e["own_row"].get("_thin_sample")
                                                for e in rb_report if e.get("own_row")
                                            )
                                            rb_is_thin = not rb_has_solid
                                            rb_adj_mu, rb_adj_pct, rb_applied = _apply_best_signal_adjustment(mu, rb_q_score, rb_is_thin)
                                            scored = rescore_quality_mu_row_nfl(rb_adj_mu, line_val, sigma)
                                            rb_line_rows.append({
                                                "Prop": RB_LINE_PROP_LABELS[prop], "Line": line_val,
                                                "mu": round(mu, 1),
                                                "Premium Adj": f"{rb_adj_pct:+.1f}%" if rb_applied else "none (not extreme enough)",
                                                "sigma": round(sigma, 1), "Games": n_games,
                                                "p(Over)": scored["p_over"], "Edge": scored["edge"],
                                                "Backtest Reliability": rb_reliability,
                                            })
                                        st.session_state["_rb_line_compare"] = {"rows": rb_line_rows}
                                except Exception as e:
                                    st.session_state["_rb_line_compare"] = {"error": f"Line check failed: {e}"}

                            rbl = st.session_state.get("_rb_line_compare")
                            if rbl:
                                if rbl.get("error"):
                                    st.warning(rbl["error"])
                                elif not rbl.get("rows"):
                                    st.info("Enter at least one line above, then click the button.")
                                else:
                                    st.dataframe(pd.DataFrame(rbl["rows"]), width='stretch')

                            st.divider()
                            st.markdown("### Backtest This Method — Full Season (RB)")
                            st.caption(
                                "Same real backtest as the WR side: for every real game this RB "
                                "played, checks whether Rush Attempts or Rush Yards (whichever the "
                                "verdict above would've picked, using that week's REAL opponent) "
                                "actually was the better-performing prop that week."
                            )
                            RB_GAME_LOG_PROP_MAP = {"rush_attempts": "carries", "rush_yards": "rushing_yards"}

                            def _actual_best_rb_prop(tiers):
                                valid = {p: TIER_WEIGHTS[tiers[c]] for p, c in RB_GAME_LOG_PROP_MAP.items()
                                         if c in tiers and tiers[c] in TIER_WEIGHTS}
                                if not valid:
                                    return None, []
                                best = max(valid, key=valid.get)
                                ties = [p for p, s in valid.items() if p != best and (valid[best] - s) <= 10]
                                return best, ties

                            if st.button("Run RB season backtest", type="secondary"):
                                try:
                                    rbt_pstats = pull_player_stats([int(game_log_season)])
                                    rbt_sched = pull_schedules([int(game_log_season)])
                                    rbt_matches = rbt_pstats[rbt_pstats["position"].astype(str).str.upper() == "RB"]
                                    rbt_name_col = "player_display_name" if "player_display_name" in rbt_pstats.columns else (
                                        "player_name" if "player_name" in rbt_pstats.columns else None)
                                    rbt_gsis = None
                                    if rbt_name_col:
                                        rbt_hit = rbt_matches[rbt_matches[rbt_name_col].astype(str).str.lower() == p_name.lower()]
                                        if not rbt_hit.empty:
                                            rbt_gsis = rbt_hit.iloc[0]["gsis_id"]
                                    if rbt_gsis is None:
                                        st.session_state["_rb_backtest"] = {
                                            "error": f"Couldn't match '{p_name}' to a real nflreadpy record for {game_log_season}."}
                                    else:
                                        all_abbrevs = set(TEAM_ABBREV_TO_FULL.keys())
                                        rbt_log = build_coverage_crossref_game_log(
                                            rbt_gsis, "RB", all_abbrevs, rbt_pstats, rbt_sched,
                                            seasons=[int(game_log_season)], max_games=25,
                                        )
                                        rbt_rows, rbt_strict, rbt_generous, rbt_graded = [], 0, 0, 0
                                        for g in rbt_log:
                                            g_opp_full = TEAM_ABBREV_TO_FULL.get(g["opponent"])
                                            pred = _predict_best_rb_prop(rb_bundle, p_name, g_opp_full) if g_opp_full else None
                                            pred_best = pred["best"] if pred else None
                                            actual_best, actual_ties = _actual_best_rb_prop(g.get("tiers", {}))
                                            result = "—"
                                            if pred_best is not None and actual_best is not None:
                                                rbt_graded += 1
                                                if pred_best == actual_best:
                                                    rbt_strict += 1
                                                    rbt_generous += 1
                                                    result = "✅ Hit"
                                                elif pred_best in actual_ties:
                                                    rbt_generous += 1
                                                    result = "〰️ Tie"
                                                else:
                                                    result = "❌ Miss"
                                            rbt_rows.append({
                                                "Week": g["week"], "Opponent": g["opponent"],
                                                "Predicted Best": RB_PROP_LABELS.get(pred_best, "no data"),
                                                "Actual Best": RB_PROP_LABELS.get(actual_best, "no data"),
                                                "Result": result,
                                            })
                                        st.session_state["_rb_backtest"] = {
                                            "rows": rbt_rows, "strict_hits": rbt_strict,
                                            "generous_hits": rbt_generous, "graded": rbt_graded,
                                        }
                                except Exception as e:
                                    st.session_state["_rb_backtest"] = {"error": f"Backtest failed: {e}"}

                            rbt = st.session_state.get("_rb_backtest")
                            if rbt:
                                if rbt.get("error"):
                                    st.warning(rbt["error"])
                                elif rbt["graded"] == 0:
                                    st.info("No graded weeks - not enough real data to backtest this player/season.")
                                else:
                                    rsr = rbt["strict_hits"] / rbt["graded"] * 100
                                    rgr = rbt["generous_hits"] / rbt["graded"] * 100
                                    st.markdown(
                                        f"**Strict hit rate:** {rbt['strict_hits']}/{rbt['graded']} ({rsr:.0f}%) "
                                        f"&nbsp;&nbsp;|&nbsp;&nbsp; **Including near-ties:** {rbt['generous_hits']}/{rbt['graded']} ({rgr:.0f}%) "
                                        f"&nbsp;&nbsp;(random baseline with 2 props: ~50%)"
                                    )
                                    st.dataframe(pd.DataFrame(rbt["rows"]), width='stretch')

                            st.divider()
                            st.markdown("### Real Weekly Game Log — Cross-Referenced Opponents")
                            st.caption(
                                "Same idea as the WR/TE cross-reference above, adapted for a real "
                                "difference: run concept is called by the OFFENSE, so there's no "
                                "'top-N usage' to match teams on. Instead, this finds OTHER teams "
                                "whose defense-allowed grade lands in the SAME Elite/Poor direction "
                                "as this opponent, specifically on the concepts THIS RB actually "
                                "runs a lot (weighted by his real attempt share per concept) - then "
                                "pulls his real weekly game logs against those teams. Still an "
                                "APPROXIMATION, not verified per-play concept tracking - no source "
                                "tracks that."
                            )
                            rb_gl_col1, rb_gl_col2 = st.columns(2)
                            with rb_gl_col1:
                                rb_gl_season = st.number_input(
                                    "Game log season", min_value=2020, max_value=2030, value=2025, step=1,
                                    key="rb_gl_season",
                                )
                            with rb_gl_col2:
                                rb_min_match = st.number_input(
                                    "Min matching extreme concepts with another team",
                                    min_value=1, max_value=6, value=1, step=1, key="rb_min_match",
                                )

                            if st.button("Load RB cross-reference game log", type="secondary"):
                                try:
                                    own_atts = {
                                        c: rb_bundle.rb_vs_concept.get(c, {}).get(p_name, {}).get("_att", 0)
                                        for c in RB_CONCEPT_FILES
                                    }
                                    total_att = sum(own_atts.values())
                                    rb_own_weights = {c: (a / total_att) for c, a in own_atts.items() if total_att and a > 0}

                                    matches = {}
                                    for concept, w in rb_own_weights.items():
                                        opp_row = rb_bundle.def_allowed.get(concept, {}).get(opp)
                                        if opp_row is None or opp_row.get("_thin_sample"):
                                            continue
                                        opp_tier = opp_row.get("_tiers", {}).get("YDS")
                                        if opp_tier not in ("Elite", "Poor"):
                                            continue
                                        for team, row in rb_bundle.def_allowed.get(concept, {}).items():
                                            if team == opp or row.get("_thin_sample"):
                                                continue
                                            if row.get("_tiers", {}).get("YDS") == opp_tier:
                                                matches[team] = matches.get(team, 0) + 1
                                    cross_teams_full = sorted([t for t, c in matches.items() if c >= int(rb_min_match)])

                                    full_to_abbrevs = {}
                                    for abbr, full in TEAM_ABBREV_TO_FULL.items():
                                        full_to_abbrevs.setdefault(full, set()).add(abbr)
                                    cross_team_abbrevs = set()
                                    for full_name in cross_teams_full:
                                        cross_team_abbrevs |= full_to_abbrevs.get(full_name, set())

                                    rb_pstats = pull_player_stats([int(rb_gl_season)])
                                    rb_sched = pull_schedules([int(rb_gl_season)])
                                    rb_pbp = pull_pbp([int(rb_gl_season)])
                                    rb_matches_df = rb_pstats[rb_pstats["position"].astype(str).str.upper() == "RB"]
                                    rb_name_col = "player_display_name" if "player_display_name" in rb_pstats.columns else (
                                        "player_name" if "player_name" in rb_pstats.columns else None)
                                    rb_gsis = None
                                    if rb_name_col:
                                        rb_hit = rb_matches_df[rb_matches_df[rb_name_col].astype(str).str.lower() == p_name.lower()]
                                        if not rb_hit.empty:
                                            rb_gsis = rb_hit.iloc[0]["gsis_id"]

                                    if rb_gsis is None or not cross_team_abbrevs:
                                        st.session_state["_rb_crossref_game_log"] = {
                                            "error": (f"Couldn't match '{p_name}' to a real nflreadpy player record."
                                                      if rb_gsis is None else
                                                      "No cross-reference teams found at this threshold.")
                                        }
                                    else:
                                        rb_log = build_coverage_crossref_game_log(
                                            rb_gsis, "RB", cross_team_abbrevs, rb_pstats, rb_sched,
                                            seasons=[int(rb_gl_season)], pbp_df=rb_pbp,
                                        )
                                        st.session_state["_rb_crossref_game_log"] = {
                                            "log": rb_log, "cross_teams": sorted(cross_team_abbrevs),
                                        }
                                except Exception as e:
                                    st.session_state["_rb_crossref_game_log"] = {"error": f"Game log lookup failed: {e}"}

                            rb_crossref = st.session_state.get("_rb_crossref_game_log")
                            if rb_crossref:
                                if rb_crossref.get("error"):
                                    st.warning(rb_crossref["error"])
                                else:
                                    rb_gl_log = rb_crossref.get("log", [])
                                    st.caption(f"Cross-reference teams at this threshold: {', '.join(rb_crossref.get('cross_teams', [])) or 'none'}")
                                    if not rb_gl_log:
                                        st.info("No real games found against a cross-reference team at this threshold/season.")
                                    for g in rb_gl_log:
                                        g_tiers = g.get("tiers", {})
                                        g_stats = g.get("stats", {})
                                        g_qs = _quality_score(g_tiers)
                                        g_qs_badge = (
                                            f'<span class="tier-badge {_score_badge_class(g_qs)}">Quality {g_qs:.0f}</span>'
                                            if g_qs is not None else ""
                                        )
                                        g_rows_html = "".join(
                                            f'<div class="stat-row"><span class="stat-label">{s}</span>'
                                            f'<span class="stat-value">{g_stats.get(s)}'
                                            f'<span class="tier-badge {TIER_CLASS.get(g_tiers.get(s), "tier-average")}">{g_tiers.get(s, "-")}</span>'
                                            f'</span></div>'
                                            for s in g_stats
                                        )
                                        g_note_html = (
                                            f'<div class="cov-fit-warning">{g["sample_size_note"]}</div>'
                                            if g.get("sample_size_note") else ""
                                        )
                                        rb_game_card_html = (
                                            '<div class="cov-card">'
                                            f'<div class="cov-card-header">{g["season"]} Week {g["week"]} '
                                            f'— {g["team"]} vs {g["opponent"]}{g_qs_badge}</div>'
                                            f'{g_note_html}'
                                            f'<div class="cov-col">{g_rows_html}</div>'
                                            '</div>'
                                        )
                                        st.markdown(rb_game_card_html, unsafe_allow_html=True)

elif mode not in ("Draft Rankings", "Coverage Matchup (premium data)"):
    st.info("Click the button above to load this week's props.")
