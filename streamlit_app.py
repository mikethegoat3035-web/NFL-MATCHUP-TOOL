"""
streamlit_app.py
NFL Matchup Tool - main UI. Scans a week's slate, shows every prop with
mu-based inputs, and lets you type in a line per row to get live edge/p_over,
same workflow as the MLB tool's adjustable Best Edges table.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import itertools
import concurrent.futures
from datetime import datetime

# Real fix - the actual reported symptom ("loads for a bit then just stops
# scanning, no error") is the signature of a hung network call, not a
# crash: nflreadpy's underlying HTTP pulls don't have an explicit timeout
# anywhere in this codebase, so one slow/stalled request can block the
# entire week loop forever with nothing to show for it - no exception, no
# progress movement, nothing. This wraps one unit of work (one week) in a
# background thread with a hard wall-clock limit - if it doesn't finish in
# time, the loop gives up on that week and moves to the next one instead
# of hanging indefinitely.
BT_PER_WEEK_TIMEOUT_SECONDS = 180


def _run_with_timeout(fn, args, kwargs, timeout_seconds):
    """
    Runs fn(*args, **kwargs) in a background thread; returns (result, timed_out).
    Real bug caught and fixed during testing (see the MLB tool's identical
    fix) - using the executor as a context manager blocks on exit until
    the hung thread actually finishes, defeating the entire timeout.
    Managing the executor manually and calling shutdown(wait=False)
    detaches the still-running thread instead of waiting for it.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout_seconds)
        executor.shutdown(wait=False)
        return result, False
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        return None, True
from nfl_model_combined import (
    scan_full_slate_nfl, rescore_quality_mu_row_nfl, backtest_week, build_season_accuracy_report,
    build_season_simulation_backtest_report, build_combined_readiness_and_simulation_report,
    build_combined_report_from_raw, score_week_against_actuals,
    score_partial_game_week_against_actuals, add_simulation_columns_to_backtest_rows,
    diagnose_participation_data, get_player_matchup_explanation, diagnose_injuries_data,
    diagnose_alignment_data, pull_player_stats, pull_schedules, pull_pbp,
    build_coverage_crossref_game_log, diagnose_player_stats_for_game_log,
    build_longest_play_by_game, build_week_games_list,
    build_team_offense, simulate_matchup_n_times, real_over_rate_from_simulation,
    pull_rosters,
    # Real, merged in from coverage_matchup.py and rb_matchup.py (per direct
    # request, single-file consolidation matching the MLB tool's structure) -
    # both files' real content now lives directly inside nfl_model_combined.py.
    load_full_dataset, get_matchup, TEAM_ABBREV_TO_FULL, COVERAGE_FIELDS,
    _same_team, check_alignment_fit, ALIGNMENT_RTE_COLUMNS, ALIGNMENTS,
    load_full_rb_dataset, get_rb_matchup, CONCEPT_FILES as RB_CONCEPT_FILES,
    CRUCIAL_RB_STATS,
    build_defense_coverage_tendency_profile, calc_original_method_match_nfl,
    rescore_via_direct_hit_rate, build_rb_concept_usage_ranks,
    calc_original_method_match_nfl_for_prop, NFL_PROP_ORIGINAL_METHOD_STATS, _to_float,
    TEAM_ABBREV_TO_FULL_RB, scan_stage1_pass_catch_survivors, scan_stage1_rush_survivors,
    stage2_pass_catch_cross_reference, stage2_rush_cross_reference,
    scan_full_slate_simulation_nfl,
)

st.set_page_config(page_title="NFL Matchup Tool", layout="wide", page_icon="🏈")

# -----------------------------------------------------------------------
# DARK BLUE THEME - matches the MLB tool's look (dark background, blue
# accents), replacing the earlier Cowboys navy/silver/WHITE-background
# theme. Real fix, not just a rename: the old palette had two kinds of
# color reference mixed together - some through these variables (easy to
# redirect), but several were hardcoded light-mode assumptions directly
# in the CSS below (#fafafa card backgrounds, #f0f0f0 borders, #8a8a8a/
# #5a6b7a/#444 gray text meant to read on a WHITE page) that would have
# looked genuinely broken - light boxes, low-contrast text - sitting on
# a dark page if only the named variables were swapped. Every one of
# those got a real dark-mode equivalent below, not left as-is.
# This only restyles chrome (header, buttons, tabs, dataframe accents) -
# the scanner itself still covers every NFL team/matchup; it doesn't
# change what data is pulled or how anything is scored.
# -----------------------------------------------------------------------
DARK_BG = "#0E1117"
CARD_BG = "#1A1F2B"
ACCENT_BLUE = "#3B82F6"
TEXT_LIGHT = "#FAFAFA"
MUTED_TEXT = "#9CA3AF"
BORDER_DARK = "#2D3340"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {DARK_BG};
    }}
    [data-testid="stHeader"] {{
        background-color: {CARD_BG};
    }}
    h1, h2, h3 {{
        color: {TEXT_LIGHT} !important;
    }}
    .stRadio > label, .stNumberInput > label, .stSelectbox > label {{
        color: {TEXT_LIGHT} !important;
        font-weight: 600;
    }}
    div.stButton > button {{
        background-color: {ACCENT_BLUE};
        color: {TEXT_LIGHT};
        border: 1px solid {BORDER_DARK};
        border-radius: 6px;
        font-weight: 600;
    }}
    div.stButton > button:hover {{
        background-color: {CARD_BG};
        color: {ACCENT_BLUE};
        border: 1px solid {ACCENT_BLUE};
    }}
    /* Body text (markdown, write, caption) reads light against the dark
       app background - mirrors the same fix the old theme needed, just
       inverted for a dark base instead of a light one. */
    .stApp, .stApp p, .stApp li, .stApp span,
    .stMarkdown, [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {{
        color: {TEXT_LIGHT} !important;
    }}
    input, textarea,
    .stTextArea textarea, .stTextInput input, .stNumberInput input,
    [data-testid="stTextAreaContainer"] textarea,
    [data-testid="stTextInputRootElement"] input {{
        color: {TEXT_LIGHT} !important;
        -webkit-text-fill-color: {TEXT_LIGHT} !important;
    }}
    /* Coverage Matchup cards - real dark-mode equivalents, not just a
       renamed light palette. Card backgrounds were #fff/#fafafa (light
       boxes on white); now use the same dark card color as everywhere
       else. Borders were light grays (#ddd/#f0f0f0) meant to be subtle
       against white - now a dark border that's subtle against dark
       instead. Gray label text (#5a6b7a/#8a8a8a/#444) was tuned for
       readability on white - now uses MUTED_TEXT, tuned for readability
       on dark. */
    .cov-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_DARK};
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }}
    .cov-card-header {{
        font-size: 19px;
        font-weight: 700;
        color: {TEXT_LIGHT};
        margin-bottom: 2px;
    }}
    .cov-card-usage {{
        font-size: 13px;
        color: {MUTED_TEXT};
        margin-bottom: 6px;
    }}
    .cov-z-badge {{
        display: inline-block;
        padding: 1px 9px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: 700;
        color: {TEXT_LIGHT};
        background: {ACCENT_BLUE};
        margin-left: 6px;
    }}
    .cov-fit-warning {{
        font-size: 12px;
        color: #ff6b6b;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    .cov-grid {{
        display: flex;
        gap: 28px;
        margin-top: 10px;
    }}
    .cov-align-block {{
        background: {DARK_BG};
        border: 1px solid {BORDER_DARK};
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
    }}
    .cov-align-header {{
        font-weight: 700;
        color: {TEXT_LIGHT};
        font-size: 13px;
        margin-bottom: 4px;
    }}
    .cov-col {{
        flex: 1;
        min-width: 0;
    }}
    .cov-col-title {{
        font-weight: 700;
        color: {TEXT_LIGHT};
        font-size: 14px;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 2px solid {ACCENT_BLUE};
    }}
    .cov-thin-flag {{
        font-size: 11px;
        font-weight: 700;
        color: {TEXT_LIGHT};
        background: #b02a37;
        padding: 1px 6px;
        border-radius: 8px;
        margin-left: 6px;
    }}
    .cov-no-data {{
        font-size: 13px;
        color: {MUTED_TEXT};
        font-style: italic;
    }}
    .stat-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid {BORDER_DARK};
        font-size: 13px;
    }}
    .stat-label {{
        color: {MUTED_TEXT};
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
        color: {TEXT_LIGHT};
        white-space: nowrap;
    }}
    .tier-elite {{ background: #1b7a3d; }}
    .tier-above-avg {{ background: #66bb6a; }}
    .tier-average {{ background: #6b7280; }}
    .tier-below-avg {{ background: #ef8c1e; }}
    .tier-poor {{ background: #c0392b; }}
    .cov-more {{
        margin-top: 6px;
    }}
    .cov-more summary {{
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
        color: {MUTED_TEXT};
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
        color: {TEXT_LIGHT};
    }}
    .nfl-banner {{
        background: linear-gradient(90deg, {CARD_BG} 0%, {ACCENT_BLUE} 100%);
        padding: 14px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
    }}
    .nfl-banner h1 {{
        color: {TEXT_LIGHT} !important;
        margin: 0;
        font-size: 28px;
    }}
    .nfl-banner p {{
        color: {TEXT_LIGHT} !important;
        margin: 2px 0 0 0;
        opacity: 0.9;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="nfl-banner"><h1>🏈 NFL Matchup Tool</h1>'
    '<p>Scan a week\'s slate across the league, then type in lines per row for live edge/probability.</p></div>',
    unsafe_allow_html=True,
)


# DEPLOY VERSION MARKER - bump this string on every file delivered, so a
# glance at the app tells you in 5 seconds whether a new deploy actually
# took effect, instead of waiting through a full readiness-report run to
# find out indirectly. If this doesn't match what was just sent, the
# deploy didn't land - no need to test anything further until it does.
DEPLOY_VERSION = "v51-play-by-play-sim-merged-scrambles-rushconcepts-2026-08-29"
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

# REAL, SAFE CONSOLIDATION (per direct request) - the Mode selector
# radio is gone. Draft Rankings and Coverage Matchup (premium data)
# have both been removed, so the only real choice left was Weekly
# Scan vs Season Backtest - now both always render on one continuous
# page instead of a toggle, matching the MLB tool's single-page
# layout (scan at the top, backtest at the bottom).
#
# IMPORTANT: mode is set to a value that does NOT equal "Weekly Scan /
# Draft Rankings" on purpose - every real, live scan feature (the
# real week-number input, "Scan one game at a time", "Scan full
# slate") lives in the ELSE branch of "if mode == 'Weekly Scan /
# Draft Rankings'" checks throughout this file (that string has meant
# "show draft-rankings-style inputs" this whole time, a confusing
# legacy naming quirk from an earlier consolidation, confirmed by
# checking each comparison directly rather than assumed). Setting
# mode to match that string here would have SILENTLY DISABLED the
# real scan functionality - caught and fixed before delivery by
# re-checking which branch actually contains the live scan code.
mode = "__scan__"


if True:
    col1, col2 = st.columns(2)
    with col1:
        if mode == "Weekly Scan / Draft Rankings":
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
        if mode == "Weekly Scan / Draft Rankings":
            st.caption("Draft Rankings uses the prior completed season as the projection basis - "
                       "week isn't used in this mode.")
            week = None
        else:
            week = st.number_input("Week", min_value=1, max_value=18, value=10, step=1)

if "slate_df" not in st.session_state:
    st.session_state.slate_df = None
if "sim_scan_df" not in st.session_state:
    st.session_state.sim_scan_df = None
if "backtest_mode" not in st.session_state:
    st.session_state.backtest_mode = False
if "draft_rankings_df" not in st.session_state:
    st.session_state.draft_rankings_df = None
if "season_report" not in st.session_state:
    st.session_state.season_report = None
if "qs_report" not in st.session_state:
    st.session_state.qs_report = None
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

if mode == "Weekly Scan / Draft Rankings":
    pass  # REAL, SAFE REMOVAL (per direct request) - Draft Rankings removed
    # entirely. The "League Settings" UI and "Build Draft Rankings" button
    # that used to live here (which populated st.session_state.
    # draft_rankings_df) have been deleted - draft_rankings_df now stays
    # None permanently, which makes the rankings-display branch further
    # down in this file (originally reached via "if draft_rankings_df is
    # not None") correctly, safely unreachable without needing to touch
    # that branch's own code directly.
else:
    # REAL BUG FOUND AND FIXED: this whole section (including the Season
    # Readiness Report / "Run Readiness Report for this week range"
    # button) was gated behind mode.startswith("Backtest") - but the
    # separate "Backtest" mode option was removed from the mode selector
    # a while back (consolidated elsewhere), and this check was never
    # updated to match. Since this else-branch is only ever reached for
    # "Season Backtest" mode now (Draft Rankings and Coverage
    # Matchup are both handled by earlier branches above), btn_col2 -
    # and the real, working build_season_accuracy_report tool inside it -
    # has been permanently unreachable dead code, not something tonight's
    # changes broke. Always create both columns now.
    # ---------------------------------------------------------------------
    # Per-game scan picker - REAL per-game scanning (uses team_filter to
    # skip the expensive per-player scoring loop for every team not
    # picked), not just a display filter on an already-fully-scanned week.
    # Uses the real schedule + build_week_games_list, which already
    # existed in the backend but was never wired into any UI until now.
    # ---------------------------------------------------------------------
    st.subheader("Scan one game at a time")
    try:
        _picker_sched = pull_schedules([int(season)])
        _week_games = build_week_games_list(int(season), int(week), _picker_sched)
    except Exception:
        _week_games = pd.DataFrame(columns=["away_team", "home_team", "matchup"])

    if _week_games.empty:
        st.caption("No games found for this season/week yet - check back once the schedule posts, or use "
                   "\"Scan full slate\" below for everyone at once.")
    else:
        st.caption("Each box scans only that game's two teams - real per-game scanning, not a display "
                   "filter on an already-scanned week. Use \"Scan full slate\" below instead if you want everyone at once.")
        game_cols = st.columns(4)
        for i, g in enumerate(_week_games.itertuples()):
            with game_cols[i % 4]:
                if st.button(g.matchup, key=f"game_box_{g.away_team}_{g.home_team}", use_container_width=True):
                    with st.spinner(f"Scanning just {g.matchup}..."):
                        try:
                            st.session_state.slate_df = scan_full_slate_nfl(
                                int(season), int(week),
                                coverage_bundle=st.session_state.get("coverage_bundle"),
                                rb_bundle=st.session_state.get("rb_bundle"),
                                team_filter=[g.away_team, g.home_team],
                            )
                            st.session_state.backtest_mode = False
                            st.session_state.show_season_report = False
                            st.success(f"Loaded {len(st.session_state.slate_df)} prop rows for {g.matchup}.")
                        except Exception as e:
                            st.error(f"Scan failed: {e}")

    st.divider()

    # REAL FIX (confirmed bug, found via direct user report) - the main
    # scan silently ran with ZERO coverage data whenever the separate
    # "Load coverage dataset" button (in a different section of the app)
    # hadn't been clicked first - no error, no warning, just a
    # degraded result that looked complete but wasn't (every coverage
    # column blank, mu never actually coverage-adjusted). This is a
    # loud, impossible-to-miss warning right where the scan button
    # lives, instead of a silent gap discovered only after the fact.
    if st.session_state.get("coverage_bundle") is None:
        st.warning(
            "⚠️ Coverage dataset not loaded yet - scanning now will run WITHOUT any coverage "
            "data (every coverage-related column will be blank, and mu won't get the real, "
            "direct coverage adjustment). Scroll to the 'Coverage data folder' section further "
            "down, click 'Load coverage dataset' there first, then come back and scan."
        )

    button_label = "Scan full slate"

    if st.button(button_label, type="primary"):
        with st.spinner(f"Pulling and scoring Week {week}, {season}..."):
            try:
                if mode.startswith("Backtest"):
                    st.session_state.slate_df = backtest_week(
                        season, week,
                        coverage_bundle=st.session_state.get("coverage_bundle"),
                        rb_bundle=st.session_state.get("rb_bundle"),
                    )
                    st.session_state.backtest_mode = True
                    st.session_state.show_season_report = False
                else:
                    st.session_state.slate_df = scan_full_slate_nfl(
                        season, week,
                        coverage_bundle=st.session_state.get("coverage_bundle"),
                        rb_bundle=st.session_state.get("rb_bundle"),
                    )
                    st.session_state.backtest_mode = False
                    st.session_state.show_season_report = False
                st.success(f"Loaded {len(st.session_state.slate_df)} prop rows.")
            except Exception as e:
                st.error(f"{'Backtest' if mode.startswith('Backtest') else 'Scan'} failed: {e}")
                st.session_state.slate_df = None

    st.divider()
    st.header("🎲 Monte Carlo Simulation Scan")
    st.caption(
        "Real, genuine Monte Carlo simulation - the same architecture as the MLB tool. "
        "For every real player on this week's slate, simulates 1000 real games "
        "(target-by-target for receivers, carry-by-carry for backs, attempt-by-attempt "
        "for QBs), using his own real per-coverage/per-concept rates against the "
        "opponent's real, matchup-specific tendencies. Then applies the same real "
        "z-score/CV filter already proven throughout this tool - not a separate or "
        "lesser standard."
    )
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_n_simulations_nfl = st.number_input("Simulations per player", min_value=100, max_value=5000,
                                                  value=1000, step=100, key="sim_n_nfl")
    with sim_col2:
        sim_min_zscore_nfl = st.slider("Minimum real edge (z-score)", 0.0, 3.0, 1.5, step=0.1, key="sim_min_z_nfl")
    with sim_col3:
        sim_max_cv_nfl = st.slider("Maximum real CV (consistency)", 0.1, 2.0, 1.0, step=0.1, key="sim_max_cv_nfl")

    if st.button("Run Monte Carlo simulation scan", type="primary"):
        if st.session_state.get("coverage_bundle") is None or st.session_state.get("rb_bundle") is None:
            st.error("Both the coverage dataset AND the RB concept dataset need to be loaded first - "
                     "scroll down to their sections, load both, then come back and run this.")
        else:
            with st.spinner(f"Running {sim_n_simulations_nfl} real, simulated games for every real "
                             f"player in Week {week}, {season}..."):
                try:
                    sim_schedules = pull_schedules([int(season)])
                    sim_games = build_week_games_list(int(season), int(week), sim_schedules)
                    sim_opponent_pc, sim_opponent_rb = {}, {}
                    for _, g in sim_games.iterrows():
                        away_full = TEAM_ABBREV_TO_FULL.get(g["away_team"], g["away_team"])
                        home_full = TEAM_ABBREV_TO_FULL.get(g["home_team"], g["home_team"])
                        sim_opponent_pc[away_full] = home_full
                        sim_opponent_pc[home_full] = away_full
                        away_rb = TEAM_ABBREV_TO_FULL_RB.get(g["away_team"], g["away_team"])
                        home_rb = TEAM_ABBREV_TO_FULL_RB.get(g["home_team"], g["home_team"])
                        sim_opponent_rb[away_rb] = home_rb
                        sim_opponent_rb[home_rb] = away_rb
                    sim_rosters = pull_rosters([int(season)])
                    st.session_state.sim_scan_df = scan_full_slate_simulation_nfl(
                        st.session_state.coverage_bundle, st.session_state.rb_bundle,
                        sim_rosters, sim_opponent_pc, sim_opponent_rb,
                        n_simulations=int(sim_n_simulations_nfl),
                    )
                    st.success(f"Simulated {len(st.session_state.sim_scan_df)} real player/prop combinations.")
                except Exception as e:
                    st.error(f"Simulation scan failed: {e}")
                    st.session_state.sim_scan_df = None

    if st.session_state.get("sim_scan_df") is not None and not st.session_state.sim_scan_df.empty:
        sim_survivors = st.session_state.sim_scan_df[
            (st.session_state.sim_scan_df["zscore"] >= sim_min_zscore_nfl)
            & (st.session_state.sim_scan_df["cv"] <= sim_max_cv_nfl)
        ]
        st.subheader(f"Kept simulated results ({len(sim_survivors)} of {len(st.session_state.sim_scan_df)})")
        st.dataframe(sim_survivors, width='stretch')

    st.divider()
    st.header("🎯 Stage 1 / Stage 2 - Coverage & Concept Survivors")
    st.caption(
        "Stage 1: scans the WHOLE real slate at once - does this player's own real "
        "performance clear the bar on a real MAJORITY of tonight's specific opponent's "
        "meaningfully-used coverages (pass/pass-catching props), or is this RB elite in "
        "his own dominant run concept AND facing a real defense that's genuinely weak "
        "defending that same concept (rush props)? No line needed yet - this just finds "
        "who has a genuinely high-quality real mu. Stage 2: once you enter a real line for "
        "a survivor, finds that SAME player's own real past games against OTHER teams that "
        "also share that same real tendency, and shows how many of those specific games "
        "actually cleared your line - a real '8/12' style read."
    )

    if st.session_state.get("coverage_bundle") is None or st.session_state.get("rb_bundle") is None:
        st.warning(
            "⚠️ Needs BOTH the coverage dataset AND the RB concept dataset loaded first "
            "(scroll down to their sections) - Stage 1 can't run without either one."
        )
    elif st.button("Scan Stage 1 Survivors", key="stage1_scan_btn"):
        with st.spinner("Scanning the whole real slate for coverage & concept survivors..."):
            try:
                schedules_df = pull_schedules([season])
                games_df = build_week_games_list(season, week, schedules_df)
                opponent_by_team_pc = {}
                opponent_by_team_rush = {}
                for _, g in games_df.iterrows():
                    away_pc = TEAM_ABBREV_TO_FULL.get(g["away_team"], g["away_team"])
                    home_pc = TEAM_ABBREV_TO_FULL.get(g["home_team"], g["home_team"])
                    opponent_by_team_pc[away_pc] = home_pc
                    opponent_by_team_pc[home_pc] = away_pc
                    away_rb = TEAM_ABBREV_TO_FULL_RB.get(g["away_team"], g["away_team"])
                    home_rb = TEAM_ABBREV_TO_FULL_RB.get(g["home_team"], g["home_team"])
                    opponent_by_team_rush[away_rb] = home_rb
                    opponent_by_team_rush[home_rb] = away_rb

                rosters_df = pull_rosters([season])
                week_rosters_df = rosters_df[rosters_df["position"].isin(["QB", "RB", "WR", "TE"])].drop_duplicates("gsis_id")

                pc_survivors = scan_stage1_pass_catch_survivors(
                    st.session_state.coverage_bundle, week_rosters_df, opponent_by_team_pc,
                )
                rush_survivors = scan_stage1_rush_survivors(
                    st.session_state.rb_bundle, week_rosters_df, opponent_by_team_rush,
                )
                st.session_state.stage1_pc_survivors = pc_survivors
                st.session_state.stage1_rush_survivors = rush_survivors
                st.success(
                    f"Stage 1 complete - {len(pc_survivors)} pass/pass-catching survivors, "
                    f"{len(rush_survivors)} rush-concept survivors."
                )
            except Exception as e:
                st.error(f"Stage 1 scan failed: {e}")

    pc_survivors = st.session_state.get("stage1_pc_survivors")
    rush_survivors = st.session_state.get("stage1_rush_survivors")

    if (pc_survivors is not None and not pc_survivors.empty) or (rush_survivors is not None and not rush_survivors.empty):
        st.subheader("Stage 1 survivors")
        if pc_survivors is not None and not pc_survivors.empty:
            st.write("Pass / pass-catching")
            st.dataframe(
                pc_survivors[["player", "team", "opponent", "prop_type", "coverages_qualifying", "coverages_scored", "read"]],
                width="stretch", hide_index=True,
            )
        if rush_survivors is not None and not rush_survivors.empty:
            st.write("Rush concept")
            st.dataframe(
                rush_survivors[["player", "team", "opponent", "prop_type", "dominant_concept",
                                 "own_percentile", "defense_allowed_percentile", "read"]],
                width="stretch", hide_index=True,
            )

        st.subheader("Stage 2 - enter a real line for any survivor above")
        stage2_col1, stage2_col2, stage2_col3 = st.columns(3)
        with stage2_col1:
            all_survivor_names = []
            if pc_survivors is not None and not pc_survivors.empty:
                all_survivor_names += (pc_survivors["player"] + " - " + pc_survivors["prop_type"]).tolist()
            if rush_survivors is not None and not rush_survivors.empty:
                all_survivor_names += (rush_survivors["player"] + " - " + rush_survivors["prop_type"]).tolist()
            stage2_pick = st.selectbox("Survivor", all_survivor_names, key="stage2_pick") if all_survivor_names else None
        with stage2_col2:
            stage2_line = st.number_input("Real line", min_value=0.0, value=50.0, step=0.5, key="stage2_line")
        with stage2_col3:
            stage2_years_back = st.number_input("Seasons of history to check", min_value=1, max_value=5, value=2, key="stage2_years")

        if stage2_pick and st.button("Run Stage 2 cross-reference", key="stage2_run_btn"):
            with st.spinner("Cross-referencing real past games..."):
                try:
                    player_name_picked, prop_picked = stage2_pick.rsplit(" - ", 1)
                    player_stats_hist = pull_player_stats(list(range(season - stage2_years_back, season)))

                    pc_match = pc_survivors[
                        (pc_survivors["player"] == player_name_picked) & (pc_survivors["prop_type"] == prop_picked)
                    ] if pc_survivors is not None and not pc_survivors.empty else pd.DataFrame()
                    rush_match = rush_survivors[
                        (rush_survivors["player"] == player_name_picked) & (rush_survivors["prop_type"] == prop_picked)
                    ] if rush_survivors is not None and not rush_survivors.empty else pd.DataFrame()

                    if not pc_match.empty:
                        row = pc_match.iloc[0]
                        team_full = TEAM_ABBREV_TO_FULL.get(row["team"], row["team"])
                        result = stage2_pass_catch_cross_reference(
                            row["gsis_id"], row["prop_type"], row["qualifying_coverage_fields"][0],
                            st.session_state.coverage_bundle, stage2_line, player_stats_hist,
                            exclude_team_full=team_full,
                        )
                    elif not rush_match.empty:
                        row = rush_match.iloc[0]
                        team_full = TEAM_ABBREV_TO_FULL_RB.get(row["team"], row["team"])
                        result = stage2_rush_cross_reference(
                            row["gsis_id"], row["prop_type"], row["dominant_concept"],
                            st.session_state.rb_bundle, stage2_line, player_stats_hist,
                            exclude_team_full=team_full,
                        )
                    else:
                        result = {"usable": False, "reason": "survivor not found - try re-running Stage 1"}

                    if result.get("usable"):
                        st.success(result["read"])
                        st.metric("Real hit rate", f"{result['hits']}/{result['total']}", f"{result['hit_rate']*100:.0f}%")
                    else:
                        st.warning(result.get("reason", "Not usable."))
                except Exception as e:
                    st.error(f"Stage 2 cross-reference failed: {e}")


# -----------------------------------------------------------------------
# SEASON READINESS REPORT DISPLAY - pulled into a real function so it can
# render in TWO places: standalone (if only the backtest has ever been
# run, slate_df still empty) AND stacked directly below the live scan
# results/Slip Builder/Locked Slips when BOTH have data - matching the
# MLB tool's layout (live scan on top, backtest below, both visible
# together on one page) instead of the old mutually-exclusive toggle
# where running one hid the other entirely.
# -----------------------------------------------------------------------
def _render_season_report(report):
    raw = report["raw"]

    if raw.empty:
        st.warning("No scoreable rows came back for this season - check that the season has completed weeks with real player_stats data.")
        return
    st.subheader(f"Season Readiness Report — {report.get('season', season)}")
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
if (st.session_state.slate_df is None or st.session_state.slate_df.empty) \
        and st.session_state.season_report is not None:
    # Standalone case: backtest has been run, but the live scan never has
    # (or its results were cleared) - nothing to stack it below yet, so it
    # renders on its own, same as before. Checks season_report directly,
    # NOT show_season_report (that flag flips back to False the instant
    # the regular scan button is clicked, regardless of whether real
    # backtest data still exists - reusing it here would silently hide
    # the backtest again the moment a new scan runs, the exact problem
    # this whole change is meant to fix).
    _render_season_report(st.session_state.season_report)

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
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
    with fcol1:
        prop_types = ["All"] + sorted(df["prop_type"].dropna().unique().tolist())
        prop_filter = st.selectbox("Prop type", prop_types)
    with fcol2:
        positions = ["All"] + sorted(df["position"].dropna().unique().tolist())
        position_filter = st.selectbox("Position", positions)
    with fcol3:
        if not st.session_state.backtest_mode:
            # Defaulted to 0.10, matching the MLB tool's real base filter -
            # was 0.0 (off) before, real gap: NFL had no baseline edge floor
            # at all here, unlike MLB's established .10/10 starting point.
            min_edge_filter = st.slider("Minimum edge (after entering lines)", 0.0, 1.0, 0.10, 0.05)
        else:
            min_edge_filter = 0.0
    with fcol4:
        # Combined with min_edge_filter below (both must clear, when set) -
        # quality_score and edge are otherwise completely independent
        # numbers (edge comes purely from mu/line/sigma; quality_score
        # never feeds into that calculation) - this is the actual gate
        # that makes "only show me the highest quality matchups" real
        # instead of something you have to eyeball across two separate
        # columns yourself. Applies across EVERY prop_type, Scan and
        # Backtest alike (not just the old pass_yards/rec_yards-only
        # Best Quality Matchups panel below).
        # REAL FIX - defaulted from 0 (show everything) to 55, mirroring
        # the same real lesson from MLB's hitter/pitcher threshold work:
        # a filter default of 0 means every player shows up, defeating
        # the point of a "show me the great ones" quality filter. Checked
        # the real, actual scale first rather than guessing - quality_
        # score never reaches 70-100 the way a naive 0-100 slider
        # implies (QB/RB cap at 66, WR/TE rarely exceed 70), and the
        # real 90th percentile clusters at 54-57 fairly evenly across
        # all four positions - unlike MLB, this one didn't need a
        # position-specific split, just a default that matches the
        # real, observed scale.
        min_quality_filter = st.slider("Minimum quality_score", 0, 100, 60, 5,
                                        help="Applies to every prop type. 0 = off. Real quality_score "
                                             "never reaches 70-100 in practice - 60 reflects a tighter, "
                                             "more selective real cut across every position.")
    with fcol5:
        # REAL GAP FOUND: no games_sampled floor existed anywhere in this
        # section at all before - MLB's real base filter is edge>=.10 AND
        # games>=10 together, not edge alone. Defaulted to 10 to match
        # that same real baseline, using games_sampled_current (this
        # player's own real current-season sample, already computed via
        # get_data_confidence elsewhere in this file).
        min_games_filter = st.slider("Minimum games_sampled", 0, 17, 10, 1,
                                      help="Real games behind this player's own current-season mu. 0 = off.")

    filtered = df.copy()
    if prop_filter != "All":
        filtered = filtered[filtered["prop_type"] == prop_filter]
    if position_filter != "All":
        filtered = filtered[filtered["position"] == position_filter]
    if min_quality_filter > 0 and "quality_score" in filtered.columns:
        filtered = filtered[filtered["quality_score"].fillna(0) >= min_quality_filter]
    # REAL FIX (confirmed bug) - was filtering on games_sampled_current
    # alone, which is always 0 at week 1 (mathematically correct in
    # isolation, but hides the real, substantial prior-season data
    # actually backing the mu) - forced sliding this filter to 0 just to
    # see any real results. games_sampled_total (current + fallback,
    # confirmed mutually exclusive) reflects the real, total sample size.
    if min_games_filter > 0 and "games_sampled_total" in filtered.columns:
        filtered = filtered[filtered["games_sampled_total"].fillna(0) >= min_games_filter]
    elif min_games_filter > 0 and "games_sampled_current" in filtered.columns:
        filtered = filtered[filtered["games_sampled_current"].fillna(0) >= min_games_filter]

    # REAL, NEW - per direct, explicit request, built and validated
    # tonight (confirmed end-to-end: Maye's longest_completion vs Cover 4
    # correctly shows True here, matching what was found by hand earlier)
    # - requires a genuine MAJORITY of the player's own real, relevant
    # metrics to be Elite (90th percentile, properly sample-filtered) for
    # an over, or Poor (10th percentile) for an under. This is the
    # strictest available filter - real backtesting tonight found this
    # standard returns very few results most weeks (that's expected and
    # correct, not a bug - see the "1 of 6 checks passes" finding).
    stage1_elite_only = st.checkbox(
        "Stage 1: require HIS OWN metrics to be majority-Elite/Poor",
        value=False,
        help="The strictest real filter available. Confirmed via testing tonight that this "
             "returns very few real results most weeks - that's genuine selectivity, not a bug.",
    )
    if stage1_elite_only and "stage1_player_majority_elite" in filtered.columns:
        filtered = filtered[
            (filtered["p_over"] >= 0.5) & (filtered["stage1_player_majority_elite"].fillna(False))
            | (filtered["p_over"] < 0.5) & (filtered["stage1_player_majority_poor"].fillna(False))
        ]

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

        # Curated view by default, same MLB-style "player/team/opp/mu/edge"
        # shape, not a full column dump - see full note further down where
        # this same toggle also controls the post-scoring display table.
        show_full_diagnostics = st.checkbox(
            "Show full diagnostic columns (coverage breakdown, advanced grades, "
            "play-action/personnel/alignment signals)",
            value=False,
            help="Off by default for a cleaner, MLB-style slate view. Turn on to "
                 "see exactly which signals feed a specific row's quality_score.",
        )

        # REAL, NEW continuity filter - per direct request, lets the
        # user hide rows that don't have FULL CONTINUITY (own QB/OC
        # unchanged AND opponent's real DC unchanged) - real signal
        # only matters weeks 1-5, since every team should have enough
        # real current-season data by then that this stops mattering.
        # Defaults ON for weeks 1-5, OFF after, per direct instruction.
        if "continuity_confidence" in filtered.columns:
            default_continuity_filter = week is not None and week <= 5
            only_full_continuity = st.checkbox(
                "Only show FULL CONTINUITY rows (own QB/play-caller unchanged AND "
                "opponent's real DC unchanged)",
                value=default_continuity_filter,
                help="Most relevant weeks 1-5, while mu still leans on prior-season "
                     "fallback data - after that, every team should have enough real "
                     "current-season data that this stops being the deciding factor.",
            )
            if only_full_continuity:
                filtered = filtered[
                    (filtered["continuity_confidence"] == "FULL CONTINUITY")
                    | (filtered["continuity_confidence"].isna())  # kickers/etc - concept doesn't apply, don't hide them
                ]

        core_editor_cols = ["player_display_name", "team", "opponent", "matchup",
                             "position", "prop_type", "line", "mu", "sigma",
                             "continuity_confidence", "data_confidence", "games_sampled_total", "games_sampled_current", "quality_score"]
        editor_col_order = core_editor_cols if not show_full_diagnostics else None
        editor_col_order = [c for c in editor_col_order if c in filtered.columns] if editor_col_order else None

        edited = st.data_editor(
            filtered,
            column_config={
                "line": st.column_config.NumberColumn("line", help="Enter the book/DFS line for this prop"),
            },
            column_order=editor_col_order,
            disabled=[c for c in filtered.columns if c not in ("line",)],
            num_rows="fixed",
            width='stretch',
            key="slate_editor",
        )

        # Real, one-time cache - player_stats_df is needed by the direct
        # hit-rate dispatcher below; pulled once per session for the
        # real season being scanned plus the prior season (needed for
        # any cross-season fallback inside those functions).
        _real_seasons_needed = sorted(set(
            int(s) for s in edited["season"].dropna().unique().tolist()
        )) if "season" in edited.columns else []
        _cache_key = tuple(_real_seasons_needed)
        if _real_seasons_needed and st.session_state.get("player_stats_df_cache_key") != _cache_key:
            _real_all_seasons = sorted(set(_real_seasons_needed + [s - 1 for s in _real_seasons_needed]))
            _cache_df = pull_player_stats(_real_all_seasons)
            # REAL FIX (critical bug found via a full, direct end-to-end
            # test before finalizing) - this cache never had the real
            # longest-play data merged in at all, meaning Stage 2
            # rescoring for longest_completion/longest_reception would
            # always fail with "real stat data missing" in real, live
            # use - confirmed directly by testing Drake Maye's real row
            # through this exact path. Merges in both real longest-play
            # columns now, the same validated way used throughout tonight.
            try:
                _cache_pbp = pull_pbp(_real_all_seasons)
                _longest_qb = build_longest_play_by_game(_cache_pbp, "QB")
                _longest_wr = build_longest_play_by_game(_cache_pbp, "WR")
                _cache_df = _cache_df.merge(
                    _longest_qb[["gsis_id", "season", "week", "longest_play"]],
                    on=["gsis_id", "season", "week"], how="left",
                ).rename(columns={"longest_play": "longest_completion"})
                _cache_df = _cache_df.merge(
                    _longest_wr[["gsis_id", "season", "week", "longest_play"]],
                    on=["gsis_id", "season", "week"], how="left",
                ).rename(columns={"longest_play": "longest_reception"})
            except Exception:
                pass  # real, graceful degradation - everything else still works if this merge fails
            st.session_state.player_stats_df_cache = _cache_df
            st.session_state.player_stats_df_cache_key = _cache_key

        # Real, one-time cache - rb_concept_usage is expensive to build
        # (scans every real team across every real concept), so it's
        # computed once per session rather than once per row.
        _rb_bundle_for_scoring = st.session_state.get("rb_bundle")
        if _rb_bundle_for_scoring is not None and "rb_concept_usage_cache" not in st.session_state:
            st.session_state.rb_concept_usage_cache = build_rb_concept_usage_ranks(_rb_bundle_for_scoring)
        _rb_concept_usage = st.session_state.get("rb_concept_usage_cache")
        _coverage_bundle_for_scoring = st.session_state.get("coverage_bundle")

        results = []
        for _, row in edited.iterrows():
            mu = row.get("mu")
            line = row.get("line")
            sigma = row.get("sigma")
            if pd.notna(line) and pd.notna(mu) and pd.notna(sigma):
                # REAL FIX (per direct, explicit request) - uses the real,
                # empirical direct hit-rate system instead of the old
                # normal-distribution approach, for every prop that has
                # one; falls back automatically (inside the dispatcher)
                # for pass_attempts/rush_attempts and any matchup with a
                # genuinely thin real sample.
                scored = rescore_via_direct_hit_rate(
                    row.get("gsis_id"), row.get("prop_type"), row.get("position"), line,
                    row.get("season"), row.get("week"), row.get("opponent"),
                    st.session_state.get("player_stats_df_cache"),
                    coverage_bundle=_coverage_bundle_for_scoring, rb_bundle=_rb_bundle_for_scoring,
                    rb_concept_usage=_rb_concept_usage, mu=mu, sigma=sigma,
                )
                # REAL, NEW - per direct request, pulls the real hit-rate
                # detail (e.g. "Cleared 79.5 in 7/9 real past games") out
                # of the nested dict it was buried in and into its own,
                # real, readable columns, so it's actually visible in the
                # table instead of computed-but-hidden.
                real_detail = scored.pop("real_hit_rate_detail", None) or {}
                real_read = real_detail.get("read", "")
                real_hit_count = real_detail.get("real_hit_count")
                real_sample_size = real_detail.get("real_sample_size")
                # REAL, NEW - per direct, repeated request (validated
                # against real, independently-known numbers: Drake
                # Maye's real YPA vs Cover 6 confirmed at 7.39, Deep
                # Throw % at 12.1) - formats the real supporting metrics
                # into one clean, readable string so the hit-rate isn't
                # standing alone; backs it up with his own real skill
                # metrics at each real target coverage.
                supporting = real_detail.get("supporting_metrics") or {}
                supporting_parts = []
                for coverage_field, metrics in supporting.items():
                    metric_strs = []
                    for k, v in metrics.items():
                        if k.startswith("_"):
                            continue  # internal flags (_majority_elite/_majority_poor), not a per-metric detail dict
                        if "percentile" in v and v.get("percentile") is not None:
                            metric_strs.append(f"{k}={v['value']} (p{v['percentile']})")
                        elif "real_opponent_allowed_percentile" in v and v.get("real_opponent_allowed_percentile") is not None:
                            his_pct = v.get("his_own_percentile")
                            his_pct_str = f" (p{his_pct})" if his_pct is not None else ""
                            metric_strs.append(f"{k}: his={v['his_own_value']}{his_pct_str}, opp allows={v['real_opponent_allowed_value']} (p{v['real_opponent_allowed_percentile']})")
                    if metric_strs:
                        supporting_parts.append(f"{coverage_field}: " + ", ".join(metric_strs))
                real_supporting_read = " | ".join(supporting_parts)
                # REAL, NEW - per direct request, surfaces the new Stage 1
                # "his own real metrics are majority-elite/poor" standard
                # as a real, visible column - built and validated tonight,
                # not just applied manually in conversation.
                player_stage1_elite = real_detail.get("player_stage1_majority_elite", False)
                player_stage1_poor = real_detail.get("player_stage1_majority_poor", False)
                results.append({
                    **row.to_dict(), **scored,
                    "real_hit_rate_read": real_read,
                    "real_hit_count": real_hit_count,
                    "real_sample_size": real_sample_size,
                    "real_supporting_metrics": real_supporting_read,
                    "stage1_player_majority_elite": player_stage1_elite,
                    "stage1_player_majority_poor": player_stage1_poor,
                })
            else:
                results.append({**row.to_dict(), "p_over": np.nan, "edge": np.nan})

        scored_df = pd.DataFrame(results)
        if min_edge_filter > 0 and "edge" in scored_df.columns:
            scored_df = scored_df[scored_df["edge"].fillna(0) >= min_edge_filter]

        if scored_df.empty:
            # REAL BUG FOUND+FIXED this session: an empty `results` list
            # (every row filtered out upstream by quality_score/games_
            # sampled_current/prop/position - genuinely possible for a
            # week 1 scan, where games_sampled_current is 0 for everyone
            # until real current-season data accumulates) produces
            # pd.DataFrame([]) - ZERO ROWS AND ZERO COLUMNS, not just zero
            # rows. "edge" then isn't a column at all, so it silently drops
            # out of display_cols below and .sort_values("edge") crashes
            # with a raw KeyError instead of showing a clear message.
            # Short-circuit here with the same graceful-degrade pattern
            # used throughout this file rather than let it reach the sort.
            st.info(
                "No rows match the current filters. Early in a season (week 1-3 "
                "especially) most players still show games_sampled_current = 0, "
                "so a min-games or min-quality filter can wipe out the whole "
                "slate - try lowering those filters."
            )
            st.stop()

        # REAL FIX: this table was showing every diagnostic column at once
        # (coverage breakdown, every advanced metric grade, alignment
        # data...) by default - genuinely useful for debugging a specific
        # row, but not what you want scanning a slate for plays, and not
        # how the MLB tool's Best Edges table works (player/team/opp/mu/
        # edge, nothing else, by default). Curated core view now shown by
        # default; the full diagnostic column set (coverage/grade/
        # alignment breakdown) moves behind an explicit toggle, off by
        # default - same "curated view + full raw available on request"
        # pattern already used in the Backtest section's "Every scored row
        # (raw)" expander, just applied here too.
        core_display_cols = ["player_display_name", "team", "opponent", "matchup",
                              "position", "prop_type", "line", "mu", "sigma",
                              "p_over", "edge", "real_hit_rate_read", "real_supporting_metrics",
                              "stage1_player_majority_elite", "stage1_player_majority_poor",
                              "quality_score", "data_confidence",
                              "games_sampled_total"]

        # Compute these UNCONDITIONALLY - a later line (the color-gradient
        # styling below) references them regardless of the checkbox state.
        # REAL BUG FOUND+FIXED: previously only computed inside the
        # `if show_full_diagnostics:` branch, so leaving the checkbox at
        # its default (off) hit a genuine NameError on that later
        # unconditional reference - confirmed live via the exact
        # traceback (line ~1126, `+ grade_cols + cov_breakdown_cols`).
        cov_breakdown_cols = sorted([c for c in scored_df.columns if c.startswith("opp_cov_")])
        grade_cols = sorted([c for c in scored_df.columns if c.endswith("_grade")])

        if show_full_diagnostics:
            extra_display_cols = ["opp_dominant_coverage",
                                   "opp_dominant_coverage_pct", "opp_num_elevated_coverages",
                                   "opp_man_pct", "opp_zone_pct",
                                   "opp_box_stack_pct", "opp_box_elevated",
                                   "playaction_exploit_strength", "playaction_used_coverage_specific_data",
                                   "personnel_exploit_strength", "dominant_personnel",
                                   "grade_matchup_strength",
                                   "role_verification_score", "role_trend_ratio"]
            # Include the full individual coverage-type breakdown AND every
            # advanced metric grade (QB/WR/TE/RB own performance grades,
            # plus opponent defense grades) - column names vary by what's
            # actually available for a given player/week, so these are
            # picked up dynamically rather than hardcoded.
            display_cols = core_display_cols + extra_display_cols + grade_cols + cov_breakdown_cols
        else:
            display_cols = core_display_cols
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

        st.divider()

        # ---------------------------------------------------------------
        # Quality/edge/confidence gate + Slip Builder + Locked Slips + Top-up
        # - direct port of the MLB tool's version, same architecture, same
        # conflict rules. Real differences from the MLB version, on
        # purpose, not oversights:
        #   - "matchup" (e.g. "LAR @ KC") plays the role of MLB's game_pk -
        #     it's the real, exact per-game identifier already on every
        #     row here, so the same-game conflict rule reuses it directly.
        #   - No batting-order tiebreak - there's no NFL equivalent, so the
        #     strength score here is just quality_score + edge, nothing
        #     invented to fill that slot.
        #   - Gate defaults to 0/0/0 (off) here, NOT MLB's 70/70/0.20 -
        #     those numbers were never validated against NFL's own real
        #     edge/quality distribution, so defaulting to MLB's tuned
        #     values here would be presenting an unproven number as if it
        #     were calibrated. Set your own once you have a feel for what
        #     real NFL edge/quality looks like.
        #   - Deliberately NOT added: any cross-game correlation rule
        #     beyond literal same-matchup (e.g. flagging two different
        #     games with a similar projected script/weather) - there's no
        #     real weather or win-probability signal in this model to back
        #     that rule honestly, so it's left out rather than faked.
        # ---------------------------------------------------------------
        st.header("🎯 Minimum bar (quality/confidence/edge)")
        nfl_g1, nfl_g2, nfl_g3 = st.columns(3)
        with nfl_g1:
            nfl_min_quality_gate = st.number_input("Min quality_score", min_value=0, max_value=100, value=0, step=1, key="nfl_min_quality_gate")
        with nfl_g2:
            nfl_min_prob_gate = st.number_input("Min confidence % (whichever direction it leans)",
                                                 min_value=50, max_value=100, value=50, step=1, key="nfl_min_prob_gate")
        with nfl_g3:
            nfl_min_edge_gate = st.number_input("Min edge", min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="nfl_min_edge_gate")

        nfl_confidence = scan_sorted["p_over"].apply(lambda p: max(p, 1 - p) if pd.notna(p) else np.nan)
        nfl_qualified_df = scan_sorted[
            (scan_sorted["quality_score"].fillna(0) >= nfl_min_quality_gate)
            & (nfl_confidence.fillna(0) >= nfl_min_prob_gate / 100.0)
            & (scan_sorted["edge"].fillna(0) >= nfl_min_edge_gate)
        ].copy()
        st.caption(f"{len(nfl_qualified_df)} of {len(scan_sorted)} rows clear the bar above.")

        # REAL, NEW - per direct, explicit request, computes the real
        # Stage 1 standalone check (does HIS OWN metrics show a genuine
        # majority-elite/poor profile) for this filtered, pre-line-entry
        # subset - confirmed this can run without any line at all, unlike
        # the earlier version which was incorrectly bundled inside the
        # line-requiring Stage 2 function. Runs on the already-filtered,
        # much smaller subset for real performance, not the full raw slate.
        _stage1_elite_list = []
        _stage1_poor_list = []
        _rb_bundle_stage1 = st.session_state.get("rb_bundle")
        _coverage_bundle_stage1 = st.session_state.get("coverage_bundle")
        if _rb_bundle_stage1 is not None and "rb_concept_usage_cache" not in st.session_state:
            st.session_state.rb_concept_usage_cache = build_rb_concept_usage_ranks(_rb_bundle_stage1)
        _rb_concept_usage_stage1 = st.session_state.get("rb_concept_usage_cache")
        for _, _row in nfl_qualified_df.iterrows():
            try:
                _s1 = calc_stage1_player_elite_check(
                    _row.get("gsis_id"), _row.get("player_display_name"), _row.get("prop_type"),
                    _row.get("position"), _row.get("opponent"),
                    coverage_bundle=_coverage_bundle_stage1, rb_bundle=_rb_bundle_stage1,
                    rb_concept_usage=_rb_concept_usage_stage1,
                )
                _stage1_elite_list.append(_s1.get("majority_elite", False))
                _stage1_poor_list.append(_s1.get("majority_poor", False))
            except Exception:
                _stage1_elite_list.append(False)
                _stage1_poor_list.append(False)
        nfl_qualified_df["stage1_player_majority_elite"] = _stage1_elite_list
        nfl_qualified_df["stage1_player_majority_poor"] = _stage1_poor_list

        # Color-coded read-only view, same 3-scheme style as the MLB tool -
        # data_editor itself can't render color (Streamlit limitation), so
        # this sits alongside the actual checkbox-editing table below as a
        # visual reference, not a second data source.
        def _nfl_color_edge(val):
            if pd.isna(val):
                return ""
            intensity = min(val / 0.5, 1.0)
            return f"background-color: rgba(0, 200, 0, {intensity * 0.6})"

        def _nfl_color_prob(val):
            if pd.isna(val):
                return ""
            if val >= 0.5:
                intensity = min((val - 0.5) / 0.5, 1.0)
                return f"background-color: rgba(0, 200, 0, {intensity * 0.6})"
            intensity = min((0.5 - val) / 0.5, 1.0)
            return f"background-color: rgba(200, 0, 0, {intensity * 0.6})"

        def _nfl_color_quality(val):
            if pd.isna(val):
                return ""
            intensity = min(val / 100, 1.0)
            return f"background-color: rgba(0, 150, 220, {intensity * 0.5})"

        nfl_preview_cols = ["player_display_name", "team", "matchup", "prop_type", "line",
                            "mu", "edge", "p_over", "quality_score", "games_sampled_total",
                            "stage1_player_majority_elite", "stage1_player_majority_poor"]
        nfl_styled_preview = (nfl_qualified_df[nfl_preview_cols].style
                              .map(_nfl_color_edge, subset=["edge"])
                              .map(_nfl_color_prob, subset=["p_over"])
                              .map(_nfl_color_quality, subset=["quality_score"]))
        st.dataframe(nfl_styled_preview, width='stretch', hide_index=True)

        nfl_qualified_df.insert(0, "Include", False)

        nfl_checked = st.data_editor(
            nfl_qualified_df[["Include", "player_display_name", "team", "matchup", "prop_type",
                              "line", "mu", "edge", "p_over", "quality_score", "games_sampled_total"]],
            column_config={"Include": st.column_config.CheckboxColumn(
                "Include", help="Check to add this leg to the slip builder below")},
            disabled=["player_display_name", "team", "matchup", "prop_type", "line", "mu",
                      "edge", "p_over", "quality_score", "games_sampled_total"],
            width='stretch', key="nfl_include_editor",
        )

        st.header("🎰 Slip Builder")
        nfl_target_size = st.selectbox("Target slip size", [3, 2, 4], index=0, key="nfl_slip_size")
        nfl_selected = nfl_checked[nfl_checked["Include"] == True].copy()
        if nfl_selected.empty:
            st.caption("Check the Include box on legs above to start building slips.")
        else:
            nfl_selected["_strength"] = nfl_selected["quality_score"].fillna(50) + nfl_selected["edge"].fillna(0) * 100 * 0.5
            nfl_selected = nfl_selected.sort_values("_strength", ascending=False).reset_index(drop=True)

            n = len(nfl_selected)
            base = nfl_target_size
            if n < base:
                nfl_slip_sizes = [n] if n > 0 else []
            else:
                n_slips, remainder = n // base, n % base
                nfl_slip_sizes = [base] * n_slips
                if remainder:
                    if remainder + base <= 4:
                        nfl_slip_sizes[-1] += remainder
                    else:
                        nfl_slip_sizes.append(max(2, remainder))

            nfl_slips = [[] for _ in nfl_slip_sizes]
            nfl_slip_games = [set() for _ in nfl_slip_sizes]
            nfl_slip_players = [set() for _ in nfl_slip_sizes]
            nfl_leftover = []
            for _, leg in nfl_selected.iterrows():
                placed = False
                for i, size in enumerate(nfl_slip_sizes):
                    if len(nfl_slips[i]) >= size:
                        continue
                    if leg["matchup"] in nfl_slip_games[i] or leg["player_display_name"] in nfl_slip_players[i]:
                        continue
                    nfl_slips[i].append(leg)
                    nfl_slip_games[i].add(leg["matchup"])
                    nfl_slip_players[i].add(leg["player_display_name"])
                    placed = True
                    break
                if not placed:
                    nfl_leftover.append(leg)

            for i, slip in enumerate(nfl_slips):
                if not slip:
                    continue
                avg_q = sum(l["quality_score"] for l in slip if pd.notna(l["quality_score"])) / max(len(slip), 1)
                st.subheader(f"Slip {i + 1} — {len(slip)}-man (avg quality {avg_q:.0f})")
                st.dataframe(pd.DataFrame(slip)[["player_display_name", "team", "matchup", "prop_type",
                                                  "line", "quality_score", "edge", "games_sampled_total"]],
                            width='stretch', hide_index=True)

            if nfl_leftover:
                st.warning(f"{len(nfl_leftover)} checked leg(s) couldn't be placed without breaking the "
                           f"same-game/same-player rule against every open slip slot - shown below, "
                           f"add manually or check a different combination of legs.")
                st.dataframe(pd.DataFrame(nfl_leftover)[["player_display_name", "team", "matchup", "prop_type",
                                                          "line", "quality_score", "edge"]],
                            width='stretch', hide_index=True)

            if st.button("🔒 Lock in these slips", key="nfl_lock_slips_btn"):
                if "nfl_locked_slips" not in st.session_state:
                    st.session_state.nfl_locked_slips = []
                new_locked = [pd.DataFrame(slip)[["player_display_name", "team", "prop_type", "line",
                                                   "quality_score", "edge", "games_sampled_total", "matchup"]]
                              for slip in nfl_slips if slip]
                st.session_state.nfl_locked_slips.extend(new_locked)
                st.success(f"Locked in {len(new_locked)} slip(s) - they'll now survive a rescan.")

        if st.session_state.get("nfl_locked_slips"):
            st.divider()
            st.header("🔒 Locked Slips (survive a rescan)")
            st.caption("Saved copies - rescanning above won't touch these. Survives a rescan, "
                       "not a full app reboot/redeploy (that restarts everything from scratch).")
            nfl_all_locked = pd.concat(
                st.session_state.nfl_locked_slips,
                keys=range(1, len(st.session_state.nfl_locked_slips) + 1), names=["slip_number"]
            ).reset_index(level=0)
            nfl_locked_display_cols = [c for c in nfl_all_locked.columns if c != "matchup"]
            nfl_locked_csv = nfl_all_locked[nfl_locked_display_cols].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download ALL locked slips as CSV", nfl_locked_csv,
                               file_name=f"nfl_locked_slips_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", key="nfl_dl_locked_slips")

            nfl_currently_checked = nfl_qualified_df.merge(
                nfl_checked[nfl_checked["Include"] == True][["player_display_name", "prop_type", "line"]],
                on=["player_display_name", "prop_type", "line"], how="inner",
            ) if not nfl_checked.empty else pd.DataFrame()
            nfl_growable = [i for i, s in enumerate(st.session_state.nfl_locked_slips) if len(s) < 4]
            if not nfl_currently_checked.empty and nfl_growable:
                st.subheader("Add a checked leg to an existing locked slip")
                nfl_target_idx = st.selectbox(
                    "Which locked slip?", nfl_growable,
                    format_func=lambda i: f"Locked Slip {i + 1} ({len(st.session_state.nfl_locked_slips[i])}-man, room for {4 - len(st.session_state.nfl_locked_slips[i])} more)",
                    key="nfl_topup_target",
                )
                nfl_leg_opts = list(nfl_currently_checked["player_display_name"] + " - " + nfl_currently_checked["prop_type"])
                nfl_picks = st.multiselect("Which checked leg(s) to add?", nfl_leg_opts, key="nfl_topup_legs")
                if st.button("Add to locked slip", key="nfl_topup_btn") and nfl_picks:
                    target = st.session_state.nfl_locked_slips[nfl_target_idx]
                    existing_games = set(target["matchup"])
                    existing_players = set(target["player_display_name"])
                    added, skipped = 0, []
                    for pick in nfl_picks:
                        row = nfl_currently_checked[
                            (nfl_currently_checked["player_display_name"] + " - " + nfl_currently_checked["prop_type"]) == pick
                        ].iloc[0]
                        if len(target) >= 4:
                            skipped.append((pick, "slip already at 4-man max")); continue
                        if row["player_display_name"] in existing_players:
                            skipped.append((pick, "same player already in this slip")); continue
                        if row["matchup"] in existing_games:
                            skipped.append((pick, "another leg from this same game is already in this slip")); continue
                        target = pd.concat([target, pd.DataFrame([row[
                            ["player_display_name", "team", "prop_type", "line", "quality_score",
                             "edge", "games_sampled_total", "matchup"]
                        ]])], ignore_index=True)
                        existing_players.add(row["player_display_name"]); existing_games.add(row["matchup"])
                        added += 1
                    st.session_state.nfl_locked_slips[nfl_target_idx] = target
                    if added:
                        st.success(f"Added {added} leg(s) to Locked Slip {nfl_target_idx + 1}.")
                    if skipped:
                        st.warning("Skipped: " + ", ".join(f"{p} ({r})" for p, r in skipped))

            for i, locked_slip in enumerate(st.session_state.nfl_locked_slips):
                lcol1, lcol2 = st.columns([5, 1])
                with lcol1:
                    st.subheader(f"Locked Slip {i + 1} — {len(locked_slip)}-man")
                with lcol2:
                    if st.button("Remove", key=f"nfl_remove_locked_{i}"):
                        st.session_state.nfl_locked_slips.pop(i)
                        st.rerun()
                display_cols_locked = [c for c in locked_slip.columns if c != "matchup"]
                st.dataframe(locked_slip[display_cols_locked], width='stretch', hide_index=True)
            if st.button("Clear ALL locked slips", key="nfl_clear_all_locked"):
                st.session_state.nfl_locked_slips = []
                st.rerun()

    # Backtest stacked directly below the live scan/Slip Builder/Locked
    # Slips, matching the MLB tool's layout (scan on top, backtest below,
    # both visible together) - only when the backtest has actually been
    # run at least once; otherwise nothing extra shows here.
    if st.session_state.season_report is not None:
        st.divider()
        _render_season_report(st.session_state.season_report)

    # ---------------------------------------------------------------
    # Full Matchup Simulation - real, direct port of the MLB tool's
    # section, using the full play-by-play game engine (simulate_one_game/
    # simulate_matchup_n_times/build_team_offense) that already existed in
    # nfl_model_combined.py but was never wired into any live UI before
    # this. Two real, severe bugs were found and fixed getting this
    # working: (1) build_team_offense only looked at CURRENT-season pbp to
    # identify who the QB/top rushers/top targets even are, so week 1
    # always produced a completely empty offense (confirmed directly:
    # qb_gsis_id None, 0 rushers, 0 targets) - fixed with the same
    # prior-season fallback pattern used throughout this session. (2) same
    # cold-start class of bug already fixed in the advanced-metrics/grade
    # builders.
    # STATUS, stated plainly: mechanically verified against real data
    # (produces sane, real stat-line distributions - confirmed directly,
    # e.g. a real QB's simulated pass_yards/completions/attempts came back
    # in a realistic range) but NOT YET backtested/calibrated for accuracy
    # - same honest standard as everywhere else in this project.
    #
    # REAL DESIGN CHANGE (per direct request/pushback): quality_score and
    # the simulation's own empirical read are two DIFFERENT signals - one
    # answers "how much do I trust this projection's inputs" (data
    # confidence), the other answers "how consistent/big is this specific
    # edge across many random game outcomes" (real empirical
    # variance/rate). They should NOT be blended into one number, but they
    # SHOULD be shown side by side on the same row so you can actually
    # cross-check them together, instead of living in two disconnected
    # places. Both Stage 1 and Stage 2 now pull quality_score for the same
    # (gsis_id, prop) via scan_full_slate_nfl (team-filtered to just this
    # matchup's two teams, so it's a fast, small scan, not a full-league
    # one) and show it as a real column - a genuinely "confirmed" play is
    # one where BOTH signals agree (real quality_score AND a tight,
    # consistent real simulated edge), not one where either is trusted
    # alone.
    # ---------------------------------------------------------------
    st.divider()
    st.header("🎮 Full Matchup Simulation")
    st.caption(
        "Real, full-game simulation (down-and-distance state machine, real "
        "bootstrap-sampled outcomes from each player's own real play-by-play "
        "history, tilted by the same real coverage/alignment/concept exploit "
        "signals used elsewhere in this tool) - not a single formula's one "
        "answer. Runs the whole game many times and shows the real, "
        "empirical rate each player's props actually clear a real line, "
        "with quality_score shown alongside for the same row so you can "
        "cross-check both signals together rather than trusting either "
        "alone."
    )
    st.caption(
        "Honest status: mechanically verified against real data (produces "
        "real, plausible stat-line distributions) but not yet backtested "
        "for calibrated accuracy - treat Stage 1/2 results here as a real, "
        "additional signal to weigh, not yet a fully proven one, until a "
        "real backtest against completed games confirms it."
    )

    nfl_sim_season = st.number_input("Season", min_value=2020, max_value=2030,
                                       value=datetime.now().year, step=1, key="nfl_sim_season")
    nfl_sim_week = st.number_input("Week", min_value=1, max_value=18, value=1, step=1, key="nfl_sim_week")

    if st.button("Load this week's real games", key="nfl_sim_load_games"):
        try:
            schedules_for_sim = pull_schedules([nfl_sim_season])
            st.session_state.nfl_sim_games_df = build_week_games_list(nfl_sim_season, nfl_sim_week, schedules_for_sim)
        except Exception as e:
            st.error(f"Couldn't pull the real schedule: {e}")
            st.session_state.nfl_sim_games_df = pd.DataFrame()

    sim_games_df = st.session_state.get("nfl_sim_games_df")
    if sim_games_df is None or sim_games_df.empty:
        st.info("Click \"Load this week's real games\" above to pick a real matchup.")
    else:
        sim_game_label = st.selectbox("Pick a real game", sim_games_df["matchup"].tolist(), key="nfl_sim_game_select")
        sim_row = sim_games_df[sim_games_df["matchup"] == sim_game_label].iloc[0]
        sim_home, sim_away = sim_row["home_team"], sim_row["away_team"]

        sim_n_games = st.slider("Number of simulated games", 100, 2000, 500, step=100, key="nfl_sim_n_games_slider",
                                 help="500 is a reasonable balance of speed vs statistical tightness for this "
                                      "engine's per-play bootstrap sampling - more takes longer with diminishing "
                                      "real precision gains past this point.")

        if st.button("Run full matchup simulation", key="nfl_sim_run_button"):
            with st.spinner("Pulling real pbp/roster data and building both real offenses..."):
                try:
                    sim_pbp_df = pull_pbp([nfl_sim_season])
                    sim_prior_pbp_df = pull_pbp([nfl_sim_season - 1])
                    sim_rosters_df = pull_rosters([nfl_sim_season])
                    sim_coverage_bundle = st.session_state.get("coverage_bundle")
                    sim_rb_bundle = st.session_state.get("rb_bundle")

                    home_offense = build_team_offense(
                        sim_home, sim_away, nfl_sim_season, nfl_sim_week,
                        sim_pbp_df, sim_prior_pbp_df, sim_rosters_df,
                        coverage_bundle=sim_coverage_bundle, rb_bundle=sim_rb_bundle,
                    )
                    away_offense = build_team_offense(
                        sim_away, sim_home, nfl_sim_season, nfl_sim_week,
                        sim_pbp_df, sim_prior_pbp_df, sim_rosters_df,
                        coverage_bundle=sim_coverage_bundle, rb_bundle=sim_rb_bundle,
                    )

                    if home_offense.qb_gsis_id is None or away_offense.qb_gsis_id is None:
                        st.warning("Couldn't identify a real starting QB for one or both teams (even after the "
                                   "prior-season fallback) - results below may be incomplete for that side.")

                    name_lookup_df = sim_rosters_df[sim_rosters_df["season"] == nfl_sim_season][
                        ["gsis_id", "full_name"]].drop_duplicates()
                    name_lookup = name_lookup_df.set_index("gsis_id")["full_name"].to_dict()
                except Exception as e:
                    st.error(f"Couldn't build one or both real offenses: {e}")
                    home_offense = away_offense = None

            if home_offense is not None and away_offense is not None:
                with st.spinner(f"Running {sim_n_games} real simulated games..."):
                    try:
                        sim_results = simulate_matchup_n_times(home_offense, away_offense, n_simulations=sim_n_games)
                        st.session_state.nfl_sim_results = sim_results
                        st.session_state.nfl_sim_name_lookup = name_lookup

                        # Real, direct team tagging - every gsis_id that ended
                        # up in either offense's pools gets tagged with its
                        # real team, so Stage 1 can show it and a same-team
                        # conflict check works for parlay building later.
                        team_map = {}
                        if home_offense.qb_gsis_id:
                            team_map[home_offense.qb_gsis_id] = sim_home
                        if away_offense.qb_gsis_id:
                            team_map[away_offense.qb_gsis_id] = sim_away
                        for gid, _ in home_offense.rushers:
                            team_map[gid] = sim_home
                        for gid, _ in home_offense.targets:
                            team_map[gid] = sim_home
                        for gid, _ in away_offense.rushers:
                            team_map[gid] = sim_away
                        for gid, _ in away_offense.targets:
                            team_map[gid] = sim_away
                        st.session_state.nfl_sim_team_lookup = team_map

                        # Real fix (per direct request) - also run the real,
                        # actual live scan for just these two teams
                        # (team_filter keeps this fast - a 2-team scan, not a
                        # full-league one) so quality_score is available to
                        # cross-check against the simulation's own numbers,
                        # instead of the two signals living in totally
                        # separate, disconnected places.
                        try:
                            quality_slate = scan_full_slate_nfl(
                                nfl_sim_season, nfl_sim_week,
                                coverage_bundle=sim_coverage_bundle, rb_bundle=sim_rb_bundle,
                                team_filter=[sim_home, sim_away],
                            )
                            quality_lookup = {}
                            if not quality_slate.empty and "gsis_id" in quality_slate.columns:
                                for _, qrow in quality_slate.iterrows():
                                    quality_lookup[(qrow["gsis_id"], qrow["prop_type"])] = qrow.get("quality_score")
                            st.session_state.nfl_sim_quality_lookup = quality_lookup
                        except Exception as e:
                            st.warning(f"Simulation ran, but pulling quality_score for cross-checking failed: {e} "
                                       f"- Stage 1/2 below will show sim results without a quality_score column.")
                            st.session_state.nfl_sim_quality_lookup = {}

                        st.success(f"Ran {sim_n_games} real simulated games for {sim_away} @ {sim_home}.")
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")

        if st.session_state.get("nfl_sim_results"):
            st.subheader("Enter each real line to check against the simulated games")
            sim_results = st.session_state.nfl_sim_results
            name_lookup = st.session_state.get("nfl_sim_name_lookup", {})
            team_lookup = st.session_state.get("nfl_sim_team_lookup", {})
            quality_lookup = st.session_state.get("nfl_sim_quality_lookup", {})

            nfl_sim_props_wanted = st.multiselect(
                "Which props to show",
                ["pass_attempts", "pass_completions", "pass_yards", "pass_tds", "interceptions",
                 "rush_attempts", "rush_yards", "rush_tds"],
                default=["pass_yards", "rush_yards", "pass_tds"],
                key="nfl_sim_props_multiselect",
            )

            if not nfl_sim_props_wanted:
                st.info("Pick at least one prop above.")
            else:
                st.subheader("Stage 1 - who actually stayed great across the simulation")
                st.caption(
                    "No line needed yet. Ranks every real player against the rest of THIS matchup's "
                    "own field (z-score) for real edge + consistency, with the real, live quality_score "
                    "for that same (player, prop) shown alongside for cross-checking - not blended in, "
                    "just visible together."
                )
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    nfl_min_zscore = st.slider("Minimum edge (real std devs above this matchup's own field)",
                                                0.0, 2.0, 0.3, step=0.1, key="nfl_sim_min_zscore")
                with fcol2:
                    nfl_max_cv = st.slider("Maximum coefficient of variation (lower = more consistent)",
                                            0.1, 1.5, 0.7, step=0.05, key="nfl_sim_max_cv")
                with fcol3:
                    nfl_min_quality_for_flag = st.slider(
                        "Quality_score bar for the \"confirmed\" flag", 0, 100, 70, step=5,
                        key="nfl_sim_min_quality_flag",
                        help="Doesn't filter anything out - just controls which rows get marked "
                             "\"confirmed\" (real sim edge AND real quality_score both clearing their bars) "
                             "in the table below.",
                    )

                stage1_rows = []
                for gid, props in sim_results.items():
                    player_name = name_lookup.get(gid, gid)
                    team = team_lookup.get(gid, "?")
                    for prop in nfl_sim_props_wanted:
                        series = props.get(prop, [])
                        if not series:
                            continue
                        avg = sum(series) / len(series)
                        std = (sum((v - avg) ** 2 for v in series) / len(series)) ** 0.5
                        cv = round(std / avg, 3) if avg else None
                        q_score = quality_lookup.get((gid, prop))
                        stage1_rows.append({"gsis_id": gid, "player": player_name, "team": team, "prop": prop,
                                             "real_avg": round(avg, 2), "cv": cv, "quality_score": q_score})

                stage1_df = pd.DataFrame(stage1_rows)
                if stage1_df.empty:
                    st.warning("No real data to rank yet.")
                else:
                    stage1_df["field_mean"] = stage1_df.groupby("prop")["real_avg"].transform("mean")
                    stage1_df["field_std"] = stage1_df.groupby("prop")["real_avg"].transform("std").fillna(0.01)
                    stage1_df["zscore"] = round((stage1_df["real_avg"] - stage1_df["field_mean"]) / stage1_df["field_std"], 2)

                    survivors = stage1_df[
                        (stage1_df["zscore"] >= nfl_min_zscore) & (stage1_df["cv"].fillna(99) <= nfl_max_cv)
                    ].sort_values("zscore", ascending=False).copy()

                    # Real, visible cross-check column - "confirmed" means
                    # BOTH signals agree (real sim edge/consistency already
                    # required to be a survivor here, AND a real quality_score
                    # clearing the bar above) - this is the direct answer to
                    # "how do we know a 70+ play is consistent": it's this
                    # column, not a blended number.
                    survivors["confirmed"] = survivors["quality_score"].apply(
                        lambda q: "✅ both agree" if pd.notna(q) and q >= nfl_min_quality_for_flag else "sim only")

                    st.dataframe(survivors[["player", "team", "prop", "real_avg", "cv", "zscore",
                                             "quality_score", "confirmed"]], width='stretch')
                    st.caption(f"{len(survivors)} of {len(stage1_df)} real (player, prop) combinations cleared "
                               f"the sim's own edge/consistency bars. \"confirmed\" additionally requires real "
                               f"quality_score >= {nfl_min_quality_for_flag} for that same row - a genuinely "
                               f"cross-checked play, not just one signal trusted alone.")

                    with st.expander(f"See all {len(stage1_df)} real (player, prop) combinations, unfiltered"):
                        st.dataframe(stage1_df[["player", "team", "prop", "real_avg", "cv", "zscore", "quality_score"]]
                                     .sort_values("zscore", ascending=False), width='stretch')

                    if survivors.empty:
                        st.info("Nothing cleared the bar - try lowering the sliders above, or this matchup "
                                "genuinely doesn't have a standout edge tonight.")
                    else:
                        st.subheader("Stage 2 - enter each real line for the survivors above")

                        def _round_half(x):
                            return math.floor(x) + 0.5 if x is not None else 1.5

                        base_rows = [{"gsis_id": r["gsis_id"], "player": r["player"], "team": r["team"],
                                      "prop": r["prop"], "quality_score": r["quality_score"],
                                      "your_line": _round_half(r["real_avg"])}
                                     for _, r in survivors.iterrows()]
                        base_df = pd.DataFrame(base_rows)
                        edited_lines = st.data_editor(
                            base_df, key="nfl_sim_lines_editor", width='stretch', hide_index=True,
                            column_order=["player", "team", "prop", "quality_score", "your_line"],
                            disabled=["gsis_id", "player", "team", "prop", "quality_score"],
                            column_config={"your_line": st.column_config.NumberColumn("Real line (edit me)", step=0.5)},
                        )

                        # Real fix - joins back to sim_results by the REAL
                        # gsis_id carried through from Stage 1, not a fragile
                        # reverse name lookup (which would silently break for
                        # any two players sharing a display name).
                        result_rows = []
                        for _, row in edited_lines.iterrows():
                            gid = row["gsis_id"]
                            series = sim_results.get(gid, {}).get(row["prop"], [])
                            # Real bug fixed before delivery - real_over_rate_
                            # from_simulation returns "mean", not "avg" - the
                            # original draft of this section referenced "avg"
                            # directly, which would have crashed the instant
                            # anyone entered a line here. Confirmed the real
                            # function signature before shipping this time.
                            r = real_over_rate_from_simulation(series, row["your_line"])
                            result_rows.append({"player": row["player"], "team": row["team"], "prop": row["prop"],
                                                 "quality_score": row["quality_score"], "line": row["your_line"],
                                                 "sim_avg": r["mean"], "over_rate_pct": round(r["over_rate"] * 100, 1),
                                                 "n_simulations": r["n"]})

                        result_df = pd.DataFrame(result_rows).sort_values("over_rate_pct", ascending=False, na_position="last")
                        result_df["under_rate_pct"] = round(100 - result_df["over_rate_pct"], 1)
                        result_df["lean"] = result_df["over_rate_pct"].apply(
                            lambda v: "OVER" if v > 50 else ("UNDER" if v < 50 else "COIN FLIP"))
                        result_df["confirmed"] = result_df["quality_score"].apply(
                            lambda q: "✅ both agree" if pd.notna(q) and q >= nfl_min_quality_for_flag else "sim only")
                        st.dataframe(
                            result_df[["player", "team", "prop", "quality_score", "line", "sim_avg",
                                       "over_rate_pct", "under_rate_pct", "lean", "confirmed"]],
                            width='stretch')
                        st.caption("over_rate_pct is the real, empirical rate across the simulated games - "
                                   "'over in 78 of 100', not a formula's single calculated probability. "
                                   "\"confirmed\" means this row's real quality_score also clears the bar set "
                                   f"above ({nfl_min_quality_for_flag}) - the real, visible cross-check between "
                                   "the two signals.")

elif mode != "Weekly Scan / Draft Rankings":
    st.info("Click the button above to load this week's props.")

if True:  # was: if mode == "Season Backtest": - now always renders below the scan section, matching MLB's one-page layout
    st.divider()
    # ---------------------------------------------------------------
    # Season backtest controls - moved to the BOTTOM of the scan page,
    # matching the MLB tool's layout (live scan/Slip Builder/Locked Slips
    # up top, backtest trigger + results below). Previously sat right
    # next to the daily scan button near the top; the actual per-game
    # scan boxes above are untouched by this move.
    # ---------------------------------------------------------------
    st.subheader("Season backtest (2025 or a completed 2026 range)")
    st.caption(
        "Runs a range of weeks, not necessarily the whole season - start small "
        "(e.g. a 4-6 week range) to confirm it works within Streamlit Cloud's free-"
        "tier memory limit before attempting the full season in one run."
    )
    # REAL BUG FOUND AND FIXED: this section used to silently reuse the
    # TOP section's "season" widget - meaning setting season=2026 up top
    # for a live scan also forced this backtest to run against 2026,
    # whether or not enough real games existed yet, with no way to
    # independently check 2025's completed season while scanning 2026
    # live. Genuinely separate, own widget now - defaults to 2025 (the
    # last fully-completed season, the safe default for methodology
    # checks), switchable to 2026 later once real weeks have actually
    # been played, completely independent of whatever the top section
    # is currently scanning.
    bt_season = st.number_input(
        "Backtest season", min_value=2020, max_value=2030, value=2025, step=1,
        help="Independent of the Season picker above - set this to 2025 to validate "
             "methodology on a complete season, or to 2026 later once enough real "
             "weeks exist to check progress mid-season.",
        key="bt_season_input",
    )
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        report_start_week = st.number_input(
            "Report start week", min_value=1, max_value=18, value=1, step=1,
            help="Week 1 used to be skipped automatically (no prior-week history to "
                 "project from). That's no longer true - this season's prior-season "
                 "fallback bridges (mu/sigma, role trend, longest-play) now give week 1 "
                 "a real projection to test, confirmed working live this session.",
        )
    with rcol2:
        report_end_week = st.number_input(
            "Report end week", min_value=2, max_value=18, value=6, step=1,
        )
    st.subheader("Real, combined backtest - readiness + 1000-sample simulation, one real pass")
    st.caption(
        "Runs the real, expensive scoring step ONCE per week and builds both reports "
        "from the same shared results - per direct request, since the readiness "
        "check and the simulation backtest genuinely need each other and were "
        "previously two separate buttons silently re-scoring the same real weeks "
        "twice. Includes a real 1000-sample Monte Carlo per player-prop (negative "
        "binomial for count props, normal for continuous props, both matched to the "
        "model's own real mu AND sigma). Same real, honest limitation throughout: no "
        "free historical NFL player-prop-line archive exists, so the simulation's "
        "hypothetical line is the model's own real average (floored to a genuine "
        ".5), not an actual historical book line."
    )
    st.caption(
        "Real, honest note on which signals are actually feeding mu: this uses "
        "whatever ENABLE_*_IN_QUALITY_SCORE flags are currently on elsewhere in the "
        "model - alignment vs coverage is currently on, QB-vs-coverage and RB run-"
        "concepts are currently off (still working through their own one-at-a-time "
        "rollout). Won't retroactively include a signal that's switched off."
    )
    st.markdown("**Real toggles for QB-vs-coverage / RB run-concepts (per direct request - no more manual code edits needed)**")
    st.caption(
        "Both require your own real, premium FantasyPoints CSV data "
        "(coverage_bundle/rb_bundle) to actually do anything - if that data "
        "isn't loaded, flipping these on won't error, but won't change the "
        "real results either, since there's nothing for the signal to compute "
        "from. Run this same week range once with a toggle off and once on, "
        "then compare the by_quality_tier_by_prop table for the relevant real "
        "props (pass_yards/pass_completions/pass_attempts for QB-coverage, "
        "rush_yards/rush_attempts for RB run-concepts) - that's the real, "
        "direct evidence for whether a signal genuinely helps."
    )
    toggle_col1, toggle_col2 = st.columns(2)
    with toggle_col1:
        toggle_qb_coverage = st.checkbox(
            "Enable QB-vs-coverage in quality_score", value=False, key="nfl_toggle_qb_coverage",
        )
    with toggle_col2:
        toggle_run_concept = st.checkbox(
            "Enable RB run-concepts in quality_score", value=False, key="nfl_toggle_run_concept",
        )
    combo_col1, combo_col2 = st.columns(2)
    with combo_col1:
        sim_n = st.number_input("Simulations per prop", min_value=100, max_value=5000, value=1000, step=100,
                                 key="nfl_sim_n_simulations")
    with combo_col2:
        combined_min_quality = st.number_input(
            "Minimum quality_score to backtest (0 = all rows)", min_value=0, max_value=100, value=0, step=5,
            key="nfl_combined_min_quality",
            help="Per direct request - only simulate/backtest rows at or above this "
                 "quality_score, to directly check whether a 'best of best' floor "
                 "genuinely produces more reliable results. 0 runs every real row, "
                 "unchanged from the default.",
        )
    # Real fix - this used to be ONE blocking call across the whole week
    # range (build_combined_readiness_and_simulation_report), holding
    # every week's scored rows in memory until the entire range finished,
    # then only writing session_state at the very end. Real-world use
    # showed weeks 1-7 "won't load" - the app was actually hitting
    # Streamlit Community Cloud's free-tier memory ceiling mid-run
    # (each week takes ~90-100+ real seconds and scoring several weeks'
    # worth of full slates plus a 1000-sample simulation at the end is
    # genuinely heavy) and getting silently restarted, which wipes
    # session_state completely with zero warning. Same root cause and
    # same fix pattern as the MLB tool's backtest: score ONE week at a
    # time, save that week's raw rows to disk the instant it finishes,
    # and skip any week already saved so a restart only costs whatever
    # week was actually in flight - not the whole range.
    NFL_BT_RESULTS_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"nfl_bt_raw_{bt_season}.csv")

    if "nfl_bt_raw" not in st.session_state or st.session_state.get("nfl_bt_raw_season") != bt_season:
        if os.path.exists(NFL_BT_RESULTS_FILE):
            try:
                st.session_state.nfl_bt_raw = pd.read_csv(NFL_BT_RESULTS_FILE)
            except Exception:
                st.session_state.nfl_bt_raw = pd.DataFrame()
        else:
            st.session_state.nfl_bt_raw = pd.DataFrame()
        st.session_state.nfl_bt_raw_season = bt_season

    st.caption(
        "Results save to disk, per real week, the moment each week finishes - "
        "if the app restarts mid-run (this platform's free tier can do that "
        "under memory pressure on a long range), just click Run again with "
        "the same season/week range and it'll only process the weeks not "
        "already saved. Note: a saved week reflects whatever QB-coverage/RB-"
        "concept toggle state was active when IT was scored - if you change "
        "a toggle, clear the accumulated data below before re-running, or "
        "you'll get a real mix of old-toggle and new-toggle weeks with no "
        "warning which is which."
    )

    if st.button("Run combined backtest for this week range", type="secondary"):
        if report_end_week < report_start_week:
            st.error("End week must be >= start week.")
        else:
            # Real, runtime toggle - sets the module's own real global flags
            # directly (confirmed this actually works - functions inside
            # nfl_model_combined.py read these as their own module's global,
            # so a change made here is genuinely seen by them, not just a
            # local copy). Reset to their real, current defaults after the
            # run either way, so this doesn't silently leave a flag flipped
            # for anything else that runs later in the same session.
            import nfl_model_combined as _nfl_module
            _prior_qb_coverage = _nfl_module.ENABLE_QB_COVERAGE_IN_QUALITY_SCORE
            _prior_run_concept = _nfl_module.ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE
            _nfl_module.ENABLE_QB_COVERAGE_IN_QUALITY_SCORE = toggle_qb_coverage
            _nfl_module.ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE = toggle_run_concept
            weeks_to_run = list(range(report_start_week, report_end_week + 1))

            already_done_weeks = set()
            if not st.session_state.nfl_bt_raw.empty and "week" in st.session_state.nfl_bt_raw.columns:
                already_done_weeks = set(st.session_state.nfl_bt_raw["week"].dropna().astype(int).tolist())
            weeks_remaining = [w for w in weeks_to_run if w not in already_done_weeks]
            skipped = len(weeks_to_run) - len(weeks_remaining)

            try:
                if not weeks_remaining:
                    st.info(f"All {len(weeks_to_run)} week(s) in this range are already saved from a "
                             f"prior run - nothing new to score. Clear the accumulated data below if "
                             f"you want to re-run them (e.g. after changing a toggle).")
                else:
                    if skipped:
                        st.info(f"Skipping {skipped} week(s) already saved from a prior run - "
                                 f"scoring the remaining {len(weeks_remaining)}.")
                    progress = st.progress(0.0, text="Starting...")
                    total_new_rows = 0
                    timed_out_weeks = []
                    for i, wk in enumerate(weeks_remaining):
                        # Real fix - hard timeout per week. A hung
                        # nflreadpy/network pull used to freeze the WHOLE
                        # remaining run with no error shown at all - this
                        # gives up on just this one week after
                        # BT_PER_WEEK_TIMEOUT_SECONDS and keeps going, so a
                        # single bad week doesn't take the rest of the range
                        # down with it.
                        try:
                            wk_df, timed_out = _run_with_timeout(
                                score_week_against_actuals,
                                (bt_season, wk),
                                {"starters_only": True,
                                 "coverage_bundle": st.session_state.get("coverage_bundle"),
                                 "rb_bundle": st.session_state.get("rb_bundle")},
                                BT_PER_WEEK_TIMEOUT_SECONDS,
                            )
                        except Exception as e:
                            st.warning(f"Week {wk} failed to score, skipped: {e}")
                            wk_df, timed_out = pd.DataFrame(), False

                        if timed_out:
                            timed_out_weeks.append(wk)
                            wk_df = pd.DataFrame()

                        # Real fix - save THIS week's rows to session_state
                        # AND disk immediately, right after it finishes,
                        # instead of waiting for every requested week to
                        # complete. session_state alone only protects a
                        # rerun within the same session (tab switches) - the
                        # disk file is what survives the app process itself
                        # restarting, which wipes session_state completely.
                        if not wk_df.empty:
                            if st.session_state.nfl_bt_raw.empty:
                                st.session_state.nfl_bt_raw = wk_df
                            else:
                                st.session_state.nfl_bt_raw = pd.concat(
                                    [st.session_state.nfl_bt_raw, wk_df], ignore_index=True)
                            file_exists = os.path.exists(NFL_BT_RESULTS_FILE)
                            wk_df.to_csv(NFL_BT_RESULTS_FILE, mode="a", header=not file_exists, index=False)
                            total_new_rows += len(wk_df)

                        progress.progress((i + 1) / len(weeks_remaining),
                                           text=f"Scored {i+1}/{len(weeks_remaining)} weeks "
                                                f"({total_new_rows} real rows saved so far"
                                                + (f", {len(timed_out_weeks)} timed out" if timed_out_weeks else "")
                                                + ")...")
                    progress.empty()

                    if timed_out_weeks:
                        st.warning(f"Week(s) {timed_out_weeks} took longer than "
                                   f"{BT_PER_WEEK_TIMEOUT_SECONDS}s to pull/score and were skipped instead "
                                   f"of freezing the whole run - this is what was previously showing as "
                                   f"'stops scanning' with no error. Click Run again to retry just these "
                                   f"weeks (everything else stays saved).")

                    st.success(f"Scored {total_new_rows} new real rows this run - "
                               f"{len(st.session_state.nfl_bt_raw)} total accumulated for "
                               f"season {bt_season}.")

                # Real, cheap step - rebuilds every summary table from
                # whatever's accumulated in session_state right now, even if
                # this run only added a few weeks or got cut short. Doesn't
                # require every requested week to be present - a partial
                # range still produces a real, usable report instead of
                # nothing until the full range finishes.
                st.session_state.combined_report = build_combined_report_from_raw(
                    st.session_state.nfl_bt_raw,
                    n_simulations=int(sim_n),
                    min_quality_score=combined_min_quality if combined_min_quality > 0 else None,
                )
                st.session_state.backtest_mode = True
            except Exception as e:
                st.error(f"Combined backtest failed: {e}")
            finally:
                # Real, guaranteed reset - runs whether the backtest
                # succeeded or failed, so a toggle used for one real test
                # never silently stays flipped for anything that runs
                # later in the same session.
                _nfl_module.ENABLE_QB_COVERAGE_IN_QUALITY_SCORE = _prior_qb_coverage
                _nfl_module.ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE = _prior_run_concept

    if st.session_state.get("nfl_bt_raw") is not None and not st.session_state.nfl_bt_raw.empty:
        if st.button("Clear accumulated backtest data for this season", key="nfl_bt_clear_button"):
            st.session_state.nfl_bt_raw = pd.DataFrame()
            st.session_state.combined_report = None
            if os.path.exists(NFL_BT_RESULTS_FILE):
                os.remove(NFL_BT_RESULTS_FILE)
            st.rerun()

    if st.session_state.get("combined_report") is not None and not st.session_state.combined_report["raw"].empty:
        creport = st.session_state.combined_report
        st.markdown("**Readiness: real accuracy (|mu - actual|) by quality_score tier, all props**")
        st.dataframe(creport["by_quality_tier"], width='stretch')
        st.markdown("**Readiness: same tier breakdown, split by prop_type**")
        st.dataframe(creport["by_quality_tier_by_prop"], width='stretch')
        st.caption(
            "The real, direct 'is quality_score actually meaningful' check - if the "
            "80-100 tier's mean_abs_miss/mean_match_ratio isn't meaningfully better "
            "than the <40 tier's, quality_score isn't earning its keep yet for that "
            "prop, even if it looks fine when every prop is pooled together."
        )
        st.markdown("**Readiness: mean absolute miss by prop_type**")
        st.dataframe(creport["by_prop_type"], width='stretch')
        if not pd.isna(creport["adjustment_direction_accuracy"]):
            st.metric("Coverage/box-count adjustment direction accuracy",
                      f"{creport['adjustment_direction_accuracy']*100:.1f}%",
                      help="Should clear 50% by a real margin - if it doesn't, the "
                           "adjustment isn't adding signal as currently weighted.")
        if not creport["bucket_summary"].empty:
            st.markdown("**Simulation: real hit-rate by prop_type and gap-pct bucket**")
            st.dataframe(creport["bucket_summary"], width='stretch')
            st.caption(
                "Split by prop_type from the start - different NFL props likely need "
                "genuinely different real gap-pct thresholds. Needs a real, decent "
                "sample per bucket before trusting it."
            )

    st.divider()
    st.subheader("Real 1Q / 1H prop backtest (Option A - genuinely independent historical build)")
    st.caption(
        "Every prop above (rush attempts, longest rush, rush yards, rush TDs, "
        "receptions, targets, rec yards, rec TDs, longest catch, pass attempts, "
        "completions, pass yards, pass TDs, longest completion), now also computed "
        "for just the 1st quarter or 1st half - built from real play-by-play data, "
        "filtered to the real, actual plays inside that time window, with its own "
        "genuinely independent real historical mu/sigma - not a fraction or estimate "
        "derived from the full-game number. Uses the same, unmodified mu/sigma logic "
        "(recency weighting, shrinkage) the full-game props already use - real, honest "
        "note: that logic hasn't been separately tuned for partial-game variance, "
        "which may genuinely behave differently (a smaller, choppier real sample) - "
        "worth watching once this runs against real data."
    )
    pw_col1, pw_col2 = st.columns(2)
    with pw_col1:
        partial_time_window = st.radio("Time window", ["1q", "1h"], horizontal=True, key="nfl_partial_window")
    with pw_col2:
        partial_n_sim = st.number_input("Simulations per prop", min_value=100, max_value=5000, value=1000,
                                         step=100, key="nfl_partial_n_simulations")
    if st.button(f"Run {report_start_week}-{report_end_week} week 1Q/1H backtest", type="secondary"):
        weeks_to_run = list(range(report_start_week, report_end_week + 1))
        all_partial_rows = []
        with st.spinner(f"Scoring real {partial_time_window.upper()} props across weeks "
                         f"{report_start_week}-{report_end_week} of {bt_season}..."):
            for wk in weeks_to_run:
                try:
                    wk_scored = score_partial_game_week_against_actuals(
                        bt_season, wk, partial_time_window, starters_only=True)
                    if wk_scored.empty:
                        continue
                    wk_sim = add_simulation_columns_to_backtest_rows(wk_scored, n_simulations=int(partial_n_sim))
                    if not wk_sim.empty:
                        all_partial_rows.append(wk_sim)
                except Exception as e:
                    st.warning(f"Week {wk}: {e}")
                    continue
        if not all_partial_rows:
            st.warning("No real, comparable 1Q/1H rows came back for this range.")
        else:
            partial_raw = pd.concat(all_partial_rows, ignore_index=True)
            st.session_state.nfl_partial_backtest_raw = partial_raw
            st.success(f"Scored {len(partial_raw)} real {partial_time_window.upper()} rows across weeks "
                       f"{report_start_week}-{report_end_week} of {bt_season}.")

    if st.session_state.get("nfl_partial_backtest_raw") is not None:
        praw = st.session_state.nfl_partial_backtest_raw
        bins = [0, 5, 10, 15, 20, 30, 1000]
        labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
        praw = praw.copy()
        praw["gap_bucket"] = pd.cut(praw["gap_pct"], bins=bins, labels=labels, right=False)
        psummary = praw.groupby(["prop_type", "gap_bucket"], observed=True).agg(
            n=("real_cleared_line", "size"), real_hit_rate=("real_cleared_line", "mean"),
        ).reset_index()
        psummary["real_hit_rate"] = round(psummary["real_hit_rate"] * 100, 1)
        st.markdown("**Real hit-rate by 1Q/1H prop_type and gap-pct bucket**")
        st.dataframe(psummary, width='stretch')
        pmiss_summary = praw.groupby("prop_type", observed=True).agg(
            n=("abs_miss", "size"), mean_abs_miss=("abs_miss", "mean"),
        ).reset_index()
        pmiss_summary["mean_abs_miss"] = round(pmiss_summary["mean_abs_miss"], 2)
        st.markdown("**Real mu accuracy by 1Q/1H prop_type**")
        st.dataframe(pmiss_summary, width='stretch')

