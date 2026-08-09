"""
TEMPORARY diagnostic version of streamlit_app.py.
Purpose: pull real data from nflreadpy and display actual column names
so we can correct the placeholder column names in nfl_model_combined.py.
Once we've confirmed everything, this gets replaced by the real scanner UI.
"""

import streamlit as st
import nflreadpy as nfl

st.title("NFL Matchup Tool — Data Diagnostic")
st.write("Pulling real data to confirm column names before building the scanner.")

YEARS = [2025]  # last completed season - safest for a first pull test

st.header("1. Play-by-play (load_pbp) — filtered to relevant columns")
try:
    pbp = nfl.load_pbp(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(pbp):,} rows, {len(pbp.columns)} total columns")

    candidates = [
        "play_type", "posteam", "defteam", "down", "ydstogo", "yardline_100",
        "rush_attempt", "rushing_yards", "rusher_player_id", "rusher_player_name",
        "run_location", "run_gap",
        "pass_attempt", "passing_yards", "passer_player_id", "passer_player_name",
        "complete_pass", "incomplete_pass", "interception",
        "receiver_player_id", "receiver_player_name", "receiving_yards",
        "air_yards", "yards_after_catch", "pass_location",
        "touchdown", "rush_touchdown", "pass_touchdown",
        "field_goal_attempt", "field_goal_result", "kick_distance",
        "extra_point_attempt", "extra_point_result",
        "sack", "qb_hit", "epa", "wp", "game_id", "week", "season", "game_date",
        "nflverse_game_id", "play_id",
    ]
    found = [c for c in candidates if c in pbp.columns]
    missing = [c for c in candidates if c not in pbp.columns]

    st.write("FOUND (usable as-is):", found)
    st.write("MISSING (need a different name — check the full list below):", missing)

    st.write("Full column list (371 total), for cross-checking missing ones:")
    st.code("\n".join(sorted(pbp.columns.tolist())))
except Exception as e:
    st.error(f"pbp pull failed: {e}")

st.header("2. Next Gen Stats — passing")
try:
    ngs_pass = nfl.load_nextgen_stats(stat_type="passing", seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(ngs_pass):,} rows")
    st.write("Columns:", sorted(ngs_pass.columns.tolist()))
    st.dataframe(ngs_pass.head(3))
except Exception as e:
    st.error(f"NGS passing pull failed: {e}")

st.header("3. Next Gen Stats — rushing")
try:
    ngs_rush = nfl.load_nextgen_stats(stat_type="rushing", seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(ngs_rush):,} rows")
    st.write("Columns:", sorted(ngs_rush.columns.tolist()))
    st.dataframe(ngs_rush.head(3))
except Exception as e:
    st.error(f"NGS rushing pull failed: {e}")

st.header("4. Next Gen Stats — receiving")
try:
    ngs_rec = nfl.load_nextgen_stats(stat_type="receiving", seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(ngs_rec):,} rows")
    st.write("Columns:", sorted(ngs_rec.columns.tolist()))
    st.dataframe(ngs_rec.head(3))
except Exception as e:
    st.error(f"NGS receiving pull failed: {e}")

st.header("5. Player stats (load_player_stats) — filtered to relevant columns")
try:
    player_stats = nfl.load_player_stats(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(player_stats):,} rows, {len(player_stats.columns)} total columns")

    candidates = [
        "player_id", "gsis_id", "player_display_name", "player_name",
        "position", "team", "recent_team", "season", "week",
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "fantasy_points", "fantasy_points_ppr",
    ]
    found = [c for c in candidates if c in player_stats.columns]
    missing = [c for c in candidates if c not in player_stats.columns]

    st.write("FOUND (usable as-is):", found)
    st.write("MISSING (need a different name — check the full list below):", missing)

    st.write("Full column list, for cross-checking missing ones:")
    st.code("\n".join(sorted(player_stats.columns.tolist())))
except Exception as e:
    st.error(f"player_stats pull failed: {e}")

st.header("6. FTN charting (load_ftn_charting)")
try:
    ftn = nfl.load_ftn_charting(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(ftn):,} rows")
    st.write("Columns:", sorted(ftn.columns.tolist()))
    st.dataframe(ftn.head(3))
except Exception as e:
    st.error(f"FTN charting pull failed: {e}")

st.header("7. Participation (load_participation)")
try:
    participation = nfl.load_participation(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(participation):,} rows")
    st.write("Columns:", sorted(participation.columns.tolist()))
    st.dataframe(participation.head(3))
except Exception as e:
    st.error(f"participation pull failed: {e}")

st.header("8. Snap counts (load_snap_counts)")
try:
    snaps = nfl.load_snap_counts(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(snaps):,} rows")
    st.write("Columns:", sorted(snaps.columns.tolist()))
    st.dataframe(snaps.head(3))
except Exception as e:
    st.error(f"snap_counts pull failed: {e}")

st.header("9. Depth charts (load_depth_charts)")
try:
    depth = nfl.load_depth_charts(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(depth):,} rows")
    st.write("Columns:", sorted(depth.columns.tolist()))
    st.dataframe(depth.head(3))
except Exception as e:
    st.error(f"depth_charts pull failed: {e}")

st.header("10. Rosters (load_rosters)")
try:
    rosters = nfl.load_rosters(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(rosters):,} rows")
    st.write("Columns:", sorted(rosters.columns.tolist()))
    st.dataframe(rosters.head(3))
except Exception as e:
    st.error(f"rosters pull failed: {e}")

st.header("11. Schedules (load_schedules) — checking gameday column")
try:
    schedules = nfl.load_schedules(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(schedules):,} rows")
    st.write("Columns:", sorted(schedules.columns.tolist()))
    st.dataframe(schedules.head(3))
except Exception as e:
    st.error(f"schedules pull failed: {e}")

st.info("Once every section above shows real columns with no red errors, "
        "screenshot each section (or copy the column list text) and send it back "
        "so the backend file can be corrected to match the real data.")
