"""
draft_rankings_diagnostic.py
TEMPORARY diagnostic - checks draft_rankings.py's unverified assumptions
against real data:
  1. Defensive stat column names in player_stats (def_sacks, def_interceptions, etc.)
  2. load_ff_rankings() actual column names (player name field, rank field)
Once confirmed, draft_rankings.py gets corrected against real columns,
same process used for the main scanner build.
"""

import streamlit as st
import nflreadpy as nfl

st.title("Draft Rankings — Diagnostic")
st.write("Checking unverified column assumptions against real data.")

YEARS = [2025]

st.header("1. Defensive stat columns in player_stats")
try:
    player_stats = nfl.load_player_stats(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(player_stats):,} rows, {len(player_stats.columns)} total columns")

    candidates = [
        "def_sacks", "def_interceptions", "def_fumbles", "def_fumbles_forced",
        "fumble_recovery_tds", "def_tds", "special_teams_tds", "def_safeties",
        "def_tackles_solo", "def_tackles_with_assist", "def_qb_hits",
        "def_pass_defended", "def_sack_yards", "position",
    ]
    found = [c for c in candidates if c in player_stats.columns]
    missing = [c for c in candidates if c not in player_stats.columns]

    st.write("FOUND (usable as-is):", found)
    st.write("MISSING (need a different name):", missing)

    # Check if defensive positions (DL/LB/DB) actually appear in this data
    if "position" in player_stats.columns:
        def_positions_present = player_stats[
            player_stats["position"].isin(["DL", "LB", "DB", "CB", "S", "DE", "DT", "OLB", "ILB"])
        ]
        st.write(f"Rows with defensive positions: {len(def_positions_present):,}")
        if not def_positions_present.empty:
            st.write("Sample defensive player row:")
            st.dataframe(def_positions_present.head(3))
except Exception as e:
    st.error(f"player_stats pull failed: {e}")

st.header("2. load_ff_rankings() column names")
try:
    ff_rankings = nfl.load_ff_rankings().to_pandas()
    st.success(f"Pulled {len(ff_rankings):,} rows")
    st.write("Columns:", sorted(ff_rankings.columns.tolist()))
    st.write("Sample rows:")
    st.dataframe(ff_rankings.head(10))
except Exception as e:
    st.error(f"load_ff_rankings failed: {e}")

st.header("3. Rosters — checking full_name and position values for K/DEF")
try:
    rosters = nfl.load_rosters(seasons=YEARS).to_pandas()
    st.success(f"Pulled {len(rosters):,} rows")
    unique_positions = sorted(rosters["position"].dropna().unique().tolist())
    st.write("All unique position values in rosters:", unique_positions)
except Exception as e:
    st.error(f"rosters pull failed: {e}")

st.info("Send back what shows for all 3 sections so draft_rankings.py can be corrected against real columns.")
