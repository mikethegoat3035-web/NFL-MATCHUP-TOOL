"""
NFL PREMIUM TOOL - Coverage Matchup Streamlit Tab
=====================================================
Renders a UI on top of coverage_matchup.py's data + logic. This is a
SEPARATE file from coverage_matchup.py on purpose - that file stays pure
data/logic (already tested standalone all session), this file is just
the display layer on top of it.

HOW TO WIRE THIS INTO YOUR EXISTING APP (2 lines):
1. Add this file to your repo as coverage_matchup_tab.py (same folder as
   streamlit_app.py, coverage_matchup.py, and the data/ folder).
2. In streamlit_app.py, near your other tab imports/setup, add:

       from coverage_matchup_tab import render_coverage_matchup_tab

   Then wherever you create your tabs (likely a st.tabs([...]) call),
   add one more tab and call render_coverage_matchup_tab() inside it.
   Example pattern (adjust names to match your actual tab setup):

       tab_scan, tab_backtest, tab_draft, tab_coverage = st.tabs(
           ["Scan", "Backtest", "Draft Rankings", "Coverage Matchup"]
       )
       with tab_coverage:
           render_coverage_matchup_tab()

That's the only wiring needed - everything else (loading the 72 CSVs,
the matchup logic, tiering, thin-sample flags) is already built and
tested in coverage_matchup.py.
"""

import streamlit as st
from coverage_matchup import load_full_dataset, get_matchup, ALIGNMENTS, TEAM_ABBREV_TO_FULL


@st.cache_resource
def _get_bundle():
    """Loads all 72 CSVs from the data/ folder ONCE per app session,
    not on every interaction - st.cache_resource keeps it in memory.
    'data' is relative to the repo root, matching where the CSVs were
    uploaded (data/wide_vs_cover0.csv etc)."""
    return load_full_dataset(data_dir="data")


def render_coverage_matchup_tab():
    st.subheader("Coverage Matchup Lookup")
    st.caption(
        "Auto-detects each defense's real statistical outlier coverage(s) "
        "and shows the player's own history plus what that defense actually "
        "allows in that specific coverage - built from FantasyPoints.com data."
    )

    bundle = _get_bundle()

    if bundle.missing:
        with st.expander(f"⚠️ {len(bundle.missing)} file(s) not found in data/", expanded=False):
            for m in bundle.missing:
                st.text(m)

    all_teams = sorted(bundle.def_coverage.keys())

    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Player name", placeholder="e.g. Matthew Stafford")
        position = st.selectbox("Position", ["QB", "WR", "TE", "RB"])
        player_team_abbrev = st.text_input(
            "Player's team (abbreviation, optional but recommended)",
            placeholder="e.g. LA, SEA, DAL",
            help="Prevents building a nonsense report if the player happens to face his own team."
        )

    with col2:
        opponent_team = st.selectbox("Opponent (defense)", all_teams)
        alignment = None
        if position != "QB":
            alignment = st.selectbox("Alignment", [a.capitalize() for a in ALIGNMENTS]).lower()

    if st.button("Get Matchup", type="primary"):
        if not player_name.strip():
            st.warning("Enter a player name first.")
            return

        report = get_matchup(
            bundle, player_name.strip(), position, opponent_team,
            player_team=player_team_abbrev.strip() or None,
            alignment=alignment,
        )

        if report and "error" in report[0]:
            st.error(report[0]["error"])
            return
        if report and "note" in report[0]:
            st.info(report[0]["note"])
            return

        for entry in report:
            st.markdown(
                f"### {opponent_team} runs **{entry['coverage']}** at "
                f"{entry['opponent_usage_pct']:.1f}% "
                f"(z={entry['opponent_z_score']:+.2f} vs league — real outlier)"
            )

            own_data = entry.get("qb_data") or entry.get("receiver_data")
            own_confidence = entry.get("confidence")
            vol_key = "ATT" if position == "QB" else "TGT"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{player_name} (own history)**")
                if own_data is None:
                    st.text("No recorded attempts/targets vs this coverage.")
                else:
                    thin = own_confidence == "thin_sample"
                    st.caption(f"{own_data['_att']} {vol_key}" + ("  🔸 THIN SAMPLE" if thin else ""))
                    if entry.get("alignment_fit_warning"):
                        st.caption(f"⚠️ Only {entry['alignment_fit_pct']:.0f}% of routes are "
                                   f"this alignment - may not represent usual usage")
                    highlight = (["CMP %", "YPA", "TD", "INT", "RATE", "CPOE", "FP/G"] if position == "QB"
                                 else ["CR %", "YPRR", "TD", "CTGT %", "RATE", "FP/G"])
                    for stat in highlight:
                        if stat in own_data:
                            tier = own_data.get("_tiers", {}).get(stat, "-")
                            st.text(f"{stat}: {own_data[stat]}  ({tier})")

            with c2:
                st.markdown(f"**{opponent_team} allows**")
                def_data = entry.get("defense_allows")
                if def_data is None:
                    st.text("No data.")
                else:
                    def_thin = entry.get("defense_confidence") == "thin_sample"
                    st.caption(f"{def_data['_att']} {vol_key} allowed" + ("  🔸 THIN SAMPLE" if def_thin else ""))
                    highlight = (["CMP %", "YPA", "TD", "INT", "RATE"] if position == "QB"
                                 else ["CR %", "YPRR", "TD", "CTGT %", "RATE"])
                    for stat in highlight:
                        if stat in def_data:
                            tier = def_data.get("_tiers", {}).get(stat, "-")
                            st.text(f"{stat}: {def_data[stat]}  ({tier})")

            st.divider()
