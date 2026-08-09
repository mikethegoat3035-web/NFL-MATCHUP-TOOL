"""
nfl_model_combined.py
Backend for the NFL matchup/prop analysis tool ("quality mu" style),
mirroring the structure of prop_model_combined.py from the MLB tool.

Data sources (all free, via nflreadpy):
  - load_pbp()              -> play-by-play (down/distance, target depth, EPA, run_location, FG/XP)
  - load_nextgen_stats()    -> official NGS passing/rushing/receiving efficiency
  - load_player_stats()     -> game/season aggregates (targets, receptions, rush attempts, etc.)
  - load_snap_counts()      -> snap share / route participation proxy
  - load_ftn_charting()     -> FTN charting: coverage type, man/zone, box count, motion, play-action
  - load_participation()    -> also carries defense_man_zone_type / defense_coverage_type, time_to_throw, was_pressure

NOTE: nflreadpy returns Polars DataFrames. We convert to pandas immediately
after each pull so the rest of the codebase (styling, Streamlit, scoring)
stays consistent with the pandas-based MLB tool.

RESOLVED ID KEY MISMATCH ACROSS TABLES: player_stats natively keys players on
`player_id`, while NGS/rosters/depth_charts key on `gsis_id`. pull_player_stats()
renames player_id -> gsis_id immediately on pull, so every downstream function
in this file (detect_role_change, calc_receiving_mu, calc_kicking_mu, etc.)
can safely join/filter on `gsis_id` consistently across all tables.
"""

import pandas as pd
import numpy as np

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None  # allows this file to be imported/tested without the package present


# ---------------------------------------------------------------------------
# 1. DATA PULL FUNCTIONS
# ---------------------------------------------------------------------------

def pull_pbp(years: list[int]) -> pd.DataFrame:
    """Play-by-play data for the given seasons, converted to pandas."""
    df = nfl.load_pbp(seasons=years)
    return df.to_pandas()


def pull_ngs(stat_type: str, years: list[int]) -> pd.DataFrame:
    """
    stat_type: 'passing', 'rushing', or 'receiving'
    Returns official Next Gen Stats for the given seasons.
    """
    df = nfl.load_nextgen_stats(stat_type=stat_type, seasons=years)
    return df.to_pandas()


def pull_player_stats(years: list[int]) -> pd.DataFrame:
    """
    Game-level player stats (targets, receptions, rush att, pass yds, etc.).

    CONFIRMED: player_stats keys players on `player_id`, while NGS/rosters/
    depth_charts all key on `gsis_id`. Renaming here so every other function
    in this file can join on `gsis_id` consistently without re-checking which
    table uses which name.
    """
    df = nfl.load_player_stats(seasons=years).to_pandas()
    df = df.rename(columns={"player_id": "gsis_id"})
    return df


def pull_snap_counts(years: list[int]) -> pd.DataFrame:
    """Snap counts by player/game - used as a route-participation / opportunity proxy."""
    df = nfl.load_snap_counts(seasons=years)
    return df.to_pandas()


def pull_ftn_charting(years: list[int]) -> pd.DataFrame:
    """
    FTN manual charting data (free, 2022-onward).
    Key columns: n_defense_box, n_offense_backfield, is_motion, is_play_action,
    is_screen_pass, is_no_huddle, qb_location.
    """
    df = nfl.load_ftn_charting(seasons=years)
    return df.to_pandas()


def pull_participation(years: list[int]) -> pd.DataFrame:
    """
    Participation data - carries defense_man_zone_type, defense_coverage_type,
    time_to_throw, was_pressure. This is where coverage-shell % comes from.
    """
    df = nfl.load_participation(seasons=years)
    return df.to_pandas()


def pull_schedules(years: list[int]) -> pd.DataFrame:
    df = nfl.load_schedules(seasons=years)
    return df.to_pandas()


def pull_rosters(years: list[int]) -> pd.DataFrame:
    df = nfl.load_rosters(seasons=years)
    return df.to_pandas()


def pull_depth_charts(years: list[int]) -> pd.DataFrame:
    df = nfl.load_depth_charts(seasons=years)
    return df.to_pandas()


# ---------------------------------------------------------------------------
# 2. COVERAGE % AGGREGATION (per defense, by team)
# ---------------------------------------------------------------------------

def build_coverage_profile(participation_df: pd.DataFrame, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates defense_coverage_type and defense_man_zone_type into
    per-team usage rates.

    CONFIRMED real participation columns: defenders_in_box, defense_coverage_type,
    defense_man_zone_type, defense_personnel, offense_formation, offense_personnel,
    route, time_to_throw, was_pressure, possession_team, nflverse_game_id, play_id.

    CONFIRMED join key fix: pbp_df has NO nflverse_game_id column - only
    game_id (which IS the nflverse-format ID, e.g. "2025_01_KC_BAL") and
    old_game_id (legacy numeric). participation_df's nflverse_game_id
    corresponds to pbp's game_id, not a column of the same name - so the
    join uses left_on/right_on across the differently-named columns.

    Returns one row per defteam with columns like:
      cover_1_pct, cover_2_pct, cover_3_pct, cover_4_pct, cover_6_pct,
      man_pct, zone_pct, n_plays
    """
    merged = participation_df.merge(
        pbp_df[["game_id", "play_id", "defteam", "posteam"]],
        left_on=["nflverse_game_id", "play_id"],
        right_on=["game_id", "play_id"],
        how="left",
    )
    df = merged.dropna(subset=["defense_coverage_type", "defteam"])

    coverage_counts = (
        df.groupby(["defteam", "defense_coverage_type"])
        .size()
        .reset_index(name="n")
    )
    totals = df.groupby("defteam").size().reset_index(name="n_plays")

    pivot = coverage_counts.pivot(
        index="defteam", columns="defense_coverage_type", values="n"
    ).fillna(0)
    pivot = pivot.merge(totals, on="defteam")

    # normalize each coverage type column into a % of total plays
    coverage_cols = [c for c in pivot.columns if c not in ("defteam", "n_plays")]
    for col in coverage_cols:
        pivot[f"{col}_pct"] = (pivot[col] / pivot["n_plays"]).round(3)

    man_zone = (
        df.groupby(["defteam", "defense_man_zone_type"])
        .size()
        .reset_index(name="n")
        .pivot(index="defteam", columns="defense_man_zone_type", values="n")
        .fillna(0)
    )
    man_zone_pct = man_zone.div(man_zone.sum(axis=1), axis=0).round(3)
    man_zone_pct.columns = [f"{c}_pct" for c in man_zone_pct.columns]

    result = pivot.merge(man_zone_pct, on="defteam", how="left")
    return result


def build_box_count_profile(ftn_df: pd.DataFrame, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates n_defense_box into a per-team stacked-box rate.

    FIX: ftn_df has NO defteam/posteam columns directly - only
    nflverse_game_id and nflverse_play_id. Joins to pbp_df on those keys
    (pbp's game_id/play_id) to pull in defteam/posteam, same fix as
    build_coverage_profile() needed for participation_df.

    Returns avg box count and % of plays with 7+ / 8+ defenders in the box,
    split by defteam (and separately, offense's box counts faced, by posteam).
    """
    merged = ftn_df.merge(
        pbp_df[["game_id", "play_id", "defteam", "posteam"]],
        left_on=["nflverse_game_id", "nflverse_play_id"],
        right_on=["game_id", "play_id"],
        how="left",
    )
    df = merged.dropna(subset=["n_defense_box", "defteam"]).copy()

    def_profile = (
        df.groupby("defteam")
        .agg(
            avg_box_count=("n_defense_box", "mean"),
            pct_stacked_7plus=("n_defense_box", lambda x: (x >= 7).mean()),
            pct_stacked_8plus=("n_defense_box", lambda x: (x >= 8).mean()),
            n_plays=("n_defense_box", "count"),
        )
        .reset_index()
    )

    off_profile = (
        df.dropna(subset=["posteam"])
        .groupby("posteam")
        .agg(
            avg_box_faced=("n_defense_box", "mean"),
            pct_faced_stacked_7plus=("n_defense_box", lambda x: (x >= 7).mean()),
            n_plays_off=("n_defense_box", "count"),
        )
        .reset_index()
    )

    return def_profile, off_profile


# ---------------------------------------------------------------------------
# 3. EXPLOSIVE-PLAY / TAIL-RISK RATES (rush + pass + rec)
# ---------------------------------------------------------------------------

def build_explosive_rates(pbp_df: pd.DataFrame) -> dict:
    """
    Computes explosive-play rates needed for tail-heavy props
    (longest rush, rec yds, pass yds).
    """
    df = pbp_df.copy()

    rush_explosive = (
        df[df["play_type"] == "run"]
        .groupby("rusher_player_id")
        .agg(
            explosive_10plus_rate=("rushing_yards", lambda x: (x >= 10).mean()),
            explosive_15plus_rate=("rushing_yards", lambda x: (x >= 15).mean()),
            max_rush_yards=("rushing_yards", "max"),
            n_carries=("rushing_yards", "count"),
        )
        .reset_index()
    )

    pass_explosive = (
        df[df["play_type"] == "pass"]
        .groupby("passer_player_id")
        .agg(
            explosive_20plus_rate=("passing_yards", lambda x: (x >= 20).mean()),
            explosive_40plus_rate=("passing_yards", lambda x: (x >= 40).mean()),
            n_attempts=("passing_yards", "count"),
        )
        .reset_index()
    )

    rec_explosive = (
        df[df["play_type"] == "pass"]
        .dropna(subset=["receiver_player_id"])
        .groupby("receiver_player_id")
        .agg(
            explosive_15plus_rate=("receiving_yards", lambda x: (x >= 15).mean()),
            explosive_20plus_rate=("receiving_yards", lambda x: (x >= 20).mean()),
            n_targets=("receiving_yards", "count"),
        )
        .reset_index()
    )

    return {
        "rush_explosive": rush_explosive,
        "pass_explosive": pass_explosive,
        "rec_explosive": rec_explosive,
    }


# ---------------------------------------------------------------------------
# 4. COORDINATOR TENDENCY MAPPING (manual lookup - free data can't supply this)
# ---------------------------------------------------------------------------

# Maintain this manually - update whenever a team hires/fires an OC or DC.
# When a coordinator moves teams, their historical tendency profile
# (computed from posteam/defteam while they were at their OLD team)
# can be applied to their NEW team before enough current-season
# data has accumulated under them.
COORDINATOR_MAP = {
    # "team_abbr": {"oc": "Coordinator Name", "dc": "Coordinator Name"},
    # Fill in each offseason / after news of a hire/fire.
}


def get_coordinator_tendency_profile(coach_name: str, tendency_df: pd.DataFrame,
                                      coordinator_history: dict) -> pd.DataFrame:
    """
    coordinator_history: {"Coordinator Name": ["team_abbr_year1", "team_abbr_year2", ...]}
    Pulls the tendency rows (PROE, box count, coverage rate, personnel, motion, pace)
    for every team/season that coordinator called plays for, so their profile
    travels with them to a new team.
    """
    teams_seasons = coordinator_history.get(coach_name, [])
    if not teams_seasons:
        return pd.DataFrame()
    mask = tendency_df["team_season_key"].isin(teams_seasons)
    return tendency_df[mask]


# ---------------------------------------------------------------------------
# 4b. ROLE / VOLUME BRIDGE FOR TRADES, NEW STARTERS, NEW COORDINATORS
# ---------------------------------------------------------------------------

def detect_role_change(player_gsis_id: str, current_season: int, current_week: int,
                        rosters_df: pd.DataFrame, depth_charts_df: pd.DataFrame,
                        player_stats_df: pd.DataFrame, schedules_df: pd.DataFrame) -> dict:
    """
    Flags whether a player's team/role changed recently (trade, new starter
    promotion, depth chart shift) so downstream mu calc knows to bridge
    volume from depth chart position rather than trust trailing stat history.

    CONFIRMED real column fixes vs original draft:
      - player ID key is `gsis_id` (not player_id) - consistent across
        rosters, depth_charts, and NGS data.
      - depth_charts_df has NO season/week columns - only a `dt` (date) field
        and `pos_slot`/`pos_rank` (not `depth_position`). We match the closest
        `dt` on/before the target game's date (pulled from schedules_df) instead
        of filtering by season/week directly.
      - rosters_df confirmed to have `season`, `week`, `team`, `gsis_id` - that
        part of the original logic holds.

    Returns a dict like:
      {
        "team_changed": bool,
        "current_team": str,
        "games_on_current_team": int,
        "depth_chart_slot": str,   # from pos_slot (e.g. "WR1", "RB2")
        "use_depth_chart_estimate": bool,
      }
    """
    roster_row = rosters_df[
        (rosters_df["gsis_id"] == player_gsis_id) & (rosters_df["season"] == current_season)
    ]
    if roster_row.empty:
        return {"team_changed": None, "current_team": None, "games_on_current_team": 0,
                "depth_chart_slot": None, "use_depth_chart_estimate": True}

    current_team = roster_row.iloc[0].get("team")

    games_on_team = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["team"] == current_team)
        & (player_stats_df["season"] == current_season)
        & (player_stats_df["week"] < current_week)
    ].shape[0]

    # depth_charts_df has no season/week - find the game date for this
    # season/week from schedules_df, then match the closest dt on/before it
    game_date_row = schedules_df[
        (schedules_df["season"] == current_season) & (schedules_df["week"] == current_week)
    ]
    depth_slot = None
    if not game_date_row.empty:
        target_date = game_date_row.iloc[0].get("gameday")
        player_depth_rows = depth_charts_df[
            (depth_charts_df["gsis_id"] == player_gsis_id)
            & (depth_charts_df["dt"] <= target_date)
        ].sort_values("dt", ascending=False)
        if not player_depth_rows.empty:
            depth_slot = player_depth_rows.iloc[0].get("pos_slot")

    prior_team_row = rosters_df[
        (rosters_df["gsis_id"] == player_gsis_id) & (rosters_df["season"] == current_season - 1)
    ]
    prior_team = prior_team_row.iloc[0].get("team") if not prior_team_row.empty else None
    team_changed = (prior_team is not None) and (prior_team != current_team)

    return {
        "team_changed": team_changed,
        "current_team": current_team,
        "games_on_current_team": games_on_team,
        "depth_chart_slot": depth_slot,
        "use_depth_chart_estimate": games_on_team < 3,
    }


def blend_volume_estimate(stat_based_volume: float, depth_chart_volume_estimate: float,
                           games_on_current_team: int, full_confidence_games: int = 3) -> float:
    """
    Bayesian-style shrinkage between a depth-chart-based volume estimate
    (used when a player is new to a team/role) and real accumulated
    stat-based volume, shifting weight toward real data as games pile up.
    Same shrinkage idea as the MLB tool's hitter window escalation.
    """
    if games_on_current_team <= 0:
        return depth_chart_volume_estimate
    weight_real = min(games_on_current_team / full_confidence_games, 1.0)
    return (weight_real * stat_based_volume) + ((1 - weight_real) * depth_chart_volume_estimate)


def blend_scheme_baseline(current_season_tendency: float, prior_baseline_tendency: float,
                           games_played_this_season: int, full_confidence_games: int = 5) -> float:
    """
    Bayesian shrinkage between this-season accumulating team/coordinator
    tendency and a prior baseline (last season's team data, or a new
    coordinator's tendency profile from his old team). Weight shifts toward
    current-season data as the season progresses - mirrors MLB's
    since-June/rolling-window blending approach rather than a hard cutover.
    """
    if games_played_this_season <= 0:
        return prior_baseline_tendency
    weight_current = min(games_played_this_season / full_confidence_games, 1.0)
    return (weight_current * current_season_tendency) + ((1 - weight_current) * prior_baseline_tendency)


# ---------------------------------------------------------------------------
# 4c. TEAM-LEVEL TENDENCY BLENDING (coverage %, box-stack rate) ACROSS SEASONS
# ---------------------------------------------------------------------------

def blend_team_tendency_profiles(current_profile_df: pd.DataFrame, prior_profile_df: pd.DataFrame,
                                  key_col: str, games_played_by_team: dict,
                                  full_confidence_games: int = 5) -> pd.DataFrame:
    """
    Blends a current-season team tendency table (e.g. coverage % by team,
    still thin early in the season) with the prior season's full-season
    table, using the same shrinkage logic as blend_scheme_baseline() -
    weight shifts toward current-season data as that team's real game count
    grows, rather than a fixed week cutover.

    current_profile_df / prior_profile_df: output of build_coverage_profile()
    or build_box_count_profile() - one row per team, numeric tendency columns.
    key_col: the team column name (e.g. "defteam" or "posteam").
    games_played_by_team: {team_abbr: games_played_this_season} - used to
    decide how much to trust the current-season numbers for that specific team.

    Returns one row per team with each numeric column blended. Teams present
    in only one of the two tables pass through unblended (using whichever
    table has them).
    """
    merged = prior_profile_df.merge(
        current_profile_df, on=key_col, how="outer", suffixes=("_prior", "_current")
    )

    numeric_cols = [c for c in prior_profile_df.columns if c != key_col]
    result = merged[[key_col]].copy()

    for col in numeric_cols:
        prior_col = f"{col}_prior"
        current_col = f"{col}_current"
        if prior_col not in merged.columns or current_col not in merged.columns:
            continue

        def _blend_row(row):
            team = row[key_col]
            games = games_played_by_team.get(team, 0)
            prior_val = row.get(prior_col)
            current_val = row.get(current_col)
            if pd.isna(current_val):
                return prior_val
            if pd.isna(prior_val):
                return current_val
            return blend_scheme_baseline(current_val, prior_val, games, full_confidence_games)

        result[col] = merged.apply(_blend_row, axis=1)

    return result


def build_blended_coverage_profile(season: int, week: int) -> pd.DataFrame:
    """
    Builds the coverage-% profile for the target week using a blend of this
    season's completed weeks (weeks < target week only - avoids leaking
    future data) and last season's full-season profile, weighted by how
    many real games each team has played so far this season.
    """
    current_pbp = pull_pbp([season])
    current_pbp = current_pbp[current_pbp["week"] < week]
    current_participation = pull_participation([season])
    current_participation = current_participation[
        current_participation["nflverse_game_id"].isin(current_pbp["game_id"])
    ]
    current_coverage = build_coverage_profile(current_participation, current_pbp)

    prior_pbp = pull_pbp([season - 1])
    prior_participation = pull_participation([season - 1])
    prior_coverage = build_coverage_profile(prior_participation, prior_pbp)

    games_played_by_team = current_pbp.groupby("defteam")["game_id"].nunique().to_dict()

    return blend_team_tendency_profiles(
        current_coverage, prior_coverage, "defteam", games_played_by_team
    )


def build_blended_box_profile(season: int, week: int) -> tuple:
    """
    Same blending approach as build_blended_coverage_profile(), applied to
    the box-stack rate profile (both defensive and offensive-side views).
    """
    current_ftn = pull_ftn_charting([season])
    current_ftn = current_ftn[current_ftn["week"] < week]
    current_pbp = pull_pbp([season])
    current_pbp = current_pbp[current_pbp["week"] < week]
    current_def_profile, current_off_profile = build_box_count_profile(current_ftn, current_pbp)

    prior_ftn = pull_ftn_charting([season - 1])
    prior_pbp = pull_pbp([season - 1])
    prior_def_profile, prior_off_profile = build_box_count_profile(prior_ftn, prior_pbp)

    games_played_by_defteam = current_pbp[current_pbp["defteam"].notna()].groupby("defteam")["game_id"].nunique().to_dict()
    games_played_by_posteam = current_pbp[current_pbp["posteam"].notna()].groupby("posteam")["game_id"].nunique().to_dict()

    blended_def = blend_team_tendency_profiles(
        current_def_profile, prior_def_profile, "defteam", games_played_by_defteam
    )
    blended_off = blend_team_tendency_profiles(
        current_off_profile, prior_off_profile, "posteam", games_played_by_posteam
    )
    return blended_def, blended_off


# ---------------------------------------------------------------------------
# 5. MU CALCULATION PER PROP TYPE
# ---------------------------------------------------------------------------

def calc_passing_mu(qb_ngs_row, def_coverage_profile_row, team_total_attempts):
    """
    mu = expected pass attempts (volume) x efficiency, adjusted for defense's
    coverage tendency and pressure rate.

    CONFIRMED real NGS passing columns (via nflreadpy load_nextgen_stats):
      attempts, completions, pass_yards, pass_touchdowns, interceptions,
      completion_percentage, completion_percentage_above_expectation,
      expected_completion_percentage, avg_time_to_throw, avg_completed_air_yards,
      avg_intended_air_yards, avg_air_yards_differential, avg_air_yards_to_sticks,
      aggressiveness, passer_rating, max_air_distance, max_completed_air_distance

    NOTE: there is no pre-built "PROE" (pass rate over expected) field in NGS -
    raw volume is just `attempts`. True PROE needs to be derived from pbp
    (comparing actual pass rate to expected pass rate by down/distance/score).
    For now, use the QB's raw `attempts` as volume; swap in a real PROE-adjusted
    figure once that's built from pbp data.
    """
    raw_attempts = qb_ngs_row.get("attempts", np.nan)
    cpoe = qb_ngs_row.get("completion_percentage_above_expectation", np.nan)
    adot = qb_ngs_row.get("avg_intended_air_yards", np.nan)
    aggressiveness = qb_ngs_row.get("aggressiveness", np.nan)
    # defense adjustment factor from coverage profile / pressure rate goes here
    return raw_attempts, cpoe, adot, aggressiveness


def calc_rushing_mu(rb_ngs_row, team_total_rush_attempts, def_box_profile_row):
    """
    mu = rush share x efficiency (yards over expected), adjusted for
    how often this defense stacks the box against this offense.

    CONFIRMED real NGS rushing columns:
      rush_attempts, rush_yards, rush_touchdowns, efficiency, avg_rush_yards,
      avg_time_to_los, expected_rush_yards, rush_yards_over_expected,
      rush_yards_over_expected_per_att, rush_pct_over_expected,
      percent_attempts_gte_eight_defenders

    NOTE: there is no pre-built "rush_attempt_share" field - compute manually
    as rb_ngs_row["rush_attempts"] / team_total_rush_attempts (sum of all RBs'
    rush_attempts for that team/week). Also note NGS rushing already includes
    percent_attempts_gte_eight_defenders per player - this can be used directly
    instead of (or alongside) the FTN-derived box-count profile.
    """
    rush_attempts = rb_ngs_row.get("rush_attempts", np.nan)
    rush_share = (
        rush_attempts / team_total_rush_attempts
        if team_total_rush_attempts else np.nan
    )
    efficiency = rb_ngs_row.get("rush_yards_over_expected_per_att", np.nan)
    box_stack_pct_faced = rb_ngs_row.get("percent_attempts_gte_eight_defenders", np.nan)
    return rush_share, efficiency, box_stack_pct_faced


def calc_receiving_mu(wr_ngs_row, wr_player_stats_row, def_coverage_profile_row):
    """
    mu = target share x catch efficiency, adjusted for defense's coverage
    shell tendency and how this specific offense/WR performs against it.

    UPDATE: player_stats already has target_share and wopr (weighted
    opportunity rating) built in - no manual computation from team totals
    needed after all. Also has air_yards_share, racr, pacr as bonus
    efficiency-share metrics. NGS still supplies adot/yac_oe (not in
    player_stats).

    CONFIRMED real NGS receiving columns: avg_intended_air_yards (= aDOT),
    avg_yac_above_expectation, avg_separation, avg_cushion.
    CONFIRMED real player_stats columns: target_share, wopr, air_yards_share,
    racr, receiving_epa.
    """
    target_share = wr_player_stats_row.get("target_share", np.nan)
    wopr = wr_player_stats_row.get("wopr", np.nan)
    adot = wr_ngs_row.get("avg_intended_air_yards", np.nan)
    yac_oe = wr_ngs_row.get("avg_yac_above_expectation", np.nan)
    separation = wr_ngs_row.get("avg_separation", np.nan)
    return target_share, wopr, adot, yac_oe, separation


def calc_kicking_mu(kicker_player_stats_row: dict) -> dict:
    """
    FG/XP mu pulled directly from player_stats' pre-built distance-bucket
    columns - no manual pbp derivation needed.

    CONFIRMED real player_stats columns (much richer than initially assumed):
      fg_att, fg_made, fg_pct, fg_long,
      fg_made_0_19, fg_made_20_29, fg_made_30_39, fg_made_40_49,
      fg_made_50_59, fg_made_60_,
      fg_missed_0_19, fg_missed_20_29, fg_missed_30_39, fg_missed_40_49,
      fg_missed_50_59, fg_missed_60_,
      pat_att, pat_made, pat_missed, pat_pct, pat_blocked,
      gwfg_att, gwfg_made (game-winning FG specific)
    """
    return {
        "fg_pct_overall": kicker_player_stats_row.get("fg_pct", np.nan),
        "fg_pct_0_39": None,  # combine fg_made_0_19 + fg_made_20_29 + fg_made_30_39 vs attempts in that range once we have team-level FG attempt distribution
        "fg_made_40_49": kicker_player_stats_row.get("fg_made_40_49", np.nan),
        "fg_made_50_59": kicker_player_stats_row.get("fg_made_50_59", np.nan),
        "fg_long": kicker_player_stats_row.get("fg_long", np.nan),
        "pat_pct": kicker_player_stats_row.get("pat_pct", np.nan),
        "pat_att": kicker_player_stats_row.get("pat_att", np.nan),
    }


# ---------------------------------------------------------------------------
# 5b. FANTASY POINTS CALCULATION (offense + kicker)
# ---------------------------------------------------------------------------

def calc_offense_fantasy_points(player_stats_row: dict) -> float:
    """
    Full PPR offensive fantasy scoring, using confirmed real player_stats columns:
      passing_yards, passing_tds, passing_interceptions,
      rushing_yards, rushing_tds,
      receptions, receiving_yards, receiving_tds,
      rushing_fumbles_lost, receiving_fumbles_lost, sack_fumbles_lost,
      passing_2pt_conversions, rushing_2pt_conversions, receiving_2pt_conversions

    Scoring rules (as provided):
      Passing Yards: 0.04/yd | Passing TD: 4 | INT: -1
      Rushing Yards: 0.1/yd | Rushing TD: 6
      Receptions: 1 (Full PPR) | Receiving Yards: 0.1/yd | Receiving TD: 6
      Fumbles Lost: -1 | 2-Point Conversion: 2
      Offensive Fumble Recovery TD: 6 | Kick/Punt/FG Return TD: 6

    NOTE: qualifying rule (1+ offensive snap or return TD) should be checked
    upstream using snap_counts (offense_snaps > 0) before calling this, since
    player_stats alone doesn't carry a snap-participation flag.
    """
    r = player_stats_row
    points = 0.0
    points += r.get("passing_yards", 0) * 0.04
    points += r.get("passing_tds", 0) * 4
    points += r.get("passing_interceptions", 0) * -1
    points += r.get("rushing_yards", 0) * 0.1
    points += r.get("rushing_tds", 0) * 6
    points += r.get("receptions", 0) * 1
    points += r.get("receiving_yards", 0) * 0.1
    points += r.get("receiving_tds", 0) * 6

    fumbles_lost = (
        r.get("rushing_fumbles_lost", 0)
        + r.get("receiving_fumbles_lost", 0)
        + r.get("sack_fumbles_lost", 0)
    )
    points += fumbles_lost * -1

    two_pt = (
        r.get("passing_2pt_conversions", 0)
        + r.get("rushing_2pt_conversions", 0)
        + r.get("receiving_2pt_conversions", 0)
    )
    points += two_pt * 2

    # Return TDs (special_teams_tds) and offensive fumble recovery TDs are not
    # cleanly broken out in player_stats as separate columns - special_teams_tds
    # exists and can be added at 6pts/each; offensive fumble recovery TD isn't
    # a distinct column and would need pbp-level detection if you want it exact.
    points += r.get("special_teams_tds", 0) * 6

    return round(points, 2)


def calc_kicker_fantasy_points(player_stats_row: dict) -> float:
    """
    Kicker fantasy scoring, using confirmed real player_stats columns:
      fg_made_0_19, fg_made_20_29, fg_made_30_39 (all = "0-39 yard" bucket, 3pts each)
      fg_made_40_49 (4pts), fg_made_50_59 + fg_made_60_ (both = "50+", 5pts each)
      fg_missed_* (any distance, -1pt each)
      pat_made (1pt), pat_missed (-1pt)
    """
    r = player_stats_row
    points = 0.0

    fg_0_39 = r.get("fg_made_0_19", 0) + r.get("fg_made_20_29", 0) + r.get("fg_made_30_39", 0)
    points += fg_0_39 * 3
    points += r.get("fg_made_40_49", 0) * 4
    fg_50_plus = r.get("fg_made_50_59", 0) + r.get("fg_made_60_", 0)
    points += fg_50_plus * 5

    fg_missed_total = (
        r.get("fg_missed_0_19", 0) + r.get("fg_missed_20_29", 0)
        + r.get("fg_missed_30_39", 0) + r.get("fg_missed_40_49", 0)
        + r.get("fg_missed_50_59", 0) + r.get("fg_missed_60_", 0)
    )
    points += fg_missed_total * -1

    points += r.get("pat_made", 0) * 1
    points += r.get("pat_missed", 0) * -1

    return round(points, 2)


# ---------------------------------------------------------------------------
# 6. PROBABILITY / EDGE / QUALITY SCORING (mirrors rescore_quality_mu_row from MLB tool)
# ---------------------------------------------------------------------------

def rescore_quality_mu_row_nfl(mu: float, line: float, sigma: float) -> dict:
    """
    Given a mu (model projection), a line (book or user-entered), and an
    estimated sigma (variance - higher for tail-heavy props like longest
    rush / pass TD), returns p_over, p_under, and edge.
    Uses a normal approximation, consistent with the MLB tool's approach
    for continuous stats; swap in Poisson for count stats (TDs, receptions)
    the same way rescore_quality_mu_row() does for MLB counting stats.
    """
    from scipy.stats import norm
    if sigma <= 0 or np.isnan(mu) or np.isnan(line):
        return {"p_over": np.nan, "p_under": np.nan, "edge": np.nan}

    z = (line - mu) / sigma
    p_under = norm.cdf(z)
    p_over = 1 - p_under
    edge = abs(p_over - 0.5) * 2  # 0 = coinflip, 1 = max conviction, same shape as MLB edge
    return {"p_over": round(p_over, 3), "p_under": round(p_under, 3), "edge": round(edge, 3)}


def calc_quality_score(matchup_exploit_strength: float, sample_size_games: int,
                        coverage_confidence: float) -> float:
    """
    Placeholder quality_score formula, same 0-100 scale as MLB tool.
    matchup_exploit_strength: how much this specific offense/player profile
        beats this specific defense's tendency (e.g. high aDOT WR vs man-heavy defense)
    sample_size_games: games backing the mu (fewer games early season = lower confidence)
    coverage_confidence: how much of the play sample has charted coverage data
    """
    base = matchup_exploit_strength * 70
    sample_bonus = min(sample_size_games / 10, 1.0) * 20
    coverage_bonus = coverage_confidence * 10
    return round(min(base + sample_bonus + coverage_bonus, 100), 1)


def build_weekly_slate(season: int, week: int) -> pd.DataFrame:
    """
    Pulls and merges every data source needed for one week's slate, returning
    a single player-level DataFrame with mu inputs for every prop type ready
    to score. This does NOT include lines - lines are entered/adjusted
    manually per row in the Streamlit UI, same as the MLB tool's adjustable
    Best Edges table (avoids repeating the unreliable Underdog auto-pull
    issue; PrizePicks auto-pull decided against too - staying fully manual
    for lines).

    Returns columns including (not exhaustive):
      gsis_id, player_display_name, team, position, prop_type, mu, sigma
    """
    schedules_df = pull_schedules([season])
    rosters_df = pull_rosters([season])
    depth_charts_df = pull_depth_charts([season]) if nfl else pd.DataFrame()
    player_stats_df = pull_player_stats([season])
    ngs_pass_df = pull_ngs("passing", [season])
    ngs_rush_df = pull_ngs("rushing", [season])
    ngs_rec_df = pull_ngs("receiving", [season])
    pbp_df = pull_pbp([season])
    participation_df = pull_participation([season])
    ftn_df = pull_ftn_charting([season])

    coverage_profile = build_blended_coverage_profile(season, week)
    box_def_profile, box_off_profile = build_blended_box_profile(season, week)
    explosive_rates = build_explosive_rates(pbp_df)
    fallback_sigmas = build_league_fallback_sigmas(player_stats_df, season, week)
    fallback_mus = build_league_fallback_mus(player_stats_df, season, week)

    this_week_games = schedules_df[
        (schedules_df["season"] == season) & (schedules_df["week"] == week)
    ]
    teams_this_week = pd.concat([
        this_week_games["home_team"], this_week_games["away_team"]
    ]).unique().tolist()

    # Eligible players come from ROSTERS (who's on the team this week),
    # NOT from this week's own NGS/player_stats rows - those don't exist
    # yet for an upcoming week. This fixes the original bug where scanning
    # a future week returned zero rows.
    week_rosters = rosters_df[
        (rosters_df["season"] == season) & (rosters_df["team"].isin(teams_this_week))
    ]

    rows = []

    # --- Passing props ---
    qb_pool = week_rosters[week_rosters["position"] == "QB"]
    for _, qb in qb_pool.iterrows():
        gsis_id = qb.get("gsis_id")
        mu = calc_prop_mu(
            gsis_id, "passing_yards", player_stats_df, season, week,
            league_fallback_mu=fallback_mus.get(("QB", "passing_yards")),
        )
        sigma = calc_player_sigma(
            gsis_id, "passing_yards", player_stats_df, season, week,
            league_fallback_sigma=fallback_sigmas.get(("QB", "passing_yards")),
        )
        rows.append({
            "gsis_id": gsis_id, "player_display_name": qb.get("full_name"),
            "team": qb.get("team"), "position": "QB", "prop_type": "pass_yards",
            "mu": mu, "sigma": sigma,
        })

    # --- Rushing props ---
    rush_pool = week_rosters[week_rosters["position"].isin(["RB", "QB"])]
    for _, rb in rush_pool.iterrows():
        gsis_id = rb.get("gsis_id")
        position = rb.get("position")
        mu = calc_prop_mu(
            gsis_id, "rushing_yards", player_stats_df, season, week,
            league_fallback_mu=fallback_mus.get((position, "rushing_yards")),
        )
        sigma = calc_player_sigma(
            gsis_id, "rushing_yards", player_stats_df, season, week,
            league_fallback_sigma=fallback_sigmas.get((position, "rushing_yards")),
        )
        if pd.notna(mu):  # skip QBs/RBs with no real rushing history at all
            rows.append({
                "gsis_id": gsis_id, "player_display_name": rb.get("full_name"),
                "team": rb.get("team"), "position": position, "prop_type": "rush_yards",
                "mu": mu, "sigma": sigma,
            })

    # --- Receiving props ---
    rec_pool = week_rosters[week_rosters["position"].isin(["WR", "TE", "RB"])]
    for _, wr in rec_pool.iterrows():
        gsis_id = wr.get("gsis_id")
        position = wr.get("position")
        mu = calc_prop_mu(
            gsis_id, "receiving_yards", player_stats_df, season, week,
            league_fallback_mu=fallback_mus.get((position, "receiving_yards")),
        )
        sigma = calc_player_sigma(
            gsis_id, "receiving_yards", player_stats_df, season, week,
            league_fallback_sigma=fallback_sigmas.get((position, "receiving_yards")),
        )
        if pd.notna(mu):
            rows.append({
                "gsis_id": gsis_id, "player_display_name": wr.get("full_name"),
                "team": wr.get("team"), "position": position, "prop_type": "rec_yards",
                "mu": mu, "sigma": sigma,
            })

    # --- Fantasy points (offense: QB, RB, WR, TE) ---
    offense_positions = ["QB", "RB", "WR", "TE"]
    fantasy_pool_roster = week_rosters[week_rosters["position"].isin(offense_positions)]
    for _, pr in fantasy_pool_roster.iterrows():
        gsis_id = pr.get("gsis_id")
        recent_games = player_stats_df[
            (player_stats_df["gsis_id"] == gsis_id) & (player_stats_df["season"] == season)
            & (player_stats_df["week"] < week)
        ].sort_values("week", ascending=False).head(6)
        if len(recent_games) < 2:
            # bridge across season boundary for Week 1-2 of a new season
            prior_season_games = player_stats_df[
                (player_stats_df["gsis_id"] == gsis_id) & (player_stats_df["season"] == season - 1)
            ].sort_values("week", ascending=False).head(6)
            recent_games = pd.concat([recent_games, prior_season_games])
        if recent_games.empty:
            continue
        fantasy_pts_per_game = recent_games.apply(
            lambda r: calc_offense_fantasy_points(r.to_dict()), axis=1
        )
        mu_fantasy = round(fantasy_pts_per_game.mean(), 2)
        sigma = round(fantasy_pts_per_game.std(ddof=1), 2) if len(fantasy_pts_per_game) >= 2 else np.nan
        rows.append({
            "gsis_id": gsis_id, "player_display_name": pr.get("full_name"),
            "team": pr.get("team"), "position": pr.get("position"), "prop_type": "fantasy_points",
            "mu": mu_fantasy, "sigma": sigma,
        })

    # --- Kicker fantasy + FG/XP props ---
    kicker_pool = week_rosters[week_rosters["position"] == "K"]
    for _, kr in kicker_pool.iterrows():
        gsis_id = kr.get("gsis_id")
        recent_games = player_stats_df[
            (player_stats_df["gsis_id"] == gsis_id) & (player_stats_df["season"] == season)
            & (player_stats_df["week"] < week)
        ].sort_values("week", ascending=False).head(6)
        if len(recent_games) < 2:
            prior_season_games = player_stats_df[
                (player_stats_df["gsis_id"] == gsis_id) & (player_stats_df["season"] == season - 1)
            ].sort_values("week", ascending=False).head(6)
            recent_games = pd.concat([recent_games, prior_season_games])
        if recent_games.empty:
            continue
        kicker_pts_per_game = recent_games.apply(
            lambda r: calc_kicker_fantasy_points(r.to_dict()), axis=1
        )
        mu_kicker = round(kicker_pts_per_game.mean(), 2)
        sigma = round(kicker_pts_per_game.std(ddof=1), 2) if len(kicker_pts_per_game) >= 2 else np.nan
        rows.append({
            "gsis_id": gsis_id, "player_display_name": kr.get("full_name"),
            "team": kr.get("team"), "position": "K", "prop_type": "kicker_fantasy",
            "mu": mu_kicker, "sigma": sigma,
        })

    return pd.DataFrame(rows)


def calc_prop_mu(player_gsis_id: str, prop_column: str, player_stats_df: pd.DataFrame,
                  season: int, current_week: int, lookback_games: int = 6,
                  min_games: int = 2, league_fallback_mu: float = None) -> float:
    """
    Computes mu as the average of a player's own recent real games for a
    given stat column (e.g. "passing_yards", "rushing_yards",
    "receiving_yards"), using player_stats history from weeks BEFORE
    current_week only.

    CROSS-SEASON FIX: for Week 1 of a new season (e.g. season=2026, week=1),
    there is ZERO history within that season yet - player_stats for the new
    season doesn't exist until games are actually played. Without this fix,
    every Week 1 row would come back NaN. This now falls back to the end of
    the PRIOR season's games when the current season doesn't have enough
    history, so Week 1 uses last season's most recent form as a starting
    point - same bridging idea as blend_scheme_baseline(), applied here
    directly rather than left unused.

    THIS ALSO FIXES A REAL BUG from an earlier version: the original scanner
    filtered NGS data by week == target_week to find both the list of
    eligible players AND their stat inputs, which only works retroactively
    (scanning an already-played week) and would also be data leakage (using
    the target week's own result as an input).

    Returns NaN if there's no usable history in either season and no
    league_fallback_mu is provided - the row should be flagged low-confidence
    in the UI rather than scored with a guessed mu.
    """
    current_season_history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season)
        & (player_stats_df["week"] < current_week)
    ].sort_values("week", ascending=False).head(lookback_games)

    if len(current_season_history) >= min_games:
        return round(current_season_history[prop_column].mean(), 2)

    # Not enough current-season games (e.g. Week 1-2, or right after a trade) -
    # bridge with the end of the prior season instead of returning NaN outright.
    prior_season_history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season - 1)
    ].sort_values("week", ascending=False).head(lookback_games)

    combined = pd.concat([current_season_history, prior_season_history])
    if len(combined) >= min_games:
        return round(combined[prop_column].mean(), 2)

    return league_fallback_mu if league_fallback_mu is not None else np.nan


def build_league_fallback_mus(player_stats_df: pd.DataFrame, season: int,
                               through_week: int) -> dict:
    """
    Position-level average mu fallback (e.g. "what does an average starting
    RB rush for per game this season") for players without enough of their
    own history yet (rookies, recent trades, Week 1-2). Same structure as
    build_league_fallback_sigmas().
    """
    prop_by_position = {
        "QB": ["passing_yards", "rushing_yards"],
        "RB": ["rushing_yards", "receiving_yards"],
        "WR": ["receiving_yards", "rushing_yards"],
        "TE": ["receiving_yards"],
    }
    df = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < through_week)
    ]
    fallback = {}
    for position, columns in prop_by_position.items():
        pos_df = df[df["position"] == position]
        for col in columns:
            per_player_avg = (
                pos_df.groupby("gsis_id")[col]
                .agg(["mean", "count"])
                .query("count >= 2")
            )
            if not per_player_avg.empty:
                fallback[(position, col)] = round(per_player_avg["mean"].mean(), 2)
    return fallback


# ---------------------------------------------------------------------------
# 6b. SIGMA (VARIANCE) ESTIMATION PER PROP TYPE
# ---------------------------------------------------------------------------

def calc_player_sigma(player_gsis_id: str, prop_column: str, player_stats_df: pd.DataFrame,
                       season: int, current_week: int, lookback_games: int = 8,
                       min_games: int = 3, league_fallback_sigma: float = None) -> float:
    """
    Computes a player's own game-to-game standard deviation for a given prop
    column (e.g. "rushing_yards", "receiving_yards", "passing_yards") using
    their real weekly history from player_stats, up to `lookback_games` most
    recent games before current_week.

    CROSS-SEASON FIX (same as calc_prop_mu): falls back to the end of the
    prior season's games when the current season doesn't have enough history
    yet (Week 1-2 of a new season) rather than returning NaN immediately.

    This is the missing piece rescore_quality_mu_row_nfl() needs - without
    a real sigma, mu can't be turned into p_over/edge.

    Returns league_fallback_sigma if there's no usable history in either
    season - otherwise NaN, and the row should be flagged as low-confidence
    in the UI rather than scored with a guessed sigma.
    """
    current_season_history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season)
        & (player_stats_df["week"] < current_week)
    ].sort_values("week", ascending=False).head(lookback_games)

    if len(current_season_history) >= min_games:
        return round(current_season_history[prop_column].std(ddof=1), 3)

    prior_season_history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season - 1)
    ].sort_values("week", ascending=False).head(lookback_games)

    combined = pd.concat([current_season_history, prior_season_history])
    if len(combined) >= min_games:
        return round(combined[prop_column].std(ddof=1), 3)

    return league_fallback_sigma if league_fallback_sigma is not None else np.nan


def build_league_fallback_sigmas(player_stats_df: pd.DataFrame, season: int,
                                  through_week: int) -> dict:
    """
    Builds position-level fallback sigma values (e.g. "what's the typical
    game-to-game std dev for an average starting RB's rushing_yards this
    season") to use when an individual player doesn't have enough history
    yet (rookies, recent trades, Week 1-2). Computed as the average
    within-player std dev across all players at that position with enough
    games, NOT the spread across different players (that would conflate
    variance between players with variance within one player's games).

    Returns a dict like:
      {
        ("RB", "rushing_yards"): 22.4,
        ("WR", "receiving_yards"): 19.1,
        ("QB", "passing_yards"): 48.7,
        ...
      }
    """
    prop_by_position = {
        "QB": ["passing_yards", "rushing_yards"],
        "RB": ["rushing_yards", "receiving_yards"],
        "WR": ["receiving_yards", "rushing_yards"],
        "TE": ["receiving_yards"],
    }

    df = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < through_week)
    ]

    fallback = {}
    for position, columns in prop_by_position.items():
        pos_df = df[df["position"] == position]
        for col in columns:
            per_player_std = (
                pos_df.groupby("gsis_id")[col]
                .agg(["std", "count"])
                .query("count >= 3")  # only players with enough games to get a real std
            )
            if not per_player_std.empty:
                fallback[(position, col)] = round(per_player_std["std"].mean(), 3)

    return fallback


# ---------------------------------------------------------------------------
# 7. FULL SLATE SCAN (mirrors scan_full_slate_quality_mu from MLB tool)
# ---------------------------------------------------------------------------

def scan_full_slate_nfl(season: int, week: int) -> pd.DataFrame:
    """
    Weekly full-slate scanner. Builds the slate (see build_weekly_slate),
    but does NOT auto-fill lines or compute edge/p_over - those are added
    in the Streamlit UI via an adjustable "line" column per row, same as
    the MLB tool's adjustable Best Edges table. quality_score and mu
    components are pre-computed here; edge/p_over recompute live in the UI
    whenever the user edits a line.
    """
    slate_df = build_weekly_slate(season, week)
    slate_df["line"] = np.nan  # user fills this in per row in the UI
    slate_df["p_over"] = np.nan
    slate_df["edge"] = np.nan
    return slate_df


# ---------------------------------------------------------------------------
# 8. BACKTEST MODE - compare projected mu against actual results for a
#    completed week (no real lines needed - tests mu accuracy directly)
# ---------------------------------------------------------------------------

def backtest_week(season: int, week: int) -> pd.DataFrame:
    """
    Runs the scanner for a week that's already been played, then joins in
    each player's REAL result for that week, so you can compare mu (what
    the model projected using only prior weeks) against what actually
    happened - no betting line needed for this.

    Only meaningful for a week where player_stats already has real results
    (i.e. week has been played). Running this on a genuinely upcoming week
    will just show NaN in the actual/miss columns since the result doesn't
    exist yet.

    Returns the same columns as scan_full_slate_nfl(), plus:
      actual: the player's real stat for that prop_type in that week
      miss: mu - actual (positive = model overprojected, negative = underprojected)
      abs_miss: absolute value of miss, for sorting worst-to-best
    """
    slate_df = build_weekly_slate(season, week)
    player_stats_df = pull_player_stats([season])

    actual_week = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] == week)
    ].set_index("gsis_id")

    prop_to_stat_column = {
        "pass_yards": "passing_yards",
        "rush_yards": "rushing_yards",
        "rec_yards": "receiving_yards",
        "fantasy_points": "fantasy_points_ppr",
    }

    def _lookup_actual(row):
        prop_type = row["prop_type"]
        gsis_id = row["gsis_id"]
        if gsis_id not in actual_week.index:
            return np.nan
        if prop_type == "kicker_fantasy":
            return calc_kicker_fantasy_points(actual_week.loc[gsis_id].to_dict())
        stat_col = prop_to_stat_column.get(prop_type)
        if stat_col is None:
            return np.nan
        val = actual_week.loc[gsis_id]
        if isinstance(val, pd.DataFrame):  # duplicate index safety
            val = val.iloc[0]
        return val.get(stat_col, np.nan)

    slate_df["actual"] = slate_df.apply(_lookup_actual, axis=1)
    slate_df["miss"] = slate_df["mu"] - slate_df["actual"]
    slate_df["abs_miss"] = slate_df["miss"].abs()

    return slate_df.drop(columns=["line", "p_over", "edge"], errors="ignore")
