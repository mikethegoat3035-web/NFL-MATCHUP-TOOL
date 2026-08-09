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


def build_box_count_profile(ftn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates n_defense_box into a per-team stacked-box rate.
    Returns avg box count and % of plays with 7+ / 8+ defenders in the box,
    split by defteam (and separately, offense's box counts faced, by posteam).
    """
    df = ftn_df.dropna(subset=["n_defense_box"]).copy()

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
        df.groupby("posteam")
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

    NOTE: expects tendency_df to have a "team_season_key" column - this needs
    to be built once we construct a season-level tendency table (not yet
    implemented as of this version).
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
        rosters, depth_charts, and NGS data. player_stats_df is renamed to
        gsis_id at pull time (see pull_player_stats), so this join works too.
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


# ---------------------------------------------------------------------------
# 7. FULL SLATE SCAN (mirrors scan_full_slate_quality_mu from MLB tool)
# ---------------------------------------------------------------------------

def scan_full_slate_nfl(week: int, season: int) -> pd.DataFrame:
    """
    Placeholder for the weekly full-slate scanner. NFL has far fewer
    confirmed players/games per week than MLB (1 game per team per week,
    ~14-16 games max vs MLB's larger nightly slate), so this should be
    lighter-weight than scan_full_slate_quality_mu() while following the
    same shape: pull data -> compute mu per prop per player -> merge live
    lines -> score edge/p_over/quality -> return one combined DataFrame.
    """
    raise NotImplementedError("Wire this up once data pulls are confirmed live.")
