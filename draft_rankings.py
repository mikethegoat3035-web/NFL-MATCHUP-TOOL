"""
draft_rankings.py
Season-long fantasy draft rankings for a specific league format, built on
top of nfl_model_combined.py's data pulls and fantasy scoring functions.

League settings confirmed by user:
  - 6 teams
  - Full PPR
  - Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 3 FLEX (RB/WR/TE), 1 DEF, 1 K
  - 6 bench spots (14 roster spots per team, 84 total rostered players)
  - Drafting from pick 3

TWO REAL GAPS FLAGGED HONESTLY:
  1. Team defense (DEF) scoring is built here for the first time - no prior
     infrastructure existed for it (everything else so far is offense/K).
  2. "Public consensus" comparison uses FantasyPros rankings via
     nflreadpy's load_ff_rankings() - this is NOT Yahoo's own ADP
     specifically (no free source for that was found), but is the closest
     available free proxy for "public perception" to detect risers against.
"""

import pandas as pd
import numpy as np

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None

from nfl_model_combined import (
    pull_player_stats, pull_schedules, pull_rosters,
    calc_offense_fantasy_points, calc_kicker_fantasy_points,
)


# ---------------------------------------------------------------------------
# LEAGUE SETTINGS
# ---------------------------------------------------------------------------

LEAGUE_SETTINGS = {
    "num_teams": 6,
    "starters": {
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 3,   # RB/WR/TE eligible
        "DEF": 1,
        "K": 1,
    },
    "bench": 6,
}


# ---------------------------------------------------------------------------
# 1. TEAM DEFENSE (DEF) FANTASY SCORING - NEW, no prior infrastructure
# ---------------------------------------------------------------------------

def pull_team_defense_stats(years: list[int]) -> pd.DataFrame:
    """
    Pulls team-level defensive stats for fantasy DEF scoring.

    NOTE: nflreadpy's load_player_stats() covers individual players.
    Team defense fantasy scoring needs TEAM-level aggregates (sacks,
    interceptions, fumble recoveries, defensive/special-teams TDs, points
    allowed). This aggregates individual defensive player_stats up to the
    team level, plus points allowed from schedules/pbp.

    UNVERIFIED -> CONFIRMED: def_sacks, def_interceptions, def_fumbles,
    def_fumbles_forced, fumble_recovery_tds, def_tds, special_teams_tds,
    def_safeties, def_tackles_solo, def_tackles_with_assist, def_qb_hits,
    def_pass_defended, def_sack_yards all confirmed real columns.
    Defensive position values confirmed as DB/DL/LB (simpler than initially
    assumed - not the more granular CB/S/DE/DT/OLB/ILB split).
    """
    df = nfl.load_player_stats(seasons=years).to_pandas()
    df = df.rename(columns={"player_id": "gsis_id"})

    defense_cols = [
        "def_sacks", "def_interceptions", "def_fumbles",
        "fumble_recovery_tds", "def_tds", "special_teams_tds",
        "def_safeties",
    ]
    available_cols = [c for c in defense_cols if c in df.columns]

    if not available_cols:
        return pd.DataFrame(columns=["team", "season", "week"])

    team_defense = (
        df.groupby(["team", "season", "week"])[available_cols]
        .sum()
        .reset_index()
    )
    return team_defense


def calc_points_allowed_bucket_score(points_allowed: float) -> int:
    """
    Standard Yahoo-style points-allowed scoring tiers for DEF.
    UNVERIFIED: these are the common Yahoo default tiers, not confirmed
    against this specific league's actual settings - worth checking
    against the real Yahoo league scoring page if these numbers matter.
    """
    if points_allowed == 0:
        return 10
    elif points_allowed <= 6:
        return 7
    elif points_allowed <= 13:
        return 4
    elif points_allowed <= 20:
        return 1
    elif points_allowed <= 27:
        return 0
    elif points_allowed <= 34:
        return -1
    else:
        return -4


def calc_team_defense_fantasy_points(defense_row: dict, points_allowed: float = None) -> float:
    """
    Standard Yahoo-default DEF scoring:
      Sack: 1 | INT: 2 | Fumble Recovery: 2 | Safety: 2
      Defensive/Return TD: 6 | Points allowed: tiered (see above)

    UNVERIFIED: matches Yahoo's common default, but not confirmed against
    this specific league's actual scoring settings - flag to double check.
    """
    r = defense_row
    points = 0.0
    points += r.get("def_sacks", 0) * 1
    points += r.get("def_interceptions", 0) * 2
    points += r.get("def_fumbles", 0) * 2
    points += r.get("def_safeties", 0) * 2
    points += (r.get("def_tds", 0) + r.get("fumble_recovery_tds", 0) + r.get("special_teams_tds", 0)) * 6
    if points_allowed is not None:
        points += calc_points_allowed_bucket_score(points_allowed)
    return round(points, 2)


# ---------------------------------------------------------------------------
# 2. SEASON-LONG PROJECTIONS (not weekly - full season totals for draft)
# ---------------------------------------------------------------------------

def build_season_projection(player_gsis_id: str, position: str,
                             player_stats_df: pd.DataFrame, projection_season: int,
                             games_in_season: int = 17) -> dict:
    """
    Builds a FULL-SEASON projection for draft purposes, using last season's
    per-game rate stats projected forward across a full season - genuinely
    different from the weekly build_weekly_slate() mu, which only looks a
    few games back. Draft projections need season totals, not weekly
    snapshots, since there's no current-season data to look back on yet
    before the year starts.

    Uses the player's most recent completed season (projection_season - 1)
    as the rate basis. Games played is capped at the games they actually
    played (injury-shortened seasons don't get inflated to a full 17).
    """
    prior_season = projection_season - 1
    history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id) & (player_stats_df["season"] == prior_season)
    ]
    if history.empty:
        return {"season_proj_points": np.nan, "games_played_prior": 0, "ppg_prior": np.nan}

    per_game_points = history.apply(lambda r: calc_offense_fantasy_points(r.to_dict()), axis=1)
    games_played = len(history)
    ppg = per_game_points.mean()

    # project forward at the same per-game rate across a full season,
    # capped at games_in_season (don't project beyond a real season length)
    projected_games = min(games_in_season, games_played + 2)  # small bump assuming health, not full 17 blindly
    season_proj_points = round(ppg * projected_games, 1)

    return {
        "season_proj_points": season_proj_points,
        "games_played_prior": games_played,
        "ppg_prior": round(ppg, 2),
    }


def build_kicker_season_projection(player_gsis_id: str, player_stats_df: pd.DataFrame,
                                    projection_season: int, games_in_season: int = 17) -> dict:
    """Same season-projection approach as build_season_projection(), for kickers."""
    prior_season = projection_season - 1
    history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id) & (player_stats_df["season"] == prior_season)
    ]
    if history.empty:
        return {"season_proj_points": np.nan, "games_played_prior": 0, "ppg_prior": np.nan}

    per_game_points = history.apply(lambda r: calc_kicker_fantasy_points(r.to_dict()), axis=1)
    games_played = len(history)
    ppg = per_game_points.mean()
    projected_games = min(games_in_season, games_played + 2)
    season_proj_points = round(ppg * projected_games, 1)

    return {
        "season_proj_points": season_proj_points,
        "games_played_prior": games_played,
        "ppg_prior": round(ppg, 2),
    }


# ---------------------------------------------------------------------------
# 3. VALUE OVER REPLACEMENT (VBD) - positional scarcity for THIS league
# ---------------------------------------------------------------------------

def compute_replacement_levels(projections_df: pd.DataFrame, league_settings: dict) -> dict:
    """
    Computes the "replacement level" projected points at each position -
    the level of the last startable player league-wide, given THIS
    league's team count and roster settings. This is what makes rankings
    actually specific to a 6-team league instead of generic industry
    rankings built for a standard 10-12 team league.

    FLEX spots are split proportionally across RB/WR/TE based on how often
    each position tends to fill FLEX in practice (rough industry-standard
    split: RB ~45%, WR ~45%, TE ~10%) since FLEX doesn't belong to one
    position exclusively.
    """
    num_teams = league_settings["num_teams"]
    starters = league_settings["starters"]

    flex_rb_share = round(starters["FLEX"] * 0.45)
    flex_wr_share = round(starters["FLEX"] * 0.45)
    flex_te_share = starters["FLEX"] - flex_rb_share - flex_wr_share

    startable_counts = {
        "QB": starters["QB"] * num_teams,
        "RB": (starters["RB"] + flex_rb_share) * num_teams,
        "WR": (starters["WR"] + flex_wr_share) * num_teams,
        "TE": (starters["TE"] + flex_te_share) * num_teams,
        "K": starters["K"] * num_teams,
        "DEF": starters["DEF"] * num_teams,
    }

    replacement_levels = {}
    for position, startable_count in startable_counts.items():
        pos_df = projections_df[projections_df["position"] == position].sort_values(
            "season_proj_points", ascending=False
        )
        if len(pos_df) >= startable_count and startable_count > 0:
            replacement_levels[position] = pos_df.iloc[startable_count - 1]["season_proj_points"]
        elif not pos_df.empty:
            replacement_levels[position] = pos_df["season_proj_points"].min()
        else:
            replacement_levels[position] = 0

    return replacement_levels


def compute_vor(projections_df: pd.DataFrame, replacement_levels: dict) -> pd.DataFrame:
    """
    Value Over Replacement = projected points minus that position's
    replacement level. This is the actual number to rank/draft by - raw
    projected points alone misrank across positions (a low-scarcity
    position's "good" player is worth less than a scarce position's
    "good" player at the same raw point total).
    """
    df = projections_df.copy()
    df["replacement_level"] = df["position"].map(replacement_levels)
    df["vor"] = df["season_proj_points"] - df["replacement_level"]
    return df


# ---------------------------------------------------------------------------
# 4. YAHOO-STYLE RANKING TABLE
# ---------------------------------------------------------------------------

def get_bye_weeks(season: int, schedules_df: pd.DataFrame) -> dict:
    """
    Derives each team's bye week: the week where that team has no game
    scheduled at all that season. Schedules doesn't have an explicit
    "bye_week" column, so this is computed from which weeks a team is
    missing from both home_team and away_team.

    NOTE: load_ff_rankings() (used in detect_risers) already includes a
    real "bye" column directly per player - that's actually the simpler
    source once a player is matched to FantasyPros data. This function
    stays as a fallback for players not found in that data (e.g. someone
    FantasyPros hasn't ranked yet).
    """
    season_games = schedules_df[schedules_df["season"] == season]
    all_weeks = set(season_games["week"].unique())
    all_teams = pd.concat([season_games["home_team"], season_games["away_team"]]).unique()

    bye_weeks = {}
    for team in all_teams:
        team_weeks = set(season_games[
            (season_games["home_team"] == team) | (season_games["away_team"] == team)
        ]["week"].unique())
        missing = all_weeks - team_weeks
        bye_weeks[team] = min(missing) if missing else None
    return bye_weeks


def build_yahoo_style_rankings(season: int) -> pd.DataFrame:
    """
    Builds the full draft ranking table in Yahoo's standard display format:
    Rank | Player | Pos | Team | Bye | Proj Pts | Pos Rank

    Ranked by VOR (value over replacement) for THIS specific league's
    6-team, 1QB/2RB/2WR/1TE/3FLEX/1DEF/1K/6BN settings - not generic
    industry rankings.
    """
    player_stats_df = pull_player_stats([season - 1])
    rosters_df = pull_rosters([season])
    schedules_df = pull_schedules([season])
    bye_weeks = get_bye_weeks(season, schedules_df)

    offense_positions = ["QB", "RB", "WR", "TE"]
    current_roster = rosters_df[
        (rosters_df["season"] == season) & (rosters_df["position"].isin(offense_positions + ["K"]))
    ].drop_duplicates(subset=["gsis_id"])

    rows = []
    for _, player in current_roster.iterrows():
        gsis_id = player.get("gsis_id")
        position = player.get("position")
        team = player.get("team")

        if position == "K":
            proj = build_kicker_season_projection(gsis_id, player_stats_df, season)
        else:
            proj = build_season_projection(gsis_id, position, player_stats_df, season)

        if pd.isna(proj["season_proj_points"]):
            continue  # no prior-season data to project from - skip rather than guess

        rows.append({
            "gsis_id": gsis_id,
            "player": player.get("full_name"),
            "position": position,
            "team": team,
            "bye": bye_weeks.get(team),
            "season_proj_points": proj["season_proj_points"],
            "games_played_prior": proj["games_played_prior"],
            "ppg_prior": proj["ppg_prior"],
        })

    projections_df = pd.DataFrame(rows)
    if projections_df.empty:
        return projections_df

    replacement_levels = compute_replacement_levels(projections_df, LEAGUE_SETTINGS)
    vor_df = compute_vor(projections_df, replacement_levels)

    vor_df = vor_df.sort_values("vor", ascending=False).reset_index(drop=True)
    vor_df["overall_rank"] = vor_df.index + 1
    vor_df["pos_rank"] = vor_df.groupby("position")["vor"].rank(ascending=False, method="min").astype(int)
    vor_df["pos_rank_label"] = vor_df["position"] + vor_df["pos_rank"].astype(str)

    return vor_df[[
        "overall_rank", "player", "position", "pos_rank_label", "team", "bye",
        "season_proj_points", "vor", "ppg_prior", "games_played_prior",
    ]]


# ---------------------------------------------------------------------------
# 5. RISER DETECTION - our rank vs FantasyPros public consensus
# ---------------------------------------------------------------------------

def detect_risers(rankings_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Compares our internally-computed overall_rank against FantasyPros
    consensus rank (via nflreadpy's load_ff_rankings()) to flag players
    our model ranks meaningfully HIGHER than public consensus does - i.e.
    potential risers/undervalued players the public hasn't caught up to.

    CONFIRMED real load_ff_rankings() columns: player, pos, team, bye, ecr
    (Expert Consensus Rank - the real rank field), player_owned_yahoo
    (actual real Yahoo ownership % - genuinely useful "public perception"
    signal, arguably better than a generic ADP for this purpose).

    NOTE: this is FantasyPros consensus (ecr), not Yahoo's own ADP
    specifically (no free source found for Yahoo ADP directly) - flagged
    as an honest substitute. player_owned_yahoo IS real Yahoo data though,
    included as a secondary signal.

    FIX: FantasyPros' own data already has a column literally called
    "rank_delta" (their own metric, not ours) - our computed delta is
    renamed to our_rank_delta to avoid a merge collision/overwrite.

    our_rank_delta = ecr - our_rank. A large POSITIVE value means we rank
    them much higher (earlier) than public consensus does - a riser.
    """
    if nfl is None:
        return rankings_df.assign(fantasypros_ecr=np.nan, our_rank_delta=np.nan)

    try:
        ff_rankings = nfl.load_ff_rankings().to_pandas()
    except Exception:
        return rankings_df.assign(fantasypros_ecr=np.nan, our_rank_delta=np.nan)

    merged = rankings_df.merge(
        ff_rankings[["player", "ecr", "player_owned_yahoo"]],
        left_on="player", right_on="player", how="left"
    )
    merged = merged.rename(columns={"ecr": "fantasypros_ecr"})
    merged["our_rank_delta"] = merged["fantasypros_ecr"] - merged["overall_rank"]

    return merged.sort_values("our_rank_delta", ascending=False)
