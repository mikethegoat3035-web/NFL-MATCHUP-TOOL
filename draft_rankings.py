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
# LEAGUE SETTINGS - now fully adjustable, not a fixed constant
# ---------------------------------------------------------------------------

def build_league_settings(num_teams: int = 6, draft_position: int = 3,
                           qb: int = 1, rb: int = 2, wr: int = 2, te: int = 1,
                           flex: int = 3, def_: int = 1, k: int = 1, bench: int = 6,
                           ppr_value: float = 1.0) -> dict:
    """
    Builds a league settings dict from adjustable parameters - replaces the
    old hardcoded LEAGUE_SETTINGS constant so every draft-relevant setting
    (roster composition, team count, PPR value, draft slot) can be changed
    per-league from the UI instead of being fixed to one specific league.
    """
    return {
        "num_teams": num_teams,
        "draft_position": draft_position,
        "starters": {"QB": qb, "RB": rb, "WR": wr, "TE": te, "FLEX": flex, "DEF": def_, "K": k},
        "bench": bench,
        "ppr_value": ppr_value,
    }


# Default settings matching the league originally described - kept as a
# fallback / example, but build_yahoo_style_rankings() now takes settings
# as a parameter rather than always using this fixed dict.
LEAGUE_SETTINGS = build_league_settings()


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
    CONFIRMED against user's actual Yahoo league scoring settings (screenshots
    checked): Sack 1, INT 2, Fumble Recovery 2, TD 6, Safety 2, Block Kick 2,
    Kickoff/Punt Return TD 6, Points Allowed tiers 10/7/4/1/0/-1/-4 for
    0/1-6/7-13/14-20/21-27/28-34/35+ - all confirmed exact matches to
    real league settings except Block Kick, which is now added.

    HONEST FLAG: block_kicks is NOT a confirmed real column in player_stats -
    our earlier column diagnostic never checked for a blocked-kick stat
    specifically. It may not exist as a pre-aggregated column at all (blocked
    kicks might only be derivable from pbp's field_goal_result/punt_blocked
    play-level flags, which would need a separate aggregation function). This
    uses .get() with a default of 0, so it silently contributes nothing if
    the column doesn't exist rather than crashing - but that also means
    block kicks may not actually be counted until this is verified/built out.
    """
    r = defense_row
    points = 0.0
    points += r.get("def_sacks", 0) * 1
    points += r.get("def_interceptions", 0) * 2
    points += r.get("def_fumbles", 0) * 2
    points += r.get("def_safeties", 0) * 2
    points += (r.get("def_tds", 0) + r.get("fumble_recovery_tds", 0) + r.get("special_teams_tds", 0)) * 6
    points += r.get("block_kicks", 0) * 2  # UNVERIFIED column - see docstring
    if points_allowed is not None:
        points += calc_points_allowed_bucket_score(points_allowed)
    return round(points, 2)


# ---------------------------------------------------------------------------
# 2. SEASON-LONG PROJECTIONS (not weekly - full season totals for draft)
# ---------------------------------------------------------------------------

def build_season_projection(player_gsis_id: str, position: str,
                             player_stats_df: pd.DataFrame, projection_season: int,
                             games_in_season: int = 17, ppr_value: float = 1.0) -> dict:
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
    ppr_value is now passed through to scoring so PPR/half-PPR/standard
    leagues get correctly different projections, not just full PPR always.
    """
    prior_season = projection_season - 1
    history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id) & (player_stats_df["season"] == prior_season)
        & (player_stats_df["season_type"] == "REG")
    ]
    if history.empty:
        return {"season_proj_points": np.nan, "games_played_prior": 0, "ppg_prior": np.nan}

    per_game_points = history.apply(lambda r: calc_offense_fantasy_points(r.to_dict(), ppr_value), axis=1)
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
        & (player_stats_df["season_type"] == "REG")
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


def build_defense_season_projection(team: str, player_stats_df: pd.DataFrame, schedules_df: pd.DataFrame,
                                     projection_season: int, games_in_season: int = 17) -> dict:
    """
    Season-long DEF projection - was completely missing from the rankings
    pipeline before (calc_team_defense_fantasy_points existed but nothing
    ever called it in build_yahoo_style_rankings, so zero defenses showed
    up in rankings at all despite the league requiring 1 DEF starter).

    Points allowed per game is derived from schedules_df (the opponent's
    final score that game), since player_stats doesn't carry a points-
    allowed field directly.
    """
    prior_season = projection_season - 1
    team_defense_stats = pull_team_defense_stats([prior_season])
    team_games = team_defense_stats[
        (team_defense_stats["team"] == team) & (team_defense_stats["season"] == prior_season)
    ]
    if team_games.empty:
        return {"season_proj_points": np.nan, "games_played_prior": 0, "ppg_prior": np.nan}

    season_games = schedules_df[schedules_df["season"] == prior_season]

    def _points_allowed(week):
        game = season_games[
            ((season_games["home_team"] == team) | (season_games["away_team"] == team))
            & (season_games["week"] == week)
        ]
        if game.empty:
            return None
        g = game.iloc[0]
        return g["away_score"] if g["home_team"] == team else g["home_score"]

    weekly_points = []
    for _, row in team_games.iterrows():
        pa = _points_allowed(row["week"])
        weekly_points.append(calc_team_defense_fantasy_points(row.to_dict(), pa))

    games_played = len(weekly_points)
    ppg = sum(weekly_points) / games_played if games_played else np.nan
    projected_games = min(games_in_season, games_played + 2)
    season_proj_points = round(ppg * projected_games, 1) if pd.notna(ppg) else np.nan

    return {
        "season_proj_points": season_proj_points,
        "games_played_prior": games_played,
        "ppg_prior": round(ppg, 2) if pd.notna(ppg) else np.nan,
    }


def pull_draft_picks(years: list[int]) -> pd.DataFrame:
    """
    Pulls NFL draft class data (round, pick, position, gsis_id) for the
    given draft years.

    UNVERIFIED: nflreadpy's load_draft_picks() function and its exact
    column names haven't been confirmed against real live output yet -
    this follows the standard nflverse naming convention (round, pick,
    gsis_id, position, season = draft year), same as every other function
    in this build that needs live verification before fully trusting it.
    """
    if nfl is None:
        return pd.DataFrame()
    try:
        df = nfl.load_draft_picks(seasons=years).to_pandas()
        if "player_id" in df.columns and "gsis_id" not in df.columns:
            df = df.rename(columns={"player_id": "gsis_id"})
        return df
    except Exception:
        return pd.DataFrame()


def build_rookie_production_curve(position: str, current_season: int, player_stats_df: pd.DataFrame,
                                   draft_picks_df: pd.DataFrame, ppr_value: float = 1.0,
                                   lookback_years: int = 5) -> dict:
    """
    THE ROOKIE FIX: since a rookie has zero prior-NFL-season stats,
    build_season_projection() would return NaN for them and they'd be
    silently skipped from rankings entirely - meaning every incoming
    rookie class was completely absent, a real gap for an actual draft.

    Instead of guessing, this uses REAL historical data: for the past
    `lookback_years` draft classes at this position, what did players
    drafted in each round actually score as rookies (real fantasy points,
    real games)? This builds a genuine draft-capital-to-production curve
    from real outcomes, not a fabricated estimate.

    Returns {round_number: avg_rookie_season_points}. Round 8 = undrafted
    (treated as replacement-level/waiver-wire value).
    """
    curve = {}
    for round_num in range(1, 8):
        round_players = []
        for draft_year in range(current_season - lookback_years, current_season):
            picks_this_round = draft_picks_df[
                (draft_picks_df["season"] == draft_year) & (draft_picks_df["round"] == round_num)
                & (draft_picks_df["position"] == position)
            ]
            for gsis_id in picks_this_round["gsis_id"].dropna().unique():
                rookie_year_stats = player_stats_df[
                    (player_stats_df["gsis_id"] == gsis_id) & (player_stats_df["season"] == draft_year)
                    & (player_stats_df["season_type"] == "REG")
                ]
                if not rookie_year_stats.empty:
                    pts = rookie_year_stats.apply(
                        lambda r: calc_offense_fantasy_points(r.to_dict(), ppr_value), axis=1
                    ).sum()
                    round_players.append(pts)
        curve[round_num] = round(sum(round_players) / len(round_players), 1) if round_players else np.nan

    return curve


def build_rookie_season_projection(player_gsis_id: str, position: str, current_season: int,
                                    draft_picks_df: pd.DataFrame, production_curve: dict) -> dict:
    """
    Projects an incoming rookie using the draft-capital production curve
    (build_rookie_production_curve) rather than skipping them entirely.
    Clearly a rougher proxy than stats-based projections for veterans -
    draft capital predicts opportunity/talent evaluation reasonably well
    on average, but obviously can't capture individual situation/fit the
    way real stats can once a player has actually played.
    """
    pick_row = draft_picks_df[
        (draft_picks_df["season"] == current_season) & (draft_picks_df["gsis_id"] == player_gsis_id)
    ]
    if pick_row.empty:
        return {"season_proj_points": np.nan, "draft_round": None, "is_rookie_projection": False}

    draft_round = pick_row.iloc[0].get("round")
    proj_points = production_curve.get(draft_round, np.nan)

    return {
        "season_proj_points": proj_points,
        "draft_round": draft_round,
        "is_rookie_projection": True,
    }


# ---------------------------------------------------------------------------
# 5. YAHOO-STYLE RANKING TABLE
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


def build_yahoo_style_rankings(season: int, league_settings: dict = None) -> pd.DataFrame:
    """
    Builds the full draft ranking table in Yahoo's standard display format:
    Rank | Player | Pos | Team | Bye | Proj Pts | Pos Rank

    Ranked by VOR (value over replacement) for the GIVEN league_settings -
    fully adjustable now (team count, roster composition, PPR value) rather
    than fixed to one hardcoded league. Falls back to the default 6-team
    example settings if none provided.
    """
    if league_settings is None:
        league_settings = LEAGUE_SETTINGS
    ppr_value = league_settings.get("ppr_value", 1.0)

    player_stats_df = pull_player_stats([season - 1])
    rosters_df = pull_rosters([season])
    schedules_df = pull_schedules([season])
    draft_picks_df = pull_draft_picks([season])
    bye_weeks = get_bye_weeks(season, schedules_df)

    # Build rookie production curves once per position (real historical
    # draft-round-to-production data), used as a fallback for any player
    # with zero prior-season history (rookies) instead of skipping them.
    rookie_curves = {
        pos: build_rookie_production_curve(pos, season, player_stats_df, draft_picks_df, ppr_value)
        for pos in ["QB", "RB", "WR", "TE"]
    } if not draft_picks_df.empty else {}

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
            proj = build_season_projection(gsis_id, position, player_stats_df, season, ppr_value=ppr_value)

        is_rookie_projection = False
        if pd.isna(proj["season_proj_points"]) and position in rookie_curves and not draft_picks_df.empty:
            # THE ROOKIE FIX: no prior-season stats means this is likely a
            # rookie (or a player who didn't play last season) - use the
            # draft-capital production curve instead of skipping them.
            rookie_proj = build_rookie_season_projection(gsis_id, position, season, draft_picks_df, rookie_curves[position])
            if pd.notna(rookie_proj["season_proj_points"]):
                proj = rookie_proj
                is_rookie_projection = True

        if pd.isna(proj["season_proj_points"]):
            continue  # genuinely no data to project from either way - skip rather than guess

        rows.append({
            "gsis_id": gsis_id,
            "player": player.get("full_name"),
            "position": position,
            "team": team,
            "bye": bye_weeks.get(team),
            "season_proj_points": proj["season_proj_points"],
            "games_played_prior": proj.get("games_played_prior", 0),
            "ppg_prior": proj.get("ppg_prior", np.nan),
            "is_rookie_projection": is_rookie_projection,
        })

    projections_df = pd.DataFrame(rows)
    if projections_df.empty:
        return projections_df

    # Add team DEFENSES - this was completely missing before (the scoring
    # function existed but nothing ever called it here), so zero defenses
    # showed up in rankings despite the league requiring 1 DEF starter.
    all_teams = rosters_df[rosters_df["season"] == season]["team"].dropna().unique().tolist()
    def_rows = []
    for team in all_teams:
        proj = build_defense_season_projection(team, player_stats_df, schedules_df, season)
        if pd.isna(proj["season_proj_points"]):
            continue
        def_rows.append({
            "gsis_id": f"DEF_{team}",  # synthetic ID since defenses aren't individual players
            "player": f"{team} DEF",
            "position": "DEF",
            "team": team,
            "bye": bye_weeks.get(team),
            "season_proj_points": proj["season_proj_points"],
            "games_played_prior": proj["games_played_prior"],
            "ppg_prior": proj["ppg_prior"],
        })
    if def_rows:
        projections_df = pd.concat([projections_df, pd.DataFrame(def_rows)], ignore_index=True)

    # Flag position competition (e.g. Kamara/Etienne both landing on the same
    # Saints backfield): when 2+ players at the same position on the same
    # CURRENT-season team both have a real prior-season track record, their
    # individual projections are each based on their OWN last-season workload
    # as if they'd keep it entirely - but they're about to split one team's
    # worth of volume. We deliberately do NOT try to guess the exact split
    # (that's genuinely uncertain, even for real analysts, and a fabricated
    # number would look more rigorous than it is) - instead this just flags
    # it so you can manually discount rather than trust an inflated number.
    def _flag_competition(row):
        teammates = projections_df[
            (projections_df["team"] == row["team"])
            & (projections_df["position"] == row["position"])
            & (projections_df["gsis_id"] != row["gsis_id"])
            & (projections_df["games_played_prior"] >= 6)
        ]
        if row["games_played_prior"] < 6 or teammates.empty:
            return None
        return ", ".join(teammates["player"].tolist())

    projections_df["shares_backfield_with"] = projections_df.apply(_flag_competition, axis=1)

    replacement_levels = compute_replacement_levels(projections_df, league_settings)
    vor_df = compute_vor(projections_df, replacement_levels)

    vor_df = vor_df.sort_values("vor", ascending=False).reset_index(drop=True)
    vor_df["overall_rank"] = vor_df.index + 1
    vor_df["pos_rank"] = vor_df.groupby("position")["vor"].rank(ascending=False, method="min").astype(int)
    vor_df["pos_rank_label"] = vor_df["position"] + vor_df["pos_rank"].astype(str)

    return vor_df[[
        "gsis_id", "overall_rank", "player", "position", "pos_rank_label", "team", "bye",
        "season_proj_points", "vor", "ppg_prior", "games_played_prior",
        "shares_backfield_with", "is_rookie_projection",
    ]]


# ---------------------------------------------------------------------------
# 5b. SNAKE DRAFT - your pick number each round, and who to target
# ---------------------------------------------------------------------------

def get_snake_pick_numbers(num_teams: int, draft_position: int, total_rounds: int) -> list:
    """
    Computes YOUR overall pick number in each round of a snake draft.
    Snake draft alternates direction each round: Round 1 goes 1->num_teams,
    Round 2 goes num_teams->1, Round 3 goes 1->num_teams again, etc.

    Example: 6-team league, drafting 3rd -> picks are 3, 10, 15, 22, 27, 34...
    """
    picks = []
    for round_num in range(1, total_rounds + 1):
        if round_num % 2 == 1:  # odd round - normal order
            overall_pick = (round_num - 1) * num_teams + draft_position
        else:  # even round - reversed order
            overall_pick = (round_num - 1) * num_teams + (num_teams - draft_position + 1)
        picks.append({"round": round_num, "overall_pick": overall_pick})
    return picks


def build_snake_draft_targets(rankings_df: pd.DataFrame, league_settings: dict,
                               num_targets_per_round: int = 5,
                               sort_column: str = "vor", sort_ascending: bool = False) -> pd.DataFrame:
    """
    For each of YOUR picks in a snake draft, shows the top remaining
    candidates at that point in the draft.

    FIX: previously always sorted by pure stats-only "vor", completely
    bypassing the blended_rank correction (which combines our stats with
    FantasyPros public consensus to catch situational blind spots our
    stats-only model can't see - e.g. a player whose target share is about
    to jump because a teammate left in free agency, like Egbuka after
    Mike Evans signed with the 49ers). That meant the snake draft
    simulation could show a real riser going far too late, even though
    the Full Rankings view (which DOES apply the blend) would rank them
    more reasonably. sort_column/sort_ascending now let the caller pass
    "blended_score" (ascending=True, since lower = better there) instead
    of always defaulting to "vor" (descending, higher = better).

    SIMPLIFYING ASSUMPTION FLAGGED: to know who's "still available" at your
    pick, this simulates every OTHER team's pick as "take the best remaining
    player by the same sort_column" - a reasonable default, but real
    drafters have positional needs, personal preferences, and reach for
    sleepers, so actual availability at your real draft will differ from
    this simulation.
    """
    num_teams = league_settings["num_teams"]
    draft_position = league_settings["draft_position"]
    starters = league_settings["starters"]
    total_rounds = sum(starters.values()) + league_settings["bench"]

    your_picks = get_snake_pick_numbers(num_teams, draft_position, total_rounds)
    your_pick_numbers = {p["overall_pick"] for p in your_picks}

    pool = rankings_df.sort_values(sort_column, ascending=sort_ascending).reset_index(drop=True).copy()
    drafted_gsis_ids = set()

    results = []
    overall_pick_counter = 0
    for round_info in your_picks:
        round_num = round_info["round"]
        target_pick = round_info["overall_pick"]

        # simulate every pick leading up to (and not including) your pick
        while overall_pick_counter < target_pick - 1:
            overall_pick_counter += 1
            remaining = pool[~pool["gsis_id"].isin(drafted_gsis_ids)]
            if remaining.empty:
                break
            top_pick = remaining.iloc[0]
            drafted_gsis_ids.add(top_pick["gsis_id"])

        # now show top remaining candidates at YOUR pick
        remaining = pool[~pool["gsis_id"].isin(drafted_gsis_ids)]
        top_candidates = remaining.head(num_targets_per_round)

        for _, candidate in top_candidates.iterrows():
            results.append({
                "round": round_num,
                "your_overall_pick": target_pick,
                "player": candidate["player"],
                "position": candidate["position"],
                "team": candidate["team"],
                "season_proj_points": candidate["season_proj_points"],
                "vor": candidate["vor"],
                "blended_rank": candidate.get("blended_rank"),
                "overall_rank": candidate.get("overall_rank"),
                "fantasypros_ecr": candidate.get("fantasypros_ecr"),
            })

        # advance simulation assuming you take the top remaining player,
        # so later rounds' availability keeps making sense
        overall_pick_counter += 1
        if not remaining.empty:
            drafted_gsis_ids.add(remaining.iloc[0]["gsis_id"])

    return pd.DataFrame(results)


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

    # FIX: load_ff_rankings() has MULTIPLE rows per player (rankings get
    # re-published with different scrape_date snapshots through the
    # offseason). Merging without deduplicating first multiplies every row
    # in rankings_df once per matching FantasyPros row - this was causing
    # each player to show up repeated 5x (or however many snapshots exist).
    # Keep only the most recent snapshot per player before merging.
    if "scrape_date" in ff_rankings.columns:
        ff_rankings = ff_rankings.sort_values("scrape_date").drop_duplicates(subset=["player"], keep="last")
    else:
        ff_rankings = ff_rankings.drop_duplicates(subset=["player"], keep="last")

    # FIX: normalize names before matching, so suffix variations (e.g. our
    # roster data saying "Travis Etienne" vs FantasyPros saying "Travis
    # Etienne Jr.") don't silently fail to match and fall back to
    # stats-only without any warning.
    def _normalize_name(name):
        if pd.isna(name):
            return name
        name = str(name).strip()
        for suffix in [" Jr.", " Jr", " Sr.", " Sr", " II", " III", " IV"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)].strip()
        return name.replace(".", "").replace("'", "").lower()

    rankings_df = rankings_df.copy()
    ff_rankings = ff_rankings.copy()
    rankings_df["_merge_key"] = rankings_df["player"].apply(_normalize_name)
    ff_rankings["_merge_key"] = ff_rankings["player"].apply(_normalize_name)

    merged = rankings_df.merge(
        ff_rankings[["_merge_key", "ecr", "player_owned_yahoo"]],
        on="_merge_key", how="left"
    ).drop(columns=["_merge_key"])
    merged = merged.rename(columns={"ecr": "fantasypros_ecr"})

    # FALLBACK MATCHING: if the exact normalized name didn't match (still
    # NaN), try a looser "first initial + last name" match before giving up
    # entirely. Exact-name matching can silently fail on minor spelling/
    # formatting differences between our roster data and FantasyPros -
    # this catches more real matches instead of leaving a player unblended
    # (which was one of two suspected causes of players like Lamb/Egbuka/
    # Rice showing up unrealistically late in round-by-round targeting).
    def _loose_key(name):
        if pd.isna(name):
            return name
        parts = _normalize_name(name).split()
        if len(parts) < 2:
            return _normalize_name(name)
        return f"{parts[0][0]}_{parts[-1]}"  # first initial + last name

    still_missing = merged["fantasypros_ecr"].isna()
    if still_missing.any():
        rankings_df["_loose_key"] = rankings_df["player"].apply(_loose_key)
        ff_rankings["_loose_key"] = ff_rankings["player"].apply(_loose_key)
        loose_lookup = ff_rankings.drop_duplicates(subset=["_loose_key"])[
            ["_loose_key", "ecr", "player_owned_yahoo"]
        ].rename(columns={"ecr": "fantasypros_ecr_loose", "player_owned_yahoo": "player_owned_yahoo_loose"})

        merged = merged.merge(
            rankings_df[["gsis_id", "_loose_key"]], on="gsis_id", how="left"
        ).merge(loose_lookup, on="_loose_key", how="left").drop(columns=["_loose_key"])

        merged["fantasypros_ecr"] = merged["fantasypros_ecr"].fillna(merged["fantasypros_ecr_loose"])
        merged["player_owned_yahoo"] = merged["player_owned_yahoo"].fillna(merged["player_owned_yahoo_loose"])
        merged = merged.drop(columns=["fantasypros_ecr_loose", "player_owned_yahoo_loose"])

    merged["our_rank_delta"] = merged["fantasypros_ecr"] - merged["overall_rank"]

    return merged.sort_values("our_rank_delta", ascending=False)


def build_draft_rankings_backtest(test_season: int, league_settings: dict = None) -> pd.DataFrame:
    """
    Tests the STATS-ONLY portion of the draft methodology by building
    projections "as if drafting" for test_season (using test_season-1
    stats + test_season rosters - the exact same logic build_yahoo_style_
    rankings uses for a real upcoming draft), then comparing against REAL
    test_season performance.

    HONEST LIMITATION FLAGGED: this can only validate the pure stats-based
    projection, NOT the blended_rank from compute_blended_rankings().
    load_ff_rankings() only returns TODAY'S live FantasyPros snapshot -
    there's no free historical archive of what FantasyPros said before a
    past season, so there's no way to reconstruct what the blend would
    have said for a prior draft. This tells you whether our own
    historical-rate projection tends to be accurate on its own - useful
    context for deciding how much weight to give it vs. public consensus,
    but not a direct test of the blend itself.
    """
    if league_settings is None:
        league_settings = LEAGUE_SETTINGS
    ppr_value = league_settings.get("ppr_value", 1.0)

    projected = build_yahoo_style_rankings(test_season, league_settings)
    if projected.empty:
        return projected

    player_stats_df = pull_player_stats([test_season])
    actual_season = player_stats_df[
        (player_stats_df["season"] == test_season) & (player_stats_df["season_type"] == "REG")
    ]

    def _actual_points(gsis_id, position):
        history = actual_season[actual_season["gsis_id"] == gsis_id]
        if history.empty:
            return np.nan, 0
        if position == "K":
            pts = history.apply(lambda r: calc_kicker_fantasy_points(r.to_dict()), axis=1).sum()
        else:
            pts = history.apply(lambda r: calc_offense_fantasy_points(r.to_dict(), ppr_value), axis=1).sum()
        return round(pts, 1), len(history)

    actual_points_list, actual_games_list = [], []
    for _, row in projected.iterrows():
        pts, games = _actual_points(row["gsis_id"], row["position"])
        actual_points_list.append(pts)
        actual_games_list.append(games)

    projected["actual_season_points"] = actual_points_list
    projected["actual_games_played"] = actual_games_list
    projected["projection_miss"] = projected["season_proj_points"] - projected["actual_season_points"]

    return projected.sort_values("projection_miss", key=lambda s: s.abs(), ascending=False)


def compute_blended_rankings(rankings_df: pd.DataFrame, our_weight: float = 0.5) -> pd.DataFrame:
    """
    Blends our pure stats-based rank (overall_rank, built entirely from last
    season's rate stats) with FantasyPros consensus rank (fantasypros_ecr).

    ADAPTIVE WEIGHTING FIX: a flat 50/50 blend wasn't correcting severely
    tanked stats-only ranks (e.g. a player coming off an injury-shortened
    season, or a rookie whose situation just improved) - the correction was
    too weak to overcome a large gap. A big disagreement between our stats
    rank and public consensus is ITSELF evidence that situational factors
    our stats-only model can't see (injury recovery outlook, new role, new
    scheme) are likely driving the difference - so the bigger the gap, the
    more weight shifts toward public consensus, up to a floor of 20% our
    stats weight even for extreme disagreements (never fully abandons our
    own signal entirely).

    our_weight is now the BASE weight used when our rank and FantasyPros
    roughly agree - it gets reduced automatically as disagreement grows.
    """
    df = rankings_df.copy()
    if "fantasypros_ecr" not in df.columns:
        df["blended_rank"] = df["overall_rank"]
        return df

    max_rank = max(df["overall_rank"].max(), df["fantasypros_ecr"].max(skipna=True))

    def _blend(row):
        our_pct = row["overall_rank"] / max_rank
        if pd.isna(row.get("fantasypros_ecr")):
            return row["overall_rank"]  # no public data to blend with - use our rank alone
        public_pct = row["fantasypros_ecr"] / max_rank

        # adaptive weight: shrink our_weight as disagreement grows
        disagreement = abs(our_pct - public_pct)  # 0 = perfect agreement, up to ~1 = max disagreement
        adaptive_weight = max(our_weight * (1 - disagreement), 0.2 * our_weight)

        blended_pct = (adaptive_weight * our_pct) + ((1 - adaptive_weight) * public_pct)
        return blended_pct * max_rank

    df["blended_score"] = df.apply(_blend, axis=1)
    df["blended_rank"] = df["blended_score"].rank(method="min").astype(int)
    return df.sort_values("blended_rank")
