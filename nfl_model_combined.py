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
import functools

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None  # allows this file to be imported/tested without the package present


# ---------------------------------------------------------------------------
# 0. IN-PROCESS PULL CACHE
#
# PERFORMANCE FIX: build_season_accuracy_report() calls build_weekly_slate()
# once per week in a loop (16-17 times for a full season). Every pull_*
# function below re-fetches the SAME season-level data (pbp, participation,
# ftn, ngs, etc.) on every single call, since that data doesn't change
# week-to-week within a season - this was a genuine redundant-network-call
# bug (a full season report was re-downloading the entire season's raw
# data 16-17x over), not just "the data is naturally big/slow." This
# decorator makes every pull_* function fetch each unique argument
# combination exactly ONCE per process; repeat calls (e.g. every week of
# the season loop asking for the same season's pbp) hit the in-memory
# cache instead of hitting the network again. A fresh Streamlit run/rerun
# starts with an empty cache naturally, so this never serves stale data
# across separate scans - only within the SAME run's redundant re-asks.
# ---------------------------------------------------------------------------

_PULL_CACHE: dict = {}


def _cache_pull(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hashable_args = tuple(tuple(a) if isinstance(a, list) else a for a in args)
        hashable_kwargs = tuple(sorted(
            (k, tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items()
        ))
        key = (func.__name__, hashable_args, hashable_kwargs)
        if key not in _PULL_CACHE:
            _PULL_CACHE[key] = func(*args, **kwargs)
        return _PULL_CACHE[key]
    return wrapper


# ---------------------------------------------------------------------------
# 1. DATA PULL FUNCTIONS
# ---------------------------------------------------------------------------

@_cache_pull
def pull_pbp(years: list[int]) -> pd.DataFrame:
    """Play-by-play data for the given seasons, converted to pandas."""
    df = nfl.load_pbp(seasons=years)
    return df.to_pandas()


@_cache_pull
def pull_ngs(stat_type: str, years: list[int]) -> pd.DataFrame:
    """
    stat_type: 'passing', 'rushing', or 'receiving'
    Returns official Next Gen Stats for the given seasons.
    """
    df = nfl.load_nextgen_stats(stat_type=stat_type, seasons=years)
    return df.to_pandas()


@_cache_pull
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


@_cache_pull
def pull_snap_counts(years: list[int]) -> pd.DataFrame:
    """Snap counts by player/game - used as a route-participation / opportunity proxy."""
    df = nfl.load_snap_counts(seasons=years)
    return df.to_pandas()


@_cache_pull
def pull_ftn_charting(years: list[int]) -> pd.DataFrame:
    """
    FTN manual charting data (free, 2022-onward).
    Key columns: n_defense_box, n_offense_backfield, is_motion, is_play_action,
    is_screen_pass, is_no_huddle, qb_location.
    """
    df = nfl.load_ftn_charting(seasons=years)
    return df.to_pandas()


@_cache_pull
def pull_participation(years: list[int]) -> pd.DataFrame:
    """
    Participation data - carries defense_man_zone_type, defense_coverage_type,
    time_to_throw, was_pressure. This is where coverage-shell % comes from.
    """
    df = nfl.load_participation(seasons=years)
    return df.to_pandas()


@_cache_pull
def pull_schedules(years: list[int]) -> pd.DataFrame:
    df = nfl.load_schedules(seasons=years)
    return df.to_pandas()


@_cache_pull
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
# 3b. PLAY-ACTION TENDENCY + COVERAGE-SPECIFIC PLAY-ACTION VULNERABILITY
#
# Closes a real, previously-unused gap: FTN charting's is_play_action sat
# in already-pulled data completely unwired. This isn't just "does this
# team run play-action a lot" - it's the specific interaction requested:
# does a defense's coverage mix (already tracked) get specifically worse
# against play-action, AND does the offense in front of them both run PA
# often AND actually perform well in it. Two separate offense/defense
# profiles below, combined by calc_playaction_exploit_strength().
# ---------------------------------------------------------------------------

def build_qb_playaction_profile(season: int, week: int, pbp_df: pd.DataFrame,
                                 ftn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-QB play-action rate (share of his dropbacks that are play-action)
    and play-action EFFECTIVENESS (his own EPA/play on PA snaps vs his own
    EPA/play on non-PA snaps) - frequency and skill are graded separately,
    since a QB can run PA constantly without being especially good at it,
    or vice versa. Joins ftn_df (is_play_action) to pbp_df on
    (game_id, play_id), same join fix used throughout this file for FTN/
    participation data. Uses weeks BEFORE the target week only.
    """
    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)
                       & (pbp_df["play_type"] == "pass")]
    if hist_pbp.empty:
        return pd.DataFrame()

    merged = hist_pbp.merge(
        ftn_df[["nflverse_game_id", "nflverse_play_id", "is_play_action"]],
        left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner",
    )
    df = merged.dropna(subset=["passer_player_id", "is_play_action"])
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["passer_player_id", "is_play_action"]).agg(
        epa=("epa", "mean"), n=("epa", "count"),
    ).reset_index()

    pa = agg[agg["is_play_action"] == True].rename(columns={"epa": "pa_epa", "n": "pa_plays"}).drop(columns=["is_play_action"])
    non_pa = agg[agg["is_play_action"] == False].rename(columns={"epa": "non_pa_epa", "n": "non_pa_plays"}).drop(columns=["is_play_action"])

    result = pa.merge(non_pa, on="passer_player_id", how="outer").rename(columns={"passer_player_id": "gsis_id"})
    total_plays = result[["pa_plays", "non_pa_plays"]].fillna(0).sum(axis=1)
    result["pa_rate"] = (result["pa_plays"].fillna(0) / total_plays).where(total_plays > 0)
    result["pa_epa_diff"] = result["pa_epa"] - result["non_pa_epa"]  # how much better/worse he is IN play-action vs his own baseline

    for col in ["pa_rate", "pa_epa_diff"]:
        result[f"{col}_grade"] = result[col].apply(lambda v: calc_percentile_grade(v, result[col]))

    return result


def build_defense_playaction_allowed(season: int, week: int, pbp_df: pd.DataFrame,
                                      ftn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-defense play-action-allowed EPA vs non-PA-allowed EPA - the
    OVERALL (not coverage-specific) play-action vulnerability signal, used
    as a fallback when a team's dominant coverage doesn't have enough
    charted PA-specific plays yet (see build_coverage_playaction_crosswalk).
    """
    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)
                       & (pbp_df["play_type"] == "pass")]
    if hist_pbp.empty:
        return pd.DataFrame()

    merged = hist_pbp.merge(
        ftn_df[["nflverse_game_id", "nflverse_play_id", "is_play_action"]],
        left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner",
    )
    df = merged.dropna(subset=["defteam", "is_play_action"])
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["defteam", "is_play_action"]).agg(epa=("epa", "mean")).reset_index()
    pa = agg[agg["is_play_action"] == True].rename(columns={"epa": "pa_epa_allowed"}).drop(columns=["is_play_action"])
    non_pa = agg[agg["is_play_action"] == False].rename(columns={"epa": "non_pa_epa_allowed"}).drop(columns=["is_play_action"])
    result = pa.merge(non_pa, on="defteam", how="outer")
    result["pa_vulnerability_gap"] = result["pa_epa_allowed"] - result["non_pa_epa_allowed"]  # positive = allows MORE in PA than normal

    # allowed metric: lower is better defensively, same inversion convention as every other *_allowed grade in this file
    result["pa_epa_allowed_grade"] = result["pa_epa_allowed"].apply(
        lambda v: 100 - calc_percentile_grade(v, result["pa_epa_allowed"]) if pd.notna(v) else np.nan
    )
    return result


def build_coverage_playaction_crosswalk(season: int, week: int, participation_df: pd.DataFrame,
                                         ftn_df: pd.DataFrame, pbp_df: pd.DataFrame,
                                         min_plays: int = 15) -> pd.DataFrame:
    """
    The actual requested interaction: per (defteam, coverage_type),
    EPA allowed SPECIFICALLY on play-action plays run against that
    coverage - e.g. does THIS defense's Cover 3 specifically get
    exploited by play-action, not just "is this defense bad against PA
    in general." Joins participation's coverage type + ftn's
    is_play_action on the SAME play (both keyed off nflverse_game_id,
    with participation's play id column named `play_id` and ftn's named
    `nflverse_play_id` - a real naming mismatch between the two tables,
    matched explicitly here), then pulls in defteam/epa from pbp.

    Rows below min_plays are dropped (not returned as NaN) - a coverage
    type a defense rarely plays on PA specifically doesn't have a
    trustworthy sample yet, and the caller should fall back to the
    overall (non-coverage-specific) play-action-allowed signal instead
    (see calc_playaction_exploit_strength).
    """
    hist_participation = participation_df.dropna(subset=["defense_coverage_type"])
    merged = hist_participation.merge(
        ftn_df[["nflverse_game_id", "nflverse_play_id", "is_play_action"]],
        left_on=["nflverse_game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner",
    )
    merged = merged.merge(
        pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)][["game_id", "play_id", "defteam", "epa"]],
        left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner",
    )
    df = merged[(merged["is_play_action"] == True) & merged["defteam"].notna()]
    if df.empty:
        return pd.DataFrame()

    result = df.groupby(["defteam", "defense_coverage_type"]).agg(
        pa_epa_allowed_in_coverage=("epa", "mean"), n_pa_plays=("epa", "count"),
    ).reset_index()
    result = result[result["n_pa_plays"] >= min_plays]
    if result.empty:
        return result

    result["pa_epa_allowed_in_coverage_grade"] = result["pa_epa_allowed_in_coverage"].apply(
        lambda v: 100 - calc_percentile_grade(v, result["pa_epa_allowed_in_coverage"]) if pd.notna(v) else np.nan
    )
    return result


def calc_playaction_exploit_strength(qb_pa_row: dict, def_pa_row: dict,
                                      coverage_pa_crosswalk_df: pd.DataFrame,
                                      defteam: str, dominant_coverage: str) -> dict:
    """
    Combines the offense side (does this QB run PA often AND perform well
    in it) with the defense side (is this specific defense - ideally in
    its SPECIFIC dominant coverage, falling back to its overall PA-allowed
    number if that coverage doesn't have a trustworthy PA-specific sample
    yet - vulnerable to play-action) into one 0-1 exploit signal. This is
    the real interaction requested: coverage tendency x play-action
    tendency x play-action-specific vulnerability, not any of those three
    in isolation.
    """
    offense_vals = [
        qb_pa_row.get("pa_rate_grade") if qb_pa_row else None,
        qb_pa_row.get("pa_epa_diff_grade") if qb_pa_row else None,
    ]
    offense_vals = [v for v in offense_vals if pd.notna(v)]
    offense_component = (sum(offense_vals) / len(offense_vals) / 100) if offense_vals else np.nan

    # Prefer the coverage-specific number; fall back to the defense's
    # overall PA-allowed grade if that specific coverage lacks a sample.
    coverage_specific_grade = np.nan
    if not coverage_pa_crosswalk_df.empty and dominant_coverage is not None:
        match = coverage_pa_crosswalk_df[
            (coverage_pa_crosswalk_df["defteam"] == defteam)
            & (coverage_pa_crosswalk_df["defense_coverage_type"] == dominant_coverage)
        ]
        if not match.empty:
            coverage_specific_grade = match.iloc[0].get("pa_epa_allowed_in_coverage_grade")

    if pd.notna(coverage_specific_grade):
        defense_grade = coverage_specific_grade
        used_coverage_specific = True
    else:
        defense_grade = def_pa_row.get("pa_epa_allowed_grade") if def_pa_row else np.nan
        used_coverage_specific = False

    defense_component = (1 - (defense_grade / 100)) if pd.notna(defense_grade) else np.nan

    if pd.isna(offense_component) and pd.isna(defense_component):
        return {"exploit_strength": np.nan, "used_coverage_specific_playaction_data": used_coverage_specific}
    if pd.isna(offense_component):
        return {"exploit_strength": round(defense_component, 3), "used_coverage_specific_playaction_data": used_coverage_specific}
    if pd.isna(defense_component):
        return {"exploit_strength": round(offense_component, 3), "used_coverage_specific_playaction_data": used_coverage_specific}
    return {
        "exploit_strength": round(offense_component * 0.5 + defense_component * 0.5, 3),
        "used_coverage_specific_playaction_data": used_coverage_specific,
    }


# ---------------------------------------------------------------------------
# 3c. QB PRESSURE / TIME-TO-THROW PROFILE (own-side counterpart to the
#     defense's existing pressure_rate_generated - was one-sided before,
#     nothing on the QB's own side to pair against it)
# ---------------------------------------------------------------------------

def build_qb_pressure_profile(season: int, week: int, participation_df: pd.DataFrame,
                               pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    This QB's own pressure-rate-faced and average time-to-throw, joined
    from participation_df (was_pressure, time_to_throw) via pbp for the
    passer id. pressure_rate_faced is graded INVERTED (lower pressure
    faced = better QB play/protection = higher grade), same convention as
    every other *_allowed/faced metric in this file. avg_time_to_throw is
    included for context but NOT graded directionally - a fast release
    isn't unambiguously better or worse than a longer-developing deep
    shot, unlike pressure faced which is unambiguous.
    """
    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week) & (pbp_df["play_type"] == "pass")]
    if hist_pbp.empty:
        return pd.DataFrame()

    merged = participation_df.merge(
        hist_pbp[["game_id", "play_id", "passer_player_id"]],
        left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner",
    )
    df = merged.dropna(subset=["passer_player_id"])
    if df.empty or "was_pressure" not in df.columns:
        return pd.DataFrame()

    agg_dict = {"pressure_rate_faced": ("was_pressure", "mean")}
    if "time_to_throw" in df.columns:
        agg_dict["avg_time_to_throw"] = ("time_to_throw", "mean")

    result = df.groupby("passer_player_id").agg(**agg_dict).reset_index().rename(columns={"passer_player_id": "gsis_id"})
    result["pressure_rate_faced_grade"] = result["pressure_rate_faced"].apply(
        lambda v: 100 - calc_percentile_grade(v, result["pressure_rate_faced"]) if pd.notna(v) else np.nan
    )
    return result


# ---------------------------------------------------------------------------
# 3d. PROE (PASS RATE OVER EXPECTED) - previously flagged as not built,
#     raw attempt volume used as a rougher stand-in. Expected pass rate is
#     computed as the league-wide average pass rate for each (down,
#     distance-bucket) situation, rather than a full trained model - a
#     real, defensible free-data baseline, not the exact proprietary PROE
#     methodology (which uses a trained model on score/time/etc. too).
# ---------------------------------------------------------------------------

def build_proe_profile(season: int, week: int, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team PROE: actual pass rate minus the league-wide expected pass
    rate for the same (down, distance-bucket) situations this team faced,
    isolating real play-calling aggressiveness from the confound of a
    team just facing more/fewer obvious passing downs. distance-bucket:
    short (<=3), medium (4-7), long (8+). Only 1st/2nd/3rd/4th down,
    normal (non-garbage-time-only) plays with a real down/ydstogo value.
    """
    hist_pbp = pbp_df[
        (pbp_df["season"] == season) & (pbp_df["week"] < week)
        & (pbp_df["play_type"].isin(["pass", "run"]))
        & pbp_df["down"].notna() & pbp_df["ydstogo"].notna()
    ].copy()
    if hist_pbp.empty:
        return pd.DataFrame()

    hist_pbp["distance_bucket"] = pd.cut(
        hist_pbp["ydstogo"], bins=[-0.1, 3, 7, 100], labels=["short", "medium", "long"]
    )
    hist_pbp["is_pass"] = (hist_pbp["play_type"] == "pass").astype(int)

    league_expected = hist_pbp.groupby(["down", "distance_bucket"], observed=True)["is_pass"].mean().rename("expected_pass_rate")
    hist_pbp = hist_pbp.merge(league_expected, on=["down", "distance_bucket"], how="left")

    result = hist_pbp.groupby("posteam").agg(
        actual_pass_rate=("is_pass", "mean"),
        expected_pass_rate=("expected_pass_rate", "mean"),
        n_plays=("is_pass", "count"),
    ).reset_index()
    result["proe"] = result["actual_pass_rate"] - result["expected_pass_rate"]
    result["proe_grade"] = result["proe"].apply(lambda v: calc_percentile_grade(v, result["proe"]))
    return result


# ---------------------------------------------------------------------------
# 3e. MOTION / NO-HUDDLE TENDENCY (display/context only - NOT wired into
#     quality_score, since neither has a paired "defense specifically
#     struggles against motion/no-huddle" metric to combine it with the
#     way play-action does. Flagged honestly rather than wired in on a
#     guess; a genuine future addition would need the same paired
#     offense-tendency x defense-vulnerability treatment PA just got.)
# ---------------------------------------------------------------------------

def build_motion_tendency_profile(season: int, week: int, ftn_df: pd.DataFrame,
                                   pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Per-offense motion rate and no-huddle rate - real, free, previously entirely unused FTN columns."""
    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)]
    merged = ftn_df.merge(
        hist_pbp[["game_id", "play_id", "posteam"]],
        left_on=["nflverse_game_id", "nflverse_play_id"], right_on=["game_id", "play_id"], how="inner",
    )
    df = merged.dropna(subset=["posteam"])
    if df.empty:
        return pd.DataFrame()

    agg_dict = {}
    if "is_motion" in df.columns:
        agg_dict["motion_rate"] = ("is_motion", "mean")
    if "is_no_huddle" in df.columns:
        agg_dict["no_huddle_rate"] = ("is_no_huddle", "mean")
    if not agg_dict:
        return pd.DataFrame()

    return df.groupby("posteam").agg(**agg_dict).reset_index()


# ---------------------------------------------------------------------------
# 3f. PERSONNEL GROUPING TENDENCY + VULNERABILITY (11/12/21 personnel etc.)
#
# Direct analog to the play-action crosswalk above, using offense_personnel/
# defense_personnel - confirmed real participation_df columns that sat
# completely unused. Same real question as PA: does this offense line up
# in one specific personnel grouping most of the time, and is THIS
# opponent specifically bad against that exact grouping (not just bad in
# general).
# ---------------------------------------------------------------------------

def build_offense_personnel_tendency(season: int, week: int, participation_df: pd.DataFrame,
                                      pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team DOMINANT offense personnel grouping (e.g. '11 Personnel',
    whatever the real charted label is) and how often they actually use
    it - the offense-tendency half of the crosswalk. Same join pattern
    used throughout this file for participation data: nflverse_game_id +
    play_id -> pbp's game_id + play_id for posteam.
    """
    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)]
    merged = participation_df.merge(
        hist_pbp[["game_id", "play_id", "posteam"]],
        left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner",
    )
    df = merged.dropna(subset=["posteam", "offense_personnel"])
    if df.empty:
        return pd.DataFrame()

    counts = df.groupby(["posteam", "offense_personnel"]).size().reset_index(name="n")
    totals = df.groupby("posteam").size().reset_index(name="n_total")
    counts = counts.merge(totals, on="posteam")
    counts["usage_pct"] = counts["n"] / counts["n_total"]

    dominant = counts.loc[counts.groupby("posteam")["usage_pct"].idxmax()][
        ["posteam", "offense_personnel", "usage_pct"]
    ].rename(columns={"offense_personnel": "dominant_personnel", "usage_pct": "dominant_personnel_pct"})
    return dominant


def build_defense_personnel_allowed(season: int, week: int, participation_df: pd.DataFrame,
                                     pbp_df: pd.DataFrame, min_plays: int = 15) -> pd.DataFrame:
    """
    Per (defteam, offense_personnel-they-faced), real EPA allowed - which
    SPECIFIC personnel grouping a defense struggles against, not just
    overall defense quality. Rows below min_plays are dropped entirely
    (too thin a sample to trust) rather than returned as noisy NaN, same
    pattern as build_coverage_playaction_crosswalk.
    """
    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)]
    merged = participation_df.merge(
        hist_pbp[["game_id", "play_id", "defteam", "epa"]],
        left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner",
    )
    df = merged.dropna(subset=["defteam", "offense_personnel", "epa"])
    if df.empty:
        return pd.DataFrame()

    result = df.groupby(["defteam", "offense_personnel"]).agg(
        epa_allowed=("epa", "mean"), n_plays=("epa", "count"),
    ).reset_index()
    result = result[result["n_plays"] >= min_plays]
    if result.empty:
        return result

    result["epa_allowed_grade"] = result["epa_allowed"].apply(
        lambda v: 100 - calc_percentile_grade(v, result["epa_allowed"]) if pd.notna(v) else np.nan
    )
    return result


def calc_personnel_exploit_strength(team: str, offense_personnel_tendency_df: pd.DataFrame,
                                     defteam: str, defense_personnel_allowed_df: pd.DataFrame) -> dict:
    """
    Looks up this offense's dominant personnel grouping, then the
    opponent's real EPA-allowed grade specifically against THAT grouping -
    a 0-1 exploit signal, same shape as calc_playaction_exploit_strength.
    Degrades to NaN (not a guessed neutral value) if either side lacks
    enough real data - the caller should treat NaN as "no signal here"
    same as every other exploit function in this file.
    """
    if offense_personnel_tendency_df.empty or defense_personnel_allowed_df.empty:
        return {"exploit_strength": np.nan, "dominant_personnel": None}

    off_row = offense_personnel_tendency_df[offense_personnel_tendency_df["posteam"] == team]
    if off_row.empty:
        return {"exploit_strength": np.nan, "dominant_personnel": None}
    dominant_personnel = off_row.iloc[0]["dominant_personnel"]

    def_row = defense_personnel_allowed_df[
        (defense_personnel_allowed_df["defteam"] == defteam)
        & (defense_personnel_allowed_df["offense_personnel"] == dominant_personnel)
    ]
    if def_row.empty:
        return {"exploit_strength": np.nan, "dominant_personnel": dominant_personnel}

    grade = def_row.iloc[0]["epa_allowed_grade"]
    if pd.isna(grade):
        return {"exploit_strength": np.nan, "dominant_personnel": dominant_personnel}
    return {"exploit_strength": round(1 - (grade / 100), 3), "dominant_personnel": dominant_personnel}


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
# 4a2. SHADOW-CORNER CONTEXT (manual lookup - free data genuinely CANNOT
#      supply this, same category as COORDINATOR_MAP above)
#
# HONESTY NOTE: true per-play defender assignment (which specific CB
# covered which specific WR) is NOT in any free NFL data source -
# nflreadpy/nflverse has no column identifying this. That's specifically
# what PFF sells as a premium "coverage/matchup" product. This is
# deliberately NOT an automatic algorithm pretending to detect real
# matchups from stats - it's a manually-maintained list, same honest
# pattern as COORDINATOR_MAP, for the small number of corners around the
# league who are PUBLICLY KNOWN to consistently shadow the opponent's
# WR1 (most defenses instead rotate by field side or play zone concepts,
# where no fixed CB-vs-WR1 assignment exists at all - don't fill in a
# team here unless that team's shadow-corner tendency is real, known
# information, not a guess).
#
# DELIBERATELY NOT wired into quality_score/mu - even when a shadow
# corner is known, the only free per-defender stat available
# (def_interceptions from player_stats) is a weak, noisy proxy for
# coverage quality (a corner can play great technique with zero picks,
# or get lucky with several despite poor coverage) - wiring a thin signal
# like that into scoring is exactly the mistake the readiness report just
# caught with the coverage/box adjustment. Shown as CONTEXT ONLY.
# ---------------------------------------------------------------------------

SHADOW_CORNER_MAP = {
    # "team_abbr": "gsis_id_of_the_corner_who_shadows_the_opponent's_true_WR1"
    # Fill in ONLY for teams with a real, known shadow-coverage tendency.
    # Leave every other team out entirely - an empty/missing entry means
    # "no known fixed assignment", not "no advantage".
}


def get_shadow_corner_context(team: str, opponent: str, receiver_target_share_rank: int,
                               player_stats_df: pd.DataFrame, season: int, week: int) -> dict:
    """
    CONTEXT ONLY - see honesty note above. If `opponent` has a known
    shadow corner in SHADOW_CORNER_MAP AND this receiver is presumed to
    be that opponent's primary target (receiver_target_share_rank == 1
    on his own team, i.e. the team's real WR1 by target share - the
    closest free-data proxy for "the corner's likely assignment"),
    returns that corner's real season interception/fumble-recovery
    counting stats as background context for a human to weigh manually -
    NOT a coverage-quality grade, and NOT added to any exploit_strength
    or quality_score calculation.
    """
    corner_gsis_id = SHADOW_CORNER_MAP.get(opponent)
    if corner_gsis_id is None or receiver_target_share_rank != 1:
        return {}

    corner_stats = player_stats_df[
        (player_stats_df["gsis_id"] == corner_gsis_id) & (player_stats_df["season"] == season)
        & (player_stats_df["week"] < week)
    ]
    if corner_stats.empty:
        return {"shadow_corner_gsis_id": corner_gsis_id, "shadow_corner_note": "Known shadow corner, no season stats yet."}

    return {
        "shadow_corner_gsis_id": corner_gsis_id,
        "shadow_corner_interceptions_season": int(corner_stats["def_interceptions"].sum()) if "def_interceptions" in corner_stats.columns else None,
        "shadow_corner_note": "Context only - real coverage-quality data isn't free. Weigh manually, not scored.",
    }


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

def calc_offense_fantasy_points(player_stats_row: dict, ppr_value: float = 1.0) -> float:
    """
    Offensive fantasy scoring, using confirmed real player_stats columns.
    ppr_value is now adjustable: 1.0 = full PPR, 0.5 = half PPR, 0.0 = standard
    (no reception points) - previously hardcoded to full PPR only.

    Scoring rules (as provided, with receptions now adjustable):
      Passing Yards: 0.04/yd | Passing TD: 4 | INT: -1
      Rushing Yards: 0.1/yd | Rushing TD: 6
      Receptions: ppr_value (default 1.0/Full PPR) | Receiving Yards: 0.1/yd | Receiving TD: 6
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
    points += r.get("receptions", 0) * ppr_value
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


def build_player_coverage_efficiency(player_gsis_id: str, role: str, season: int,
                                      participation_df: pd.DataFrame, pbp_df: pd.DataFrame,
                                      min_plays_per_bucket: int = 14, current_team: str = None,
                                      prior_participation_df: pd.DataFrame = None,
                                      prior_pbp_df: pd.DataFrame = None) -> dict:
    """
    Computes a player's REAL historical efficiency (yards per play) against
    man coverage vs zone coverage specifically, using their own play-level
    history - not an approximation, an actual data-driven split.

    CROSS-SEASON FIX (same principle as calc_prop_mu/calc_player_sigma):
    previously this ONLY looked at the current season's plays before the
    target week, with no fallback at all - meaning EVERY player, even ones
    who didn't change teams, had no coverage-specific adjustment until they
    personally racked up 8+ real plays against both man AND zone THIS
    season (often several weeks in). Now, if current_team and prior-season
    data are provided, insufficient current-season buckets are topped up
    with prior-season plays FILTERED TO THE SAME TEAM (via posteam) -
    exactly like the team-filtered mu/sigma fallback. A player who didn't
    change teams gets a much larger, more reliable sample right away. A
    traded player (e.g. AJ Brown, Eagles->Patriots June 2026) correctly
    gets NOTHING from the prior-season fallback here, since none of his
    2025 plays have posteam=="NE" - his old-team plays are excluded, same
    as intended, not a gap in this fix.

    role: "receiver" or "passer". Joins participation_df (which carries
    defense_man_zone_type per play) to pbp_df on (game_id, play_id) - same
    join fix already used in build_coverage_profile() - then filters to
    plays where this specific player was the receiver/passer.

    Returns {"man_ypp": x, "zone_ypp": y, "overall_ypp": z,
             "man_plays": n, "zone_plays": n} - ypp = yards per play.
    Buckets still below min_plays_per_bucket after the cross-season top-up
    return NaN (too small a sample to trust), and the caller should fall
    back to no adjustment rather than react to noise.
    """
    def _get_player_plays(part_df, pbp_source_df):
        merged = part_df.merge(
            pbp_source_df[["game_id", "play_id", "defteam", "posteam",
                            "receiver_player_id", "receiving_yards",
                            "passer_player_id", "passing_yards"]],
            left_on=["nflverse_game_id", "play_id"],
            right_on=["game_id", "play_id"],
            how="left",
        )
        player_col = "receiver_player_id" if role == "receiver" else "passer_player_id"
        return merged[
            (merged[player_col] == player_gsis_id) & merged["defense_man_zone_type"].notna()
        ]

    player_plays = _get_player_plays(participation_df, pbp_df)

    def _bucket_count(plays_df, coverage_type):
        return len(plays_df[plays_df["defense_man_zone_type"] == coverage_type])

    man_n = _bucket_count(player_plays, "Man")
    zone_n = _bucket_count(player_plays, "Zone")

    # top up with team-filtered prior-season plays if either bucket is short
    if (man_n < min_plays_per_bucket or zone_n < min_plays_per_bucket) \
            and current_team is not None and prior_participation_df is not None and prior_pbp_df is not None:
        prior_plays = _get_player_plays(prior_participation_df, prior_pbp_df)
        prior_plays = prior_plays[prior_plays["posteam"] == current_team]
        player_plays = pd.concat([player_plays, prior_plays])

    def _bucket_avg(coverage_type):
        bucket = player_plays[player_plays["defense_man_zone_type"] == coverage_type]
        n = len(bucket)
        if n < min_plays_per_bucket:
            return np.nan, n
        yards_col = "receiving_yards" if role == "receiver" else "passing_yards"
        return round(bucket[yards_col].mean(), 2), n

    man_avg, man_n = _bucket_avg("Man")
    zone_avg, zone_n = _bucket_avg("Zone")
    yards_col = "receiving_yards" if role == "receiver" else "passing_yards"
    overall_avg = round(player_plays[yards_col].mean(), 2) if len(player_plays) > 0 else np.nan

    return {
        "man_ypp": man_avg, "zone_ypp": zone_avg, "overall_ypp": overall_avg,
        "man_plays": man_n, "zone_plays": zone_n,
    }


def calc_coverage_adjusted_mu(base_mu: float, coverage_efficiency: dict,
                               opp_man_pct: float, opp_zone_pct: float,
                               max_adjustment: float = 0.2) -> float:
    """
    Actually ADJUSTS mu based on the player's real man/zone efficiency split
    and this week's specific opponent's man/zone tendency - not just a
    quality_score side signal, a real change to the projection itself.

    If either bucket lacks enough plays to trust (NaN from
    build_player_coverage_efficiency), falls back to base_mu unadjusted
    rather than react to a small, noisy sample.

    TIGHTENED per real 2025 backtest results: adjustment_direction_accuracy
    (did this adjustment move mu toward the real result more often than
    not) came back at 48.4% across 7,641 real rows - worse than a coinflip.
    min_plays_per_bucket raised from 8 to 14 (build_player_coverage_
    efficiency requires more real history before trusting a man/zone split
    enough to adjust mu at all) and max_adjustment tightened from 30% to
    20% (limits how much damage a still-imperfect signal can do even when
    it does fire). This limits the blast radius of a proven-unreliable
    mechanism; it is NOT a verified fix of whatever is actually causing the
    wrong-direction calls, since the data used to find this problem didn't
    include enough detail to diagnose the root cause. Re-run
    build_season_accuracy_report() on the same weeks after this change to
    see whether direction accuracy actually improves.
    """
    man_ypp = coverage_efficiency.get("man_ypp")
    zone_ypp = coverage_efficiency.get("zone_ypp")
    overall_ypp = coverage_efficiency.get("overall_ypp")

    if pd.isna(man_ypp) or pd.isna(zone_ypp) or pd.isna(overall_ypp) or overall_ypp == 0:
        return base_mu  # not enough real data to trust an adjustment

    expected_ypp_this_matchup = (opp_man_pct * man_ypp) + (opp_zone_pct * zone_ypp)
    multiplier = expected_ypp_this_matchup / overall_ypp
    multiplier = max(1 - max_adjustment, min(1 + max_adjustment, multiplier))

    return round(base_mu * multiplier, 2)


def get_opponent_this_week(team: str, season: int, week: int, schedules_df: pd.DataFrame) -> str:
    """
    Looks up who a team plays this week, using schedules_df's home_team/away_team.
    Returns None if the team has a bye or isn't found.
    """
    game = schedules_df[
        (schedules_df["season"] == season) & (schedules_df["week"] == week)
        & ((schedules_df["home_team"] == team) | (schedules_df["away_team"] == team))
    ]
    if game.empty:
        return None
    g = game.iloc[0]
    return g["away_team"] if g["home_team"] == team else g["home_team"]


def get_matchup_label(team: str, season: int, week: int, schedules_df: pd.DataFrame) -> str:
    """
    Returns the "AWAY @ HOME" label for whichever game this team plays in
    this week - used to group the slate by game (see build_week_games_list)
    rather than only by prop_type/position. Same lookup shape as
    get_opponent_this_week, so both teams in a game resolve to the
    identical label regardless of which side's row is being tagged.
    """
    game = schedules_df[
        (schedules_df["season"] == season) & (schedules_df["week"] == week)
        & ((schedules_df["home_team"] == team) | (schedules_df["away_team"] == team))
    ]
    if game.empty:
        return None
    g = game.iloc[0]
    return f"{g['away_team']} @ {g['home_team']}"


def build_week_games_list(season: int, week: int, schedules_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per real game this week - away_team, home_team, matchup label,
    and gameday (date) if present - used by the UI to render a game-by-
    game picker (mirrors a scoreboard/"Gamecast" list) rather than only a
    flat prop_type/position filter. Does NOT filter out preseason games
    itself (schedules_df's game_type column, if present, can be used by
    the caller to do that) - this function just lists whatever games
    schedules_df has for that season/week.
    """
    games = schedules_df[
        (schedules_df["season"] == season) & (schedules_df["week"] == week)
    ].copy()
    if games.empty:
        return pd.DataFrame(columns=["away_team", "home_team", "matchup"])
    games["matchup"] = games["away_team"] + " @ " + games["home_team"]
    cols = [c for c in ["away_team", "home_team", "matchup", "gameday", "game_type"] if c in games.columns]
    return games[cols].reset_index(drop=True)


def get_full_coverage_breakdown(coverage_row: dict) -> dict:
    """
    Returns the FULL individual coverage-type breakdown (Cover 1 %, Cover 2 %,
    Cover 3 %, Cover 4 %, Cover 6 %, etc. - whichever coverage labels actually
    appear in the charted data), not just the single dominant one. Each
    specific coverage type gets its own real percentage from
    build_coverage_profile(), e.g. "Cover 1: 19%, Cover 2: 17.5%" - this
    surfaces all of them, prefixed opp_cov_<type>_pct, so the full grading
    is visible, not just whichever one happens to be highest.
    """
    if not coverage_row:
        return {}
    excluded = {"defteam", "n_plays", "man_pct", "zone_pct"}
    return {
        f"opp_cov_{k.replace('_pct', '')}": v
        for k, v in coverage_row.items()
        if k.endswith("_pct") and k not in excluded and pd.notna(v)
    }


def get_player_grades(gsis_id: str, metrics_df: pd.DataFrame) -> dict:
    """
    Looks up a player's row in an advanced-metrics table (built by
    build_qb_advanced_metrics/build_receiver_advanced_metrics/
    build_rb_advanced_metrics) and returns only the *_grade columns plus
    their raw values, ready to merge into a scanner row.
    """
    if metrics_df is None or metrics_df.empty or "gsis_id" not in metrics_df.columns:
        return {}
    match = metrics_df[metrics_df["gsis_id"] == gsis_id]
    if match.empty:
        return {}
    row = match.iloc[0].to_dict()
    return {k: v for k, v in row.items() if k != "gsis_id" and pd.notna(v)}


def get_defense_grades(team: str, def_metrics_df: pd.DataFrame) -> dict:
    """Same idea as get_player_grades(), but for the defense-metrics table (keyed by defteam)."""
    if def_metrics_df is None or def_metrics_df.empty or "defteam" not in def_metrics_df.columns:
        return {}
    match = def_metrics_df[def_metrics_df["defteam"] == team]
    if match.empty:
        return {}
    row = match.iloc[0].to_dict()
    return {f"opp_{k}": v for k, v in row.items() if k != "defteam" and pd.notna(v)}


def calc_percentile_grade(value: float, comparison_series: pd.Series) -> float:
    """
    Generic 0-100 percentile grade for ANY metric against its league-wide
    distribution this season - one reusable function instead of hand-coded
    grading logic per stat, so every advanced metric gets the same
    consistent, color-codable treatment.
    """
    if pd.isna(value) or comparison_series.dropna().empty:
        return np.nan
    valid = comparison_series.dropna()
    return round((valid < value).mean() * 100, 1)


def build_qb_advanced_metrics(season: int, week: int, player_stats_df: pd.DataFrame,
                               ngs_pass_df: pd.DataFrame, participation_df: pd.DataFrame,
                               pbp_df: pd.DataFrame, pass_explosive_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    QB advanced metrics: EPA/play, CPOE, success rate, passer_rating, aDOT,
    aggressiveness, air-EPA vs YAC-EPA split, pressure rate faced, and
    (when pass_explosive_df is supplied - see build_explosive_rates())
    explosive_20plus_rate - the "big-play" tendency signal, distinct from
    aDOT (average depth of target): a QB can have a modest aDOT but still
    hit explosive gains at an above-average rate via scheme/YAC, or vice
    versa, so this isn't redundant with aDOT.
    Uses weeks BEFORE the target week only (same leak-avoidance as mu).
    Each metric gets a 0-100 percentile grade against this season's QBs.
    """
    hist_stats = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < week)
        & (player_stats_df["position"] == "QB")
    ]
    if hist_stats.empty:
        return pd.DataFrame()

    agg = hist_stats.groupby("gsis_id").agg(
        passing_epa=("passing_epa", "mean"),
        passing_yards=("passing_yards", "sum"),
        attempts=("attempts", "sum"),
    ).reset_index()

    hist_ngs = ngs_pass_df[(ngs_pass_df["season"] == season) & (ngs_pass_df["week"] < week)]
    ngs_agg = hist_ngs.groupby("player_gsis_id").agg(
        cpoe=("completion_percentage_above_expectation", "mean"),
        adot=("avg_intended_air_yards", "mean"),
        aggressiveness=("aggressiveness", "mean"),
        passer_rating=("passer_rating", "mean"),
    ).reset_index().rename(columns={"player_gsis_id": "gsis_id"})

    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week) & (pbp_df["play_type"] == "pass")]
    pbp_agg = hist_pbp.groupby("passer_player_id").agg(
        success_rate=("success", "mean"),
        air_epa=("air_epa", "mean"),
        yac_epa=("yac_epa", "mean"),
    ).reset_index().rename(columns={"passer_player_id": "gsis_id"})

    merged = agg.merge(ngs_agg, on="gsis_id", how="left").merge(pbp_agg, on="gsis_id", how="left")

    if pass_explosive_df is not None and not pass_explosive_df.empty:
        exp = pass_explosive_df.rename(columns={"passer_player_id": "gsis_id"})[
            ["gsis_id", "explosive_20plus_rate", "explosive_40plus_rate"]
        ]
        merged = merged.merge(exp, on="gsis_id", how="left")

    for col in ["passing_epa", "cpoe", "success_rate", "passer_rating", "adot", "aggressiveness",
                "explosive_20plus_rate"]:
        if col in merged.columns:
            merged[f"{col}_grade"] = merged[col].apply(lambda v: calc_percentile_grade(v, merged[col]))

    return merged


def build_receiver_advanced_metrics(season: int, week: int, player_stats_df: pd.DataFrame,
                                     ngs_rec_df: pd.DataFrame, rec_explosive_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    WR/TE advanced metrics: target_share, air_yards_share, wopr, racr,
    receiving_epa (season-aggregated, already in player_stats), separation,
    cushion, catch_percentage, YAC-over-expected, and (when rec_explosive_df
    is supplied - see build_explosive_rates()) explosive_15plus_rate - the
    "big-play" tendency signal.

    NOTE ON YAC/YPR: raw YAC and yards-per-reception are deliberately NOT
    added as separate metrics - yac_above_expectation (already here, from
    NGS) is a strictly better version of the same signal, since it's
    normalized against the specific depth/difficulty of each catch rather
    than being a raw counting number that a short-target slot receiver and
    a deep-threat receiver can't be fairly compared on. Adding raw YAC/YPR
    alongside it would be redundant, not additive.
    """
    hist_stats = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < week)
        & (player_stats_df["position"].isin(["WR", "TE", "RB"]))
    ]
    if hist_stats.empty:
        return pd.DataFrame()

    agg = hist_stats.groupby("gsis_id").agg(
        target_share=("target_share", "mean"),
        air_yards_share=("air_yards_share", "mean"),
        wopr=("wopr", "mean"),
        racr=("racr", "mean"),
        receiving_epa=("receiving_epa", "mean"),
    ).reset_index()

    hist_ngs = ngs_rec_df[(ngs_rec_df["season"] == season) & (ngs_rec_df["week"] < week)]
    ngs_agg = hist_ngs.groupby("player_gsis_id").agg(
        avg_separation=("avg_separation", "mean"),
        avg_cushion=("avg_cushion", "mean"),
        catch_percentage=("catch_percentage", "mean"),
        yac_above_expectation=("avg_yac_above_expectation", "mean"),
    ).reset_index().rename(columns={"player_gsis_id": "gsis_id"})

    merged = agg.merge(ngs_agg, on="gsis_id", how="left")

    if rec_explosive_df is not None and not rec_explosive_df.empty:
        exp = rec_explosive_df.rename(columns={"receiver_player_id": "gsis_id"})[
            ["gsis_id", "explosive_15plus_rate", "explosive_20plus_rate"]
        ]
        merged = merged.merge(exp, on="gsis_id", how="left")

    for col in ["target_share", "wopr", "racr", "receiving_epa", "avg_separation",
                "catch_percentage", "yac_above_expectation", "explosive_15plus_rate"]:
        if col in merged.columns:
            merged[f"{col}_grade"] = merged[col].apply(lambda v: calc_percentile_grade(v, merged[col]))

    return merged


def build_rb_advanced_metrics(season: int, week: int, player_stats_df: pd.DataFrame,
                               ngs_rush_df: pd.DataFrame, rush_explosive_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    RB advanced metrics: rushing_epa (season-aggregated), rush_yards_over_
    expected_per_att, efficiency, avg_time_to_los, percent_attempts_gte_
    eight_defenders (box rate faced), and (when rush_explosive_df is
    supplied - see build_explosive_rates()) explosive_10plus_rate - the
    "breakaway run" tendency signal, distinct from efficiency (average
    per-carry value): a between-the-tackles grinder can have strong
    efficiency with almost no explosive runs, or vice versa.
    """
    hist_stats = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < week)
        & (player_stats_df["position"] == "RB")
    ]
    if hist_stats.empty:
        return pd.DataFrame()

    agg = hist_stats.groupby("gsis_id").agg(
        rushing_epa=("rushing_epa", "mean"),
    ).reset_index()

    hist_ngs = ngs_rush_df[(ngs_rush_df["season"] == season) & (ngs_rush_df["week"] < week)]
    ngs_agg = hist_ngs.groupby("player_gsis_id").agg(
        rush_yards_over_expected_per_att=("rush_yards_over_expected_per_att", "mean"),
        efficiency=("efficiency", "mean"),
        avg_time_to_los=("avg_time_to_los", "mean"),
        box_stack_pct_faced=("percent_attempts_gte_eight_defenders", "mean"),
    ).reset_index().rename(columns={"player_gsis_id": "gsis_id"})

    merged = agg.merge(ngs_agg, on="gsis_id", how="left")

    if rush_explosive_df is not None and not rush_explosive_df.empty:
        exp = rush_explosive_df.rename(columns={"rusher_player_id": "gsis_id"})[
            ["gsis_id", "explosive_10plus_rate", "explosive_15plus_rate"]
        ]
        merged = merged.merge(exp, on="gsis_id", how="left")

    for col in ["rushing_epa", "rush_yards_over_expected_per_att", "efficiency", "explosive_10plus_rate"]:
        if col in merged.columns:
            merged[f"{col}_grade"] = merged[col].apply(lambda v: calc_percentile_grade(v, merged[col]))

    return merged


def build_defense_explosive_allowed(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    The defense-side counterpart to build_explosive_rates(): how often THIS
    defense allows an explosive gain, split pass vs run - the piece that
    was genuinely missing before (only pass/run EPA-allowed existed on
    defense; nothing captured big-play tendency specifically, which EPA's
    average can mask - a defense can have decent average EPA allowed while
    still bleeding a high rate of explosive plays that spike variance).
    Uses weeks BEFORE the target week only - caller is expected to pass an
    already-week-filtered pbp_df, same convention as build_defense_advanced_metrics.
    """
    if pbp_df.empty:
        return pd.DataFrame()

    pass_plays = pbp_df[pbp_df["play_type"] == "pass"]
    run_plays = pbp_df[pbp_df["play_type"] == "run"]

    pass_allowed = pass_plays.groupby("defteam").agg(
        pass_explosive_allowed_rate=("passing_yards", lambda x: (x >= 20).mean()),
    ).reset_index()
    run_allowed = run_plays.groupby("defteam").agg(
        run_explosive_allowed_rate=("rushing_yards", lambda x: (x >= 10).mean()),
    ).reset_index()

    merged = pass_allowed.merge(run_allowed, on="defteam", how="outer")
    for col in ["pass_explosive_allowed_rate", "run_explosive_allowed_rate"]:
        if col in merged.columns:
            # allowed metric: lower is better defensively, invert same as
            # every other *_allowed grade in this file.
            merged[f"{col}_grade"] = merged[col].apply(
                lambda v: 100 - calc_percentile_grade(v, merged[col]) if pd.notna(v) else np.nan
            )
    return merged


def build_defense_advanced_metrics(season: int, week: int, pbp_df: pd.DataFrame,
                                    participation_df: pd.DataFrame) -> pd.DataFrame:
    """
    DEF advanced metrics: EPA allowed per play, split pass defense vs run
    defense - this is the real free equivalent of DVOA (DVOA itself is
    Football Outsiders/FTN proprietary, not available for free). Also
    success rate allowed, pressure rate generated, and explosive-play-
    allowed rate (pass + run, via build_defense_explosive_allowed) - the
    big-play-specific signal EPA's average alone doesn't isolate.
    """
    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)]
    if hist_pbp.empty:
        return pd.DataFrame()

    pass_plays = hist_pbp[hist_pbp["play_type"] == "pass"]
    run_plays = hist_pbp[hist_pbp["play_type"] == "run"]

    pass_def = pass_plays.groupby("defteam").agg(
        pass_epa_allowed=("epa", "mean"),
        pass_success_rate_allowed=("success", "mean"),
    ).reset_index()
    run_def = run_plays.groupby("defteam").agg(
        run_epa_allowed=("epa", "mean"),
        run_success_rate_allowed=("success", "mean"),
    ).reset_index()

    merged = pass_def.merge(run_def, on="defteam", how="outer")

    hist_participation = participation_df.merge(
        hist_pbp[["game_id", "play_id", "defteam"]],
        left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner",
    )
    if "was_pressure" in hist_participation.columns:
        pressure_rate = hist_participation.groupby("defteam")["was_pressure"].mean().reset_index()
        pressure_rate.columns = ["defteam", "pressure_rate_generated"]
        merged = merged.merge(pressure_rate, on="defteam", how="left")

    explosive_allowed = build_defense_explosive_allowed(hist_pbp)
    if not explosive_allowed.empty:
        merged = merged.merge(explosive_allowed, on="defteam", how="left")

    for col in ["pass_epa_allowed", "run_epa_allowed", "pressure_rate_generated"]:
        if col in merged.columns:
            # NOTE: for *_allowed metrics, LOWER is better defensively, so
            # grade is inverted (100 - percentile) to keep "high grade = good defense"
            # consistent with how every other grade in this tool works.
            if "allowed" in col:
                merged[f"{col}_grade"] = merged[col].apply(
                    lambda v: 100 - calc_percentile_grade(v, merged[col]) if pd.notna(v) else np.nan
                )
            else:
                merged[f"{col}_grade"] = merged[col].apply(lambda v: calc_percentile_grade(v, merged[col]))

    return merged




def calc_coverage_quality_score(coverage_row: dict, coverage_profile_df: pd.DataFrame = None,
                                 percentile_threshold: float = 90.0) -> dict:
    """
    FIXED per feedback: previously only looked at the SINGLE highest raw
    coverage % (e.g. if Cover 1 was 24% and Cover 3 was 22%, only the 24%
    counted - the fact that Cover 3 was ALSO unusually high got ignored).
    Also previously had no real league-wide comparison - a team's own raw
    % was used directly, with no sense of whether that % was actually
    unusual relative to the rest of the league.

    NOW: computes each coverage type's REAL percentile rank against all
    32 teams (reusing calc_percentile_grade), identifies EVERY coverage
    type that's elevated (>= percentile_threshold, default 90th percentile
    = genuinely "top 10%" league-wide, not just locally high), and combines
    ALL of them - so a defense leaning hard on BOTH Cover 1 and Cover 3
    simultaneously now correctly registers as a stronger signal than either
    one alone, instead of only counting whichever is slightly higher.

    If coverage_profile_df isn't provided (or no coverage type clears the
    threshold), falls back to the single-highest-raw-% approach as before,
    so this degrades gracefully rather than losing signal entirely for
    defenses with no single extreme tendency.
    """
    if coverage_row is None:
        return {"dominant_coverage": None, "dominant_coverage_pct": np.nan,
                "man_zone_lean": None, "elevated_coverages": [], "exploit_strength": np.nan}

    coverage_type_cols = {
        k: v for k, v in coverage_row.items()
        if k.endswith("_pct") and k not in ("man_pct", "zone_pct") and pd.notna(v)
    }
    if not coverage_type_cols:
        return {"dominant_coverage": None, "dominant_coverage_pct": np.nan,
                "man_zone_lean": None, "elevated_coverages": [], "exploit_strength": np.nan}

    dominant_coverage = max(coverage_type_cols, key=coverage_type_cols.get)
    dominant_pct = coverage_type_cols[dominant_coverage]

    man_pct = coverage_row.get("man_pct", np.nan)
    zone_pct = coverage_row.get("zone_pct", np.nan)
    man_zone_lean = None
    if pd.notna(man_pct) and pd.notna(zone_pct):
        man_zone_lean = "Man-heavy" if man_pct > zone_pct else "Zone-heavy"

    elevated = []
    if coverage_profile_df is not None and not coverage_profile_df.empty:
        for cov_type, own_pct in coverage_type_cols.items():
            if cov_type not in coverage_profile_df.columns:
                continue
            league_percentile = calc_percentile_grade(own_pct, coverage_profile_df[cov_type])
            if pd.notna(league_percentile) and league_percentile >= percentile_threshold:
                elevated.append({"coverage_type": cov_type, "own_pct": own_pct, "league_percentile": league_percentile})

    if elevated:
        # combine ALL elevated coverage types, not just the single max
        exploit_strength = sum(e["league_percentile"] for e in elevated) / len(elevated) / 100
    else:
        # graceful fallback: no coverage type is genuinely league-extreme,
        # use the old single-highest-raw-% signal (weaker, but not zero)
        exploit_strength = dominant_pct

    return {
        "dominant_coverage": dominant_coverage,
        "dominant_coverage_pct": dominant_pct,
        "man_zone_lean": man_zone_lean,
        "elevated_coverages": elevated,
        "num_elevated_coverages": len(elevated),
        "exploit_strength": exploit_strength,
    }


# ---------------------------------------------------------------------------
# 6b. BOX-COUNT STRUCTURAL EXPLOIT + REAL RUSH-SPLIT MU ADJUSTMENT
#     (run-game equivalent of the coverage-exploit / coverage-adjusted-mu
#     pair above - same elevated-percentile logic, same real per-player
#     efficiency split, same capped mu adjustment)
# ---------------------------------------------------------------------------

def calc_box_quality_score(box_row: dict, box_profile_df: pd.DataFrame = None,
                            percentile_threshold: float = 90.0) -> dict:
    """
    Same elevated-percentile approach as calc_coverage_quality_score(), but
    for stacked-box rate (pct_stacked_7plus from build_box_count_profile).

    Directionally the OPPOSITE of coverage: coverage's exploit_strength
    rewards a specific player's profile matching an elevated tendency,
    but a genuinely league-extreme box-stack rate is a suppressing signal
    for run volume/efficiency in general, so exploit_strength is inverted
    here (elevated stacking -> LOWER exploit_strength, tougher matchup).
    """
    if not box_row:
        return {"box_stack_pct": np.nan, "box_elevated": False, "exploit_strength": np.nan}

    stack_pct = box_row.get("pct_stacked_7plus", np.nan)
    if pd.isna(stack_pct):
        return {"box_stack_pct": np.nan, "box_elevated": False, "exploit_strength": np.nan}

    league_percentile = np.nan
    elevated = False
    if box_profile_df is not None and not box_profile_df.empty and "pct_stacked_7plus" in box_profile_df.columns:
        league_percentile = calc_percentile_grade(stack_pct, box_profile_df["pct_stacked_7plus"])
        elevated = pd.notna(league_percentile) and league_percentile >= percentile_threshold

    exploit_strength = (1 - (league_percentile / 100)) if pd.notna(league_percentile) else (1 - stack_pct)
    return {
        "box_stack_pct": stack_pct,
        "box_elevated": elevated,
        "league_percentile": league_percentile,
        "exploit_strength": round(exploit_strength, 3),
    }


def build_player_rush_box_efficiency(player_gsis_id: str, season: int,
                                      ftn_df: pd.DataFrame, pbp_df: pd.DataFrame,
                                      min_plays_per_bucket: int = 14, current_team: str = None,
                                      prior_ftn_df: pd.DataFrame = None,
                                      prior_pbp_df: pd.DataFrame = None) -> dict:
    """
    Same approach as build_player_coverage_efficiency() (man/zone), applied
    to box counts: this RB's REAL rushing-yards-per-carry against a light
    box (<7 defenders) vs a stacked box (7+), joining ftn_df's
    n_defense_box to pbp_df on (game_id, play_id) - same join fix used
    everywhere else in this file for ftn/participation data. Same
    cross-season, team-filtered top-up as the coverage version (a traded
    player correctly gets nothing from a prior team's plays), and the same
    min_plays_per_bucket safety net (NaN bucket if too small a sample).
    """
    def _get_player_plays(ftn_source_df, pbp_source_df):
        merged = ftn_source_df.merge(
            pbp_source_df[["game_id", "play_id", "defteam", "posteam",
                            "rusher_player_id", "rushing_yards"]],
            left_on=["nflverse_game_id", "nflverse_play_id"],
            right_on=["game_id", "play_id"],
            how="left",
        )
        return merged[
            (merged["rusher_player_id"] == player_gsis_id) & merged["n_defense_box"].notna()
        ].copy()

    player_plays = _get_player_plays(ftn_df, pbp_df)
    if not player_plays.empty:
        player_plays["box_bucket"] = np.where(player_plays["n_defense_box"] >= 7, "stacked", "light")
    else:
        player_plays["box_bucket"] = pd.Series(dtype=object)

    if current_team is not None and prior_ftn_df is not None and prior_pbp_df is not None:
        light_n = len(player_plays[player_plays["box_bucket"] == "light"])
        stacked_n = len(player_plays[player_plays["box_bucket"] == "stacked"])
        if light_n < min_plays_per_bucket or stacked_n < min_plays_per_bucket:
            prior_plays = _get_player_plays(prior_ftn_df, prior_pbp_df)
            prior_plays = prior_plays[prior_plays["posteam"] == current_team].copy()
            if not prior_plays.empty:
                prior_plays["box_bucket"] = np.where(prior_plays["n_defense_box"] >= 7, "stacked", "light")
            player_plays = pd.concat([player_plays, prior_plays])

    def _bucket_avg(bucket):
        sub = player_plays[player_plays["box_bucket"] == bucket]
        n = len(sub)
        if n < min_plays_per_bucket:
            return np.nan, n
        return round(sub["rushing_yards"].mean(), 2), n

    light_avg, light_n = _bucket_avg("light")
    stacked_avg, stacked_n = _bucket_avg("stacked")
    overall_avg = round(player_plays["rushing_yards"].mean(), 2) if len(player_plays) > 0 else np.nan

    return {
        "light_box_ypc": light_avg, "stacked_box_ypc": stacked_avg,
        "overall_ypc": overall_avg, "light_plays": light_n, "stacked_plays": stacked_n,
    }


def calc_box_adjusted_mu(base_mu: float, box_efficiency: dict, opp_stacked_pct: float,
                          max_adjustment: float = 0.2) -> float:
    """
    Real mu adjustment (not just a quality_score side signal), same shape
    as calc_coverage_adjusted_mu(): uses this RB's own real light-vs-stacked
    box yards-per-carry split, weighted by THIS week's specific opponent's
    stacked-box rate, capped at +/- max_adjustment. Falls back to base_mu
    unadjusted if either bucket lacks a trustworthy sample.
    """
    light_ypc = box_efficiency.get("light_box_ypc")
    stacked_ypc = box_efficiency.get("stacked_box_ypc")
    overall_ypc = box_efficiency.get("overall_ypc")

    if (pd.isna(light_ypc) or pd.isna(stacked_ypc) or pd.isna(overall_ypc)
            or overall_ypc == 0 or pd.isna(opp_stacked_pct)):
        return base_mu

    expected_ypc_this_matchup = (opp_stacked_pct * stacked_ypc) + ((1 - opp_stacked_pct) * light_ypc)
    multiplier = expected_ypc_this_matchup / overall_ypc
    multiplier = max(1 - max_adjustment, min(1 + max_adjustment, multiplier))
    return round(base_mu * multiplier, 2)


# ---------------------------------------------------------------------------
# 6c. GRADE-BASED MATCHUP CROSSWALK - the NFL equivalent of the MLB tool's
#     pitch-type-usage x hitter-vulnerability crosswalk. Each prop gets its
#     OWN tailored offense-grade / defense-grade list (same "per-prop
#     tailored quality_score" fix already applied on the MLB side, where a
#     single reused composite score was the confirmed bug).
# ---------------------------------------------------------------------------

PROP_METRIC_CROSSWALK = {
    "pass_yards": {
        "offense_grades": ["passing_epa_grade", "cpoe_grade", "success_rate_grade", "adot_grade",
                            "explosive_20plus_rate_grade", "pa_rate_grade", "pa_epa_diff_grade",
                            "pressure_rate_faced_grade", "proe_grade"],
        "defense_grades": ["opp_pass_epa_allowed_grade", "opp_pressure_rate_generated_grade",
                            "opp_pass_explosive_allowed_rate_grade", "opp_pa_epa_allowed_grade"],
    },
    "rush_yards": {
        "offense_grades": ["rushing_epa_grade", "rush_yards_over_expected_per_att_grade", "efficiency_grade",
                            "explosive_10plus_rate_grade"],
        "defense_grades": ["opp_run_epa_allowed_grade", "opp_run_explosive_allowed_rate_grade"],
    },
    "rec_yards": {
        "offense_grades": ["target_share_grade", "wopr_grade", "receiving_epa_grade",
                            "avg_separation_grade", "yac_above_expectation_grade",
                            "explosive_15plus_rate_grade"],
        "defense_grades": ["opp_pass_epa_allowed_grade", "opp_pressure_rate_generated_grade",
                            "opp_pass_explosive_allowed_rate_grade", "opp_pa_epa_allowed_grade"],
    },
}


def calc_grade_matchup_strength(row: dict, prop_type: str, offense_weight: float = 0.5) -> float:
    """
    Averages whichever of this prop's tailored offense grades are present
    on `row` (own-skill signal, 0-100) and whichever tailored defense
    grades are present (already inverted upstream so high = good defense),
    then combines into a single 0-1 exploit signal: player's own grade UP
    and defense's allowed-grade DOWN (bad defense = more exploitable) both
    push this higher.

    Missing individual metrics are skipped rather than treated as 0 - the
    average is over whatever's actually available (a rookie with no NGS
    separation data yet still gets a signal from his other grades), same
    graceful-degrade pattern used throughout this file. Returns np.nan only
    if NEITHER side has anything available, so the caller can fall back to
    the structural-only signal.
    """
    spec = PROP_METRIC_CROSSWALK.get(prop_type)
    if spec is None:
        return np.nan

    offense_vals = [row.get(k) for k in spec["offense_grades"] if pd.notna(row.get(k))]
    defense_vals = [row.get(k) for k in spec["defense_grades"] if pd.notna(row.get(k))]

    if not offense_vals and not defense_vals:
        return np.nan

    offense_component = (sum(offense_vals) / len(offense_vals) / 100) if offense_vals else np.nan
    defense_component = (1 - (sum(defense_vals) / len(defense_vals) / 100)) if defense_vals else np.nan

    if pd.isna(offense_component):
        return round(defense_component, 3)
    if pd.isna(defense_component):
        return round(offense_component, 3)
    return round(offense_component * offense_weight + defense_component * (1 - offense_weight), 3)


# ---------------------------------------------------------------------------
# 6d. ROLE/USAGE TREND VERIFICATION - the NFL equivalent of the MLB tool's
#     lineup_verification_score() (checking whether TONIGHT'S real role/
#     lineup context backs up a player's season-long profile, not just
#     trusting the season average blindly), blended 60/40 with the
#     structural + grade matchup signal above.
# ---------------------------------------------------------------------------

def build_role_trend(gsis_id: str, metric_col: str, source_df: pd.DataFrame, id_col: str,
                      season: int, week: int, recent_games: int = 3) -> dict:
    """
    Compares a player's recent (last `recent_games`, weeks < target week)
    usage metric against their full-season average over that same window -
    the NFL analog of MLB's real-lineup check, built from data this file
    already reliably pulls rather than snap_counts (see note below).

    NOTE ON SNAP COUNTS: pull_snap_counts() exists in this file but is
    deliberately NOT used for role verification. nflverse's snap_counts
    table keys players on `pfr_player_id`, a DIFFERENT id system than the
    `gsis_id` used consistently everywhere else here (NGS, rosters, depth
    charts, player_stats after the rename at the top of this file). There's
    no verified gsis_id<->pfr_player_id crosswalk wired into this codebase,
    so joining snap_counts in here would risk a silent bad join - same
    failure category as the id-mismatch bugs already caught and fixed
    elsewhere in this file. target_share (player_stats) and rush_attempts
    (NGS rushing) are used instead - both confirmed to key on gsis_id.
    """
    hist = source_df[
        (source_df["season"] == season) & (source_df["week"] < week)
        & (source_df[id_col] == gsis_id)
    ].sort_values("week", ascending=False)
    if hist.empty or metric_col not in hist.columns:
        return {"recent_value": np.nan, "season_value": np.nan, "trend_ratio": np.nan, "games": 0}

    recent = hist.head(recent_games)[metric_col].mean()
    season_avg = hist[metric_col].mean()
    trend_ratio = np.nan
    if pd.notna(recent) and pd.notna(season_avg) and season_avg > 0:
        trend_ratio = recent / season_avg

    return {
        "recent_value": round(recent, 3) if pd.notna(recent) else np.nan,
        "season_value": round(season_avg, 3) if pd.notna(season_avg) else np.nan,
        "trend_ratio": round(trend_ratio, 3) if pd.notna(trend_ratio) else np.nan,
        "games": len(hist),
    }


def calc_role_verification_score(role_trend: dict, min_games: int = 2) -> float:
    """
    Converts a role trend dict into a 0-1 score: a steady/growing role
    (trend_ratio >= 1.0) scores highest, a fading role (<=0.5x season
    average) scores lowest, linear between. Returns a neutral 0.5 (no
    penalty, no bonus) if there isn't enough history to trust the trend
    yet - same graceful-degrade shape as calc_coverage_quality_score's
    fallback, so a rookie/new-role player isn't punished for thin data.
    """
    if role_trend.get("games", 0) < min_games or pd.isna(role_trend.get("trend_ratio")):
        return 0.5
    ratio = role_trend["trend_ratio"]
    return round(max(0.0, min(1.0, (ratio - 0.5) / 0.5)), 3)


def calc_blended_matchup_strength(structural_exploit: float, grade_exploit: float,
                                   role_verification_score: float,
                                   structural_weight: float = 0.5,
                                   matchup_weight: float = 0.35) -> float:
    """
    Combines the structural tendency signal (coverage-elevation or
    box-count exploit strength, 0-1) with the grade-based crosswalk signal
    (calc_grade_matchup_strength, 0-1) into one matchup signal, then blends
    that with the role-verification score.

    REWEIGHTED per real 2025 backtest results (build_season_accuracy_report):
    role_verification_score showed a real, strong effect (fading-role rows
    missed by ~37.6 yards on average vs ~21.5 for steady/growing-role rows -
    nearly 2x), while adjustment_direction_accuracy (whether the coverage/
    box-driven mu adjustment moved toward the real result) came back at
    48.4% across 7,641 real rows - WORSE than a coinflip. That's real
    evidence the structural+grade matchup_signal isn't earning the 60%
    weight it originally had, and role_verification_score deserves more
    than its original 40%. matchup_weight now defaults to 0.35 (was 0.6),
    role_verification gets the remaining 0.65 (was 0.4).

    NOT a claim that the root cause inside the coverage/box logic itself
    has been found and fixed - the backtest export used to find this
    didn't include mu_before_coverage_adj/mu_before_box_adj, so WHY the
    adjustment is wrong that often isn't diagnosed yet, only THAT it is.
    This reweighting is a data-justified damage-limitation move (trust the
    proven signal more, the unproven/underperforming one less), not a
    verified root-cause fix. Re-run build_season_accuracy_report on the
    same week range after this change to see whether it actually helped.

    Degrades gracefully: a missing structural or grade component just
    reweights across whatever IS available; a completely absent matchup
    signal falls back to neutral (0.5) rather than zeroing the whole score
    out.
    """
    parts = [(structural_exploit, structural_weight), (grade_exploit, 1 - structural_weight)]
    valid = [(v, w) for v, w in parts if pd.notna(v)]
    if valid:
        total_w = sum(w for _, w in valid)
        matchup_signal = sum(v * w for v, w in valid) / total_w
    else:
        matchup_signal = 0.5

    if pd.isna(role_verification_score):
        return round(matchup_signal, 3)
    return round(matchup_signal * matchup_weight + role_verification_score * (1 - matchup_weight), 3)


def build_weekly_slate(season: int, week: int) -> pd.DataFrame:
    """
    Pulls and merges every data source needed for one week's slate, returning
    a single player-level DataFrame with mu inputs for every prop type ready
    to score. This does NOT include lines - lines are entered/adjusted
    manually per row in the Streamlit UI, same as the MLB tool's adjustable
    Best Edges table (avoids repeating the unreliable Underdog auto-pull
    issue; PrizePicks auto-pull can be tested later once this core scanner
    is proven out).

    Returns columns including (not exhaustive):
      gsis_id, player_display_name, team, position, prop_type,
      mu, sigma_estimate, quality_score, games_sampled,
      team_changed, use_depth_chart_estimate
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
    fallback_sigmas = build_league_fallback_sigmas(player_stats_df, season, week)
    fallback_mus = build_league_fallback_mus(player_stats_df, season, week)

    # Filter to weeks BEFORE the target week only, for the same reason
    # calc_prop_mu does - using this week's own plays to predict this
    # week's own result would be data leakage, not a real projection.
    pbp_history_df = pbp_df[pbp_df["week"] < week]

    # BUGFIX: explosive_rates was previously computed from the full-season
    # pbp_df (including the target week itself and every week after it) -
    # genuine data leakage, same category as the leak calc_prop_mu already
    # guards against. Now built from pbp_history_df, same as every other
    # weeks-before-target computation in this file.
    explosive_rates = build_explosive_rates(pbp_history_df)

    # Prior-season pulls for the cross-season, team-filtered coverage/box
    # efficiency fallbacks (build_player_coverage_efficiency,
    # build_player_rush_box_efficiency) - lets players who DIDN'T change
    # teams use last season's plays for a much better sample early in a
    # new season, while still correctly excluding a traded player's
    # old-team plays.
    prior_participation_df = pull_participation([season - 1])
    prior_pbp_df = pull_pbp([season - 1])
    prior_ftn_df = pull_ftn_charting([season - 1])

    # Collects each player's quality_score(s) across pass/rush/rec rows so
    # the fantasy_points row below can average them, the same way the MLB
    # tool's Fantasy quality_score averages its underlying prop scores.
    quality_scores_by_gsis: dict = {}

    def _record_quality_score(gsis_id, score):
        if pd.notna(score):
            quality_scores_by_gsis.setdefault(gsis_id, []).append(score)

    # Advanced metrics tables - computed once per scan, merged into each
    # position's rows below. Each metric gets a 0-100 percentile grade
    # against this season's league-wide distribution (calc_percentile_grade),
    # so everything is color-codable the same consistent way. Explosive-play
    # rate tables (per-player big-play tendency, per-defense big-play-
    # allowed tendency) are merged in here too - see build_explosive_rates()
    # / build_defense_explosive_allowed().
    qb_metrics = build_qb_advanced_metrics(
        season, week, player_stats_df, ngs_pass_df, participation_df, pbp_history_df,
        pass_explosive_df=explosive_rates["pass_explosive"],
    )
    rec_metrics = build_receiver_advanced_metrics(
        season, week, player_stats_df, ngs_rec_df,
        rec_explosive_df=explosive_rates["rec_explosive"],
    )
    rb_metrics = build_rb_advanced_metrics(
        season, week, player_stats_df, ngs_rush_df,
        rush_explosive_df=explosive_rates["rush_explosive"],
    )
    def_metrics = build_defense_advanced_metrics(season, week, pbp_history_df, participation_df)

    # Play-action tendency/vulnerability, QB pressure profile, and PROE -
    # closes the previously-flagged gaps (FTN's is_play_action/is_motion
    # sat unused, QB had no own-side pressure metric to pair against the
    # defense's, PROE wasn't built). qb_pa_profile/qb_pressure_profile/
    # proe_profile merge into qb_metrics by gsis_id/team so they ride along
    # with get_player_grades() automatically; def_pa_profile merges into
    # def_metrics by defteam the same way. coverage_pa_crosswalk is used
    # directly per-matchup below (dominant-coverage-specific, not a static
    # per-team column).
    qb_pa_profile = build_qb_playaction_profile(season, week, pbp_history_df, ftn_df)
    def_pa_profile = build_defense_playaction_allowed(season, week, pbp_history_df, ftn_df)
    coverage_pa_crosswalk = build_coverage_playaction_crosswalk(season, week, participation_df, ftn_df, pbp_history_df)
    qb_pressure_profile = build_qb_pressure_profile(season, week, participation_df, pbp_history_df)
    proe_profile = build_proe_profile(season, week, pbp_history_df)
    offense_personnel_tendency = build_offense_personnel_tendency(season, week, participation_df, pbp_history_df)
    defense_personnel_allowed = build_defense_personnel_allowed(season, week, participation_df, pbp_history_df)

    if not qb_pa_profile.empty and not qb_metrics.empty:
        qb_metrics = qb_metrics.merge(
            qb_pa_profile[["gsis_id", "pa_rate", "pa_epa_diff", "pa_rate_grade", "pa_epa_diff_grade"]],
            on="gsis_id", how="left",
        )
    if not qb_pressure_profile.empty and not qb_metrics.empty:
        qb_metrics = qb_metrics.merge(
            qb_pressure_profile[["gsis_id", "pressure_rate_faced", "pressure_rate_faced_grade"]],
            on="gsis_id", how="left",
        )
    if not def_pa_profile.empty and not def_metrics.empty:
        def_metrics = def_metrics.merge(
            def_pa_profile[["defteam", "pa_epa_allowed", "pa_vulnerability_gap", "pa_epa_allowed_grade"]],
            on="defteam", how="left",
        )

    this_week_games = schedules_df[
        (schedules_df["season"] == season) & (schedules_df["week"] == week)
    ]
    teams_this_week = pd.concat([
        this_week_games["home_team"], this_week_games["away_team"]
    ]).unique().tolist()

    # Team -> "AWAY @ HOME" matchup label, precomputed once so every row
    # below can tag itself with a simple dict lookup instead of a fresh
    # schedules_df filter per player - lets the UI group/filter the slate
    # game-by-game (see build_week_games_list) instead of only by
    # prop_type/position.
    week_games = build_week_games_list(season, week, schedules_df)
    team_to_matchup = {}
    for _, g in week_games.iterrows():
        team_to_matchup[g["away_team"]] = g["matchup"]
        team_to_matchup[g["home_team"]] = g["matchup"]

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
        team = qb.get("team")
        mu = calc_prop_mu(
            gsis_id, "passing_yards", player_stats_df, season, week, current_team=team,
            league_fallback_mu=fallback_mus.get(("QB", "passing_yards")),
        )
        sigma = calc_player_sigma(
            gsis_id, "passing_yards", player_stats_df, season, week, current_team=team,
            league_fallback_sigma=fallback_sigmas.get(("QB", "passing_yards")),
        )

        opponent = get_opponent_this_week(team, season, week, schedules_df)
        opp_coverage_row = None
        if opponent is not None and not coverage_profile.empty:
            match = coverage_profile[coverage_profile["defteam"] == opponent]
            if not match.empty:
                opp_coverage_row = match.iloc[0].to_dict()
        coverage_info = calc_coverage_quality_score(opp_coverage_row, coverage_profile)
        n_plays = opp_coverage_row.get("n_plays", 0) if opp_coverage_row else 0

        # Play-action exploit: does THIS QB run PA often and perform well
        # in it, AND is the opponent (specifically in whichever coverage
        # they lean on most - falls back to their overall PA-allowed
        # number if that coverage lacks a PA-specific sample) actually
        # vulnerable to it. Averaged with the structural coverage-elevation
        # signal above into one combined structural component, rather than
        # replacing it - both are real, separate tendency signals.
        qb_pa_row = qb_pa_profile[qb_pa_profile["gsis_id"] == gsis_id]
        qb_pa_row = qb_pa_row.iloc[0].to_dict() if not qb_pa_row.empty else {}
        def_pa_row = def_pa_profile[def_pa_profile["defteam"] == opponent]
        def_pa_row = def_pa_row.iloc[0].to_dict() if not def_pa_row.empty else {}
        playaction_info = calc_playaction_exploit_strength(
            qb_pa_row, def_pa_row, coverage_pa_crosswalk, opponent, coverage_info.get("dominant_coverage")
        )
        structural_parts = [v for v in [coverage_info.get("exploit_strength"), playaction_info.get("exploit_strength")] if pd.notna(v)]
        combined_structural_exploit = (sum(structural_parts) / len(structural_parts)) if structural_parts else np.nan

        # ACTUAL mu adjustment (not just a quality_score side signal) using
        # this QB's own real man/zone efficiency split from their play
        # history, weighted by this specific opponent's man/zone tendency.
        adjusted_mu = mu
        if pd.notna(mu) and opp_coverage_row:
            man_pct = opp_coverage_row.get("man_pct")
            zone_pct = opp_coverage_row.get("zone_pct")
            if pd.notna(man_pct) and pd.notna(zone_pct):
                coverage_eff = build_player_coverage_efficiency(
                    gsis_id, "passer", season, participation_df, pbp_history_df,
                    current_team=team, prior_participation_df=prior_participation_df,
                    prior_pbp_df=prior_pbp_df,
                )
                adjusted_mu = calc_coverage_adjusted_mu(mu, coverage_eff, man_pct, zone_pct)

        confidence_info = get_data_confidence(gsis_id, player_stats_df, season, week, current_team=team)
        own_grades = get_player_grades(gsis_id, qb_metrics)
        def_grades = get_defense_grades(opponent, def_metrics)

        # PROE is team-level (posteam), not per-player, so it doesn't ride
        # along with get_player_grades() the way gsis_id-keyed metrics do -
        # looked up by team and merged in directly here.
        if not proe_profile.empty:
            team_proe = proe_profile[proe_profile["posteam"] == team]
            if not team_proe.empty:
                own_grades["proe_grade"] = team_proe.iloc[0].get("proe_grade")
                own_grades["proe"] = team_proe.iloc[0].get("proe")

        # Grade-based crosswalk (own skill grades vs opponent's allowed
        # grades, tailored to pass_yards - see PROP_METRIC_CROSSWALK) and
        # real-role verification (recent vs season pass-attempt volume),
        # blended with the combined structural (coverage + play-action)
        # exploit signal above - mirrors the MLB tool's pitch-crosswalk +
        # lineup_verification blend.
        grade_exploit = calc_grade_matchup_strength({**own_grades, **def_grades}, "pass_yards")
        role_trend = build_role_trend(gsis_id, "attempts", ngs_pass_df, "player_gsis_id", season, week)
        role_score = calc_role_verification_score(role_trend)
        blended_exploit = calc_blended_matchup_strength(
            combined_structural_exploit, grade_exploit, role_score
        )
        quality_score = calc_quality_score(
            matchup_exploit_strength=blended_exploit,
            sample_size_games=min(n_plays / 60, 10),  # rough plays-to-games conversion
            coverage_confidence=min(n_plays / 300, 1.0),
        )
        _record_quality_score(gsis_id, quality_score)

        rows.append({
            "gsis_id": gsis_id, "player_display_name": qb.get("full_name"),
            "team": team, "position": "QB", "prop_type": "pass_yards",
            "matchup": team_to_matchup.get(team),
            "mu": adjusted_mu, "mu_before_coverage_adj": mu, "sigma": sigma, "opponent": opponent,
            "opp_man_pct": opp_coverage_row.get("man_pct") if opp_coverage_row else np.nan,
            "opp_zone_pct": opp_coverage_row.get("zone_pct") if opp_coverage_row else np.nan,
            "opp_dominant_coverage": coverage_info["dominant_coverage"],
            "opp_dominant_coverage_pct": coverage_info["dominant_coverage_pct"],
            "opp_num_elevated_coverages": coverage_info.get("num_elevated_coverages", 0),
            "playaction_exploit_strength": playaction_info.get("exploit_strength"),
            "playaction_used_coverage_specific_data": playaction_info.get("used_coverage_specific_playaction_data"),
            "quality_score": quality_score,
            "grade_matchup_strength": grade_exploit,
            "role_verification_score": role_score,
            "role_trend_ratio": role_trend.get("trend_ratio"),
            "data_confidence": confidence_info["data_confidence"],
            "games_sampled_current": confidence_info["games_sampled_current"],
            **get_full_coverage_breakdown(opp_coverage_row),
            **own_grades,
            **def_grades,
        })

    # --- Rushing props ---
    rush_pool = week_rosters[week_rosters["position"].isin(["RB", "QB"])]
    for _, rb in rush_pool.iterrows():
        gsis_id = rb.get("gsis_id")
        position = rb.get("position")
        rb_team = rb.get("team")
        mu = calc_prop_mu(
            gsis_id, "rushing_yards", player_stats_df, season, week, current_team=rb_team,
            league_fallback_mu=fallback_mus.get((position, "rushing_yards")),
        )
        sigma = calc_player_sigma(
            gsis_id, "rushing_yards", player_stats_df, season, week, current_team=rb_team,
            league_fallback_sigma=fallback_sigmas.get((position, "rushing_yards")),
        )
        if pd.notna(mu):  # skip QBs/RBs with no real rushing history at all
            rb_opponent = get_opponent_this_week(rb_team, season, week, schedules_df)
            opp_box_row = None
            if rb_opponent is not None and not box_def_profile.empty:
                match = box_def_profile[box_def_profile["defteam"] == rb_opponent]
                if not match.empty:
                    opp_box_row = match.iloc[0].to_dict()
            box_info = calc_box_quality_score(opp_box_row, box_def_profile)
            n_box_plays = opp_box_row.get("n_plays", 0) if opp_box_row else 0

            # ACTUAL mu adjustment using this RB's own real light-vs-stacked
            # box yards-per-carry split, weighted by this week's opponent's
            # stacked-box rate - run-game equivalent of the QB/WR coverage
            # adjustment above.
            adjusted_rush_mu = mu
            if opp_box_row and pd.notna(box_info.get("box_stack_pct")):
                box_eff = build_player_rush_box_efficiency(
                    gsis_id, season, ftn_df, pbp_history_df,
                    current_team=rb_team, prior_ftn_df=prior_ftn_df, prior_pbp_df=prior_pbp_df,
                )
                adjusted_rush_mu = calc_box_adjusted_mu(mu, box_eff, box_info.get("box_stack_pct"))

            rb_confidence_info = get_data_confidence(gsis_id, player_stats_df, season, week, current_team=rb_team)
            own_grades = get_player_grades(gsis_id, rb_metrics)
            def_grades = get_defense_grades(rb_opponent, def_metrics)

            grade_exploit = calc_grade_matchup_strength({**own_grades, **def_grades}, "rush_yards")
            role_trend = build_role_trend(gsis_id, "rush_attempts", ngs_rush_df, "player_gsis_id", season, week)
            role_score = calc_role_verification_score(role_trend)
            blended_exploit = calc_blended_matchup_strength(
                box_info.get("exploit_strength"), grade_exploit, role_score
            )
            rush_quality_score = calc_quality_score(
                matchup_exploit_strength=blended_exploit,
                sample_size_games=min(n_box_plays / 60, 10),
                coverage_confidence=min(n_box_plays / 300, 1.0),
            )
            _record_quality_score(gsis_id, rush_quality_score)

            rows.append({
                "gsis_id": gsis_id, "player_display_name": rb.get("full_name"),
                "team": rb.get("team"), "position": position, "prop_type": "rush_yards",
                "matchup": team_to_matchup.get(rb_team),
                "mu": adjusted_rush_mu, "mu_before_box_adj": mu, "sigma": sigma, "opponent": rb_opponent,
                "opp_box_stack_pct": box_info.get("box_stack_pct"),
                "opp_box_elevated": box_info.get("box_elevated"),
                "quality_score": rush_quality_score,
                "grade_matchup_strength": grade_exploit,
                "role_verification_score": role_score,
                "role_trend_ratio": role_trend.get("trend_ratio"),
                "data_confidence": rb_confidence_info["data_confidence"],
                "games_sampled_current": rb_confidence_info["games_sampled_current"],
                **own_grades,
                **def_grades,
            })

    # --- Receiving props ---
    rec_pool = week_rosters[week_rosters["position"].isin(["WR", "TE", "RB"])]
    for _, wr in rec_pool.iterrows():
        gsis_id = wr.get("gsis_id")
        position = wr.get("position")
        team = wr.get("team")
        mu = calc_prop_mu(
            gsis_id, "receiving_yards", player_stats_df, season, week, current_team=team,
            league_fallback_mu=fallback_mus.get((position, "receiving_yards")),
        )
        sigma = calc_player_sigma(
            gsis_id, "receiving_yards", player_stats_df, season, week, current_team=team,
            league_fallback_sigma=fallback_sigmas.get((position, "receiving_yards")),
        )
        if pd.notna(mu):
            opponent = get_opponent_this_week(team, season, week, schedules_df)
            opp_coverage_row = None
            if opponent is not None and not coverage_profile.empty:
                match = coverage_profile[coverage_profile["defteam"] == opponent]
                if not match.empty:
                    opp_coverage_row = match.iloc[0].to_dict()
            coverage_info = calc_coverage_quality_score(opp_coverage_row, coverage_profile)
            n_plays = opp_coverage_row.get("n_plays", 0) if opp_coverage_row else 0

            # Personnel-grouping exploit: does this team's dominant
            # personnel package (11/12/21 etc.) match up against a real
            # weakness in THIS specific opponent's defense against that
            # exact grouping - same crosswalk pattern as play-action for
            # pass_yards, applied to personnel here since it's the more
            # directly relevant tendency signal for receiving props.
            personnel_info = calc_personnel_exploit_strength(
                team, offense_personnel_tendency, opponent, defense_personnel_allowed
            )
            structural_parts = [v for v in [coverage_info.get("exploit_strength"), personnel_info.get("exploit_strength")] if pd.notna(v)]
            combined_structural_exploit = (sum(structural_parts) / len(structural_parts)) if structural_parts else np.nan

            # ACTUAL mu adjustment using this receiver's own real man/zone
            # efficiency split, weighted by this specific opponent's tendency.
            adjusted_mu = mu
            man_pct = opp_coverage_row.get("man_pct") if opp_coverage_row else None
            zone_pct = opp_coverage_row.get("zone_pct") if opp_coverage_row else None
            if pd.notna(man_pct) and pd.notna(zone_pct):
                coverage_eff = build_player_coverage_efficiency(
                    gsis_id, "receiver", season, participation_df, pbp_history_df,
                    current_team=team, prior_participation_df=prior_participation_df,
                    prior_pbp_df=prior_pbp_df,
                )
                adjusted_mu = calc_coverage_adjusted_mu(mu, coverage_eff, man_pct, zone_pct)

            rec_confidence_info = get_data_confidence(gsis_id, player_stats_df, season, week, current_team=team)
            own_grades = get_player_grades(gsis_id, rec_metrics)
            def_grades = get_defense_grades(opponent, def_metrics)

            grade_exploit = calc_grade_matchup_strength({**own_grades, **def_grades}, "rec_yards")
            role_trend = build_role_trend(gsis_id, "target_share", player_stats_df, "gsis_id", season, week)
            role_score = calc_role_verification_score(role_trend)
            blended_exploit = calc_blended_matchup_strength(
                combined_structural_exploit, grade_exploit, role_score
            )
            quality_score = calc_quality_score(
                matchup_exploit_strength=blended_exploit,
                sample_size_games=min(n_plays / 60, 10),
                coverage_confidence=min(n_plays / 300, 1.0),
            )
            _record_quality_score(gsis_id, quality_score)

            rows.append({
                "gsis_id": gsis_id, "player_display_name": wr.get("full_name"),
                "team": team, "position": position, "prop_type": "rec_yards",
                "matchup": team_to_matchup.get(team),
                "mu": adjusted_mu, "mu_before_coverage_adj": mu, "sigma": sigma, "opponent": opponent,
                "opp_man_pct": opp_coverage_row.get("man_pct") if opp_coverage_row else np.nan,
                "opp_zone_pct": opp_coverage_row.get("zone_pct") if opp_coverage_row else np.nan,
                "opp_dominant_coverage": coverage_info["dominant_coverage"],
                "opp_dominant_coverage_pct": coverage_info["dominant_coverage_pct"],
                "opp_num_elevated_coverages": coverage_info.get("num_elevated_coverages", 0),
                "personnel_exploit_strength": personnel_info.get("exploit_strength"),
                "dominant_personnel": personnel_info.get("dominant_personnel"),
                **get_full_coverage_breakdown(opp_coverage_row),
                "quality_score": quality_score,
                "grade_matchup_strength": grade_exploit,
                "role_verification_score": role_score,
                "role_trend_ratio": role_trend.get("trend_ratio"),
                "data_confidence": rec_confidence_info["data_confidence"],
                "games_sampled_current": rec_confidence_info["games_sampled_current"],
                **own_grades,
                **def_grades,
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

        # Fantasy quality_score = average of this player's already-computed
        # pass/rush/rec quality_scores (whichever apply to their position) -
        # same "Fantasy = average of underlying scores" approach the MLB
        # tool uses for Pitcher/Hitter Fantasy.
        component_scores = quality_scores_by_gsis.get(gsis_id, [])
        fantasy_quality_score = round(sum(component_scores) / len(component_scores), 1) if component_scores else np.nan

        rows.append({
            "gsis_id": gsis_id, "player_display_name": pr.get("full_name"),
            "team": pr.get("team"), "position": pr.get("position"), "prop_type": "fantasy_points",
            "matchup": team_to_matchup.get(pr.get("team")),
            "mu": mu_fantasy, "sigma": sigma, "quality_score": fantasy_quality_score,
        })

    # --- Kicker fantasy + FG/XP props ---
    # NOTE: deliberately NOT given a quality_score/matchup-exploit signal,
    # same design exception as the MLB tool's Pitcher Win prop - kicking
    # points are driven by team red-zone/scoring-drive volume rather than
    # a player-vs-defense skill matchup, so none of the offense/defense
    # grade crosswalk or coverage/box signals above meaningfully apply.
    # Not a gap, an intentional scope boundary.
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
            "matchup": team_to_matchup.get(kr.get("team")),
            "mu": mu_kicker, "sigma": sigma,
        })

    return pd.DataFrame(rows)


@_cache_pull
def pull_depth_charts(years: list[int]) -> pd.DataFrame:
    df = nfl.load_depth_charts(seasons=years)
    return df.to_pandas()


def get_data_confidence(player_gsis_id: str, player_stats_df: pd.DataFrame, season: int,
                         current_week: int, current_team: str = None) -> dict:
    """
    Tells you WHICH data a player's mu/sigma is actually built from right
    now - real current-season games, a team-filtered prior-season fallback,
    or the weakest league-average fallback - so you can judge confidence
    at a glance instead of having to remember the week-by-week thresholds
    (mu needs 2 current-season games, sigma needs 3, coverage-specific
    adjustment needs 8 real plays per bucket and follows no clean week
    number since it depends on target volume, not just games played).
    """
    current_season_games = len(player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id) & (player_stats_df["season"] == season)
        & (player_stats_df["week"] < current_week)
    ])

    if current_season_games >= 3:
        confidence = "Current Season (full)"
    elif current_season_games >= 2:
        confidence = "Current Season (mu only, sigma still blending)"
    else:
        prior_team_query = (
            (player_stats_df["gsis_id"] == player_gsis_id)
            & (player_stats_df["season"] == season - 1)
        )
        if current_team is not None:
            prior_team_query &= (player_stats_df["team"] == current_team)
        has_prior_team_games = not player_stats_df[prior_team_query].empty
        confidence = "Fallback: Prior Season (same team)" if has_prior_team_games else "Fallback: League Average"

    return {"games_sampled_current": current_season_games, "data_confidence": confidence}


def calc_prop_mu(player_gsis_id: str, prop_column: str, player_stats_df: pd.DataFrame,
                  season: int, current_week: int, current_team: str = None,
                  lookback_games: int = 6, min_games: int = 2,
                  league_fallback_mu: float = None) -> float:
    """
    Computes mu as the average of a player's own recent real games for a
    given stat column, using player_stats history from weeks BEFORE
    current_week only.

    TEAM-CHANGE FIX (per feedback, real example: AJ Brown's situation
    changed dramatically moving from the Titans to the Eagles - performance
    against the SAME coverages differed because of team/scheme context, not
    just random variance): the prior-season fallback below previously
    pulled a player's history by gsis_id ALONE, with no check on which team
    they played for. Right after a real trade, that meant a player's very
    first weeks on a NEW team could get quietly polluted by their OLD
    team's stale numbers - exactly backwards from what a projection should
    reflect. Now, if current_team is provided, the prior-season fallback
    is filtered to games with that SAME team only - if the player played
    for a different team last season (i.e. they were just traded/signed
    elsewhere), their old-team games are excluded rather than blended in.
    If current_team isn't provided (backward-compatible), falls back to
    the old team-agnostic behavior.

    Returns NaN if there's no usable history and no league_fallback_mu is
    provided - flagged low-confidence in the UI rather than guessed.
    """
    current_season_history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season)
        & (player_stats_df["week"] < current_week)
    ].sort_values("week", ascending=False).head(lookback_games)

    if len(current_season_history) >= min_games:
        return round(current_season_history[prop_column].mean(), 2)

    # Not enough current-season games (Week 1-2, or right after a trade) -
    # bridge with the end of the prior season, but ONLY if it was with the
    # SAME team (when current_team is known) - a traded player's old-team
    # games are excluded rather than silently blended in.
    prior_season_query = (
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season - 1)
    )
    if current_team is not None:
        prior_season_query &= (player_stats_df["team"] == current_team)
    prior_season_history = player_stats_df[prior_season_query].sort_values(
        "week", ascending=False
    ).head(lookback_games)

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
                       season: int, current_week: int, current_team: str = None,
                       lookback_games: int = 8, min_games: int = 3,
                       league_fallback_sigma: float = None) -> float:
    """
    Computes a player's own game-to-game standard deviation for a given prop
    column using their real weekly history from player_stats, up to
    `lookback_games` most recent games before current_week.

    TEAM-CHANGE FIX (same as calc_prop_mu): the prior-season fallback is
    now filtered to the SAME team when current_team is provided, so a
    traded player's sigma isn't computed off stale old-team variance mixed
    with new-team games.

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

    prior_season_query = (
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season - 1)
    )
    if current_team is not None:
        prior_season_query &= (player_stats_df["team"] == current_team)
    prior_season_history = player_stats_df[prior_season_query].sort_values(
        "week", ascending=False
    ).head(lookback_games)

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

def get_starters_for_week(season: int, week: int, depth_charts_df: pd.DataFrame,
                           schedules_df: pd.DataFrame) -> set:
    """
    Returns the set of gsis_ids who were starters at their position for the
    game nearest this season/week, using position-specific pos_rank
    thresholds rather than a flat pos_rank==1 - most offenses run 3-WR sets
    (11 personnel), so WR1/WR2/WR3 are all commonly real starters, not just
    WR1. Same logic applies loosely to RB in committee backfields.

    ASSUMPTION FLAGGED: depth_charts' pos_rank column is assumed to use the
    standard convention where 1 = first-string, 2 = second-string, etc.
    Column existence is confirmed real, but the actual values (and whether
    pos_abb reliably reads "QB"/"RB"/"WR"/"TE") haven't been verified
    against live output yet - check this once real starter/backup sets
    come back to confirm the thresholds below actually match known
    starters, and adjust if needed.

    depth_charts_df has no season/week columns - only a `dt` date field, so
    this matches the closest depth chart snapshot on/before the target
    game's date (same approach as detect_role_change()).
    """
    starter_rank_threshold = {
        "QB": 1,
        "RB": 2,   # covers committee backfields (RB1 + RB2)
        "WR": 3,   # covers standard 3-WR (11 personnel) sets
        "TE": 1,
        "K": 1,
    }

    game_date_row = schedules_df[
        (schedules_df["season"] == season) & (schedules_df["week"] == week)
    ]
    if game_date_row.empty:
        return set()

    target_date = game_date_row["gameday"].max()  # use latest game date that week as cutoff
    snapshot = depth_charts_df[depth_charts_df["dt"] <= target_date].sort_values("dt")
    if snapshot.empty:
        return set()

    # take the most recent depth chart entry per player before the cutoff
    latest_per_player = snapshot.groupby("gsis_id").tail(1).copy()
    latest_per_player["rank_threshold"] = latest_per_player["pos_abb"].map(starter_rank_threshold).fillna(1)
    starters = latest_per_player[latest_per_player["pos_rank"] <= latest_per_player["rank_threshold"]]
    return set(starters["gsis_id"].dropna().tolist())


def score_week_against_actuals(season: int, week: int, starters_only: bool = True) -> pd.DataFrame:
    """
    Shared core of backtest_week(): builds the week's slate, looks up each
    player's REAL result, and attaches miss/abs_miss/match_ratio - but
    returns EVERY row (no match_ratio filter), so this can feed either
    backtest_week()'s "biggest surprises" view or a season-wide accuracy/
    calibration report that needs the full distribution, not just outliers.
    """
    slate_df = build_weekly_slate(season, week)
    player_stats_df = pull_player_stats([season])
    depth_charts_df = pull_depth_charts([season]) if nfl else pd.DataFrame()
    schedules_df = pull_schedules([season])

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

    def _games_sampled(row):
        gsis_id = row["gsis_id"]
        history = player_stats_df[
            (player_stats_df["gsis_id"] == gsis_id) & (player_stats_df["season"] == season)
            & (player_stats_df["week"] < week)
        ]
        return len(history)

    slate_df["actual"] = slate_df.apply(_lookup_actual, axis=1)
    slate_df["games_sampled"] = slate_df.apply(_games_sampled, axis=1)

    # Drop non-participants: no result at all, OR a literal 0 (backup who
    # barely got in, inactive, etc. - a real starter essentially never
    # posts a true 0 in these stat categories).
    slate_df = slate_df.dropna(subset=["actual"])
    slate_df = slate_df[slate_df["actual"] != 0].copy()

    if starters_only:
        starter_ids = get_starters_for_week(season, week, depth_charts_df, schedules_df)
        if starter_ids:
            slate_df = slate_df[slate_df["gsis_id"].isin(starter_ids)]

    slate_df["miss"] = slate_df["mu"] - slate_df["actual"]
    slate_df["abs_miss"] = slate_df["miss"].abs()
    slate_df["match_ratio"] = slate_df.apply(
        lambda r: (r["abs_miss"] / r["sigma"]) if pd.notna(r.get("sigma")) and r.get("sigma", 0) > 0 else np.nan,
        axis=1,
    )
    slate_df["season"] = season
    slate_df["week"] = week
    return slate_df.drop(columns=["line", "p_over", "edge"], errors="ignore")


def backtest_week(season: int, week: int) -> pd.DataFrame:
    """
    Runs the scanner for a week that's already been played, then joins in
    each player's REAL result for that week, so you can compare mu (what
    the model projected using only prior weeks) against what actually
    happened - no betting line needed for this.

    Filters (on top of score_week_against_actuals's participant/starter
    filtering) to ONLY significant discrepancies (match_ratio >= 2.0) -
    close matches (mu ≈ actual) aren't useful for spotting mispriced-line
    opportunities, since a line near mu would've been a coinflip either
    way. Only the big over/underperformances matter here. For a full
    accuracy/calibration view across every row (not just outliers), see
    build_season_accuracy_report() instead.

    Returns columns: player_display_name, team, position, prop_type, mu,
    sigma, actual, miss, abs_miss, match_ratio, games_sampled - sorted by
    biggest surprise (match_ratio) first.
    """
    result = score_week_against_actuals(season, week, starters_only=True)
    result = result[result["match_ratio"] >= 2.0]
    return result.sort_values("match_ratio", ascending=False, na_position="last")


# ---------------------------------------------------------------------------
# 9. FULL-SEASON READINESS REPORT - runs the model across an entire
#    completed season and checks whether it's actually calibrated, not
#    just whether it runs. This is the pre-season sanity check: is
#    quality_score meaningfully predictive, are the coverage/box mu
#    adjustments moving mu the right direction more than a coinflip, is
#    accuracy uneven across prop types/positions.
# ---------------------------------------------------------------------------

def get_completed_weeks_with_data(season: int, through_week: int = 18) -> list:
    """
    Returns the list of weeks in `season` that actually have real
    player_stats rows (i.e. have been played) up through through_week -
    lets build_season_accuracy_report() run against however much of a
    season is actually complete without the caller having to know that
    number in advance.
    """
    player_stats_df = pull_player_stats([season])
    weeks_with_data = sorted(
        player_stats_df[
            (player_stats_df["season"] == season) & (player_stats_df["week"] <= through_week)
        ]["week"].unique().tolist()
    )
    # Week 1 needs week 0 history to project from, which doesn't exist -
    # score_week_against_actuals will just return an empty/fallback-only
    # slate for it, so skip it rather than report a meaningless number.
    return [w for w in weeks_with_data if w >= 2]


def build_season_accuracy_report(season: int, weeks: list = None, through_week: int = 18) -> dict:
    """
    Runs score_week_against_actuals() across every completed week of a
    season (or an explicit `weeks` list) and returns calibration
    diagnostics - the actual "is this model ready" check, not just "does
    it run."

    NOTE ON EDGE/LEAN: there's no free historical NFL player-prop-line
    archive (confirmed real gap, unlike MLB where Underdog/PrizePicks
    lines were at least manually testable), so edge/p_over/lean can't be
    backtested against a real market line the way MLB's Tier 1/Tier 2 hit
    rate could be - there's no historical line to compute edge FROM. What
    CAN be tested without a line, and what this function reports:

      1. "by_prop_type" / "by_position": raw mu accuracy (mean absolute
         miss, mean signed miss = bias) broken out by prop_type and
         position - tells you if a specific category (e.g. rush_yards,
         or TE specifically) is systematically worse than the rest.
      2. "by_quality_tier": rows bucketed by quality_score
         (80-100/60-80/40-60/<40) with mean absolute miss + mean
         match_ratio per bucket. THIS is the core "is quality_score
         actually meaningful" check - if the high-quality-score bucket
         doesn't show tighter/more favorable misses than the low bucket,
         quality_score isn't earning its keep as currently weighted.
      3. "adjustment_direction_accuracy": for every row where the
         coverage or box-count mu adjustment actually moved mu (up or
         down) from mu_before_coverage_adj/mu_before_box_adj, checks
         whether that move was in the same direction the real result
         ended up relative to the unadjusted number. Should clear 50% by
         a real margin - if it doesn't, the adjustment isn't adding
         signal as currently built and should be reweighted or dropped
         rather than trusted as-is.
      4. "role_verification_check": mean absolute miss split by
         role_verification_score >= 0.5 vs < 0.5 - confirms whether the
         recent-usage-trend signal is adding real accuracy or just noise.
      5. "raw": every scored row across every week, for any further
         manual slicing.

    Cannot be run in this build environment (no network access to pull
    real season data) - built and structured to run wherever nflreadpy
    can actually reach the network (local machine or the deployed
    Streamlit Cloud app), then bring the output back for review.
    """
    if weeks is None:
        weeks = get_completed_weeks_with_data(season, through_week)

    week_results = []
    for wk in weeks:
        try:
            wk_df = score_week_against_actuals(season, wk, starters_only=True)
            if not wk_df.empty:
                week_results.append(wk_df)
        except Exception as e:
            # A single bad/missing week (e.g. a bye-heavy early week, or a
            # data source hiccup) shouldn't sink the whole report - skip
            # and keep going, same graceful-degrade approach used
            # throughout this file.
            print(f"Skipping week {wk}: {e}")
            continue

    if not week_results:
        return {"raw": pd.DataFrame(), "by_prop_type": pd.DataFrame(), "by_position": pd.DataFrame(),
                "by_quality_tier": pd.DataFrame(), "adjustment_direction_accuracy": np.nan,
                "role_verification_check": pd.DataFrame()}

    raw = pd.concat(week_results, ignore_index=True)

    by_prop_type = raw.groupby("prop_type").agg(
        mean_abs_miss=("abs_miss", "mean"), mean_bias=("miss", "mean"), n=("abs_miss", "count"),
    ).reset_index().sort_values("mean_abs_miss")

    by_position = raw.groupby("position").agg(
        mean_abs_miss=("abs_miss", "mean"), mean_bias=("miss", "mean"), n=("abs_miss", "count"),
    ).reset_index().sort_values("mean_abs_miss")

    by_quality_tier = pd.DataFrame()
    if "quality_score" in raw.columns:
        tier_df = raw.dropna(subset=["quality_score"]).copy()
        tier_df["quality_tier"] = pd.cut(
            tier_df["quality_score"], bins=[-0.1, 40, 60, 80, 100],
            labels=["<40", "40-60", "60-80", "80-100"],
        )
        by_quality_tier = tier_df.groupby("quality_tier", observed=True).agg(
            mean_abs_miss=("abs_miss", "mean"), mean_match_ratio=("match_ratio", "mean"),
            n=("abs_miss", "count"),
        ).reset_index()

    # Adjustment direction accuracy: unify the two "before adjustment"
    # columns (pass/rec use mu_before_coverage_adj, rush uses
    # mu_before_box_adj) into one check.
    before_col = None
    if "mu_before_coverage_adj" in raw.columns or "mu_before_box_adj" in raw.columns:
        raw["mu_before_adjustment"] = raw.get("mu_before_coverage_adj")
        if "mu_before_box_adj" in raw.columns:
            raw["mu_before_adjustment"] = raw["mu_before_adjustment"].fillna(raw["mu_before_box_adj"])
        before_col = "mu_before_adjustment"

    adjustment_direction_accuracy = np.nan
    if before_col is not None:
        adj = raw.dropna(subset=[before_col, "mu", "actual"]).copy()
        adj = adj[adj["mu"] != adj[before_col]]  # only rows where an adjustment actually moved mu
        if not adj.empty:
            adj_direction = np.sign(adj["mu"] - adj[before_col])
            actual_direction = np.sign(adj["actual"] - adj[before_col])
            valid = actual_direction != 0
            if valid.any():
                adjustment_direction_accuracy = round(
                    (adj_direction[valid] == actual_direction[valid]).mean(), 3
                )

    role_verification_check = pd.DataFrame()
    if "role_verification_score" in raw.columns:
        rv = raw.dropna(subset=["role_verification_score"]).copy()
        rv["role_bucket"] = np.where(rv["role_verification_score"] >= 0.5, "role >= 0.5 (steady/growing)",
                                      "role < 0.5 (fading)")
        role_verification_check = rv.groupby("role_bucket").agg(
            mean_abs_miss=("abs_miss", "mean"), n=("abs_miss", "count"),
        ).reset_index()

    return {
        "raw": raw,
        "by_prop_type": by_prop_type,
        "by_position": by_position,
        "by_quality_tier": by_quality_tier,
        "adjustment_direction_accuracy": adjustment_direction_accuracy,
        "role_verification_check": role_verification_check,
    }
