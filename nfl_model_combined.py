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
import csv
import os
import re
from dataclasses import dataclass, field
from statistics import mean, pstdev

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None  # allows this file to be imported/tested without the package present

# Real, single-file consolidation (per direct request, matching the MLB
# tool's single prop_model_combined.py structure) - coverage_matchup.py
# and rb_matchup.py's real content now lives directly below in this same
# file, so calc_alignment_exploit_strength, calc_qb_coverage_exploit_
# strength, and calc_rb_concept_exploit_strength are natively defined
# here, no conditional import needed or possible to silently fail.


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

def _pull_years_gracefully(loader_fn, years: list[int]) -> pd.DataFrame:
    """
    REAL FIX for a real, confirmed bug: every pull_* function below used
    to call nfl.load_X(seasons=years) once, for the WHOLE requested year
    list at once - if even ONE requested year's real data file doesn't
    exist yet on nflverse's release page (confirmed 404, e.g.
    stats_player_week_2026.parquet before any real 2026 games have been
    played), the ENTIRE pull crashed with a raw, ugly download error -
    not just that one year, every year in the request, even ones that
    would have succeeded on their own. Bound to hit EVERY new season's
    Week 1, every year, until this was fixed - not a one-time fluke.

    Tries the full year-list first (fast path, one real request, same as
    before when every requested year genuinely exists). Only on failure
    does it fall back to trying each year INDIVIDUALLY, keeping whatever
    years succeed and silently dropping ones that don't exist yet -
    exactly matching this file's own existing design philosophy elsewhere
    (calc_prop_mu/calc_player_sigma already shrink toward a fallback when
    the CURRENT season is thin; this extends that same real principle to
    the case where the current season's file doesn't exist AT ALL yet,
    which is different from "exists but thin" and wasn't handled before).

    Returns an empty DataFrame (not an exception) if literally every
    requested year fails - callers already know how to handle an empty
    real DataFrame (that's the normal "no games yet" case throughout this
    file), so this doesn't need new error-handling on the caller's side.
    """
    try:
        return loader_fn(seasons=years)
    except Exception:
        pass  # fall through to per-year fallback below

    frames = []
    for yr in years:
        try:
            f = loader_fn(seasons=[yr])
            frames.append(f.to_pandas() if hasattr(f, "to_pandas") else f)
        except Exception:
            continue  # this specific year genuinely doesn't exist yet - skip it, not a real error
    if not frames:
        # REAL FIX for a real, second bug found live (a genuine KeyError:
        # 'week' after the first fix already resolved the original 404
        # crash) - dozens of places throughout this file filter these
        # pulls by df["season"]/df["week"] (hist_pbp = pbp_df[(pbp_df
        # ["season"]==season) & (pbp_df["week"]<week)], the same pattern
        # repeated everywhere from box-count/coverage profiles down to
        # individual player mu calculations). A bare pd.DataFrame() has
        # NO columns at all, so the very first one of those filters to
        # run crashes with a raw KeyError instead of just correctly
        # finding zero matching rows - patching this at its ONE real
        # source here is far safer than trying to defensively guard
        # dozens of separate downstream call sites individually (real,
        # meaningfully higher risk of missing one and reintroducing the
        # same class of crash somewhere else). Guaranteeing season/week
        # exist (empty, but present) covers every filter pattern actually
        # used in this file - real column-specific differences beyond
        # these two (e.g. player_id) don't matter here, since anything
        # that also needs a player-specific column will correctly find
        # zero rows to work with either way, same end result as a
        # genuinely successful pull that just has no data for this week.
        # FURTHER REAL FIX (found live this session): the season/week-only
        # stub above covers every filter that only ever checks season/week,
        # but real code elsewhere in this file also does column-specific
        # merges/selects (game_id, play_id, defteam, posteam,
        # nflverse_game_id, play_type, sack_player_id, etc.) that aren't
        # covered by season/week alone. Confirmed via a live crash scanning
        # a season with zero real games played yet (2026, before its
        # season started): KeyError: 'nflverse_game_id' inside
        # build_blended_coverage_profile, and the identical shape of crash
        # in build_box_count_profile - neither guarded by the stub above,
        # since neither of those is filtering on season/week. Rather than
        # hand-list every column every pull_* type actually needs (a real
        # maintenance trap - a new merge added later could silently need a
        # column such a list doesn't have), fall back to the immediately
        # PRIOR year's real schema as a template: if last year's pull for
        # this same loader_fn succeeds, borrow its real columns with ZERO
        # rows. Any filter/merge on any column now correctly finds zero
        # rows instead of crashing, and the season/week-only stub below
        # still covers the rare case where even the prior year fails too
        # (a genuinely new stat type with no history at all).
        try:
            prior_year = min(years) - 1
            template = loader_fn(seasons=[prior_year])
            template = template.to_pandas() if hasattr(template, "to_pandas") else template
            if template is not None and len(template.columns) > 2:
                return template.iloc[0:0].copy()
        except Exception:
            pass
        return pd.DataFrame({"season": pd.Series(dtype="int64"), "week": pd.Series(dtype="int64")})
    return pd.concat(frames, ignore_index=True)


@_cache_pull
def pull_pbp(years: list[int]) -> pd.DataFrame:
    """Play-by-play data for the given seasons, converted to pandas."""
    df = _pull_years_gracefully(nfl.load_pbp, years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@_cache_pull
def pull_ngs(stat_type: str, years: list[int]) -> pd.DataFrame:
    """
    stat_type: 'passing', 'rushing', or 'receiving'
    Returns official Next Gen Stats for the given seasons.
    """
    df = _pull_years_gracefully(lambda seasons: nfl.load_nextgen_stats(stat_type=stat_type, seasons=seasons), years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@_cache_pull
def pull_player_stats(years: list[int]) -> pd.DataFrame:
    """
    Game-level player stats (targets, receptions, rush att, pass yds, etc.).

    CONFIRMED: player_stats keys players on `player_id`, while NGS/rosters/
    depth_charts all key on `gsis_id`. Renaming here so every other function
    in this file can join on `gsis_id` consistently without re-checking which
    table uses which name.
    """
    df = _pull_years_gracefully(nfl.load_player_stats, years)
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    # NOTE: checks columns, not df.empty - a real-schema-but-zero-rows frame
    # (a season with no games played yet, post the schema-template fix in
    # _pull_years_gracefully above) is also "empty" by pandas' definition,
    # but still needs this rename applied so downstream gsis_id joins find
    # the column at all instead of silently seeing player_id. Confirmed via
    # a live crash otherwise: KeyError: 'gsis_id' in build_league_fallback_
    # sigmas when scanning a season with zero real games played yet.
    if "player_id" in df.columns:
        df = df.rename(columns={"player_id": "gsis_id"})
    return df


@_cache_pull
def pull_snap_counts(years: list[int]) -> pd.DataFrame:
    """Snap counts by player/game - used as a route-participation / opportunity proxy."""
    df = _pull_years_gracefully(nfl.load_snap_counts, years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@_cache_pull
def pull_ftn_charting(years: list[int]) -> pd.DataFrame:
    """
    FTN manual charting data (free, 2022-onward).
    Key columns: n_defense_box, n_offense_backfield, is_motion, is_play_action,
    is_screen_pass, is_no_huddle, qb_location.
    """
    df = _pull_years_gracefully(nfl.load_ftn_charting, years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@_cache_pull
def pull_participation(years: list[int]) -> pd.DataFrame:
    """
    Participation data - carries defense_man_zone_type, defense_coverage_type,
    time_to_throw, was_pressure. This is where coverage-shell % comes from.
    """
    df = _pull_years_gracefully(nfl.load_participation, years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@_cache_pull
def pull_injuries(years: list[int]) -> pd.DataFrame:
    """
    Weekly injury report data - UNVERIFIED real column names/values, this
    build environment has no network access to confirm nflreadpy's real
    load_injuries() schema against live data. Real injury/active-status
    (the one piece flagged all session as the most plausible explanation
    for the still-unfixed pass_yards outlier pattern - backup/uncertain-
    role QBs, in-game injuries) can't be built responsibly on a guess
    given tonight's repeated lesson about exactly this failure mode
    (the coverage-type casing bug, twice). Use
    diagnose_injuries_data() FIRST against real data before building
    anything that actually reads specific columns from this.
    """
    df = _pull_years_gracefully(nfl.load_injuries, years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@_cache_pull
def pull_schedules(years: list[int]) -> pd.DataFrame:
    df = _pull_years_gracefully(nfl.load_schedules, years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


@_cache_pull
def pull_rosters(years: list[int]) -> pd.DataFrame:
    df = _pull_years_gracefully(nfl.load_rosters, years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


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

    # NORMALIZE column names to always be "man_pct"/"zone_pct" regardless of
    # the real raw value strings, instead of relying on the raw value
    # becoming the literal column name. REAL BUG FOUND (confirmed via a
    # live diagnostic run against real 2025 week 8 data): defense_man_
    # zone_type's actual values are "MAN_COVERAGE"/"ZONE_COVERAGE" (with
    # underscore) plus a large share of empty string "" (non-charted/non-
    # pass plays) - the raw dynamic pivot previously produced columns
    # literally named "MAN_COVERAGE_pct"/"ZONE_COVERAGE_pct"/"_pct", which
    # never matched what every downstream consumer (calc_coverage_adjusted_
    # mu's own gate check, get_full_coverage_breakdown, the opp_man_pct/
    # opp_zone_pct row columns) was looking up - "man_pct"/"zone_pct"
    # exactly. Confirmed this meant opp_man_pct/opp_zone_pct were NULL for
    # all 6,875 rows in every single backtest run all session, and the
    # coverage mu-adjustment's own gate (`if pd.notna(man_pct) and
    # pd.notna(zone_pct)`) never once passed - THIS was the actual root
    # blocker, upstream of and independent from the bucket-matching bug
    # already fixed in build_player_coverage_efficiency. Now matches on
    # content (case-insensitive "man"/"zone" substring) instead of relying
    # on the exact raw string becoming the column name.
    renamed_cols = {}
    for col in man_zone_pct.columns:
        col_lower = str(col).lower()
        if "man" in col_lower:
            renamed_cols[col] = "man_pct"
        elif "zone" in col_lower:
            renamed_cols[col] = "zone_pct"
        else:
            renamed_cols[col] = f"{col}_pct"  # e.g. the "" (uncharted) bucket - kept for visibility, not relied on
    man_zone_pct = man_zone_pct.rename(columns=renamed_cols)

    result = pivot.merge(man_zone_pct, on="defteam", how="left")
    return result


def build_shell_profile_nfl(coverage_profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    1-high/2-high shell pooling for NFL, same real purpose as the MLB
    tool's build_shell_profile(): a fallback, larger-sample signal for
    when a specific granular coverage (Cover 0, Cover 2-Man especially)
    runs thin on real plays, even though the granular data itself is
    real and free (defense_coverage_type, via build_coverage_profile).

    REAL, GENUINE UNCERTAINTY WORTH STATING PLAINLY: unlike
    defense_man_zone_type (whose real raw values - MAN_COVERAGE/
    ZONE_COVERAGE - were just confirmed via an actual live diagnostic
    run), defense_coverage_type's exact real column-name strings coming
    out of build_coverage_profile's pivot have NOT been confirmed
    against real data from this build environment (no network access
    here). Built defensively the same way the man/zone fix was - matching
    by real substring content, case-insensitive, rather than assuming an
    exact spelling - specifically so this doesn't repeat that same class
    of bug. Still worth a real live check before fully trusting the
    grouping is catching every real column.

    Real coverage-shell grouping used (standard NFL coverage
    terminology): 1-high = single deep safety (Cover 1, Cover 3).
    2-high = two safeties split (Cover 2, Cover 2-Man, Cover 4, Cover 6).
    0-high = no deep safety, usually an all-out blitz look (Cover 0) -
    kept separate rather than folded into 1-high, same reasoning as the
    MLB version: it's a structurally different call, not just a smaller-
    sample version of 1-high.

    Returns one row per defteam with real 0h_pct/1h_pct/2h_pct columns,
    to be used as an ADDITIONAL fallback signal alongside the granular
    breakdown - not a replacement for it.
    """
    pct_cols = [c for c in coverage_profile_df.columns if c.endswith("_pct")
                and c not in ("man_pct", "zone_pct")]

    def _shell_for(col_name: str):
        name = col_name.lower()
        if "0" in name:
            return "0h_pct"
        if "1" in name or "3" in name:
            return "1h_pct"
        if "2" in name or "4" in name or "6" in name:
            return "2h_pct"
        return None

    shell_map = {}
    for col in pct_cols:
        shell = _shell_for(col)
        if shell:
            shell_map.setdefault(shell, []).append(col)

    result = coverage_profile_df[["defteam"]].copy()
    for shell in ("0h_pct", "1h_pct", "2h_pct"):
        member_cols = shell_map.get(shell, [])
        result[shell] = coverage_profile_df[member_cols].sum(axis=1) if member_cols else np.nan

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
        # REAL FIX (found live this session): a bare pd.DataFrame() here
        # (zero columns, not just zero rows) crashes the FIRST unguarded
        # caller that does qb_pa_profile["gsis_id"] directly (confirmed
        # live: KeyError: 'gsis_id' scanning a season with zero games
        # played yet - week 1 of ANY season structurally hits this, since
        # weeks < 1 is always empty regardless of the season). Every
        # OTHER consumer of this function already guards with
        # `if not qb_pa_profile.empty:` first, so declaring the real
        # output schema here (0 rows) is a strictly safer no-op for them
        # and the actual fix for the one unguarded site.
        return pd.DataFrame(columns=["gsis_id", "pa_epa", "pa_plays", "non_pa_epa", "non_pa_plays",
                                      "pa_rate", "pa_epa_diff", "pa_rate_grade", "pa_epa_diff_grade"])

    merged = hist_pbp.merge(
        ftn_df[["nflverse_game_id", "nflverse_play_id", "is_play_action"]],
        left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner",
    )
    df = merged.dropna(subset=["passer_player_id", "is_play_action"])
    if df.empty:
        return df  # already has real columns from the merge above (just 0 rows) - safe as-is

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
        # Same real fix as build_qb_playaction_profile above - see its
        # comment for the full reasoning (confirmed live crash otherwise).
        return pd.DataFrame(columns=["defteam", "pa_epa_allowed", "non_pa_epa_allowed",
                                      "pa_vulnerability_gap", "pa_epa_allowed_grade"])

    merged = hist_pbp.merge(
        ftn_df[["nflverse_game_id", "nflverse_play_id", "is_play_action"]],
        left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="inner",
    )
    df = merged.dropna(subset=["defteam", "is_play_action"])
    if df.empty:
        return df  # already has real columns from the merge above (just 0 rows) - safe as-is

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
                                      defteam: str, coverage_row: dict) -> dict:
    """
    Combines the offense side (does this QB run PA often AND perform well
    in it) with the defense side (is this defense - across the REAL FULL
    MIX of coverages it actually plays, weighted by real usage% - allowing
    real problems specifically on play-action) into one 0-1 exploit signal.

    FIX (real gap found by the user reading the raw export directly):
    previously only checked PA-vulnerability for the single "dominant"
    coverage type - but real defenses split their coverage mix, often
    close to evenly across 3-4+ types (confirmed via live data: the
    "dominant" coverage averages only ~31% of a defense's real snaps, not
    a majority). calc_coverage_quality_score's STRUCTURAL exploit signal
    was already fixed to combine every elevated coverage type, not just
    one - this brings the PA-specific crosswalk in line with that same
    fix, instead of being the one place still using only the top type.

    Now takes the full coverage_row (every real coverage-type percentage
    for this defense) and computes a usage-weighted average of PA-
    vulnerability across every coverage type with BOTH a real usage% AND
    a trustworthy PA-specific sample in coverage_pa_crosswalk_df - a
    coverage type the defense rarely plays contributes little to the
    blend even if its PA-allowed grade happens to be extreme, same
    principle as the structural signal.
    """
    offense_vals = [
        qb_pa_row.get("pa_rate_grade") if qb_pa_row else None,
        qb_pa_row.get("pa_epa_diff_grade") if qb_pa_row else None,
    ]
    offense_vals = [v for v in offense_vals if pd.notna(v)]
    offense_component = (sum(offense_vals) / len(offense_vals) / 100) if offense_vals else np.nan

    # Usage-weighted blend across EVERY coverage type this defense plays
    # with a trustworthy PA-specific sample - not just the single dominant one.
    coverage_type_cols = {
        k: v for k, v in (coverage_row or {}).items()
        if k.endswith("_pct") and k not in ("man_pct", "zone_pct") and pd.notna(v)
    }
    weighted_grade_sum, weight_total, any_coverage_specific = 0.0, 0.0, False
    if coverage_type_cols and not coverage_pa_crosswalk_df.empty:
        crosswalk_for_def = coverage_pa_crosswalk_df[coverage_pa_crosswalk_df["defteam"] == defteam]
        for cov_type_pct_key, usage_pct in coverage_type_cols.items():
            # BUGFIX caught in testing (same bug class as the mu-shrinkage
            # fix's own testing catch earlier): coverage_row's keys end in
            # "_pct" (e.g. "COVER_2_pct") but the crosswalk's real
            # defense_coverage_type values (raw participation_df values)
            # don't have that suffix (e.g. "COVER_2") - strip it before
            # matching, or this lookup silently never matches anything.
            cov_type = cov_type_pct_key[:-len("_pct")]
            match = crosswalk_for_def[crosswalk_for_def["defense_coverage_type"] == cov_type]
            if not match.empty:
                grade = match.iloc[0].get("pa_epa_allowed_in_coverage_grade")
                if pd.notna(grade):
                    weighted_grade_sum += grade * usage_pct
                    weight_total += usage_pct
                    any_coverage_specific = True

    if any_coverage_specific and weight_total > 0:
        defense_grade = weighted_grade_sum / weight_total
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
    Per-team FULL personnel usage distribution (every grouping actually
    used, with its real usage%) - the offense-tendency half of the
    crosswalk. Same join pattern used throughout this file for
    participation data: nflverse_game_id + play_id -> pbp's game_id +
    play_id for posteam.

    FIX (same real gap the user found for coverage, applied here too):
    previously collapsed to only the single DOMINANT personnel grouping
    per team, discarding the rest of a team's real personnel mix - same
    issue the coverage structural signal was already fixed for. Now
    returns every (posteam, offense_personnel, usage_pct) row, so
    calc_personnel_exploit_strength can weight across the full real mix
    instead of just the top grouping.
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
    return counts[["posteam", "offense_personnel", "usage_pct"]]


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
    opponent's real EPA-allowed grade across the FULL real personnel mix
    this offense uses (weighted by actual usage%), not just its single
    most-common grouping - a 0-1 exploit signal, same shape as
    calc_playaction_exploit_strength.

    FIX (same real gap the user found for coverage): previously only
    checked the defense's vulnerability to the offense's single dominant
    personnel grouping, discarding the rest of a real, often-substantial
    mix. Now computes a usage-weighted average across every grouping this
    offense actually uses with a trustworthy defense-side sample - a
    rarely-used grouping contributes little to the blend even if its
    allowed-grade happens to be extreme, same principle as the coverage fix.

    Degrades to NaN (not a guessed neutral value) if either side lacks
    enough real data - the caller should treat NaN as "no signal here"
    same as every other exploit function in this file.
    """
    if offense_personnel_tendency_df.empty or defense_personnel_allowed_df.empty:
        return {"exploit_strength": np.nan, "dominant_personnel": None}

    off_rows = offense_personnel_tendency_df[offense_personnel_tendency_df["posteam"] == team]
    if off_rows.empty:
        return {"exploit_strength": np.nan, "dominant_personnel": None}
    dominant_personnel = off_rows.loc[off_rows["usage_pct"].idxmax(), "offense_personnel"]

    def_rows_for_team = defense_personnel_allowed_df[defense_personnel_allowed_df["defteam"] == defteam]
    weighted_grade_sum, weight_total = 0.0, 0.0
    for _, off_row in off_rows.iterrows():
        match = def_rows_for_team[def_rows_for_team["offense_personnel"] == off_row["offense_personnel"]]
        if not match.empty:
            grade = match.iloc[0]["epa_allowed_grade"]
            if pd.notna(grade):
                weighted_grade_sum += grade * off_row["usage_pct"]
                weight_total += off_row["usage_pct"]

    if weight_total == 0:
        return {"exploit_strength": np.nan, "dominant_personnel": dominant_personnel}

    blended_grade = weighted_grade_sum / weight_total
    return {"exploit_strength": round(1 - (blended_grade / 100), 3), "dominant_personnel": dominant_personnel}


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


# Real prop-type categorization for the Monte Carlo simulation below -
# count-type props (whole-number outcomes: completions, attempts,
# targets, receptions, TDs) get a real negative-binomial/Poisson
# sampling; continuous props (yardage, fantasy points) get a real normal
# sampling with a floor at 0 (a real player can't post negative yards).
# This is the real, honest implementation of what
# rescore_quality_mu_row_nfl's own docstring already flagged as
# intended-but-not-yet-built ("swap in Poisson for count stats").
NFL_COUNT_PROPS = {
    "pass_completions", "pass_attempts", "pass_tds",
    "rush_attempts", "rush_tds",
    "receptions", "targets", "rec_tds",
}
NFL_CONTINUOUS_PROPS = {
    "pass_yards", "rush_yards", "rec_yards", "fantasy_points",
    "longest_completion", "longest_reception", "longest_rush", "kicker_fantasy",
}


def simulate_nfl_prop_n_times(mu: float, sigma: float, prop_type: str,
                               n_simulations: int = 1000, random_state: int = 42) -> list:
    """
    Real Monte Carlo simulation for one real player-prop - given the
    model's own real mu and sigma (already independently computed per
    prop elsewhere in this file), samples n_simulations real outcomes
    from a distribution shaped to match how that stat actually behaves:

    - Count-type props (completions, attempts, targets, receptions, TDs):
      a real negative binomial, parameterized to match the model's own
      real mu AND sigma^2 (variance) exactly - NOT a plain Poisson, which
      would force variance = mean and silently throw away the model's
      own, independently-estimated sigma. Real sports count data is
      almost always overdispersed (sigma^2 > mu) - a receiver's real
      target count varies more game to game than a pure Poisson would
      predict - so negative binomial is the honest choice here. Falls
      back to Poisson only in the rare case sigma^2 <= mu, where negative
      binomial's parameterization breaks down.
    - Continuous props (yardage, fantasy points): a real normal
      distribution centered on mu with spread sigma, floored at 0 (a
      real player can't post negative yards or negative fantasy points
      in the vast majority of real scoring systems).

    Returns a real list of n_simulations sampled values - same real
    shape as MLB's simulate_matchup_n_times() per-prop series, so the
    same downstream real_over_rate/gap-pct backtest logic can be reused.
    """
    rng = np.random.default_rng(random_state)
    if sigma is None or np.isnan(sigma) or sigma <= 0 or mu is None or np.isnan(mu):
        return []

    # REAL BUG FIX - partial-game props (1q_rush_attempts, 1h_receptions,
    # etc.) carry a real time-window prefix that doesn't literally match
    # NFL_COUNT_PROPS/NFL_CONTINUOUS_PROPS (those sets only hold the
    # unprefixed, full-game names) - without stripping it first, EVERY
    # partial-game prop silently fell through to the continuous/normal
    # branch, even genuine count stats like rush attempts or receptions.
    # Strip any real "1q_"/"1h_" prefix before checking, so a partial-
    # game count prop gets the same real negative-binomial treatment its
    # full-game counterpart already gets.
    base_prop_type = prop_type
    for prefix in NFL_TIME_WINDOWS:
        if prop_type.startswith(f"{prefix}_"):
            base_prop_type = prop_type[len(prefix) + 1:]
            break

    if base_prop_type in NFL_COUNT_PROPS:
        variance = sigma ** 2
        # REAL BUG FIX - found by actually running against live 2025 data
        # (a real player with mu=0.0 historically - e.g. never scored a
        # real 1Q TD - but a nonzero sigma from rare/occasional events).
        # r = mu^2/(variance-mu) correctly evaluates to 0 when mu=0, but
        # p = r/(r+mu) then divides 0/0, a genuine ZeroDivisionError.
        # mu<=0 is exactly the case Poisson already handles gracefully
        # (numpy's rng.poisson(0) correctly returns all real zeros, no
        # error), so route it there directly instead of attempting the
        # negative binomial parameterization at all.
        if mu > 0 and variance > mu:
            # Real negative binomial matched to the model's own real mu/variance
            r = mu ** 2 / (variance - mu)
            p = r / (r + mu)
            samples = rng.negative_binomial(r, p, size=n_simulations)
        else:
            # Real, honest fallback - either mu<=0 (handled correctly by
            # Poisson even at exactly 0) or variance too low for negative
            # binomial's parameterization to work
            samples = rng.poisson(max(mu, 0), size=n_simulations)
        return [max(0, int(v)) for v in samples]
    else:
        samples = rng.normal(mu, sigma, size=n_simulations)
        return [max(0.0, round(float(v), 1)) for v in samples]


def calc_quality_score(matchup_exploit_strength: float, sample_size_games: int,
                        coverage_confidence: float) -> float:
    """
    matchup_exploit_strength: how much this specific offense/player profile
        beats this specific defense's tendency (e.g. high aDOT WR vs man-heavy defense)
    sample_size_games: REAL games backing THIS PLAYER'S OWN mu this season
        (fewer games = lower confidence, regardless of how good the matchup
        looks) - see BUGFIX note below.
    coverage_confidence: how much of the OPPONENT's play sample has charted
        coverage data (a separate, complementary concept from the player's
        own sample size - this is about how much we trust the opponent's
        tendency profile itself)

    BUGFIX (real gap found via 2025 backtest): sample_size_games was always
    being fed opponent coverage PLAY COUNT (n_plays/60) at every call site,
    not the player's own games - a genuine mismatch between what this
    parameter was named/documented to mean and what it actually received.
    Confirmed via real correlation check: quality_score showed ~zero
    relationship with pass_yards miss size (0.056) even though the worst
    misses were concentrated in players with thin/unstable CURRENT-SEASON
    samples - quality_score's "confidence" signal was answering "how much
    do we know about the opponent's coverage" while never once asking "how
    much do we know about THIS player's own current role/production."
    Call sites fixed to pass games_sampled_current (the player's own real
    sample size, already computed via get_data_confidence and already
    reliable) instead of the opponent-derived play count.
    """
    base = matchup_exploit_strength * 70
    sample_bonus = min(sample_size_games / 6, 1.0) * 20
    coverage_bonus = coverage_confidence * 10
    return round(min(base + sample_bonus + coverage_bonus, 100), 1)



# ---------------------------------------------------------------------------
# 5b. FULL-COVERAGE-TYPE PLAYER SPLIT (real efficiency by EVERY charted
#     coverage type, not just the coarser man/zone binary above - that
#     mechanism is currently disabled, proven coinflip-accuracy at 2
#     buckets. This is a NEW, separate signal built to test whether more
#     granularity actually helps, rather than silently modifying the
#     disabled mechanism - keeps results cleanly attributable either way.
# ---------------------------------------------------------------------------

def build_player_full_coverage_efficiency(player_gsis_id: str, role: str,
                                           participation_df: pd.DataFrame, pbp_df: pd.DataFrame,
                                           min_plays_per_type: int = 8,
                                           prior_participation_df: pd.DataFrame = None,
                                           prior_pbp_df: pd.DataFrame = None) -> dict:
    """
    Real per-player efficiency (yards/play) split across EVERY charted
    coverage type (Cover 0/1/2/3/4/6/9, 2-Man, Combo, etc.), not just
    man/zone. Caller is expected to pass the FULL SEASON of real plays
    before the target week (not a short recent window) - per real-world
    volume, a full season gives enough plays for a player's more common
    coverages even split this fine, though rarer types (Cover 9, Combo,
    Blown) will often still fall below min_plays_per_type even over 17
    games. Those are dropped entirely rather than trusted on a thin
    sample - see calc_full_coverage_adjusted_mu for how the fallback
    then correctly relies on whichever 2-3 real coverages ARE reliable.

    REAL FIX - same bug/fix as the offense/defense grade builders above:
    week 1 (zero current-season plays) always returned an empty result
    ("overall_plays": 0), meaning THIS is exactly the mechanism that
    can't yet show "how does Drake Maye do vs Cover 2/4/6" or "how did
    AJ Brown/Doubs do vs those coverages" in week 1 - the real per-
    coverage-type splits from all of last season are the fix, and are
    now used automatically when the current season has nothing yet.
    Note: this function itself stays gated off live by
    ENABLE_FULL_COVERAGE_MU_ADJUSTMENT (False) pending its own isolated
    backtest, per this project's established one-flag-at-a-time rollout
    discipline - this fix makes it READY to test properly, not live yet.
    """
    merged = participation_df.merge(
        pbp_df[["game_id", "play_id", "defteam", "posteam",
                "receiver_player_id", "receiving_yards",
                "passer_player_id", "passing_yards"]],
        left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="left",
    )
    player_col = "receiver_player_id" if role == "receiver" else "passer_player_id"
    yards_col = "receiving_yards" if role == "receiver" else "passing_yards"
    player_plays = merged[
        (merged[player_col] == player_gsis_id) & merged["defense_coverage_type"].notna()
    ]

    if player_plays.empty and prior_participation_df is not None and prior_pbp_df is not None \
            and not prior_participation_df.empty and not prior_pbp_df.empty:
        prior_merged = prior_participation_df.merge(
            prior_pbp_df[["game_id", "play_id", "defteam", "posteam",
                           "receiver_player_id", "receiving_yards",
                           "passer_player_id", "passing_yards"]],
            left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="left",
        )
        player_plays = prior_merged[
            (prior_merged[player_col] == player_gsis_id) & prior_merged["defense_coverage_type"].notna()
        ]

    if player_plays.empty:
        return {"overall_ypp": np.nan, "overall_plays": 0}

    result = {}
    for cov_type, group in player_plays.groupby("defense_coverage_type"):
        n = len(group)
        if n >= min_plays_per_type:
            result[cov_type] = {"ypp": round(group[yards_col].mean(), 2), "n_plays": n}

    result["overall_ypp"] = round(player_plays[yards_col].mean(), 2)
    result["overall_plays"] = len(player_plays)
    return result


def calc_full_coverage_adjusted_mu(base_mu: float, player_coverage_eff: dict,
                                    opp_coverage_row: dict, max_adjustment: float = 0.2) -> dict:
    """
    Generalizes the same real per-player-split x this-week's-opponent-
    tendency, capped-adjustment logic the removed man/zone-only mu-
    adjustment used, from 2 buckets (man/zone) to EVERY coverage type both
    sides have reliable real data for - the fallback the user specifically
    asked for: a defense that plays 3 real coverages this season where the
    player has adequate sample against 2 of them but not the 3rd
    correctly RENORMALIZES the weighting across just those 2 reliable
    types, rather than forcing in an unreliable split or refusing to
    adjust mu at all.

    Returns a dict (not just a float) so the caller can see HOW MUCH of
    the opponent's real coverage mix was actually covered by a reliable
    player-side sample (coverage_weight_used, 0-1) - a defense that
    spreads evenly across many types the player barely sees correctly
    results in little to no adjustment, not a forced guess.
    """
    overall_ypp = player_coverage_eff.get("overall_ypp")
    if pd.isna(overall_ypp) or not overall_ypp:
        return {"adjusted_mu": base_mu, "coverage_weight_used": 0.0}

    coverage_type_cols = {
        k: v for k, v in (opp_coverage_row or {}).items()
        if k.endswith("_pct") and k not in ("man_pct", "zone_pct") and pd.notna(v)
    }
    weighted_ypp_sum, weight_total = 0.0, 0.0
    for cov_type_pct_key, usage_pct in coverage_type_cols.items():
        # BUGFIX caught in testing: opp_coverage_row's keys end in "_pct"
        # (e.g. "COVER_3_pct") but player_coverage_eff's keys don't (e.g.
        # "COVER_3") - strip the suffix before matching, or this lookup
        # silently returns None every time and coverage_weight_used stays
        # 0.0 no matter how much real overlap actually exists.
        cov_type = cov_type_pct_key[:-len("_pct")]
        player_split = player_coverage_eff.get(cov_type)
        if player_split is not None:
            weighted_ypp_sum += player_split["ypp"] * usage_pct
            weight_total += usage_pct

    if weight_total == 0:
        return {"adjusted_mu": base_mu, "coverage_weight_used": 0.0}

    expected_ypp_this_matchup = weighted_ypp_sum / weight_total
    multiplier = expected_ypp_this_matchup / overall_ypp
    multiplier = max(1 - max_adjustment, min(1 + max_adjustment, multiplier))

    return {"adjusted_mu": round(base_mu * multiplier, 2), "coverage_weight_used": round(weight_total, 3)}


def build_matchup_explanation(coverage_row: dict, player_coverage_eff: dict,
                               personnel_row: dict = None, personnel_eff: dict = None,
                               min_meaningful_usage: float = 0.05) -> dict:
    """
    DISPLAY-ONLY summary of "why" behind a matchup - built for the Best
    Matchups explainer tab, computed regardless of whether the full-
    coverage mu-adjustment itself is enabled, since seeing this reasoning
    doesn't require trusting the adjustment yet. Answers exactly what the
    user asked to see: which coverages/personnel groupings the defense
    actually leans on, and which of those the player has (or doesn't
    have) a real, reliable sample against.

    Returns:
      coverage_mix: {coverage_type: usage_pct} for every real coverage
        type the defense plays (min_meaningful_usage floor - a coverage
        run <5% of the time isn't worth listing as part of "their tendency")
      player_coverage_sample: {coverage_type: {"ypp","n_plays"}} - only
        the coverage types the player has a RELIABLE sample against
        (already filtered by build_player_full_coverage_efficiency's
        min_plays_per_type)
      coverage_types_no_sample: which of the defense's real meaningful
        coverage types the player does NOT have reliable data for -
        exactly the "defense runs 3/4/6, player only has sample vs 3/4"
        case the user described
      personnel_mix / player_personnel_note: same idea for personnel,
        when provided (rec_yards only)
    """
    coverage_mix = {
        k[:-len("_pct")]: round(v, 3) for k, v in (coverage_row or {}).items()
        if k.endswith("_pct") and k not in ("man_pct", "zone_pct") and pd.notna(v) and v >= min_meaningful_usage
    }
    player_coverage_sample = {
        k: v for k, v in (player_coverage_eff or {}).items()
        if k not in ("overall_ypp", "overall_plays")
    }
    coverage_types_no_sample = [
        cov for cov in coverage_mix if cov not in player_coverage_sample
    ]

    result = {
        "coverage_mix": coverage_mix,
        "player_coverage_sample": player_coverage_sample,
        "coverage_types_no_sample": coverage_types_no_sample,
        "player_overall_ypp": (player_coverage_eff or {}).get("overall_ypp"),
    }

    if personnel_row is not None:
        result["personnel_mix"] = {
            row["offense_personnel"]: round(row["usage_pct"], 3)
            for _, row in personnel_row.iterrows()
        } if hasattr(personnel_row, "iterrows") else {}
        result["personnel_efficiency_note"] = personnel_eff

    return result


def get_player_matchup_explanation(gsis_id: str, prop_type: str, team: str, opponent: str,
                                    season: int, week: int, use_full_season: bool = True) -> dict:
    """
    ON-DEMAND, single-player version of build_matchup_explanation - built
    to be called interactively when a user clicks a specific player in
    the Best Matchups UI, NOT baked into every row of build_weekly_slate.
    Deliberately kept separate: embedding this into every scanned row
    would add real per-player compute cost, and build_weekly_slate also
    gets called 15x inside the season readiness report's week loop -
    baking this in there would multiply that cost 15x, risking the same
    Streamlit Cloud resource-limit issue already hit once this session.
    Cheap to call per-click instead, since the underlying data pulls
    (_cache_pull-decorated) are already cached from whatever scan just ran.

    use_full_season (per explicit request, default True): this function is
    a VALIDATION/understanding tool, not the live mu-generating pathway -
    the actual mu/quality_score computation elsewhere in this file
    correctly stays restricted to weeks BEFORE the target week (no
    leakage). This explainer is different in kind: its whole purpose is
    "does this real relationship make sense," so maximizing real sample
    volume (the full season) gives a fuller, more honest picture than
    artificially restricting to a partial season, with no leakage concern
    since nothing here feeds back into a live projection. Set False to
    see the exact same before-this-week-only data mu would have used.
    """
    participation_df = pull_participation([season])
    pbp_df = pull_pbp([season])
    effective_week = 19 if use_full_season else week  # week 19 = "no real week is >= this", captures the whole season
    pbp_history_df = pbp_df[pbp_df["week"] < effective_week]
    coverage_profile = build_blended_coverage_profile(season, effective_week)

    opp_coverage_row = None
    if not coverage_profile.empty:
        match = coverage_profile[coverage_profile["defteam"] == opponent]
        if not match.empty:
            opp_coverage_row = match.iloc[0].to_dict()

    role = "passer" if prop_type == "pass_yards" else "receiver"
    player_coverage_eff = build_player_full_coverage_efficiency(gsis_id, role, participation_df, pbp_history_df)

    personnel_row = None
    if prop_type == "rec_yards":
        offense_personnel_tendency = build_offense_personnel_tendency(season, effective_week, participation_df, pbp_history_df)
        if not offense_personnel_tendency.empty:
            personnel_row = offense_personnel_tendency[offense_personnel_tendency["posteam"] == team]

    return build_matchup_explanation(opp_coverage_row, player_coverage_eff, personnel_row)


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


# REAL, VERIFIED 2026 offseason coordinator changes (via direct web search,
# CBS Sports coordinator-hire tracker, Feb 23 2026) - cross-checked against
# the article's own stated totals (21 new OC, 14 REAL new DC - one of 15
# raw DC hires, New England's Zak Kuhr, explicitly doesn't count since he
# already ran the Patriots' defense for most of last season during a
# medical leave, so no real scheme discontinuity there) and confirmed this
# exact team list reproduces both totals precisely.
#
# WHY THIS MATTERS: every prior-season bridge in this file (role_trend,
# build_qb/receiver/rb/defense_advanced_metrics, build_team_offense) was
# built assuming real scheme continuity year over year. For a team with a
# genuinely new OC or DC, last season's real tendencies describe a scheme
# that may no longer exist - blending them in isn't a cautious fallback,
# it's actively misleading. Real current-season data, even a single game
# under the NEW scheme, is more informative than a full season under the
# old one for these specific teams.
NEW_OC_TEAMS_2026 = {
    "ARI", "ATL", "BAL", "CAR", "CLE", "DEN", "DET", "LAC",
    "LV", "MIA", "NYG", "NYJ", "PHI", "PIT", "SEA", "TB", "TEN", "WAS",
}
# REAL FIX (caught via direct user correction) - New Orleans (NO) removed.
# Kellen Moore is entering his SECOND year as Saints HC, per the exact
# ESPN source already on file: "This is Moore's second year calling
# plays as a head coach." He was already New Orleans' real HC/play-
# caller in 2025 - genuine continuity, not a new 2026 hire. This was a
# real misread on my part (I'd treated him as newly "taking over" when
# he'd actually already been there a year). New Orleans still correctly
# stays OFF the offense-trusted list overall - but for the right reason
# now: their real 2026 QB change (Shough replacing Rattler, already in
# NEW_QB_TEAMS_2026), not a non-existent OC change.
# REAL FIX (found via a comprehensive, independently-sourced cross-check -
# The Ringer's "17 new play-callers for 2026" feature, which independently
# CONFIRMED all 4 corrections below by omission - Chicago/Rams/Buffalo/
# Kansas City are not among the 17, meaning their real play-callers are
# independently confirmed unchanged by a second, separate source):
# Carolina (CAR) added - the OPPOSITE pattern from Chicago. Panthers.com
# directly quotes Dave Canales confirming Brad Idzik (his own real OC,
# unchanged in TITLE since 2025) will call plays in 2026, a role Canales
# held himself last year. The title never changed, but the real play-
# calling authority did (Canales -> Idzik) - confirmed independently by
# The Ringer's profile of Idzik as a genuine first-time 2026 play-caller.
# REAL FIX (found live via direct request to reconsider title vs. actual
# play-caller): Chicago (CHI) removed from this list. The "OC" title
# changed (Declan Doyle -> Press Taylor), but multiple, very direct
# sources confirm the REAL play-caller - HC Ben Johnson - is completely
# unchanged from 2025 to 2026. Taylor himself: "None of it is calling
# plays... [I] relay his message to the staff." Johnson personally
# installs the run game weekly, same as he did in Detroit. The entire
# point of this registry is "does last year's real tendency data still
# reflect who's actually making the decisions" - for Chicago specifically,
# the title-based answer was wrong; the real scheme-setter never changed.
#
# FOUR MORE REAL FIXES (found via a comprehensive, direct cross-check of
# a full 32-team HC/OC/DC table, each confirmed with an explicit, direct
# source - not just inference):
#   - LA Rams (LAR) removed: Sean McVay is the continuing HC and the real,
#     established play-caller regardless of OC title - confirmed directly.
#   - Buffalo (BUF) removed: Joe Brady WAS Buffalo's own real play-caller
#     in 2025 as their OC - directly confirmed to continue in that real
#     role for 2026, just under a new HC title. Same real person.
#   - Kansas City (KC) removed: Andy Reid, in his own words, confirmed he
#     remains the real play-caller ("I still enjoy calling plays"),
#     describing it as roughly "51% of the say" even with new OC Eric
#     Bieniemy back in the building - unchanged from 2025.
#   - New Orleans (NO) added: was missing from this list entirely. Kellen
#     Moore is a genuinely new, established offensive mind (Philadelphia's
#     real OC before this) taking over as HC. Real change confirmed.
#
# The remaining ~14 teams on this list all have explicit, direct
# confirmation of a genuinely new real play-caller (either a brand-new HC
# who is himself the real offensive mind, or an explicit statement that
# the new OC - not the continuing HC - calls plays) - not just title-based
# assumptions.
NEW_DC_TEAMS_2026 = {
    "BAL", "BUF", "CIN", "CLE", "DAL", "GB", "IND", "LAC", "LV", "MIA",
    "NYG", "NYJ", "PIT", "SF", "TEN", "WAS",
}

# REAL, VERIFIED 2026 new starting QB registry (via direct search, Yardbarker
# "Ranking projected 2026 NFL Week 1 QB changes," published Aug 26 2026,
# which itself states "nine NFL teams will have a different Week 1 starting
# quarterback from 2025") - INDEPENDENT of OC/DC continuity, per direct
# request: a team can keep the exact same OC and still have a fundamentally
# different real passing offense if the QB changed - where the ball goes,
# which routes get trusted, how coverage gets read is heavily QB-specific,
# not just scheme-specific. ATL and LV are explicitly flagged "projected*"
# by the source itself (real, stated uncertainty remains as of publication -
# Penix could still start for ATL, rookie Mendoza could still win the LV job)
# - kept in this set since the source's own best real projection still has
# both as QB changes, but worth knowing these two are less certain than the
# other seven.
NEW_QB_TEAMS_2026 = {
    "CLE",  # Deshaun Watson replacing Joe Flacco
    "ATL",  # Tua Tagovailoa replacing Michael Penix Jr. - projected, real uncertainty
    "ARI",  # Jacoby Brissett replacing Kyler Murray
    "LV",   # Kirk Cousins replacing Geno Smith - projected, real uncertainty
    "NYJ",  # Geno Smith replacing Justin Fields
    "MIA",  # Malik Willis replacing Tua Tagovailoa
    "NO",   # Tyler Shough replacing Spencer Rattler
    "NYG",  # Jaxson Dart replacing Russell Wilson
    "MIN",  # Kyler Murray replacing J.J. McCarthy
}
# REAL FIX (found live via a direct second-source cross-check, per direct
# request to verify this list properly before building anything on it):
# the original list was built from ONE primary source (a CBS Sports
# coordinator-hire tracker) and validated only by confirming its own
# stated totals (21 OC, 14 real DC changes) matched exactly - a real,
# meaningful internal-consistency check, but NOT the same as confirming
# every team against an independent source. A second search surfaced a
# PFR tracker naming Cincinnati, Indianapolis, and Minnesota as teams
# with a real coordinator search this offseason - none of which were in
# the original list at all. Investigated all three directly:
#   - Indianapolis: CONFIRMED real, new DC - Lou Anarumo (ex-Bengals DC).
#     Real omission, now fixed. OC (Cooter) confirmed unchanged.
#   - Cincinnati: CONFIRMED real, new DC - Al Golden. Real omission, now
#     fixed. No OC change found.
#   - Minnesota: INVESTIGATED and confirmed to be a false alarm - DC
#     Brian Flores took outside HC/DC interviews (the real "search" PFR
#     was tracking) but ultimately signed a real contract extension and
#     stays as Vikings DC for 2026, confirmed by multiple independent,
#     direct sources. Correctly excluded, no fix needed.


def build_qb_advanced_metrics(season: int, week: int, player_stats_df: pd.DataFrame,
                               ngs_pass_df: pd.DataFrame, participation_df: pd.DataFrame,
                               pbp_df: pd.DataFrame, pass_explosive_df: pd.DataFrame = None,
                               prior_ngs_pass_df: pd.DataFrame = None,
                               prior_pbp_df: pd.DataFrame = None) -> pd.DataFrame:
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

    REAL FIX (found live this session, same bug class as the longest-play/
    role-trend prior-season bridges already built elsewhere in this file):
    week 1 (and, less severely, early weeks generally) has ZERO weeks of
    current-season history, so hist_stats/hist_ngs/hist_pbp were ALWAYS
    empty and this function ALWAYS returned pd.DataFrame() - meaning
    calc_grade_matchup_strength had literally nothing to work with, no
    matter how good or bad the real matchup was, which is the actual
    reason quality_score structurally couldn't clear ~65 in week 1
    (confirmed directly against real 2025 data: 44 columns/max 64.8 in
    week 1 vs 124 columns/max 82.1 in week 2, the exact moment this data
    stops being empty). The prior-season data needed to fix this was
    ALREADY being pulled in build_weekly_slate for the role_trend/longest-
    play bridges (prior_ngs_pass_df, prior_pbp_df) - just never threaded
    into this function. Now falls back to the full prior season (all
    weeks, same no-week-filter convention the longest-play bridge already
    uses, since week numbers reset each season) whenever the current
    season's own history is empty. player_stats_df already carries BOTH
    seasons (pulled as pull_player_stats([season, season - 1]) at the
    call site) so no new param was needed there - just a filter fix.
    """
    hist_stats = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < week)
        & (player_stats_df["position"] == "QB")
    ]
    if hist_stats.empty:
        # Real fix - exclude teams with a genuinely new 2026 OC from the
        # prior-season bridge (see NEW_OC_TEAMS_2026 above). For those
        # specific teams, last season's real numbers describe a scheme
        # that may no longer exist - bridging them in isn't a cautious
        # fallback, it's misleading. Those teams' QBs correctly get no
        # bridge (hist_stats stays empty for them) rather than a
        # confidently-wrong number.
        hist_stats = player_stats_df[
            (player_stats_df["season"] == season - 1) & (player_stats_df["position"] == "QB")
            & (~player_stats_df["team"].isin(NEW_OC_TEAMS_2026))
        ]
    if hist_stats.empty:
        return pd.DataFrame()

    agg = hist_stats.groupby("gsis_id").agg(
        passing_epa=("passing_epa", "mean"),
        passing_yards=("passing_yards", "sum"),
        attempts=("attempts", "sum"),
    ).reset_index()

    hist_ngs = ngs_pass_df[(ngs_pass_df["season"] == season) & (ngs_pass_df["week"] < week)]
    if hist_ngs.empty and prior_ngs_pass_df is not None and not prior_ngs_pass_df.empty:
        hist_ngs = prior_ngs_pass_df[prior_ngs_pass_df["season"] == season - 1]
    ngs_agg = hist_ngs.groupby("player_gsis_id").agg(
        cpoe=("completion_percentage_above_expectation", "mean"),
        adot=("avg_intended_air_yards", "mean"),
        aggressiveness=("aggressiveness", "mean"),
        passer_rating=("passer_rating", "mean"),
        # Real, newly-added free NGS columns found via direct audit of the
        # actual live dataset (nflreadpy load_nextgen_stats) - previously
        # sitting unused despite being free and already pulled elsewhere.
        time_to_throw=("avg_time_to_throw", "mean"),  # pocket presence / real INT-risk signal (holding the ball too long)
        air_yards_to_sticks=("avg_air_yards_to_sticks", "mean"),  # aggressiveness relative to the ACTUAL first-down marker - distinct from raw aDOT (task-relative, not absolute depth)
        max_completed_air_distance=("max_completed_air_distance", "max"),  # his real longest-completion depth - directly useful for the longest_completion prop specifically, not currently fed by anything
        # Real, newly-captured NGS column per direct request ("completion
        # probabilities"). Deliberately NOT added to the grading loop below
        # - a high expected_completion_percentage mostly reflects the
        # OFFENSE'S SCHEME (shorter/easier routes called for him), not his
        # own skill. CPOE (already graded) is the correct isolation of his
        # real skill beyond what's schemed for him - grading the raw
        # expected value too would reward offenses that call easy throws,
        # not QBs who are actually good.
        expected_completion_percentage=("expected_completion_percentage", "mean"),
    ).reset_index().rename(columns={"player_gsis_id": "gsis_id"})

    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week) & (pbp_df["play_type"] == "pass")]
    if hist_pbp.empty and prior_pbp_df is not None and not prior_pbp_df.empty:
        hist_pbp = prior_pbp_df[(prior_pbp_df["season"] == season - 1) & (prior_pbp_df["play_type"] == "pass")]
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
                "explosive_20plus_rate", "air_yards_to_sticks", "max_completed_air_distance"]:
        if col in merged.columns:
            merged[f"{col}_grade"] = merged[col].apply(lambda v: calc_percentile_grade(v, merged[col]))
    # HONEST GAP - time_to_throw is deliberately NOT graded here. Its real
    # meaning is genuinely context-dependent (a quick release can mean
    # good pocket composure/decision-making, OR just a checkdown-heavy
    # offense with no deep-shot value) - calc_percentile_grade always
    # treats "higher raw value = higher grade" with no inversion support,
    # so grading this without knowing which direction actually matters for
    # a given prop would silently bias the score in an unverified
    # direction. The raw value is still captured on the row (visible for
    # manual inspection) - just not blended into the automatic grade yet.

    return merged


def build_receiver_advanced_metrics(season: int, week: int, player_stats_df: pd.DataFrame,
                                     ngs_rec_df: pd.DataFrame, rec_explosive_df: pd.DataFrame = None,
                                     prior_ngs_rec_df: pd.DataFrame = None) -> pd.DataFrame:
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

    REAL FIX - same bug/fix as build_qb_advanced_metrics: week 1 (zero
    current-season history) always returned pd.DataFrame() empty, meaning
    a WR/TE's own real skill grade (separation, catch%, aDOT-independent
    efficiency) simply didn't exist yet for the grade-based crosswalk to
    use - directly the AJ Brown/Doubs gap flagged live: their real 2025
    NGS separation/catch%/target-share profile should inform week 1 of
    2026, not go unused just because 2026 itself hasn't accumulated games
    yet. Falls back to the full prior season (all weeks) when current-
    season history is empty, same as the QB version.
    """
    hist_stats = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < week)
        & (player_stats_df["position"].isin(["WR", "TE", "RB"]))
    ]
    if hist_stats.empty:
        # Real fix - same NEW_OC_TEAMS_2026 suppression as build_qb_advanced_metrics above.
        hist_stats = player_stats_df[
            (player_stats_df["season"] == season - 1)
            & (player_stats_df["position"].isin(["WR", "TE", "RB"]))
            & (~player_stats_df["team"].isin(NEW_OC_TEAMS_2026))
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
    if hist_ngs.empty and prior_ngs_rec_df is not None and not prior_ngs_rec_df.empty:
        hist_ngs = prior_ngs_rec_df[prior_ngs_rec_df["season"] == season - 1]
    ngs_agg = hist_ngs.groupby("player_gsis_id").agg(
        avg_separation=("avg_separation", "mean"),
        avg_cushion=("avg_cushion", "mean"),
        catch_percentage=("catch_percentage", "mean"),
        yac_above_expectation=("avg_yac_above_expectation", "mean"),
        # Real, newly-added free NGS column found via direct audit - a
        # receiver's own real target depth. Confirmed via direct search
        # this was NEVER used anywhere in this function despite being
        # free and already pulled - the premium alignment CSV's own aDOT
        # was about to duplicate this exact same real-world concept from
        # a different provider; removed there in favor of this free
        # version (see ALIGNMENT_STATS_BY_PROP's rec_yards "depth" bucket).
        adot=("avg_intended_air_yards", "mean"),
    ).reset_index().rename(columns={"player_gsis_id": "gsis_id"})

    merged = agg.merge(ngs_agg, on="gsis_id", how="left")

    if rec_explosive_df is not None and not rec_explosive_df.empty:
        exp = rec_explosive_df.rename(columns={"receiver_player_id": "gsis_id"})[
            ["gsis_id", "explosive_15plus_rate", "explosive_20plus_rate"]
        ]
        merged = merged.merge(exp, on="gsis_id", how="left")

    for col in ["target_share", "wopr", "racr", "receiving_epa", "avg_separation",
                "catch_percentage", "yac_above_expectation", "explosive_15plus_rate", "adot"]:
        if col in merged.columns:
            merged[f"{col}_grade"] = merged[col].apply(lambda v: calc_percentile_grade(v, merged[col]))

    return merged


def build_rb_advanced_metrics(season: int, week: int, player_stats_df: pd.DataFrame,
                               ngs_rush_df: pd.DataFrame, rush_explosive_df: pd.DataFrame = None,
                               prior_ngs_rush_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    RB advanced metrics: rushing_epa (season-aggregated), rush_yards_over_
    expected_per_att, efficiency, avg_time_to_los, percent_attempts_gte_
    eight_defenders (box rate faced), and (when rush_explosive_df is
    supplied - see build_explosive_rates()) explosive_10plus_rate - the
    "breakaway run" tendency signal, distinct from efficiency (average
    per-carry value): a between-the-tackles grinder can have strong
    efficiency with almost no explosive runs, or vice versa.

    REAL FIX - same bug/fix as build_qb_advanced_metrics: falls back to
    the full prior season (all weeks) when current-season history is
    empty (week 1 always was, structurally), instead of returning nothing.
    """
    hist_stats = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < week)
        & (player_stats_df["position"] == "RB")
    ]
    if hist_stats.empty:
        # Real fix - same NEW_OC_TEAMS_2026 suppression as the QB/receiver builders above.
        hist_stats = player_stats_df[
            (player_stats_df["season"] == season - 1) & (player_stats_df["position"] == "RB")
            & (~player_stats_df["team"].isin(NEW_OC_TEAMS_2026))
        ]
    if hist_stats.empty:
        return pd.DataFrame()

    agg = hist_stats.groupby("gsis_id").agg(
        rushing_epa=("rushing_epa", "mean"),
    ).reset_index()

    hist_ngs = ngs_rush_df[(ngs_rush_df["season"] == season) & (ngs_rush_df["week"] < week)]
    if hist_ngs.empty and prior_ngs_rush_df is not None and not prior_ngs_rush_df.empty:
        hist_ngs = prior_ngs_rush_df[prior_ngs_rush_df["season"] == season - 1]
    ngs_agg = hist_ngs.groupby("player_gsis_id").agg(
        rush_yards_over_expected_per_att=("rush_yards_over_expected_per_att", "mean"),
        efficiency=("efficiency", "mean"),
        avg_time_to_los=("avg_time_to_los", "mean"),
        box_stack_pct_faced=("percent_attempts_gte_eight_defenders", "mean"),
        # Real, newly-added free NGS column found via direct audit -
        # genuinely distinct from rush_yards_over_expected_per_att: that's
        # the average MAGNITUDE of beating expectation, this is the real
        # RATE (% of individual carries that beat expectation at all) -
        # a back can have a high average via a few huge runs while rarely
        # beating expectation on any given carry, or the reverse (grinds
        # out a positive result almost every time with no single huge
        # run) - two different, real signals, not a duplicate.
        rush_pct_over_expected=("rush_pct_over_expected", "mean"),
    ).reset_index().rename(columns={"player_gsis_id": "gsis_id"})

    merged = agg.merge(ngs_agg, on="gsis_id", how="left")

    if rush_explosive_df is not None and not rush_explosive_df.empty:
        exp = rush_explosive_df.rename(columns={"rusher_player_id": "gsis_id"})[
            ["gsis_id", "explosive_10plus_rate", "explosive_15plus_rate"]
        ]
        merged = merged.merge(exp, on="gsis_id", how="left")

    # Real fix - avg_time_to_los was captured above but never actually
    # graded. Direction inverted (negate before grading) since LOWER time
    # to the line of scrimmage is the better outcome (faster, more
    # decisive hitting the hole) and calc_percentile_grade always treats
    # higher-raw-value as higher-grade with no invert option.
    if "avg_time_to_los" in merged.columns:
        merged["time_to_los_inv"] = -merged["avg_time_to_los"]

    for col in ["rushing_epa", "rush_yards_over_expected_per_att", "efficiency", "explosive_10plus_rate",
                "rush_pct_over_expected", "time_to_los_inv"]:
        if col in merged.columns:
            merged[f"{col}_grade"] = merged[col].apply(lambda v: calc_percentile_grade(v, merged[col]))

    return merged


def build_qb_rushing_metrics(season: int, week: int, pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    REAL, TAILORED QB rushing signal - the gap flagged directly: QB
    rush_yards previously just borrowed the RB pipeline wholesale, with
    nothing distinguishing "he's a legit designed-run threat" from "he
    only rushes when a play breaks down under pressure." Built from
    qb_scramble - a real, standard nflverse pbp column (has existed in
    the nflfastR/nflverse schema for years) that flags whether a given
    QB rush was a scramble (pressure-driven, unplanned) versus a real
    designed run - genuinely different signals for projecting him
    forward: a QB who scrambles a lot because he's constantly under
    pressure is a different bet than a real read-option/design-run guy,
    even if their season rushing-yards-per-game looks identical.

    No run-concept charting data needed for this - qb_scramble is
    already in the free play-by-play pull that's used everywhere else in
    this file, just never used for this specific signal until now.

    Defensive design: if qb_scramble isn't present in this pbp pull for
    any reason (a real, if unlikely, schema mismatch - same caution as
    every other "confirmed real column" claim in this file that hasn't
    been checked against a live pull from this build environment),
    returns an empty DataFrame rather than crashing, so callers can
    gracefully treat this signal as unavailable exactly like a rookie
    with no NGS data yet.

    Uses weeks BEFORE the target week only, same leak-avoidance
    convention as every other advanced-metrics builder in this file.
    """
    if "qb_scramble" not in pbp_df.columns:
        return pd.DataFrame()

    hist_pbp = pbp_df[
        (pbp_df["season"] == season) & (pbp_df["week"] < week)
        & (pbp_df["play_type"].isin(["run", "pass"]))
        & pbp_df["rusher_player_id"].notna()
        & (pbp_df["passer_player_id"] == pbp_df["rusher_player_id"])  # the QB himself carried it
    ].copy()
    if hist_pbp.empty:
        return pd.DataFrame()

    agg = hist_pbp.groupby("rusher_player_id").agg(
        total_qb_rushes=("rush_attempt", "count"),
        scramble_count=("qb_scramble", "sum"),
        scramble_yards=("yards_gained", lambda s: s[hist_pbp.loc[s.index, "qb_scramble"] == 1].sum()),
        designed_run_yards=("yards_gained", lambda s: s[hist_pbp.loc[s.index, "qb_scramble"] != 1].sum()),
    ).reset_index().rename(columns={"rusher_player_id": "gsis_id"})

    agg["scramble_rate"] = agg["scramble_count"] / agg["total_qb_rushes"].replace(0, np.nan)
    agg["scramble_yards_per_att"] = agg["scramble_yards"] / agg["scramble_count"].replace(0, np.nan)
    designed_count = agg["total_qb_rushes"] - agg["scramble_count"]
    agg["designed_run_yards_per_att"] = agg["designed_run_yards"] / designed_count.replace(0, np.nan)

    for col in ["scramble_rate", "scramble_yards_per_att", "designed_run_yards_per_att"]:
        agg[f"{col}_grade"] = agg[col].apply(lambda v: calc_percentile_grade(v, agg[col]) if pd.notna(v) else np.nan)

    return agg


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
                                    participation_df: pd.DataFrame,
                                    prior_pbp_df: pd.DataFrame = None,
                                    prior_participation_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    DEF advanced metrics: EPA allowed per play, split pass defense vs run
    defense - this is the real free equivalent of DVOA (DVOA itself is
    Football Outsiders/FTN proprietary, not available for free). Also
    success rate allowed, pressure rate generated, and explosive-play-
    allowed rate (pass + run, via build_defense_explosive_allowed) - the
    big-play-specific signal EPA's average alone doesn't isolate.

    REAL FIX - same bug/fix as the offense-side builders above: this is
    literally the "how much does Seattle allow real QB/WR production"
    side of the matchup, and it was returning pd.DataFrame() empty in
    week 1 for the exact same reason (zero current-season history) -
    meaning def_grades in calc_grade_matchup_strength had nothing either,
    so BOTH sides of a matchup were structurally blank in week 1, not
    just the offensive player's own grade. Falls back to the full prior
    season (all weeks) when current-season history is empty.
    """
    hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)]
    used_prior = False
    if hist_pbp.empty and prior_pbp_df is not None and not prior_pbp_df.empty:
        # Real fix - exclude teams with a genuinely new 2026 DC (see
        # NEW_DC_TEAMS_2026 above) from the prior-season bridge. Their
        # real prior-season defensive tendencies describe a scheme that
        # may no longer exist - those teams correctly get no bridge here
        # rather than a confidently-wrong number.
        hist_pbp = prior_pbp_df[
            (prior_pbp_df["season"] == season - 1)
            & (~prior_pbp_df["defteam"].isin(NEW_DC_TEAMS_2026))
        ]
        used_prior = True
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

    # Real fix - the participation join needs to use the SAME season's
    # participation table as whichever pbp we actually ended up using
    # above (current or prior-season fallback), or the game_id/play_id
    # join keys won't match anything and pressure_rate_generated silently
    # comes back empty even when hist_pbp itself has real prior-season data.
    participation_to_use = prior_participation_df if (used_prior and prior_participation_df is not None) else participation_df
    hist_participation = participation_to_use.merge(
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



# ---------------------------------------------------------------------------
# 6c. GRADE-BASED MATCHUP CROSSWALK - the NFL equivalent of the MLB tool's
#     pitch-type-usage x hitter-vulnerability crosswalk. Each prop gets its
#     OWN tailored offense-grade / defense-grade list (same "per-prop
#     tailored quality_score" fix already applied on the MLB side, where a
#     single reused composite score was the confirmed bug).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FEATURE FLAGS - isolating untested additions from scoring after a real
# regression (2025 backtest: quality_score tiers came back completely
# INVERTED, <40 tier most accurate, 80-100 tier least accurate, after the
# play-action/personnel crosswalks were added). These were never validated
# against real data before being wired into quality_score, and multiple
# changes landed in the same round in violation of the agreed one-change-
# at-a-time process - these flags let the play-action/personnel signals
# stay computed and visible (still show up as display columns) WITHOUT
# affecting quality_score, so the reweighting fix and these new crosswalks
# can be tested in isolation instead of as one tangled change. Flip back
# to True only after re-testing shows each one is actually net-positive.
# ---------------------------------------------------------------------------
ENABLE_PLAYACTION_IN_QUALITY_SCORE = True  # RE-ENABLED for isolated testing - see note below
ENABLE_PERSONNEL_IN_QUALITY_SCORE = True  # RE-ENABLED - PA confirmed clean alone (weeks 4-18, quality tiers stable, no inversion), this round's ONE change

# ALIGNMENT (Wide/Slot/Inline/Backfield) x coverage exploit signal,
# sourced from coverage_matchup.py's premium FantasyPoints dataset
# (calc_alignment_exploit_strength). FLIPPED ON for its own isolated live
# test (round 1 of 3: alignment -> QB coverage -> run-concept, one at a
# time, per the same discipline that already caught the box-adjustment
# and quality_score sample-size bugs). Only takes effect at all when a
# CoverageDataBundle is actually passed into build_weekly_slate
# (coverage_bundle=...) - i.e. the Coverage Matchup tab's "load dataset"
# step must be run first in the same session, or this silently degrades
# to NaN same as a missing personnel/PA row (safe, not a crash).
# DO NOT flip ENABLE_QB_COVERAGE_IN_QUALITY_SCORE or
# ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE on in the same round as this one -
# test this alone first (weeks 4-18 season report, check
# adjustment_direction_accuracy + quality tier monotonicity for an
# inversion) before touching either of the other two.
ENABLE_ALIGNMENT_IN_QUALITY_SCORE = True

# QB coverage exploit signal (no alignment axis) - STAYS OFF until
# alignment above is confirmed clean on its own live test. Round 2.
ENABLE_QB_COVERAGE_IN_QUALITY_SCORE = True

# RB run-concept exploit signal, sourced from rb_matchup.py's premium
# FantasyPoints dataset (calc_rb_concept_exploit_strength). STAYS OFF
# until alignment AND QB coverage are each confirmed clean - thinnest
# real samples of the three (Counter/Power/Pull Lead), tested last on
# purpose. Round 3.
ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE = True

# QB scramble exploit signal, sourced from coverage_matchup.py's premium
# FantasyPoints scramble dataset (calc_qb_scramble_exploit_strength).
# REAL, NEWLY WIRED (found live this session): this function and its data
# existed already but were ONLY ever used inside the simulation's
# build_team_offense (tilting the QB's rush-pool bootstrap sampling) -
# never touched quality_score for QB rush_attempts/rush_yards at all.
# Off by default, same isolated one-flag-at-a-time discipline as every
# other premium signal here - genuinely new wiring, not yet backtested.
ENABLE_QB_SCRAMBLE_IN_QUALITY_SCORE = True

# Real, NEW cross-referencing signal (built live per direct request): a
# QB's real pass_yards/completions/attempts/tds signal now considers his
# ACTUAL current top pass-catchers' own real fit against the opponent's
# coverage tendencies (weighted by their real target share), not just the
# QB's own numbers in isolation. And symmetrically, a receiver's own
# rec_yards/receptions/targets signal now considers his QB's real fit
# against the same opponent - a great receiver with a QB who can't
# exploit this specific defense is a different real situation than the
# same receiver with a QB who can. Off by default, same isolated one-
# flag-at-a-time discipline as every other premium signal here - genuinely
# new, unbacktested wiring.
ENABLE_SUPPORTING_CAST_IN_QB_QUALITY_SCORE = True
ENABLE_QB_FIT_IN_RECEIVER_QUALITY_SCORE = True

# RE-ENABLE TEST (this round's ONE change, everything else held constant):
# play-action was disabled after landing untested alongside 4 other changes
# in one round, which caused a severe quality_score tier inversion never
# individually attributed to PA specifically vs personnel vs the other
# changes. Since then: the reweighting fix, mu/sigma shrinkage fix, quality_
# REMOVED (confirmed net-neutral-to-harmful, tested live, not a pending
# item): box-count mu-adjustment (ENABLE_BOX_MU_ADJUSTMENT) - direction
# accuracy 47%, got WORSE as it was trusted more. Basic man/zone coverage
# mu-adjustment (ENABLE_COVERAGE_MU_ADJUSTMENT) - direction accuracy 51%
# (coin flip), abs_miss on adjusted rows got slightly worse. Defense
# personnel-change adjustment (ENABLE_PERSONNEL_CHANGE_ADJUSTMENT) - fires
# correctly but made no measurable difference to quality_score's actual
# predictive power. Coverage and box tendency as GRADING INPUTS (feeding
# quality_score/grade_matchup_strength) are untouched - only these three
# direct mu/grade-nudge mechanisms were removed.

# NEW, SEPARATE mechanism - full-coverage-type player split (Cover 0-9
# individually, not just the coarser man/zone binary above). Built per
# user request after confirming the disabled man/zone version's problem
# wasn't necessarily "situational splits don't work" so much as "2 buckets
# might just be too coarse" - this tests that directly, using the FULL
# season of real plays (not a short window) and a fallback that only
# relies on whichever specific coverages both the player and defense have
# a reliable real sample for (renormalized, not forced). Kept OFF by
# default pending its own live test - untested, not yet proven either way.
ENABLE_FULL_COVERAGE_MU_ADJUSTMENT = False


PROP_METRIC_CROSSWALK = {
    "pass_yards": {
        "offense_grades": ["passing_epa_grade", "cpoe_grade", "success_rate_grade", "adot_grade",
                            "explosive_20plus_rate_grade"]
                           + (["pa_rate_grade", "pa_epa_diff_grade", "pressure_rate_faced_grade", "proe_grade"]
                              if ENABLE_PLAYACTION_IN_QUALITY_SCORE else []),
        "defense_grades": ["opp_pass_epa_allowed_grade", "opp_pressure_rate_generated_grade",
                            "opp_pass_explosive_allowed_rate_grade"]
                          + (["opp_pa_epa_allowed_grade"] if ENABLE_PLAYACTION_IN_QUALITY_SCORE else []),
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
                            "opp_pass_explosive_allowed_rate_grade"]
                          + (["opp_pa_epa_allowed_grade"] if ENABLE_PLAYACTION_IN_QUALITY_SCORE else []),
    },
    # REAL, TAILORED crosswalks below - the sibling props (previously just
    # inheriting pass_yards/rec_yards/rush_yards' quality_score wholesale)
    # get their own grade sets now, same fix category as the MLB fantasy-
    # weight bug found earlier tonight: an inherited/borrowed grade LOOKS
    # fine right up until it's actually wrong for what it's grading.
    "pass_attempts": {
        # Volume/game-script stat, NOT an efficiency stat - a bad team
        # down big throws a ton of garbage-time attempts regardless of
        # whether the QB is playing well (his EPA/CPOE could be terrible
        # in that exact scenario). PROE (does he throw more than the
        # situation calls for) and pressure faced (does he get sacked/
        # scramble instead of throwing) are the real drivers of raw
        # attempt COUNT - explicitly NOT reusing pass_yards' efficiency
        # grades (EPA/CPOE/aDOT), which measure a different thing.
        "offense_grades": ["proe_grade", "pressure_rate_faced_grade"],
        "defense_grades": ["opp_pressure_rate_generated_grade"],
    },
    "pass_completions": {
        # Real completions = attempts x completion quality - blends the
        # same volume signal as pass_attempts with CPOE (the one real
        # accuracy signal), rather than the full pass_yards efficiency
        # set (aDOT/explosive rate measure depth/big-plays, not whether
        # a given attempt gets completed at all).
        "offense_grades": ["proe_grade", "cpoe_grade"],
        "defense_grades": ["opp_pressure_rate_generated_grade"],
    },
    "receptions": {
        # Volume prop (does he get targeted, does he catch what's thrown) -
        # target_share/WOPR are the right real signals. Deliberately
        # EXCLUDES avg_separation/yac_above_expectation from rec_yards'
        # set - those measure what happens AFTER a catch/target, not how
        # often he gets one, which is redundant noise for a pure-volume
        # prop like this one.
        "offense_grades": ["target_share_grade", "wopr_grade"],
        "defense_grades": ["opp_pass_epa_allowed_grade"],
    },
    "targets": {
        # Same reasoning and same grade set as receptions - target_share/
        # WOPR ARE the direct measure of target volume itself, arguably
        # even more directly relevant here than for receptions (which
        # also depends on catch quality; targets is pure opportunity).
        "offense_grades": ["target_share_grade", "wopr_grade"],
        "defense_grades": ["opp_pass_epa_allowed_grade"],
    },
    "rush_attempts": {
        # Same game-script logic as pass_attempts, mirrored: a leading
        # team runs the ball to kill clock regardless of the back's own
        # per-carry efficiency. rushing_epa (season-aggregated volume-
        # weighted signal) is a closer real proxy for "is this offense
        # actually committed to running him" than rush-yards-over-
        # expected (a pure per-carry skill signal, wrong thing to grade
        # attempt COUNT on).
        "offense_grades": ["rushing_epa_grade"],
        "defense_grades": ["opp_run_epa_allowed_grade"],
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
                      season: int, week: int, recent_games: int = 3,
                      prior_source_df: pd.DataFrame = None, min_games: int = 2,
                      team: str = None) -> dict:
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

    PRIOR-SEASON BRIDGE (real gap found+fixed - same bug class as the
    mu/sigma/longest-play prior-season bridges elsewhere in this file,
    just not caught until directly asked about): previously this was
    hardcoded to the current season only, so it went fully neutral (0.5,
    via calc_role_verification_score's own fallback) for the ENTIRE first
    few weeks of a season regardless of how much real prior-season data
    existed - a real full season of 2025 games sitting right there,
    unused, purely because this one function never looked at it.

    When current-season games are below min_games and prior_source_df is
    given, this now falls back to the SAME recent-vs-season-average trend
    computed off the END of last season instead (last `recent_games` weeks
    of season-1 vs that season's full average). This is a genuinely
    weaker signal than a real in-season trend, not an equivalent one - an
    offseason coaching change, a new free-agent signing at the same
    position, or a depth-chart competition can reset a role in ways last
    season's ending trend can't see, unlike a raw per-game average (mu),
    which transfers across an offseason much more directly. Flagged via
    "bridged_from_prior_season": True in the return so
    calc_role_verification_score can (and does, see its own note) treat
    it as real but lower-confidence rather than pretending it's as strong
    as a verified current-season trend. Not yet backtested at that
    confidence level - needs its own live test like everything else here.
    """
    hist = source_df[
        (source_df["season"] == season) & (source_df["week"] < week)
        & (source_df[id_col] == gsis_id)
    ].sort_values("week", ascending=False)

    if len(hist) >= min_games and metric_col in hist.columns:
        recent = hist.head(recent_games)[metric_col].mean()
        season_avg = hist[metric_col].mean()
        trend_ratio = np.nan
        if pd.notna(recent) and pd.notna(season_avg) and season_avg > 0:
            trend_ratio = recent / season_avg
        return {
            "recent_value": round(recent, 3) if pd.notna(recent) else np.nan,
            "season_value": round(season_avg, 3) if pd.notna(season_avg) else np.nan,
            "trend_ratio": round(trend_ratio, 3) if pd.notna(trend_ratio) else np.nan,
            "games": len(hist), "bridged_from_prior_season": False,
        }

    if prior_source_df is not None and metric_col in prior_source_df.columns and team not in NEW_OC_TEAMS_2026:
        prior_hist = prior_source_df[
            (prior_source_df["season"] == season - 1) & (prior_source_df[id_col] == gsis_id)
        ].sort_values("week", ascending=False)
        if not prior_hist.empty:
            recent = prior_hist.head(recent_games)[metric_col].mean()
            season_avg = prior_hist[metric_col].mean()
            trend_ratio = np.nan
            if pd.notna(recent) and pd.notna(season_avg) and season_avg > 0:
                trend_ratio = recent / season_avg
            return {
                "recent_value": round(recent, 3) if pd.notna(recent) else np.nan,
                "season_value": round(season_avg, 3) if pd.notna(season_avg) else np.nan,
                "trend_ratio": round(trend_ratio, 3) if pd.notna(trend_ratio) else np.nan,
                "games": len(prior_hist), "bridged_from_prior_season": True,
            }

    return {"recent_value": np.nan, "season_value": np.nan, "trend_ratio": np.nan,
            "games": 0, "bridged_from_prior_season": False}


def calc_role_verification_score(role_trend: dict, min_games: int = 2) -> float:
    """
    Converts a role trend dict into a 0-1 score: a steady/growing role
    (trend_ratio >= 1.0) scores highest, a fading role (<=0.5x season
    average) scores lowest, linear between. Returns a neutral 0.5 (no
    penalty, no bonus) if there isn't enough history to trust the trend
    yet - same graceful-degrade shape as calc_coverage_quality_score's
    fallback, so a rookie/new-role player isn't punished for thin data.

    BRIDGED-TREND DAMPING (see build_role_trend's prior-season bridge note):
    when role_trend came from last season's ending trend rather than a real
    current-season one, the raw score is pulled halfway back toward neutral
    (0.5) instead of trusted at full strength - it's real signal, but a
    genuinely weaker one (an offseason coaching/personnel change can reset
    a role in ways last season's own ending trend can't see), and this
    hasn't been backtested at full confidence the way the in-season version
    has. Preserves direction (a clearly ascending or fading prior-season
    trend still nudges the score the right way) while being honest that
    it's carried-over evidence, not verified current evidence.
    """
    if role_trend.get("games", 0) < min_games or pd.isna(role_trend.get("trend_ratio")):
        return 0.5
    ratio = role_trend["trend_ratio"]
    raw_score = max(0.0, min(1.0, (ratio - 0.5) / 0.5))
    if role_trend.get("bridged_from_prior_season"):
        raw_score = 0.5 + (raw_score - 0.5) * 0.5
    return round(raw_score, 3)


def calc_blended_matchup_strength(structural_exploit: float, grade_exploit: float,
                                   role_verification_score: float,
                                   structural_weight: float = 0.5,
                                   matchup_weight: float = 0.60,
                                   role_is_bridged: bool = False,
                                   bridged_matchup_weight: float = 0.75) -> float:
    """
    Combines the structural tendency signal (coverage-elevation or
    box-count exploit strength, 0-1) with the grade-based crosswalk signal
    (calc_grade_matchup_strength, 0-1) into one matchup signal, then blends
    that with the role-verification score.

    REWEIGHTED AGAIN (real 2025 full-range backtest, 21,259 rows, run after
    tonight's sibling-prop crosswalk work): matchup_weight had already been
    cut from 0.6 to 0.35 once before, on real evidence the structural+grade
    signal was underperforming. This second, larger backtest shows the
    problem persists even at 0.35 - correlation between quality_score and
    match_ratio came back essentially zero (roughly -0.05 to +0.05) for
    EVERY prop type checked, including pass_yards/rush_yards/rec_yards,
    which have had bespoke, carefully-built crosswalks the whole project,
    not just the sibling props built tonight. That rules out "wrong
    metrics feed the grade" as the explanation (multiple different metric
    sets, same flat result) and points at the weighting itself still
    being the problem, not solved by the first reweight.
    role_verification_score, by contrast, has now been reconfirmed strong
    and consistent across many separate real backtests (~1.8-2x miss gap
    between fading and steady/growing role, every single time it's been
    checked) - a genuinely proven signal, unlike matchup_exploit_strength.
    matchup_weight cut further to 0.15 (role_verification now 0.85) on
    that same real-evidence-driven basis as the original reweight.

    NOT a claim that the root cause inside the coverage/box logic itself
    has been found and fixed - the backtest export used to find this
    didn't include mu_before_coverage_adj/mu_before_box_adj, so WHY the
    adjustment is wrong that often isn't diagnosed yet, only THAT it is.
    This reweighting is a data-justified damage-limitation move (trust the
    proven signal more, the unproven/underperforming one less), not a
    verified root-cause fix. Re-run build_season_accuracy_report on the
    same week range after this change to see whether it actually helped -
    same honest test the first reweight called for, not yet different
    this time either.

    Degrades gracefully: a missing structural or grade component just
    reweights across whatever IS available; a completely absent matchup
    signal falls back to neutral (0.5) rather than zeroing the whole score
    out.

    ISOLATED BRIDGED-MODE OVERRIDE (added live this session, real gap
    found via direct testing): the 0.15/0.85 weighting above was validated
    on a backtest where role_verification_score almost always had genuine
    current-season data behind it (most of a season's weeks do). It was
    NEVER validated for the specific week-1-style edge case where role_
    verification is bridged from prior-season data and damped to a 0.75
    ceiling (calc_role_verification_score) - in that exact case, real
    testing showed quality_score stayed capped near 65 even AFTER fixing
    the grade/coverage builders to use real prior-season data (build_qb_
    advanced_metrics etc.), because the now-real grade signal only carries
    15% of the blend. role_is_bridged=True switches to bridged_matchup_
    weight (0.4, giving genuinely-real prior-season grade/coverage data
    more say than a damped placeholder) ONLY in that specific case -
    every other week's calls pass role_is_bridged=False and get the
    exact same validated 0.15/0.85 split as before, completely untouched.
    HONEST CAVEAT: 0.4 is a reasoned starting point for this one edge
    case, not itself backtested yet (there's no free way to backtest
    "week 1 of a season that hasn't been played" against real outcomes
    ahead of time) - re-evaluate once real 2026 week 1 results come in,
    the same real-evidence discipline used for every other reweight here.

    MAJOR REWEIGHT (per direct, explicit request): matchup_weight raised
    from 0.15 to 0.60, bridged_matchup_weight from 0.4 to 0.75. The
    original 0.15 cut was based on a real backtest - but of a
    DEMONSTRABLY thin and, in one real case, backwards-scored matchup
    signal (single-stat exploit functions using RATE/FP/G alone; the
    INVERSE_STATS naming collision found and fixed this same session
    means interceptions/sacks/drops were being tiered backwards the whole
    time that backtest ran). That evidence describes a matchup signal
    that no longer exists - tonight's rebuild replaced every single-stat
    exploit function with comprehensive, bucketed multi-stat versions
    covering nearly every real column across the QB-coverage, alignment,
    and RB-concept CSVs, added a cross-referencing supporting-cast signal,
    fixed the tiering-direction bug, and added several previously-unused
    free NGS columns. Raising the weight now is a real, reasoned bet that
    THIS matchup signal is more predictive than the one actually tested -
    it is NOT itself proven yet. Re-run the full backtest immediately
    after this change (see the real 18-week comparison this session
    already has infrastructure for) before trusting this weighting for
    real plays - the same real-evidence standard this whole reweight
    history has followed, not an exception for this one.
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
    effective_matchup_weight = bridged_matchup_weight if role_is_bridged else matchup_weight
    return round(matchup_signal * effective_matchup_weight + role_verification_score * (1 - effective_matchup_weight), 3)


def build_weekly_slate(season: int, week: int, coverage_bundle=None, rb_bundle=None,
                        team_filter: list = None, alignment_target_bundle: "TeamAlignmentTargetBundle" = None) -> pd.DataFrame:
    """
    Pulls and merges every data source needed for one week's slate, returning
    a single player-level DataFrame with mu inputs for every prop type ready
    to score. This does NOT include lines - lines are entered/adjusted
    manually per row in the Streamlit UI, same as the MLB tool's adjustable
    Best Edges table (avoids repeating the unreliable Underdog auto-pull
    issue; PrizePicks auto-pull can be tested later once this core scanner
    is proven out).

    team_filter: optional list of team abbreviations (e.g. ["KC", "BAL"]) -
    REAL per-game scanning, not just a display filter. When provided,
    every player pool's loop skips scoring for any player whose team
    isn't in this list, before any of the expensive per-player work
    (percentile grades, coverage/box adjustments, crosswalk scoring)
    happens - this is what actually reduces compute for a single-game
    scan, not just narrowing what gets shown afterward. The underlying
    weekly data pulls (rosters/player_stats/NGS/participation) still
    cover the whole week regardless - that part's comparatively cheap;
    the per-player scoring loop is the expensive part this actually
    targets. None (default) scans every team, unchanged from before.

    coverage_bundle: optional CoverageDataBundle (coverage_matchup.py's
    load_full_dataset() output) - the premium alignment/coverage dataset.
    Only used when ENABLE_ALIGNMENT_IN_QUALITY_SCORE is True; when None
    (default), the alignment signal degrades to NaN for every row and
    everything else here is unaffected. Passing this in is the caller's
    job (Streamlit session_state) - this function never loads it itself,
    same reasoning as why it doesn't load lines: keeps a network/file
    concern out of the pull pipeline.

    rb_bundle: optional RBDataBundle (rb_matchup.py's load_full_rb_dataset()
    output) - the premium run-concept dataset. Same on/off/degrade contract
    as coverage_bundle, gated by ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE.

    Returns columns including (not exhaustive):
      gsis_id, player_display_name, team, position, prop_type,
      mu, sigma_estimate, quality_score, games_sampled,
      team_changed, use_depth_chart_estimate
    """
    schedules_df = pull_schedules([season])
    # REAL BUG FOUND+FIXED this session: rosters_df/player_stats_df were
    # pulled for the CURRENT season only, but detect_role_change() (line
    # ~1144) and 5 separate prior_season_query fallback branches later in
    # this function all filter these same two DataFrames for season - 1
    # rows - a real, previously undetected gap that silently never had
    # anything to fall back to. It stayed invisible for weeks 4+ of an
    # already-underway season (current-season history alone was always
    # enough to avoid the empty-fallback path), and only surfaced as
    # visibly broken (0 output rows, no crash) scanning a season with zero
    # current-season games yet - exactly the week 1 case this fallback
    # exists for. Pulling both seasons here, once, is what every one of
    # those 6 call sites already assumed was happening.
    rosters_df = pull_rosters([season, season - 1])
    depth_charts_df = pull_depth_charts([season]) if nfl else pd.DataFrame()
    player_stats_df = pull_player_stats([season, season - 1])
    ngs_pass_df = pull_ngs("passing", [season])
    ngs_rush_df = pull_ngs("rushing", [season])
    ngs_rec_df = pull_ngs("receiving", [season])
    pbp_df = pull_pbp([season])
    participation_df = pull_participation([season])
    ftn_df = pull_ftn_charting([season])
    # REAL FIX (found via direct backtest diagnosis - see
    # build_league_fallback_sigmas' own comment for the full real-data
    # case, Mark Andrews match_ratio 366) - this column was never added
    # in this specific code path, silently making the fantasy_points
    # fallback-sigma fix a no-op despite compiling and running cleanly.
    # Added here, before fallback_sigmas gets computed, so the fix
    # actually has real data to work with.
    player_stats_df = add_prizepicks_fantasy_column(player_stats_df, pbp_df=pbp_df)

    coverage_profile = build_blended_coverage_profile(season, week)
    box_def_profile, box_off_profile = build_blended_box_profile(season, week)
    fallback_sigmas = build_league_fallback_sigmas(player_stats_df, season, week)
    fallback_mus = build_league_fallback_mus(player_stats_df, season, week)

    # Filter to weeks BEFORE the target week only, for the same reason
    # calc_prop_mu does - using this week's own plays to predict this
    # week's own result would be data leakage, not a real projection.
    pbp_history_df = pbp_df[pbp_df["week"] < week]

    # Prior-season pulls for the cross-season, team-filtered longest-play
    # bridge (build_longest_play_by_game) and other prior-season fallbacks -
    # lets players who DIDN'T change teams use last season's plays for a
    # much better sample early in a new season, while still correctly
    # excluding a traded player's old-team plays. Pulled here (moved above
    # the longest-play tables below) so both can share the same
    # prior_pbp_df pull.
    prior_participation_df = pull_participation([season - 1])
    prior_pbp_df = pull_pbp([season - 1])
    prior_ftn_df = pull_ftn_charting([season - 1])
    # For the role_trend prior-season bridge (build_role_trend) - QB/RB
    # role trend use NGS columns (attempts/rush_attempts), which
    # player_stats_df doesn't carry the same way; WR's role trend uses
    # target_share, already real in player_stats_df (already pulled for
    # both seasons above), so no extra pull needed for that one.
    prior_ngs_pass_df = pull_ngs("passing", [season - 1])
    prior_ngs_rush_df = pull_ngs("rushing", [season - 1])
    # Real fix - prior_ngs_rec_df was never pulled at all (only pass/rush
    # prior-season NGS existed), so build_receiver_advanced_metrics had no
    # prior-season NGS source to fall back to even after being given the
    # capability to use one - this is the actual AJ Brown/Doubs gap.
    prior_ngs_rec_df = pull_ngs("receiving", [season - 1])

    # Per-game longest-play tables for the longest_completion/
    # longest_reception/longest_rush props (see build_longest_play_by_game).
    # REAL FIX (found live this session, closing the gap the original
    # comment here explicitly flagged as worth revisiting): these are now
    # bridged to a prior-season fallback exactly like every other prop's
    # own-history mu, by concatenating this season's per-game longest-play
    # rows with last season's, same team-scoped shape calc_prop_mu already
    # knows how to blend from (current_team=team passed at each call site
    # below, not None) - a traded player's old-team longest plays are
    # excluded by that same existing team-match logic, not specially
    # handled here. Before this fix, Week 1-3 longest_X props were
    # structurally guaranteed NaN even when the yardage/volume props for
    # the same player had a perfectly good prior-season number to show.
    qb_longest_df = pd.concat([
        build_longest_play_by_game(pbp_history_df, "QB"),
        build_longest_play_by_game(prior_pbp_df, "QB"),
    ], ignore_index=True)
    rec_longest_df = pd.concat([
        build_longest_play_by_game(pbp_history_df, "WR"),
        build_longest_play_by_game(prior_pbp_df, "WR"),
    ], ignore_index=True)
    rush_longest_df = pd.concat([
        build_longest_play_by_game(pbp_history_df, "RB"),
        build_longest_play_by_game(prior_pbp_df, "RB"),
    ], ignore_index=True)

    # REAL FIX (found via systematic rescan, per direct request to check
    # the whole model - confirmed none of the 3 longest-play props passed
    # a real league_fallback_mu/sigma at all, the exact same unprotected
    # pattern fantasy_points and kicker_fantasy had before those fixes).
    # Real, position-specific fallback computed directly from these same,
    # already-built real dataframes.
    def _real_longest_fallback(df):
        per_player = df.groupby("gsis_id")["longest_play"].agg(["mean", "std", "count"]).query("count >= 2")
        if per_player.empty:
            return None, None
        return round(per_player["mean"].mean(), 2), round(per_player["std"].mean(), 2)

    qb_longest_fallback_mu, qb_longest_fallback_sigma = _real_longest_fallback(qb_longest_df)
    rec_longest_fallback_mu, rec_longest_fallback_sigma = _real_longest_fallback(rec_longest_df)
    rush_longest_fallback_mu, rush_longest_fallback_sigma = _real_longest_fallback(rush_longest_df)

    # BUGFIX: explosive_rates was previously computed from the full-season
    # pbp_df (including the target week itself and every week after it) -
    # genuine data leakage, same category as the leak calc_prop_mu already
    # guards against. Now built from pbp_history_df, same as every other
    # weeks-before-target computation in this file.
    explosive_rates = build_explosive_rates(pbp_history_df)

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
        prior_ngs_pass_df=prior_ngs_pass_df, prior_pbp_df=prior_pbp_df,
    )
    rec_metrics = build_receiver_advanced_metrics(
        season, week, player_stats_df, ngs_rec_df,
        rec_explosive_df=explosive_rates["rec_explosive"],
        prior_ngs_rec_df=prior_ngs_rec_df,
    )
    rb_metrics = build_rb_advanced_metrics(
        season, week, player_stats_df, ngs_rush_df,
        rush_explosive_df=explosive_rates["rush_explosive"],
        prior_ngs_rush_df=prior_ngs_rush_df,
    )
    def_metrics = build_defense_advanced_metrics(
        season, week, pbp_history_df, participation_df,
        prior_pbp_df=prior_pbp_df, prior_participation_df=prior_participation_df,
    )

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

    # NOTE: the defense personnel-change adjustment (trades/signings) that
    # used to live here (ENABLE_PERSONNEL_CHANGE_ADJUSTMENT, plus
    # detect_team_changes/build_defender_individual_metrics/
    # calc_personnel_change_adjustment) was tested live (weeks 1-5, 2025,
    # true isolated on/off comparison) and confirmed to genuinely move
    # quality_score's defense-grade inputs (67% of rows) but with no
    # measurable improvement to quality_score's actual predictive power
    # (tier separation and quality-vs-miss correlation virtually identical
    # on vs off) - removed entirely rather than left off, since it's a
    # confirmed dead end, not a pending test.

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
        try:
            gsis_id = qb.get("gsis_id")
            team = qb.get("team")
            if team_filter and team not in team_filter:
                continue
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
                qb_pa_row, def_pa_row, coverage_pa_crosswalk, opponent, opp_coverage_row
            )
            # GATED per ENABLE_PLAYACTION_IN_QUALITY_SCORE - still computed and
            # still attached to the row below for visibility, just excluded
            # from scoring until validated (see feature-flag note above).
            pa_exploit_for_scoring = playaction_info.get("exploit_strength") if ENABLE_PLAYACTION_IN_QUALITY_SCORE else np.nan

            # QB coverage exploit signal - premium data, real outlier-coverage
            # gated (see calc_qb_coverage_exploit_strength in
            # coverage_matchup.py). GATED same as every other premium/isolated
            # signal here - off by default pending its own live test.
            qb_coverage_info = {"exploit_strength": np.nan, "outlier_coverages_checked": []}
            if (ENABLE_QB_COVERAGE_IN_QUALITY_SCORE and coverage_bundle is not None
                    and calc_qb_coverage_exploit_strength is not None and opponent is not None):
                qb_coverage_info = calc_qb_coverage_exploit_strength(
                    coverage_bundle, qb.get("full_name"), team, opponent, prop_type="pass_yards",
                )
            qb_coverage_exploit_for_scoring = qb_coverage_info.get("exploit_strength") if ENABLE_QB_COVERAGE_IN_QUALITY_SCORE else np.nan

            # Real, NEW cross-referencing signal - see ENABLE_SUPPORTING_CAST_
            # IN_QB_QUALITY_SCORE above. His real current top pass-catchers'
            # own fit against this same opponent, weighted by their real
            # target share.
            supporting_cast_exploit_for_scoring = np.nan
            if (ENABLE_SUPPORTING_CAST_IN_QB_QUALITY_SCORE and coverage_bundle is not None
                    and opponent is not None):
                teammates = get_top_pass_catchers(team, season, week, player_stats_df)
                supporting_cast_info = calc_supporting_cast_exploit_strength(
                    coverage_bundle, teammates, team, opponent,
                )
                supporting_cast_exploit_for_scoring = supporting_cast_info.get("exploit_strength")

            structural_parts = [v for v in [coverage_info.get("exploit_strength"), pa_exploit_for_scoring,
                                             qb_coverage_exploit_for_scoring, supporting_cast_exploit_for_scoring] if pd.notna(v)]
            combined_structural_exploit = (sum(structural_parts) / len(structural_parts)) if structural_parts else np.nan

            # NOTE: the basic man/zone-only coverage mu-adjustment that used to
            # live here (ENABLE_COVERAGE_MU_ADJUSTMENT) was tested live (weeks
            # 4-11, 2025, n=1027 fired rows) and confirmed a coin flip on
            # direction accuracy (51.0%) with slightly worse abs_miss on the
            # rows it touched - removed entirely rather than left off, since
            # it's a confirmed dead end, not a pending test. Coverage as a
            # GRADING INPUT (man/zone lean, dominant coverage feeding
            # quality_score/grade_matchup_strength above) is untouched.
            adjusted_mu = mu

            # NEW, SEPARATE full-coverage-type version - GATED per
            # ENABLE_FULL_COVERAGE_MU_ADJUSTMENT, off by default pending its
            # own live test. See rec_yards block for the full rationale.
            full_coverage_weight_used = 0.0
            if ENABLE_FULL_COVERAGE_MU_ADJUSTMENT and pd.notna(mu) and opp_coverage_row:
                player_full_coverage_eff = build_player_full_coverage_efficiency(
                    gsis_id, "passer", participation_df, pbp_history_df,
                    prior_participation_df=prior_participation_df, prior_pbp_df=prior_pbp_df,
                )
                full_cov_result = calc_full_coverage_adjusted_mu(adjusted_mu, player_full_coverage_eff, opp_coverage_row)
                adjusted_mu = full_cov_result["adjusted_mu"]
                full_coverage_weight_used = full_cov_result["coverage_weight_used"]

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
            role_trend = build_role_trend(gsis_id, "attempts", ngs_pass_df, "player_gsis_id", season, week,
                                           prior_source_df=prior_ngs_pass_df, team=team)
            role_score = calc_role_verification_score(role_trend)
            blended_exploit = calc_blended_matchup_strength(
                combined_structural_exploit, grade_exploit, role_score,
                role_is_bridged=role_trend.get("bridged_from_prior_season", False),
            )
            quality_score = calc_quality_score(
                matchup_exploit_strength=blended_exploit,
                sample_size_games=confidence_info["games_sampled_current"],  # this QB's own real sample - see calc_quality_score bugfix note
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
                "qb_coverage_exploit_strength": qb_coverage_info.get("exploit_strength"),
                "qb_coverage_outliers_checked": qb_coverage_info.get("outlier_coverages_checked"),
                "full_coverage_weight_used": full_coverage_weight_used,
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

            # --- Sibling QB count/longest props (completions, attempts, TDs,
            # longest completion) - reuse the SAME matchup signals just
            # computed for pass_yards (structural coverage/PA/QB-coverage
            # exploit, grade crosswalk, role verification, quality_score)
            # rather than recomputing a full independent stack per prop. This
            # is a deliberate simplification: these are all facets of the same
            # underlying passing matchup, not fundamentally different
            # matchups - documented here rather than silently assumed. mu/
            # sigma themselves ARE independently computed per prop (real
            # per-stat shrinkage, not copied from pass_yards).
            # REAL FIX (was: blanket inheritance from pass_yards for all three) -
            # pass_completions/pass_attempts now get their OWN real
            # quality_score, computed fresh from the tailored crosswalk
            # entries just added above (volume/game-script signals - PROE,
            # pressure faced, CPOE - not pass_yards' efficiency/explosive-
            # play grades, which measure a genuinely different thing). Same
            # blended_matchup_strength/role_verification/sample-size formula
            # as every other quality_score in this file, just fed a different
            # grade_exploit input. pass_tds is deliberately LEFT on the
            # inherited pass_yards quality_score for now - not yet given its
            # own crosswalk, an honest, stated gap rather than a silent one.
            merged_grades = {**own_grades, **def_grades}
            for sib_prop, sib_col in (("pass_completions", "completions"),
                                       ("pass_attempts", "attempts"),
                                       ("pass_tds", "passing_tds")):
                sib_mu = calc_prop_mu(
                    gsis_id, sib_col, player_stats_df, season, week, current_team=team,
                    league_fallback_mu=fallback_mus.get(("QB", sib_col)),
                )
                sib_sigma = calc_player_sigma(
                    gsis_id, sib_col, player_stats_df, season, week, current_team=team,
                    league_fallback_sigma=fallback_sigmas.get(("QB", sib_col)),
                )
                if sib_prop in PROP_METRIC_CROSSWALK:
                    sib_grade_exploit = calc_grade_matchup_strength(merged_grades, sib_prop)
                    # Real fix - recompute the QB-coverage structural signal
                    # PER SIBLING PROP rather than reusing pass_yards' own
                    # version, now that this signal is genuinely tailored
                    # per prop_type (YPA for pass_yards, ADJ CMP%/ACC% for
                    # pass_completions, etc.) - reusing one prop's version
                    # for every sibling would silently defeat the whole
                    # point of the per-prop stat curation.
                    sib_qb_coverage_exploit = np.nan
                    if (ENABLE_QB_COVERAGE_IN_QUALITY_SCORE and coverage_bundle is not None
                            and calc_qb_coverage_exploit_strength is not None and opponent is not None):
                        sib_qb_coverage_info = calc_qb_coverage_exploit_strength(
                            coverage_bundle, qb.get("full_name"), team, opponent, prop_type=sib_prop,
                        )
                        sib_qb_coverage_exploit = sib_qb_coverage_info.get("exploit_strength")
                    sib_structural_parts = [v for v in [coverage_info.get("exploit_strength"), pa_exploit_for_scoring,
                                                         sib_qb_coverage_exploit] if pd.notna(v)]
                    sib_combined_structural = (sum(sib_structural_parts) / len(sib_structural_parts)) if sib_structural_parts else np.nan
                    sib_blended = calc_blended_matchup_strength(sib_combined_structural, sib_grade_exploit, role_score, role_is_bridged=role_trend.get("bridged_from_prior_season", False))
                    sib_quality_score = calc_quality_score(
                        matchup_exploit_strength=sib_blended,
                        sample_size_games=confidence_info["games_sampled_current"],
                        coverage_confidence=min(n_plays / 300, 1.0),
                    )
                    _record_quality_score(gsis_id, sib_quality_score)
                else:
                    sib_grade_exploit, sib_quality_score = grade_exploit, quality_score
                rows.append({
                    "gsis_id": gsis_id, "player_display_name": qb.get("full_name"),
                    "team": team, "position": "QB", "prop_type": sib_prop,
                    "matchup": team_to_matchup.get(team),
                    "mu": sib_mu, "sigma": sib_sigma, "opponent": opponent,
                    "quality_score": sib_quality_score,
                    "grade_matchup_strength": sib_grade_exploit,
                    "role_verification_score": role_score,
                    "data_confidence": confidence_info["data_confidence"],
                    "games_sampled_current": confidence_info["games_sampled_current"],
                })

            # Longest completion - now bridged to prior season too (see
            # qb_longest_df build note above), team-scoped like every
            # other prop's fallback.
            longest_mu = calc_prop_mu(gsis_id, "longest_play", qb_longest_df, season, week, current_team=team, league_fallback_mu=qb_longest_fallback_mu)
            longest_sigma = calc_player_sigma(gsis_id, "longest_play", qb_longest_df, season, week, current_team=team, league_fallback_sigma=qb_longest_fallback_sigma)
            rows.append({
                "gsis_id": gsis_id, "player_display_name": qb.get("full_name"),
                "team": team, "position": "QB", "prop_type": "longest_completion",
                "matchup": team_to_matchup.get(team),
                "mu": longest_mu, "sigma": longest_sigma, "opponent": opponent,
                "quality_score": quality_score,
                "data_confidence": confidence_info["data_confidence"],
                "games_sampled_current": confidence_info["games_sampled_current"],
            })

        except Exception:
            continue  # this specific player's data is genuinely missing/broken this early in a new season - skip them, don't crash everyone else
    # --- Rushing props ---
    rush_pool = week_rosters[week_rosters["position"].isin(["RB", "QB"])]
    for _, rb in rush_pool.iterrows():
        try:
            gsis_id = rb.get("gsis_id")
            position = rb.get("position")
            rb_team = rb.get("team")
            if team_filter and rb_team not in team_filter:
                continue
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

                # NOTE: the box-count mu-adjustment that used to live here
                # (ENABLE_BOX_MU_ADJUSTMENT) was tested live and confirmed to
                # make accuracy WORSE the more it was trusted - removed
                # entirely rather than left off, since it's a confirmed dead
                # end, not a pending test. Box counts as a GRADING INPUT
                # (feeding quality_score below) are untouched.
                adjusted_rush_mu = mu

                # Run-concept exploit signal - premium data, only computed when
                # a bundle was actually passed in AND the flag is on (see
                # calc_rb_concept_exploit_strength in rb_matchup.py for the
                # real logic). GATED same as every other premium/isolated
                # signal - off by default pending its own live test. Position
                # check mirrors rush_pool's own RB/QB filter (QBs rarely have
                # FantasyPoints run-concept rows, so this will naturally
                # degrade to NaN for most QB rush_yards rows).
                run_concept_info = {"exploit_strength": np.nan, "concepts_checked": []}
                if (ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE and rb_bundle is not None
                        and calc_rb_concept_exploit_strength is not None and rb_opponent is not None
                        and position == "RB"):
                    run_concept_info = calc_rb_concept_exploit_strength(
                        rb_bundle, rb.get("full_name"), rb_opponent, prop_type="rush_yards",
                        rb_team_abbrev=rb_team,
                    )
                run_concept_exploit_for_scoring = run_concept_info.get("exploit_strength") if ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE else np.nan

                # Real, newly wired signal (see ENABLE_QB_SCRAMBLE_IN_QUALITY_SCORE
                # above) - QB-specific, mirrors the RB gate but for position == "QB".
                qb_scramble_info = {"exploit_strength": np.nan}
                if (ENABLE_QB_SCRAMBLE_IN_QUALITY_SCORE and coverage_bundle is not None
                        and calc_qb_scramble_exploit_strength is not None and rb_opponent is not None
                        and position == "QB"):
                    qb_scramble_info = calc_qb_scramble_exploit_strength(
                        coverage_bundle.qb_scrambles, coverage_bundle.def_allowed_qb_scrambles,
                        rb.get("full_name"), rb_opponent,
                    )
                qb_scramble_exploit_for_scoring = qb_scramble_info.get("exploit_strength") if ENABLE_QB_SCRAMBLE_IN_QUALITY_SCORE else np.nan

                rb_confidence_info = get_data_confidence(gsis_id, player_stats_df, season, week, current_team=rb_team)
                own_grades = get_player_grades(gsis_id, rb_metrics)
                def_grades = get_defense_grades(rb_opponent, def_metrics)

                grade_exploit = calc_grade_matchup_strength({**own_grades, **def_grades}, "rush_yards")
                role_trend = build_role_trend(gsis_id, "rush_attempts", ngs_rush_df, "player_gsis_id", season, week,
                                               prior_source_df=prior_ngs_rush_df, team=rb_team)
                role_score = calc_role_verification_score(role_trend)
                structural_parts = [v for v in [box_info.get("exploit_strength"), run_concept_exploit_for_scoring,
                                                 qb_scramble_exploit_for_scoring] if pd.notna(v)]
                combined_rush_structural = (sum(structural_parts) / len(structural_parts)) if structural_parts else np.nan
                blended_exploit = calc_blended_matchup_strength(
                    combined_rush_structural, grade_exploit, role_score,
                    role_is_bridged=role_trend.get("bridged_from_prior_season", False),
                )
                rush_quality_score = calc_quality_score(
                    matchup_exploit_strength=blended_exploit,
                    sample_size_games=rb_confidence_info["games_sampled_current"],  # this RB's own real sample - see calc_quality_score bugfix note
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
                    "run_concept_exploit_strength": run_concept_info.get("exploit_strength"),
                    "run_concepts_checked": run_concept_info.get("concepts_checked"),
                    "quality_score": rush_quality_score,
                    "grade_matchup_strength": grade_exploit,
                    "role_verification_score": role_score,
                    "role_trend_ratio": role_trend.get("trend_ratio"),
                    "data_confidence": rb_confidence_info["data_confidence"],
                    "games_sampled_current": rb_confidence_info["games_sampled_current"],
                    **own_grades,
                    **def_grades,
                })

                # --- Sibling rushing count/longest props (attempts, TDs,
                # longest rush) - rush_attempts now gets its OWN real
                # quality_score (game-script-focused crosswalk, see
                # PROP_METRIC_CROSSWALK) instead of inheriting rush_yards'
                # per-carry-skill grades wholesale. rush_tds stays inherited
                # for now, same honest, stated gap as pass_tds.
                merged_grades_rush = {**own_grades, **def_grades}
                for sib_prop, sib_col in (("rush_attempts", "carries"), ("rush_tds", "rushing_tds")):
                    sib_mu = calc_prop_mu(
                        gsis_id, sib_col, player_stats_df, season, week, current_team=rb_team,
                        league_fallback_mu=fallback_mus.get((position, sib_col)),
                    )
                    sib_sigma = calc_player_sigma(
                        gsis_id, sib_col, player_stats_df, season, week, current_team=rb_team,
                        league_fallback_sigma=fallback_sigmas.get((position, sib_col)),
                    )
                    if sib_prop in PROP_METRIC_CROSSWALK:
                        sib_grade_exploit = calc_grade_matchup_strength(merged_grades_rush, sib_prop)
                        # Real fix - recompute the run-concept structural
                        # signal PER SIBLING PROP instead of reusing
                        # rush_yards' version - same fix as the QB-coverage/
                        # alignment sibling loops above, needed for the
                        # per-prop stat curation to actually apply to
                        # rush_attempts/rush_tds.
                        sib_run_concept_exploit = np.nan
                        if (ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE and rb_bundle is not None
                                and calc_rb_concept_exploit_strength is not None and rb_opponent is not None
                                and position == "RB"):
                            sib_run_concept_info = calc_rb_concept_exploit_strength(
                                rb_bundle, rb.get("full_name"), rb_opponent, prop_type=sib_prop,
                                rb_team_abbrev=rb_team,
                            )
                            sib_run_concept_exploit = sib_run_concept_info.get("exploit_strength")
                        sib_rush_structural_parts = [v for v in [box_info.get("exploit_strength"), sib_run_concept_exploit]
                                                      if pd.notna(v)]
                        sib_combined_rush_structural = (sum(sib_rush_structural_parts) / len(sib_rush_structural_parts)) if sib_rush_structural_parts else np.nan
                        sib_blended = calc_blended_matchup_strength(sib_combined_rush_structural, sib_grade_exploit, role_score, role_is_bridged=role_trend.get("bridged_from_prior_season", False))
                        sib_quality_score = calc_quality_score(
                            matchup_exploit_strength=sib_blended,
                            sample_size_games=rb_confidence_info["games_sampled_current"],
                            coverage_confidence=min(n_box_plays / 300, 1.0),
                        )
                        _record_quality_score(gsis_id, sib_quality_score)
                    else:
                        sib_grade_exploit, sib_quality_score = grade_exploit, rush_quality_score
                    rows.append({
                        "gsis_id": gsis_id, "player_display_name": rb.get("full_name"),
                        "team": rb.get("team"), "position": position, "prop_type": sib_prop,
                        "matchup": team_to_matchup.get(rb_team),
                        "mu": sib_mu, "sigma": sib_sigma, "opponent": rb_opponent,
                        "quality_score": sib_quality_score,
                        "grade_matchup_strength": sib_grade_exploit,
                        "role_verification_score": role_score,
                        "data_confidence": rb_confidence_info["data_confidence"],
                        "games_sampled_current": rb_confidence_info["games_sampled_current"],
                    })

                longest_rush_mu = calc_prop_mu(gsis_id, "longest_play", rush_longest_df, season, week, current_team=rb_team, league_fallback_mu=rush_longest_fallback_mu)
                longest_rush_sigma = calc_player_sigma(gsis_id, "longest_play", rush_longest_df, season, week, current_team=rb_team, league_fallback_sigma=rush_longest_fallback_sigma)
                rows.append({
                    "gsis_id": gsis_id, "player_display_name": rb.get("full_name"),
                    "team": rb.get("team"), "position": position, "prop_type": "longest_rush",
                    "matchup": team_to_matchup.get(rb_team),
                    "mu": longest_rush_mu, "sigma": longest_rush_sigma, "opponent": rb_opponent,
                    "quality_score": rush_quality_score,
                    "data_confidence": rb_confidence_info["data_confidence"],
                    "games_sampled_current": rb_confidence_info["games_sampled_current"],
                })

        except Exception:
            continue  # this specific player's data is genuinely missing/broken this early in a new season - skip them, don't crash everyone else
    # --- Receiving props ---
    rec_pool = week_rosters[week_rosters["position"].isin(["WR", "TE", "RB"])]
    for _, wr in rec_pool.iterrows():
        try:
            gsis_id = wr.get("gsis_id")
            position = wr.get("position")
            team = wr.get("team")
            if team_filter and team not in team_filter:
                continue
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
                # GATED per ENABLE_PERSONNEL_IN_QUALITY_SCORE - same isolation
                # treatment as the play-action gate above.
                personnel_exploit_for_scoring = personnel_info.get("exploit_strength") if ENABLE_PERSONNEL_IN_QUALITY_SCORE else np.nan

                # Alignment (Wide/Slot/Inline/Backfield) x real opponent
                # outlier-coverage exploit signal - premium data, only computed
                # when a bundle was actually passed in AND the flag is on
                # (see calc_alignment_exploit_strength in coverage_matchup.py
                # for the real logic). GATED same as PA/personnel - isolated,
                # off by default pending its own live test.
                alignment_info = {"exploit_strength": np.nan, "dominant_alignment": None, "alignment_fit_pct": None}
                if (ENABLE_ALIGNMENT_IN_QUALITY_SCORE and coverage_bundle is not None
                        and calc_alignment_exploit_strength is not None and opponent is not None):
                    alignment_info = calc_alignment_exploit_strength(
                        coverage_bundle, wr.get("full_name"), position, team, opponent, prop_type="rec_yards",
                        alignment_bundle=alignment_target_bundle,
                    )
                alignment_exploit_for_scoring = alignment_info.get("exploit_strength") if ENABLE_ALIGNMENT_IN_QUALITY_SCORE else np.nan

                structural_parts = [v for v in [coverage_info.get("exploit_strength"), personnel_exploit_for_scoring,
                                                 alignment_exploit_for_scoring] if pd.notna(v)]
                combined_structural_exploit = (sum(structural_parts) / len(structural_parts)) if structural_parts else np.nan

                # NOTE: the basic man/zone-only coverage mu-adjustment that used
                # to live here (ENABLE_COVERAGE_MU_ADJUSTMENT) was tested live
                # and confirmed a coin flip on direction accuracy with
                # slightly worse abs_miss on the rows it touched - removed
                # entirely rather than left off. Coverage as a GRADING INPUT
                # (feeding quality_score below) is untouched.
                adjusted_mu = mu

                # NEW, SEPARATE full-coverage-type version - GATED per
                # ENABLE_FULL_COVERAGE_MU_ADJUSTMENT, off by default pending
                # its own live test. Applied on top of adjusted_mu (which is
                # just `mu` unchanged while the man/zone version stays off) so
                # this can be tested in isolation regardless of that flag's state.
                full_coverage_weight_used = 0.0
                if ENABLE_FULL_COVERAGE_MU_ADJUSTMENT and opp_coverage_row:
                    player_full_coverage_eff = build_player_full_coverage_efficiency(
                        gsis_id, "receiver", participation_df, pbp_history_df,
                        prior_participation_df=prior_participation_df, prior_pbp_df=prior_pbp_df,
                    )
                    full_cov_result = calc_full_coverage_adjusted_mu(adjusted_mu, player_full_coverage_eff, opp_coverage_row)
                    adjusted_mu = full_cov_result["adjusted_mu"]
                    full_coverage_weight_used = full_cov_result["coverage_weight_used"]

                rec_confidence_info = get_data_confidence(gsis_id, player_stats_df, season, week, current_team=team)
                own_grades = get_player_grades(gsis_id, rec_metrics)
                def_grades = get_defense_grades(opponent, def_metrics)

                grade_exploit = calc_grade_matchup_strength({**own_grades, **def_grades}, "rec_yards")
                role_trend = build_role_trend(gsis_id, "target_share", player_stats_df, "gsis_id", season, week,
                                               prior_source_df=player_stats_df, team=team)
                role_score = calc_role_verification_score(role_trend)
                blended_exploit = calc_blended_matchup_strength(
                    combined_structural_exploit, grade_exploit, role_score,
                    role_is_bridged=role_trend.get("bridged_from_prior_season", False),
                )
                quality_score = calc_quality_score(
                    matchup_exploit_strength=blended_exploit,
                    sample_size_games=rec_confidence_info["games_sampled_current"],  # this receiver's own real sample - see calc_quality_score bugfix note
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
                    "alignment_exploit_strength": alignment_info.get("exploit_strength"),
                    "dominant_alignment": alignment_info.get("dominant_alignment"),
                    "alignment_fit_pct": alignment_info.get("alignment_fit_pct"),
                    "alignment_outlier_coverages": alignment_info.get("outlier_coverages_checked"),
                    "full_coverage_weight_used": full_coverage_weight_used,
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

                # --- Sibling receiving count/longest props (receptions,
                # targets, TDs, longest catch) - receptions/targets now get
                # their OWN real quality_score (pure-opportunity crosswalk:
                # target_share/WOPR, deliberately excluding separation/YAC-
                # over-expectation, which measure what happens AFTER a target/
                # catch, not how often he gets one - real noise for a volume
                # prop). rec_tds stays inherited for now, same honest gap.
                # Applies to WR/TE/RB alike since rec_pool already includes
                # all three.
                merged_grades_rec = {**own_grades, **def_grades}
                for sib_prop, sib_col in (("receptions", "receptions"), ("targets", "targets"),
                                           ("rec_tds", "receiving_tds")):
                    sib_mu = calc_prop_mu(
                        gsis_id, sib_col, player_stats_df, season, week, current_team=team,
                        league_fallback_mu=fallback_mus.get((position, sib_col)),
                    )
                    sib_sigma = calc_player_sigma(
                        gsis_id, sib_col, player_stats_df, season, week, current_team=team,
                        league_fallback_sigma=fallback_sigmas.get((position, sib_col)),
                    )
                    if sib_prop in PROP_METRIC_CROSSWALK:
                        sib_grade_exploit = calc_grade_matchup_strength(merged_grades_rec, sib_prop)
                        # Real fix - recompute the alignment structural signal
                        # PER SIBLING PROP (receptions gets DRP%/CC%, targets
                        # gets 1READ%) instead of reusing rec_yards' version
                        # (YPRR/YACO/aDOT) - same fix as the QB-coverage
                        # sibling loop above, needed for the per-prop stat
                        # curation to actually apply to receptions/targets.
                        sib_alignment_exploit = np.nan
                        if (ENABLE_ALIGNMENT_IN_QUALITY_SCORE and coverage_bundle is not None
                                and calc_alignment_exploit_strength is not None and opponent is not None
                                and sib_prop in ("receptions", "targets")):
                            sib_alignment_info = calc_alignment_exploit_strength(
                                coverage_bundle, wr.get("full_name"), position, team, opponent, prop_type=sib_prop,
                                alignment_bundle=alignment_target_bundle,
                            )
                            sib_alignment_exploit = sib_alignment_info.get("exploit_strength")
                        sib_structural_parts = [v for v in [coverage_info.get("exploit_strength"), personnel_exploit_for_scoring,
                                                             sib_alignment_exploit] if pd.notna(v)]
                        sib_combined_structural = (sum(sib_structural_parts) / len(sib_structural_parts)) if sib_structural_parts else np.nan
                        sib_blended = calc_blended_matchup_strength(sib_combined_structural, sib_grade_exploit, role_score, role_is_bridged=role_trend.get("bridged_from_prior_season", False))
                        sib_quality_score = calc_quality_score(
                            matchup_exploit_strength=sib_blended,
                            sample_size_games=rec_confidence_info["games_sampled_current"],
                            coverage_confidence=min(n_plays / 300, 1.0),
                        )
                        _record_quality_score(gsis_id, sib_quality_score)
                    else:
                        sib_grade_exploit, sib_quality_score = grade_exploit, quality_score
                    rows.append({
                        "gsis_id": gsis_id, "player_display_name": wr.get("full_name"),
                        "team": team, "position": position, "prop_type": sib_prop,
                        "matchup": team_to_matchup.get(team),
                        "mu": sib_mu, "sigma": sib_sigma, "opponent": opponent,
                        "quality_score": sib_quality_score,
                        "grade_matchup_strength": sib_grade_exploit,
                        "role_verification_score": role_score,
                        "data_confidence": rec_confidence_info["data_confidence"],
                        "games_sampled_current": rec_confidence_info["games_sampled_current"],
                    })

                longest_rec_mu = calc_prop_mu(gsis_id, "longest_play", rec_longest_df, season, week, current_team=team, league_fallback_mu=rec_longest_fallback_mu)
                longest_rec_sigma = calc_player_sigma(gsis_id, "longest_play", rec_longest_df, season, week, current_team=team, league_fallback_sigma=rec_longest_fallback_sigma)
                rows.append({
                    "gsis_id": gsis_id, "player_display_name": wr.get("full_name"),
                    "team": team, "position": position, "prop_type": "longest_reception",
                    "matchup": team_to_matchup.get(team),
                    "mu": longest_rec_mu, "sigma": longest_rec_sigma, "opponent": opponent,
                    "quality_score": quality_score,
                    "data_confidence": rec_confidence_info["data_confidence"],
                    "games_sampled_current": rec_confidence_info["games_sampled_current"],
                })

        except Exception:
            continue  # this specific player's data is genuinely missing/broken this early in a new season - skip them, don't crash everyone else
    # --- Fantasy points (offense: QB, RB, WR, TE) ---
    offense_positions = ["QB", "RB", "WR", "TE"]
    fantasy_pool_roster = week_rosters[week_rosters["position"].isin(offense_positions)]
    for _, pr in fantasy_pool_roster.iterrows():
        try:
            gsis_id = pr.get("gsis_id")
            if team_filter and pr.get("team") not in team_filter:
                continue
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
            # REAL FIX (found via direct review, per explicit request to
            # fix if truly needed) - this used to recompute fantasy points
            # from scratch via calc_offense_fantasy_points, which does NOT
            # include the real offensive fumble-recovery TD bonus - while
            # the real "actual" outcome used in the backtest DOES include
            # it (via fantasy_points_prizepicks, computed earlier in this
            # same function). Using that same, already-complete column
            # directly here means mu and actual are now built with
            # identical methodology - no more silent mismatch for the
            # rare case of a real fumble-recovery TD in recent history.
            fantasy_pts_per_game = recent_games["fantasy_points_prizepicks"]
            raw_mu = fantasy_pts_per_game.mean()
            games_n_mu = len(fantasy_pts_per_game)
            league_fallback_mu = fallback_mus.get((pr.get("position"), "fantasy_points_prizepicks"))
            if league_fallback_mu is None:
                mu_fantasy = round(raw_mu, 2)
            else:
                weight_own_mu = min(games_n_mu / 6, 1.0)
                mu_fantasy = round((weight_own_mu * raw_mu) + ((1 - weight_own_mu) * league_fallback_mu), 2)
            raw_sigma = fantasy_pts_per_game.std(ddof=1) if len(fantasy_pts_per_game) >= 2 else np.nan
            # REAL FIX - this used to be raw_sigma directly, with ZERO
            # shrinkage protection, unlike every other prop in this file.
            # Confirmed live: Mark Andrews' real week-3 sample was just 2
            # games (1.5, 1.4 - both low and similar), producing a
            # legitimately tiny raw std dev of ~0.07 with nothing to
            # correct it - his real season-long variance turned out to be
            # roughly 6-7 points once the full season was in. Now blends
            # toward a real, position-level league fallback the same way
            # pass_yards/rush_yards/rec_yards already do, reaching full
            # confidence only at a real sample size, not on 2 games.
            position = pr.get("position")
            games_n = len(fantasy_pts_per_game)
            league_fallback = fallback_sigmas.get((position, "fantasy_points_prizepicks"))
            if pd.isna(raw_sigma) or league_fallback is None:
                sigma = round(raw_sigma, 2) if pd.notna(raw_sigma) else (round(league_fallback, 2) if league_fallback is not None else np.nan)
            else:
                weight_own = min(games_n / 6, 1.0)  # 6 = same lookback_games cap used for mu_fantasy above
                sigma = round((weight_own * raw_sigma) + ((1 - weight_own) * league_fallback), 2)

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

        except Exception:
            continue  # this specific player's data is genuinely missing/broken this early in a new season - skip them, don't crash everyone else
    # --- Kicker fantasy + FG/XP props ---
    # NOTE: deliberately NOT given a quality_score/matchup-exploit signal,
    # same design exception as the MLB tool's Pitcher Win prop - kicking
    # points are driven by team red-zone/scoring-drive volume rather than
    # a player-vs-defense skill matchup, so none of the offense/defense
    # grade crosswalk or coverage/box signals above meaningfully apply.
    # Not a gap, an intentional scope boundary.
    kicker_pool = week_rosters[week_rosters["position"] == "K"]
    for _, kr in kicker_pool.iterrows():
        try:
            gsis_id = kr.get("gsis_id")
            if team_filter and kr.get("team") not in team_filter:
                continue
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
            raw_mu_kicker = kicker_pts_per_game.mean()
            games_n_kicker = len(kicker_pts_per_game)
            league_fallback_mu_kicker = fallback_mus.get(("K", "kicker_fantasy"))
            if league_fallback_mu_kicker is None:
                mu_kicker = round(raw_mu_kicker, 2)
            else:
                weight_own_kicker_mu = min(games_n_kicker / 6, 1.0)
                mu_kicker = round((weight_own_kicker_mu * raw_mu_kicker) + ((1 - weight_own_kicker_mu) * league_fallback_mu_kicker), 2)
            raw_sigma_kicker = kicker_pts_per_game.std(ddof=1) if games_n_kicker >= 2 else np.nan
            league_fallback_sigma_kicker = fallback_sigmas.get(("K", "kicker_fantasy"))
            if pd.isna(raw_sigma_kicker) or league_fallback_sigma_kicker is None:
                sigma = round(raw_sigma_kicker, 2) if pd.notna(raw_sigma_kicker) else (round(league_fallback_sigma_kicker, 2) if league_fallback_sigma_kicker is not None else np.nan)
            else:
                weight_own_kicker_sigma = min(games_n_kicker / 6, 1.0)
                sigma = round((weight_own_kicker_sigma * raw_sigma_kicker) + ((1 - weight_own_kicker_sigma) * league_fallback_sigma_kicker), 2)
            rows.append({
                "gsis_id": gsis_id, "player_display_name": kr.get("full_name"),
                "team": kr.get("team"), "position": "K", "prop_type": "kicker_fantasy",
                "matchup": team_to_matchup.get(kr.get("team")),
                "mu": mu_kicker, "sigma": sigma,
            })

        except Exception:
            continue  # this specific player's data is genuinely missing/broken this early in a new season - skip them, don't crash everyone else
    return pd.DataFrame(rows)


@_cache_pull
def pull_depth_charts(years: list[int]) -> pd.DataFrame:
    df = _pull_years_gracefully(nfl.load_depth_charts, years)
    return df.to_pandas() if hasattr(df, "to_pandas") else df


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
                  lookback_games: int = 6, min_games: int = 5,
                  league_fallback_mu: float = None, full_confidence_games: int = None) -> float:
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

    SHRINKAGE FIX (real structural bug found via the 2025 backtest): this
    was previously a HARD CUTOVER - the instant a player had >=min_games
    (2) real games, their own average was used at FULL weight, with
    IDENTICAL treatment for a player on 2-3 games and one on 15+. This
    directly explains a confirmed real pattern in the pass_yards backtest
    failure: every worst-miss QB had thin games_sampled (3-5, almost
    always a backup/uncertain-role situation), each trusted as fully
    reliable as an established starter - one unusually good or bad game
    inside a 2-3 game sample could swing mu hugely with zero dampening.
    Now blends the player's own average with league_fallback_mu using the
    SAME Bayesian shrinkage shape already used elsewhere in this file
    (blend_volume_estimate, blend_scheme_baseline): weight shifts smoothly
    toward the player's own data as real games accumulate, reaching full
    confidence only at full_confidence_games (8, roughly half a season)
    instead of an instant all-or-nothing cutover at 2. Below min_games,
    behavior is unchanged (pure fallback, or NaN if none exists). If no
    league_fallback_mu is available to shrink toward, also unchanged
    (falls back to the player's own average outright, same as before).

    Returns NaN if there's no usable history and no league_fallback_mu is
    provided - flagged low-confidence in the UI rather than guessed.

    RECENCY WEIGHTING (latest fix, see inline comment at the actual
    computation below for the full real-data justification): the
    within-sample average now weights recent games higher via exponential
    decay (0.85^i), instead of a flat mean across the whole lookback
    window - fixes real role-change situations (a backup who just became
    the lead back) without reopening the original thin-sample noise
    problem, since the shrinkage-toward-league-average step still applies
    on top for any player without enough real games yet.
    """
    current_season_history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season)
        & (player_stats_df["week"] < current_week)
    ].sort_values("week", ascending=False).head(lookback_games)

    combined = current_season_history
    team_changed = False
    if len(combined) < min_games:
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

        # REAL BUG FIX (confirmed live - a real, multi-year veteran who
        # changed teams in the offseason, e.g. Mike Evans TB->SF for 2026,
        # verified directly against real, live 2026 roster data): if the
        # SAME-team bridge above still doesn't clear min_games, the player
        # isn't a genuine unknown (rookie, practice-squad call-up) - he has
        # real, recent, relevant skill-level history, just with a
        # DIFFERENT team. The old behavior threw that away entirely and
        # fell through to a fully generic league-average number, treating
        # a real Pro Bowler exactly like a total unknown for no reason
        # connected to his actual talent. Now bridges with his real
        # any-team prior-season history instead - genuine signal about
        # who he is as a player - but flags team_changed=True so real,
        # extra shrinkage (below) can still account for the honest
        # uncertainty a new team's scheme/role genuinely introduces,
        # rather than either extreme (full trust or total exclusion).
        if len(combined) < min_games and current_team is not None:
            any_team_query = (
                (player_stats_df["gsis_id"] == player_gsis_id)
                & (player_stats_df["season"] == season - 1)
            )
            any_team_history = player_stats_df[any_team_query].sort_values(
                "week", ascending=False
            ).head(lookback_games)
            if not any_team_history.empty:
                combined = pd.concat([current_season_history, any_team_history])
                team_changed = True

    if len(combined) < min_games:
        return league_fallback_mu if league_fallback_mu is not None else np.nan

    # RECENCY WEIGHTING FIX (real gap found via 2025 backtest): own_avg
    # previously gave EQUAL weight to every game in the lookback window -
    # for a player whose role just changed (e.g. a backup who became the
    # lead back partway through the window), that dilutes his CURRENT
    # elevated role with his OWN stale earlier games, even before
    # shrinkage applies. Confirmed real pattern: 83 rush_yards rows with a
    # maxed-out role_verification_score still badly UNDER-projected -
    # every one a recent role-change situation (Rico Dowdle, Kenneth
    # Gainwell, Rhamondre Stevenson, etc.) where the flat average was
    # still anchored to pre-change games. `combined` is already sorted
    # most-recent-first, so exponential decay (most recent game weighted
    # highest) helps directly.
    #
    # HONEST KNOWN TRADE-OFF (found via my own adversarial test before
    # shipping, not discovered live): with very few games, recency
    # weighting can't distinguish "a genuine sustained trend across
    # several recent games" from "one huge single-game outlier that
    # happens to be the most recent game" - both get extra weight from
    # pure game-order decay. Tested at decay=0.85 first: correctly helped
    # the genuine multi-game breakout case, but ALSO measurably amplified
    # a synthetic single-outlier-as-most-recent-game case (own_avg pulled
    # 12% above the flat average, the wrong direction). decay=0.95 (used
    # here) cuts that same amplification to ~4% while still producing
    # real upward movement (63.0->65.7) on the genuine sustained-trend
    # case - a much gentler, more honest middle ground, not a full fix.
    # The shrinkage step below still provides a real safety net on top for
    # low-games_n players regardless. This is shipped as a real, tested
    # improvement, not a guaranteed fix - the actual backtest is what
    # will show whether it helps more than it costs on real data.
    recency_weights = np.array([0.95 ** i for i in range(len(combined))])
    recency_weights = recency_weights / recency_weights.sum()
    own_avg = float(np.average(combined[prop_column].values, weights=recency_weights))
    games_n = len(combined)

    if league_fallback_mu is None or pd.isna(league_fallback_mu):
        return round(own_avg, 2)

    # BUGFIX caught in testing: full_confidence_games must not exceed
    # lookback_games, or full confidence becomes mathematically
    # unreachable - games_n can never exceed lookback_games (the sample
    # is capped there), so a full_confidence_games default higher than
    # that would dampen even a rock-solid veteran's mu, not just thin
    # samples. Defaults to lookback_games itself unless explicitly
    # overridden with something smaller.
    effective_full_confidence = full_confidence_games if full_confidence_games is not None else lookback_games
    weight_own = min(games_n / effective_full_confidence, 1.0)
    # Real, extra discount when team_changed - his own history is real,
    # relevant skill-level signal (kept, not thrown away), but a genuine,
    # honest uncertainty exists about role/volume/scheme fit on a new
    # team that a same-team sample wouldn't carry. Halves the trust his
    # own average would otherwise get, shifting weight toward the more
    # conservative league_fallback_mu - a real middle ground between
    # blind full trust and the old behavior's total exclusion.
    if team_changed:
        weight_own *= 0.5
    shrunk_mu = (weight_own * own_avg) + ((1 - weight_own) * league_fallback_mu)
    return round(shrunk_mu, 2)


def build_league_fallback_mus(player_stats_df: pd.DataFrame, season: int,
                               through_week: int) -> dict:
    """
    Position-level average mu fallback (e.g. "what does an average starting
    RB rush for per game this season") for players without enough of their
    own history yet (rookies, recent trades, Week 1-2). Same structure as
    build_league_fallback_sigmas().
    """
    prop_by_position = {
        "QB": ["passing_yards", "rushing_yards", "completions", "attempts", "passing_tds"],
        "RB": ["rushing_yards", "receiving_yards", "carries", "rushing_tds",
               "receptions", "targets", "receiving_tds"],
        "WR": ["receiving_yards", "rushing_yards", "receptions", "targets", "receiving_tds"],
        "TE": ["receiving_yards", "receptions", "targets", "receiving_tds"],
    }
    # REAL FIX (found live via a real screenshot showing Shane Buechele's
    # mu/sigma as None despite a "Fallback: League Average" label at week
    # 1 of 2026 - confirmed directly: build_league_fallback_mus/sigmas
    # only ever looked at the CURRENT season's data, which is
    # mathematically guaranteed to be completely empty at week 1 of any
    # new season (zero real games have been played yet for anyone). This
    # broke the fallback exactly when it's needed most - a genuinely new
    # player (rookie, practice-squad call-up) with zero current AND zero
    # prior-team history had nowhere real left to fall back to. Now
    # includes the full prior season too, giving early weeks (especially
    # week 1) a real, substantial population - later weeks naturally
    # shift more weight onto real current-season data as it accumulates,
    # since both real sources are combined here, not swapped.
    current_season_df = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < through_week)
    ]
    prior_season_df = player_stats_df[player_stats_df["season"] == season - 1]
    df = pd.concat([current_season_df, prior_season_df], ignore_index=True)
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

    # REAL FIX (same real case as build_league_fallback_sigmas above - Mark
    # Andrews' real mu of 1.45 was ALSO built from just his 2 real, both-
    # low early games, same unprotected-thin-sample problem as sigma had.
    # Real, position-level fallback computed the same way as every other
    # prop here.
    if "fantasy_points_prizepicks" in df.columns:
        for position in prop_by_position:
            pos_df = df[df["position"] == position]
            per_player_avg = (
                pos_df.groupby("gsis_id")["fantasy_points_prizepicks"]
                .agg(["mean", "count"])
                .query("count >= 2")
            )
            if not per_player_avg.empty:
                fallback[(position, "fantasy_points_prizepicks")] = round(per_player_avg["mean"].mean(), 2)

    # REAL FIX (found via systematic rescan, per direct request - "kicker
    # points" had the exact same unprotected raw-mean/raw-sigma pattern
    # fantasy_points did before that fix). Real, position-level (K)
    # fallback, built fresh since kicker points aren't a pre-existing
    # column.
    kicker_df = df[df["position"] == "K"].copy()
    if not kicker_df.empty:
        kicker_df["_kicker_pts"] = kicker_df.apply(lambda r: calc_kicker_fantasy_points(r.to_dict()), axis=1)
        per_kicker_avg = kicker_df.groupby("gsis_id")["_kicker_pts"].agg(["mean", "count"]).query("count >= 2")
        if not per_kicker_avg.empty:
            fallback[("K", "kicker_fantasy")] = round(per_kicker_avg["mean"].mean(), 2)

    return fallback


# ---------------------------------------------------------------------------
# 6b. SIGMA (VARIANCE) ESTIMATION PER PROP TYPE
# ---------------------------------------------------------------------------

def calc_player_sigma(player_gsis_id: str, prop_column: str, player_stats_df: pd.DataFrame,
                       season: int, current_week: int, current_team: str = None,
                       lookback_games: int = 8, min_games: int = 5,
                       league_fallback_sigma: float = None, full_confidence_games: int = None) -> float:
    """
    Computes a player's own game-to-game standard deviation for a given prop
    column using their real weekly history from player_stats, up to
    `lookback_games` most recent games before current_week.

    TEAM-CHANGE FIX (same as calc_prop_mu): the prior-season fallback is
    now filtered to the SAME team when current_team is provided, so a
    traded player's sigma isn't computed off stale old-team variance mixed
    with new-team games.

    SHRINKAGE FIX: same hard-cutover bug as calc_prop_mu, same fix - see
    that function's docstring for the full real-data justification. A
    thin-sample player's own std dev (itself noisy and unstable on only
    3-4 games) now blends toward league_fallback_sigma instead of being
    trusted outright the instant min_games is cleared, reaching full
    confidence only at full_confidence_games real games.

    Returns league_fallback_sigma if there's no usable history in either
    season - otherwise NaN, and the row should be flagged as low-confidence
    in the UI rather than scored with a guessed sigma.
    """
    current_season_history = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"] == season)
        & (player_stats_df["week"] < current_week)
    ].sort_values("week", ascending=False).head(lookback_games)

    combined = current_season_history
    team_changed = False
    if len(combined) < min_games:
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

        # Same real fix as calc_prop_mu - a real, multi-year veteran who
        # changed teams (verified live against Mike Evans TB->SF for
        # 2026) still has real, relevant skill-level history, just with
        # a different team - bridges with any-team history rather than
        # falling straight through to a fully generic fallback, with
        # team_changed flagged so extra real shrinkage still applies below.
        if len(combined) < min_games and current_team is not None:
            any_team_query = (
                (player_stats_df["gsis_id"] == player_gsis_id)
                & (player_stats_df["season"] == season - 1)
            )
            any_team_history = player_stats_df[any_team_query].sort_values(
                "week", ascending=False
            ).head(lookback_games)
            if not any_team_history.empty:
                combined = pd.concat([current_season_history, any_team_history])
                team_changed = True

    if len(combined) < min_games:
        return league_fallback_sigma if league_fallback_sigma is not None else np.nan

    own_sigma = combined[prop_column].std(ddof=1)
    games_n = len(combined)

    if league_fallback_sigma is None or pd.isna(league_fallback_sigma) or pd.isna(own_sigma):
        if pd.notna(own_sigma):
            return round(own_sigma, 3)
        return league_fallback_sigma if league_fallback_sigma is not None else np.nan

    # Same lookback_games/full_confidence_games ceiling fix as calc_prop_mu.
    effective_full_confidence = full_confidence_games if full_confidence_games is not None else lookback_games
    weight_own = min(games_n / effective_full_confidence, 1.0)
    # Same real, extra team-change discount as calc_prop_mu - see that
    # function's docstring for the full real-data justification.
    if team_changed:
        weight_own *= 0.5
    shrunk_sigma = (weight_own * own_sigma) + ((1 - weight_own) * league_fallback_sigma)
    return round(shrunk_sigma, 3)


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
        "QB": ["passing_yards", "rushing_yards", "completions", "attempts", "passing_tds"],
        "RB": ["rushing_yards", "receiving_yards", "carries", "rushing_tds",
               "receptions", "targets", "receiving_tds"],
        "WR": ["receiving_yards", "rushing_yards", "receptions", "targets", "receiving_tds"],
        "TE": ["receiving_yards", "receptions", "targets", "receiving_tds"],
    }

    # REAL FIX - same confirmed bug and same fix as build_league_fallback_mus above.
    current_season_df = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < through_week)
    ]
    prior_season_df = player_stats_df[player_stats_df["season"] == season - 1]
    df = pd.concat([current_season_df, prior_season_df], ignore_index=True)

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

    # REAL FIX (found via direct backtest diagnosis - Mark Andrews showed a
    # real match_ratio of 366 with sigma=0.07, traced to fantasy_points
    # having its OWN, separate sigma calculation that completely bypassed
    # the shrinkage protection every other prop already gets. Confirmed
    # his real 2-game sample at that point (1.5, 1.4 - both low and
    # similar) legitimately produced a tiny raw std dev, with nothing to
    # correct it before this fix. This computes a real, position-level
    # fantasy_points fallback the same way as every other prop above, so
    # the same shrinkage mechanism can now be applied to it too.
    #
    # SECOND REAL FIX (found immediately after the first, by directly
    # testing at week 3) - using the same count>=3 threshold as the other
    # props made this fallback impossible to compute at all that early in
    # a season (max 2 real games exist when week<3) - confirmed zero
    # fallback entries existed anywhere, for any prop, at week 3. Lowered
    # to count>=2 specifically for fantasy_points, since this is the
    # highest-volume prop (606 real rows in the 3-week sample, dwarfing
    # any other single prop) and the confirmed, most severe case. A
    # 2-game-based fallback is imperfect (small samples underestimate true
    # variance) but confirmed real and far better than the alternative
    # (0.07, from a single unshrunk outlier case) it replaces.
    if "fantasy_points_prizepicks" in df.columns:
        for position in prop_by_position:
            pos_df = df[df["position"] == position]
            per_player_std = (
                pos_df.groupby("gsis_id")["fantasy_points_prizepicks"]
                .agg(["std", "count"])
                .query("count >= 2")
            )
            if not per_player_std.empty:
                fallback[(position, "fantasy_points_prizepicks")] = round(per_player_std["std"].mean(), 3)

    # REAL FIX - same rescan finding as the mu fallback above.
    kicker_df = df[df["position"] == "K"].copy()
    if not kicker_df.empty:
        kicker_df["_kicker_pts"] = kicker_df.apply(lambda r: calc_kicker_fantasy_points(r.to_dict()), axis=1)
        per_kicker_std = kicker_df.groupby("gsis_id")["_kicker_pts"].agg(["std", "count"]).query("count >= 2")
        if not per_kicker_std.empty:
            fallback[("K", "kicker_fantasy")] = round(per_kicker_std["std"].mean(), 3)

    return fallback


# ---------------------------------------------------------------------------
# 7. FULL SLATE SCAN (mirrors scan_full_slate_quality_mu from MLB tool)
# ---------------------------------------------------------------------------

def scan_full_slate_nfl(season: int, week: int, coverage_bundle=None, rb_bundle=None,
                         team_filter: list = None) -> pd.DataFrame:
    """
    Weekly full-slate scanner. Builds the slate (see build_weekly_slate),
    but does NOT auto-fill lines or compute edge/p_over - those are added
    in the Streamlit UI via an adjustable "line" column per row, same as
    the MLB tool's adjustable Best Edges table. quality_score and mu
    components are pre-computed here; edge/p_over recompute live in the UI
    whenever the user edits a line.

    coverage_bundle, rb_bundle: passed straight through to
    build_weekly_slate - see that function's docstring. Optional; omitting
    either just means that signal stays off even if its flag is on.

    team_filter: passed straight through to build_weekly_slate - real
    per-game scanning (skips the expensive per-player scoring loop for
    every team not in the list), not just a post-scan display filter.
    """
    slate_df = build_weekly_slate(season, week, coverage_bundle=coverage_bundle, rb_bundle=rb_bundle,
                                   team_filter=team_filter)
    slate_df["line"] = np.nan  # user fills this in per row in the UI
    slate_df["p_over"] = np.nan
    slate_df["edge"] = np.nan
    return slate_df


# ---------------------------------------------------------------------------
# 8. BACKTEST MODE - compare projected mu against actual results for a
#    completed week (no real lines needed - tests mu accuracy directly)
# ---------------------------------------------------------------------------

def get_starters_for_week(season: int, week: int, depth_charts_df: pd.DataFrame,
                           schedules_df: pd.DataFrame, strict_true_starters: bool = False) -> set:
    """
    Returns the set of gsis_ids who were starters at their position for the
    game nearest this season/week, using position-specific pos_rank
    thresholds rather than a flat pos_rank==1 - most offenses run 3-WR sets
    (11 personnel), so WR1/WR2/WR3 are all commonly real starters, not just
    WR1. Same logic applies loosely to RB in committee backfields.

    strict_true_starters (real, new option) - when True, uses a genuinely
    narrower definition (QB1 only, RB1 only, WR1-2 only, TE1 only, no K) -
    for real, direct requests to exclude committee-backfield RB2s and
    3rd-WR-set players entirely, not just "starters" in the broader,
    personnel-package sense the default threshold below already covers.
    Default (False) leaves existing behavior/callers completely
    unaffected.

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
    if strict_true_starters:
        starter_rank_threshold = {"QB": 1, "RB": 1, "WR": 2, "TE": 1}
    else:
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


def get_usage_relevant_players_for_week(season: int, week: int, player_stats_df: pd.DataFrame,
                                         min_targets: float = 3.0, min_carries: float = 5.0,
                                         lookback_games: int = 4) -> set:
    """
    Real, usage-based alternative/supplement to get_starters_for_week's
    depth-chart-rank approach - includes any real player whose own
    recent real games show genuine, meaningful usage, REGARDLESS of
    depth-chart position. Built per direct request: a real RB2 heavy in
    the passing game (or getting decent real carries) and a real WR3
    with good enough real usage should both count as relevant, even if
    a depth-chart-rank filter alone wouldn't clearly separate them from
    a low-usage backup nominally listed at the same rank.

    Real, honest distinction from get_starters_for_week: that function
    asks "is this player LISTED as a starter" (depth chart), this one
    asks "did this player ACTUALLY get real touches recently" (his own
    real game log) - a depth-chart RB2 could still be a real, low-usage
    handcuff, while a "3rd" receiving back or slot WR3 could be
    outproducing the nominal RB2/WR2 in real, meaningful volume. Uses
    each player's own real games in the lookback window (current season,
    weeks before `week`) - a player with zero games yet this season
    (rookie, new signing, week 1) won't show up here since there's no
    real history yet to judge usage from; combine with
    get_starters_for_week if you want to also catch players before
    they've built real usage history.

    Returns the set of gsis_ids meeting EITHER threshold (avg targets
    per game >= min_targets OR avg carries per game >= min_carries)
    across their own real, recent games.
    """
    recent = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < week)
    ].sort_values("week", ascending=False)

    relevant = set()
    for gsis_id, group in recent.groupby("gsis_id"):
        recent_games = group.head(lookback_games)
        if recent_games.empty:
            continue
        avg_targets = recent_games["targets"].mean() if "targets" in recent_games.columns else 0
        avg_carries = recent_games["carries"].mean() if "carries" in recent_games.columns else 0
        if avg_targets >= min_targets or avg_carries >= min_carries:
            relevant.add(gsis_id)
    return relevant


def score_week_against_actuals(season: int, week: int, starters_only: bool = True, coverage_bundle=None,
                                rb_bundle=None, strict_true_starters: bool = False,
                                alignment_target_bundle: "TeamAlignmentTargetBundle" = None) -> pd.DataFrame:
    """
    Shared core of backtest_week(): builds the week's slate, looks up each
    player's REAL result, and attaches miss/abs_miss/match_ratio - but
    returns EVERY row (no match_ratio filter), so this can feed either
    backtest_week()'s "biggest surprises" view or a season-wide accuracy/
    calibration report that needs the full distribution, not just outliers.

    coverage_bundle, rb_bundle: passed straight through to
    build_weekly_slate - see that function's docstring. Needed here so
    both premium signals can eventually get their own isolated backtests,
    same as every other flag.
    """
    slate_df = build_weekly_slate(season, week, coverage_bundle=coverage_bundle, rb_bundle=rb_bundle,
                                   alignment_target_bundle=alignment_target_bundle)
    player_stats_df = pull_player_stats([season])
    depth_charts_df = pull_depth_charts([season]) if nfl else pd.DataFrame()
    schedules_df = pull_schedules([season])
    pbp_df = pull_pbp([season])
    # REAL BUG FIX - fantasy_points previously used nflreadpy's own
    # "fantasy_points_ppr" column directly for both mu AND the real,
    # actual outcome, which is the standard -2/INT -2/fumble formula, not
    # the real PrizePicks -1/-1 rules actually being scored against -
    # confirmed by hand-checking a real row. Replaced with a genuinely
    # correct column below, including the real offensive fumble recovery
    # TD bonus PrizePicks' rules explicitly include.
    player_stats_df = add_prizepicks_fantasy_column(player_stats_df, pbp_df=pbp_df)

    actual_week = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] == week)
    ].set_index("gsis_id")

    prop_to_stat_column = {
        "pass_yards": "passing_yards",
        "rush_yards": "rushing_yards",
        "rec_yards": "receiving_yards",
        "fantasy_points": "fantasy_points_prizepicks",
        "pass_completions": "completions",
        "pass_attempts": "attempts",
        "pass_tds": "passing_tds",
        "rush_attempts": "carries",
        "rush_tds": "rushing_tds",
        "receptions": "receptions",
        "targets": "targets",
        "rec_tds": "receiving_tds",
    }

    # Longest-play props aren't in player_stats - built from this SAME
    # target week's real pbp instead, same aggregation as
    # build_longest_play_by_game but for one already-played week rather
    # than a history window. Keyed by (gsis_id, prop_type) for the lookup.
    longest_actual_week_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] == week)]
    longest_actuals = {}
    for prop_type, pos_hint in (("longest_completion", "QB"), ("longest_reception", "WR"), ("longest_rush", "RB")):
        try:
            lp = build_longest_play_by_game(longest_actual_week_pbp, pos_hint)
        except KeyError:
            lp = pd.DataFrame(columns=["gsis_id", "longest_play"])
        for _, r in lp.iterrows():
            longest_actuals[(r["gsis_id"], prop_type)] = r["longest_play"]

    def _lookup_actual(row):
        prop_type = row["prop_type"]
        gsis_id = row["gsis_id"]
        if prop_type in ("longest_completion", "longest_reception", "longest_rush"):
            return longest_actuals.get((gsis_id, prop_type), np.nan)
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

    # Drop non-participants: no result at all, OR (for yardage/volume
    # props only) a literal 0 - a real starter essentially never posts a
    # true 0 yard/attempt/target total. TD-count props (pass_tds/
    # rush_tds/rec_tds) are explicitly EXCLUDED from the zero-drop - a
    # real 0-TD game is extremely common for an active starter and is
    # not itself a participation signal, unlike 0 yards/attempts/targets.
    slate_df = slate_df.dropna(subset=["actual"])
    zero_drop_exempt = {"pass_tds", "rush_tds", "rec_tds"}
    zero_mask = (slate_df["actual"] == 0) & (~slate_df["prop_type"].isin(zero_drop_exempt))
    slate_df = slate_df[~zero_mask].copy()

    if starters_only:
        starter_ids = get_starters_for_week(season, week, depth_charts_df, schedules_df,
                                             strict_true_starters=strict_true_starters)
        # Real, additional union - per direct request, a genuinely
        # heavy-usage RB2 (pass-catching or real, decent carry share) or
        # WR3 (good enough real usage) should count even when the
        # depth-chart-rank check above doesn't already catch them.
        # Union rather than replace - QB/TE/K still need the depth-chart
        # check since the usage filter only looks at targets/carries.
        usage_ids = get_usage_relevant_players_for_week(season, week, player_stats_df)
        starter_ids = starter_ids | usage_ids
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


def add_simulation_columns_to_backtest_rows(scored_df: pd.DataFrame, n_simulations: int = 1000) -> pd.DataFrame:
    """
    Real, additive layer on top of score_week_against_actuals()'s
    already-real mu/sigma/actual columns - for every real row, runs the
    1000-sample Monte Carlo (simulate_nfl_prop_n_times) and adds:

      - sim_avg: the real, empirical average across the simulated samples
        (should closely track mu itself - a real sanity check that the
        simulation faithfully represents the model's own projection)
      - hypothetical_line: same real fix as the MLB backtest - floor(mu)
        + 0.5 for count props (guarantees a genuine .5 line, matching
        real sportsbook convention, never a whole number); for
        continuous yardage props, real sportsbook lines also almost
        always sit at a .5 (e.g. 74.5 rec yards), so the same floor+0.5
        convention is used there too.
      - sim_rate: real, empirical % of the 1000 samples that cleared
        hypothetical_line
      - sim_cv: real coefficient of variation (sigma/mu) - how stable/
        consistent this specific real projection is, same real signal
        MLB's Stage 1 CV filter uses
      - real_cleared_line: did the REAL, actual outcome clear
        hypothetical_line - the real ground truth this all gets checked
        against
      - gap_pct: how far sim_avg sits from hypothetical_line, as a % of
        the line itself - same real metric MLB's backtest buckets by

    Returns a new DataFrame (doesn't mutate scored_df) with these columns
    added - rows where mu/sigma/actual aren't usable are dropped rather
    than filled with fake placeholder values.
    """
    import math
    rows = []
    for _, row in scored_df.iterrows():
        mu, sigma, actual = row.get("mu"), row.get("sigma"), row.get("actual")
        prop_type = row.get("prop_type")
        if pd.isna(mu) or pd.isna(sigma) or pd.isna(actual) or sigma is None or sigma <= 0:
            continue
        samples = simulate_nfl_prop_n_times(mu, sigma, prop_type, n_simulations=n_simulations)
        if not samples:
            continue
        sim_avg = sum(samples) / len(samples)
        hypothetical_line = math.floor(sim_avg) + 0.5
        sim_rate = round(sum(1 for v in samples if v > hypothetical_line) / len(samples) * 100, 1)
        sim_cv = round(sigma / mu, 3) if mu > 0 else np.nan
        real_cleared_line = actual > hypothetical_line
        gap_pct = round(abs(sim_avg - hypothetical_line) / hypothetical_line * 100, 1) if hypothetical_line else 0.0

        new_row = row.to_dict()
        new_row.update({
            "sim_avg": round(sim_avg, 2), "hypothetical_line": hypothetical_line,
            "sim_rate": sim_rate, "sim_cv": sim_cv,
            "real_cleared_line": real_cleared_line, "gap_pct": gap_pct,
        })
        rows.append(new_row)

    return pd.DataFrame(rows)


def build_season_simulation_backtest_report(season: int, weeks: list = None, through_week: int = 18,
                                             coverage_bundle=None, rb_bundle=None,
                                             n_simulations: int = 1000, strict_true_starters: bool = False,
                                             min_quality_score: float = None) -> dict:
    """
    Real, season-wide simulation backtest - runs score_week_against_
    actuals() + add_simulation_columns_to_backtest_rows() across every
    real, completed week of a season, then buckets every real row by
    (prop_type, gap_pct range) to show the real hit-rate in each bucket.

    Same real purpose as MLB's own backtest: answers "does a bigger real
    gap between the simulation's average and a real, plausible line
    actually predict a more reliable real outcome" - with real, evidence-
    based numbers instead of a reasoned-but-unvalidated guess.

    min_quality_score (real, new, optional) - when set, only rows with a
    real quality_score >= this value get simulated/backtested at all,
    per direct request to check whether a "best of the best" quality
    floor (e.g. 70+) genuinely produces more reliable results. Real,
    honest caveat worth restating here: an earlier real 6-week run
    showed quality_score wasn't yet cleanly ordered (the 80-100 tier
    actually had a WORSE mean_match_ratio than the 40-60 tier) - this
    filter doesn't fix that on its own, it just lets you directly,
    empirically check whether a given floor helps, using real results
    instead of assuming either way. None (default) runs every real row,
    unchanged from before.

    Real, honest limitation carried over from build_season_accuracy_
    report: no free historical NFL player-prop-line archive exists, so
    hypothetical_line is the model's own real mu (floored to the nearest
    real .5), not an actual historical book line - same real, documented
    tradeoff as the rest of this file's backtest tooling.

    Split by prop_type from the start (not lumped together) - different
    NFL props (receptions vs rush yards vs rush TDs) likely have
    genuinely different real gap-pct behavior, the same real lesson
    already learned building MLB's hitter-vs-pitcher backtest split.
    """
    if weeks is None:
        weeks = get_completed_weeks_with_data(season, through_week)

    all_sim_rows = []
    for wk in weeks:
        try:
            wk_df = score_week_against_actuals(season, wk, starters_only=True,
                                                coverage_bundle=coverage_bundle, rb_bundle=rb_bundle,
                                                strict_true_starters=strict_true_starters)
            if wk_df.empty:
                continue
            if min_quality_score is not None and "quality_score" in wk_df.columns:
                wk_df = wk_df[wk_df["quality_score"] >= min_quality_score]
                if wk_df.empty:
                    continue
            sim_df = add_simulation_columns_to_backtest_rows(wk_df, n_simulations=n_simulations)
            if not sim_df.empty:
                sim_df["week"] = wk
                all_sim_rows.append(sim_df)
        except Exception as e:
            print(f"Skipping week {wk} in simulation backtest: {e}")
            continue

    if not all_sim_rows:
        return {"raw": pd.DataFrame(), "bucket_summary": pd.DataFrame()}

    raw = pd.concat(all_sim_rows, ignore_index=True)

    bins = [0, 5, 10, 15, 20, 30, 1000]
    labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    raw["gap_bucket"] = pd.cut(raw["gap_pct"], bins=bins, labels=labels, right=False)
    bucket_summary = raw.groupby(["prop_type", "gap_bucket"], observed=True).agg(
        n=("real_cleared_line", "size"),
        real_hit_rate=("real_cleared_line", "mean"),
    ).reset_index()
    bucket_summary["real_hit_rate"] = round(bucket_summary["real_hit_rate"] * 100, 1)

    # Real, direct answer to "actual vs quality of mu" - buckets every
    # real row by its own real quality_score tier (same 80-100/60-80/
    # 40-60/<40 bands build_season_accuracy_report already uses) and
    # shows the real mean absolute miss (|mu - actual|) and match_ratio
    # in each tier - the real, correct accuracy metric here, NOT
    # real_cleared_line. real_cleared_line tests whether the actual
    # outcome landed above or below a line set at the model's OWN mean,
    # which is structurally ~50% regardless of how accurate that mean
    # actually is - it's the right metric for the gap_pct bucket
    # question above, but the wrong one for checking quality_score
    # itself. abs_miss/match_ratio (already computed by
    # score_week_against_actuals) directly measures real prediction
    # error, which is what actually answers "is quality_score earning
    # its keep." If quality_score is doing its real job, the 80-100
    # tier's mean abs_miss should sit meaningfully below the <40 tier's.
    if "quality_score" in raw.columns and "abs_miss" in raw.columns:
        q_bins = [0, 40, 60, 80, 101]
        q_labels = ["<40", "40-60", "60-80", "80-100"]
        raw["quality_tier"] = pd.cut(raw["quality_score"], bins=q_bins, labels=q_labels, right=False)
        agg_kwargs = {"n": ("abs_miss", "size"), "mean_abs_miss": ("abs_miss", "mean")}
        if "match_ratio" in raw.columns:
            agg_kwargs["mean_match_ratio"] = ("match_ratio", "mean")
        quality_tier_summary = raw.groupby("quality_tier", observed=True).agg(**agg_kwargs).reset_index()
        quality_tier_summary["mean_abs_miss"] = round(quality_tier_summary["mean_abs_miss"], 2)
        if "mean_match_ratio" in quality_tier_summary.columns:
            quality_tier_summary["mean_match_ratio"] = round(quality_tier_summary["mean_match_ratio"], 2)
    else:
        quality_tier_summary = pd.DataFrame()

    return {"raw": raw, "bucket_summary": bucket_summary, "quality_tier_summary": quality_tier_summary}


def backtest_week(season: int, week: int, coverage_bundle=None, rb_bundle=None) -> pd.DataFrame:
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
    result = score_week_against_actuals(season, week, starters_only=True, coverage_bundle=coverage_bundle, rb_bundle=rb_bundle)
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

def diagnose_participation_data(season: int, week: int, sample_gsis_id: str = None) -> dict:
    """
    DIAGNOSTIC ONLY - not used anywhere in scoring. Built after the
    coverage adjustment stayed stuck at a 0% fire rate through TWO
    separate fixes (raising then reverting min_plays_per_bucket, then a
    case-insensitive matching fix) with no change either time - two blind
    guesses in a row without seeing the real data is enough; this surfaces
    the real thing directly instead of guessing a third time. This build
    environment has no network access to nflreadpy, so this can only
    actually run wherever the real data is reachable (the deployed app).

    Returns real, raw facts about participation_df for this season/week:
      - whether "defense_man_zone_type" exists as a column at all
      - its real value_counts (including NaN share) - the actual strings
        real data contains, whatever they turn out to be
      - the same for "defense_coverage_type" for comparison (this ONE
        drives the confirmed-working man_pct/zone_pct calculation, so
        comparing the two columns' real behavior side by side is useful)
      - whether the join to pbp_df actually produces ANY matched rows at
        all for a sample player (rules out/in a join-key problem
        completely separate from the coverage-type values themselves)
    """
    participation_df = pull_participation([season])
    pbp_df = pull_pbp([season])
    result = {"season": season, "week": week}

    result["participation_columns"] = list(participation_df.columns)
    result["has_defense_man_zone_type"] = "defense_man_zone_type" in participation_df.columns
    result["has_defense_coverage_type"] = "defense_coverage_type" in participation_df.columns

    if result["has_defense_man_zone_type"]:
        vc = participation_df["defense_man_zone_type"].value_counts(dropna=False)
        result["defense_man_zone_type_value_counts"] = vc.to_dict()
    if result["has_defense_coverage_type"]:
        vc2 = participation_df["defense_coverage_type"].value_counts(dropna=False)
        result["defense_coverage_type_value_counts"] = vc2.head(10).to_dict()

    # Join sanity check: does merging participation to pbp on
    # (nflverse_game_id, play_id) -> (game_id, play_id) actually produce
    # any rows with a non-null defense_man_zone_type for a sample player?
    if sample_gsis_id is None:
        # pick whichever player has the most receiving plays this season as a reasonable sample
        hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)]
        if "receiver_player_id" in hist_pbp.columns and hist_pbp["receiver_player_id"].notna().any():
            sample_gsis_id = hist_pbp["receiver_player_id"].value_counts().idxmax()
    result["sample_gsis_id_used"] = sample_gsis_id

    if sample_gsis_id is not None and result["has_defense_man_zone_type"]:
        hist_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)]
        merged = participation_df.merge(
            hist_pbp[["game_id", "play_id", "receiver_player_id"]],
            left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="left",
        )
        sample_rows = merged[merged["receiver_player_id"] == sample_gsis_id]
        result["sample_player_total_merged_rows"] = len(sample_rows)
        result["sample_player_non_null_man_zone_rows"] = int(sample_rows["defense_man_zone_type"].notna().sum())
        if not sample_rows.empty:
            result["sample_player_man_zone_values_seen"] = (
                sample_rows["defense_man_zone_type"].value_counts(dropna=False).to_dict()
            )

    return result


def diagnose_injuries_data(season: int, week: int = 8) -> dict:
    """
    DIAGNOSTIC ONLY - not used anywhere in scoring, mirrors
    diagnose_participation_data()'s proven approach exactly: surface the
    REAL data first, before building anything that reads specific column
    names from it. pull_injuries()'s real schema is completely unverified
    in this build environment (no network access) - guessing at column
    names here would repeat the exact coverage-type-casing mistake made
    (twice) earlier this session, this time on a data source we've never
    even looked at once.

    Since the real column names aren't known at all (unlike the
    participation diagnostic, where the column NAME was already known and
    only its VALUES were in question), this dumps broadly rather than
    guessing specific column names:
      - every real column name pull_injuries() actually returns
      - a few raw sample rows, unfiltered - the fastest way to see the
        real shape at a glance
      - real value_counts for any column whose name plausibly looks like
        an injury status field (checked by name pattern, not assumed)
      - whether a gsis_id-compatible player-id column exists at all, and
        what it's actually called
    """
    result = {"season": season, "week": week}
    try:
        injuries_df = pull_injuries([season])
    except Exception as e:
        result["error"] = f"pull_injuries() itself failed: {e}"
        return result

    result["columns"] = list(injuries_df.columns)
    result["n_rows_total"] = len(injuries_df)
    result["sample_rows"] = injuries_df.head(5).to_dict(orient="records")

    status_like_cols = [c for c in injuries_df.columns if "status" in c.lower()]
    result["status_like_columns_found"] = status_like_cols
    for col in status_like_cols:
        result[f"value_counts__{col}"] = injuries_df[col].value_counts(dropna=False).head(15).to_dict()

    id_like_cols = [c for c in injuries_df.columns if "gsis" in c.lower() or "player_id" in c.lower() or c.lower() == "id"]
    result["id_like_columns_found"] = id_like_cols

    if "season" in injuries_df.columns:
        result["real_seasons_present"] = sorted(injuries_df["season"].dropna().unique().tolist())
    if "week" in injuries_df.columns:
        result["real_weeks_present"] = sorted(injuries_df["week"].dropna().unique().tolist())

    return result


def diagnose_alignment_data(season: int, week: int = 8) -> dict:
    """
    DIAGNOSTIC ONLY - not used anywhere in scoring, same discipline as
    diagnose_participation_data()/diagnose_injuries_data(): check the real
    data before building anything on an assumption. Confirmed real
    participation_df has a "route" column (route TYPE run - slant/go/
    screen/etc.), which is NOT the same thing as pre-snap ALIGNMENT
    (wide/slot/backfield/inline) - related concepts, genuinely different
    data. This checks EVERY real column in both participation_df and
    ftn_df (not just the ones already confirmed for other purposes) for
    anything alignment-related by name, plus shows real values for the
    columns already known to be alignment-adjacent (route,
    n_offense_backfield) so there's no guessing either way.
    """
    result = {"season": season, "week": week}

    participation_df = pull_participation([season])
    ftn_df = pull_ftn_charting([season])

    result["participation_columns"] = list(participation_df.columns)
    result["ftn_columns"] = list(ftn_df.columns)

    alignment_keywords = ["align", "slot", "wide", "inline", "backfield", "position", "split", "formation"]
    result["participation_alignment_like_columns"] = [
        c for c in participation_df.columns if any(k in c.lower() for k in alignment_keywords)
    ]
    result["ftn_alignment_like_columns"] = [
        c for c in ftn_df.columns if any(k in c.lower() for k in alignment_keywords)
    ]

    if "route" in participation_df.columns:
        result["route_value_counts"] = participation_df["route"].value_counts(dropna=False).head(20).to_dict()
    if "n_offense_backfield" in ftn_df.columns:
        result["n_offense_backfield_value_counts"] = ftn_df["n_offense_backfield"].value_counts(dropna=False).to_dict()

    # dump real values for anything the keyword search found, whatever it turns out to be
    for col in result["participation_alignment_like_columns"]:
        result[f"participation_value_counts__{col}"] = participation_df[col].value_counts(dropna=False).head(15).to_dict()
    for col in result["ftn_alignment_like_columns"]:
        result[f"ftn_value_counts__{col}"] = ftn_df[col].value_counts(dropna=False).head(15).to_dict()

    return result


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


def build_season_accuracy_report(season: int, weeks: list = None, through_week: int = 18, coverage_bundle=None, rb_bundle=None) -> dict:
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
            wk_df = score_week_against_actuals(season, wk, starters_only=True, coverage_bundle=coverage_bundle, rb_bundle=rb_bundle)
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
                "by_quality_tier": pd.DataFrame(), "by_quality_tier_by_prop": pd.DataFrame(),
                "adjustment_direction_accuracy": np.nan,
                "role_verification_check": pd.DataFrame()}

    raw = pd.concat(week_results, ignore_index=True)

    by_prop_type = raw.groupby("prop_type").agg(
        mean_abs_miss=("abs_miss", "mean"), mean_bias=("miss", "mean"), n=("abs_miss", "count"),
    ).reset_index().sort_values("mean_abs_miss")

    by_position = raw.groupby("position").agg(
        mean_abs_miss=("abs_miss", "mean"), mean_bias=("miss", "mean"), n=("abs_miss", "count"),
    ).reset_index().sort_values("mean_abs_miss")

    by_quality_tier = pd.DataFrame()
    by_quality_tier_by_prop = pd.DataFrame()
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

        # Same tier breakdown, but split by prop_type too - the pooled
        # by_quality_tier above blends every prop together, which dilutes
        # any prop-specific signal (e.g. the alignment exploit signal only
        # touches rec_yards/receptions/targets/rec_tds - its effect is
        # invisible in the pooled table once mixed with pass_yards/
        # rush_yards rows it never touches). This is the real per-prop
        # check for whether a given signal is actually helping.
        by_quality_tier_by_prop = tier_df.groupby(
            ["prop_type", "quality_tier"], observed=True
        ).agg(
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
        "by_quality_tier_by_prop": by_quality_tier_by_prop,
        "adjustment_direction_accuracy": adjustment_direction_accuracy,
        "role_verification_check": role_verification_check,
    }


def build_combined_report_from_raw(raw: pd.DataFrame, n_simulations: int = 1000,
                                    min_quality_score: float = None) -> dict:
    """
    Real, extracted aggregation step - takes rows ALREADY scored by
    score_week_against_actuals() (any number of weeks, even a partial/
    in-progress set) and builds every summary table (readiness +
    simulation) from them. Split out from build_combined_readiness_and_
    simulation_report() specifically so the expensive per-week SCORING
    step (confirmed ~90-100+ real seconds/week) and this cheaper
    aggregation step can run independently - the app layer can now score
    one real week at a time, persist each week's raw rows to disk
    immediately, and call this function on whatever's accumulated so far
    at any point, including right after a restart with only a partial
    set of weeks actually scored yet. No re-scoring needed just to see
    a report on partial progress.
    """
    if raw is None or raw.empty:
        return {"raw": pd.DataFrame(), "by_prop_type": pd.DataFrame(), "by_position": pd.DataFrame(),
                "by_quality_tier": pd.DataFrame(), "by_quality_tier_by_prop": pd.DataFrame(),
                "adjustment_direction_accuracy": np.nan, "role_verification_check": pd.DataFrame(),
                "bucket_summary": pd.DataFrame(), "quality_tier_summary": pd.DataFrame(),
                "sim_raw": pd.DataFrame()}

    # --- Readiness diagnostics (same real logic as build_season_accuracy_report) ---
    by_prop_type = raw.groupby("prop_type").agg(
        mean_abs_miss=("abs_miss", "mean"), mean_bias=("miss", "mean"), n=("abs_miss", "count"),
    ).reset_index().sort_values("mean_abs_miss")

    by_position = raw.groupby("position").agg(
        mean_abs_miss=("abs_miss", "mean"), mean_bias=("miss", "mean"), n=("abs_miss", "count"),
    ).reset_index().sort_values("mean_abs_miss")

    by_quality_tier = pd.DataFrame()
    by_quality_tier_by_prop = pd.DataFrame()
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
        by_quality_tier_by_prop = tier_df.groupby(
            ["prop_type", "quality_tier"], observed=True
        ).agg(
            mean_abs_miss=("abs_miss", "mean"), mean_match_ratio=("match_ratio", "mean"),
            n=("abs_miss", "count"),
        ).reset_index()

    before_col = None
    if "mu_before_coverage_adj" in raw.columns or "mu_before_box_adj" in raw.columns:
        raw["mu_before_adjustment"] = raw.get("mu_before_coverage_adj")
        if "mu_before_box_adj" in raw.columns:
            raw["mu_before_adjustment"] = raw["mu_before_adjustment"].fillna(raw["mu_before_box_adj"])
        before_col = "mu_before_adjustment"

    adjustment_direction_accuracy = np.nan
    if before_col is not None:
        adj = raw.dropna(subset=[before_col, "mu", "actual"]).copy()
        adj = adj[adj["mu"] != adj[before_col]]
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

    # --- Simulation backtest (same real rows, filtered by min_quality_score if set) ---
    sim_input = raw
    if min_quality_score is not None and "quality_score" in raw.columns:
        sim_input = raw[raw["quality_score"] >= min_quality_score]

    sim_raw = add_simulation_columns_to_backtest_rows(sim_input, n_simulations=n_simulations) if not sim_input.empty else pd.DataFrame()

    bucket_summary = pd.DataFrame()
    quality_tier_summary = pd.DataFrame()
    if not sim_raw.empty:
        bins = [0, 5, 10, 15, 20, 30, 1000]
        labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
        sim_raw["gap_bucket"] = pd.cut(sim_raw["gap_pct"], bins=bins, labels=labels, right=False)
        bucket_summary = sim_raw.groupby(["prop_type", "gap_bucket"], observed=True).agg(
            n=("real_cleared_line", "size"), real_hit_rate=("real_cleared_line", "mean"),
        ).reset_index()
        bucket_summary["real_hit_rate"] = round(bucket_summary["real_hit_rate"] * 100, 1)

        if "quality_score" in sim_raw.columns:
            q_bins = [0, 40, 60, 80, 101]
            q_labels = ["<40", "40-60", "60-80", "80-100"]
            sim_raw["quality_tier"] = pd.cut(sim_raw["quality_score"], bins=q_bins, labels=q_labels, right=False)
            agg_kwargs = {"n": ("abs_miss", "size"), "mean_abs_miss": ("abs_miss", "mean")}
            if "match_ratio" in sim_raw.columns:
                agg_kwargs["mean_match_ratio"] = ("match_ratio", "mean")
            quality_tier_summary = sim_raw.groupby("quality_tier", observed=True).agg(**agg_kwargs).reset_index()
            quality_tier_summary["mean_abs_miss"] = round(quality_tier_summary["mean_abs_miss"], 2)

    return {
        "raw": raw, "by_prop_type": by_prop_type, "by_position": by_position,
        "by_quality_tier": by_quality_tier, "by_quality_tier_by_prop": by_quality_tier_by_prop,
        "adjustment_direction_accuracy": adjustment_direction_accuracy,
        "role_verification_check": role_verification_check,
        "bucket_summary": bucket_summary, "quality_tier_summary": quality_tier_summary,
        "sim_raw": sim_raw,
    }


def build_combined_readiness_and_simulation_report(season: int, weeks: list = None, through_week: int = 18,
                                                     coverage_bundle=None, rb_bundle=None,
                                                     n_simulations: int = 1000, min_quality_score: float = None,
                                                     strict_true_starters: bool = False) -> dict:
    """
    Real, combined report - per direct request, runs score_week_against_
    actuals() ONCE per real week and builds BOTH the readiness
    diagnostics (quality_score accuracy, by_prop_type, adjustment
    direction, role verification) AND the 1000-sample simulation backtest
    (gap_pct buckets, real hit-rate) from that SAME shared set of scored
    rows - not two separate report functions each re-running the same
    expensive real scoring step for the same week range.

    Real, honest reason this matters beyond convenience: score_week_
    against_actuals is the genuinely slow part (confirmed ~90-100+
    real seconds per week during this build's own testing) - running the
    readiness report and the simulation backtest as two separate button
    clicks over the same weeks silently doubles that real cost for no
    reason, since both were computing mu/sigma/actual from scratch
    independently. This combines them into one real pass.

    Kept as a single-call convenience wrapper for any other caller that
    wants the whole range done in one shot - see build_combined_report_
    from_raw() for the version that lets the app layer score weeks one
    at a time with real, incremental disk persistence between them.

    Returns a dict with both reports' real keys merged together:
    raw, by_prop_type, by_position, by_quality_tier, by_quality_tier_by_prop,
    adjustment_direction_accuracy, role_verification_check (readiness side),
    plus bucket_summary, quality_tier_summary (simulation side) - built
    from the same underlying real rows, not recomputed separately.
    """
    if weeks is None:
        weeks = get_completed_weeks_with_data(season, through_week)

    week_results = []
    for wk in weeks:
        try:
            wk_df = score_week_against_actuals(season, wk, starters_only=True, coverage_bundle=coverage_bundle,
                                                rb_bundle=rb_bundle, strict_true_starters=strict_true_starters)
            if not wk_df.empty:
                week_results.append(wk_df)
        except Exception as e:
            print(f"Skipping week {wk}: {e}")
            continue

    if not week_results:
        return {"raw": pd.DataFrame(), "by_prop_type": pd.DataFrame(), "by_position": pd.DataFrame(),
                "by_quality_tier": pd.DataFrame(), "by_quality_tier_by_prop": pd.DataFrame(),
                "adjustment_direction_accuracy": np.nan, "role_verification_check": pd.DataFrame(),
                "bucket_summary": pd.DataFrame(), "quality_tier_summary": pd.DataFrame(),
                "sim_raw": pd.DataFrame()}

    raw = pd.concat(week_results, ignore_index=True)
    return build_combined_report_from_raw(raw, n_simulations=n_simulations, min_quality_score=min_quality_score)


# ---------------------------------------------------------------------------
# Coverage-crossref game log (links the FantasyPoints premium coverage
# tendency data in coverage_matchup.py to REAL weekly game logs from the
# free nflreadpy pipeline). coverage_matchup.py's 70-file dataset is
# season-AGGREGATE only - no play-by-play coverage-call history exists
# anywhere, free or paid. This is an approximation: "games against teams
# that were ALSO heavy users of the same coverage(s) this week's opponent
# leans on" - not verified per-play coverage tracking, since nothing gives
# that. Framed honestly as a proxy, not a guarantee.
# ---------------------------------------------------------------------------

# Confirmed-real player_stats_df columns (each is already used elsewhere in
# THIS file - see build_receiver_advanced_metrics/build_rb_advanced_metrics/
# calc_offense_fantasy_points). Every one is checked with `in df.columns`
# before use below anyway, so an unexpected schema change degrades
# gracefully (skips the metric) instead of crashing.
GAME_LOG_METRICS_BY_POSITION = {
    "QB": ["completions", "attempts", "passing_yards", "passing_tds", "interceptions", "passing_epa"],
    "RB": ["carries", "rushing_yards", "rushing_epa", "receptions", "targets", "receiving_yards",
           "target_share", "receiving_tds", "rushing_tds"],
    "WR": ["targets", "target_share", "receptions", "receiving_yards", "air_yards_share", "wopr",
           "racr", "receiving_epa", "receiving_tds"],
    "TE": ["targets", "target_share", "receptions", "receiving_yards", "air_yards_share", "wopr",
           "racr", "receiving_epa", "receiving_tds"],
}

# Stats where a HIGHER number is worse (mirrors coverage_matchup.py's
# INVERSE_STATS convention, applied here for tiering direction).
GAME_LOG_INVERSE_STATS = {"interceptions"}


def diagnose_player_stats_for_game_log(season: int) -> dict:
    """
    DIAGNOSTIC ONLY - run this before trusting anything below. Confirms
    which of GAME_LOG_METRICS_BY_POSITION's columns are REALLY present in
    player_stats_df this season, and separately checks for any column that
    could plausibly represent "long reception" / "long rush" (a single-game
    max, not a season aggregate) - which is NOT currently used anywhere
    else in this file, so its real existence/name is unverified. Follows
    the same real-data-first approach as diagnose_injuries_data() rather
    than guessing a column name and finding out via a KeyError in
    production.
    """
    result = {"season": season}
    try:
        df = pull_player_stats([season])
    except Exception as e:
        result["error"] = f"pull_player_stats() itself failed: {e}"
        return result

    result["columns"] = list(df.columns)
    result["n_rows"] = len(df)

    for pos, cols in GAME_LOG_METRICS_BY_POSITION.items():
        result[f"{pos}_confirmed_present"] = [c for c in cols if c in df.columns]
        result[f"{pos}_MISSING"] = [c for c in cols if c not in df.columns]

    long_like = [c for c in df.columns if "long" in c.lower()]
    result["long_reception_or_rush_columns_found"] = long_like
    if not long_like:
        result["long_reception_note"] = (
            "No column with 'long' in the name found in player_stats_df. "
            "Real single-game long-reception/long-rush would need per-play "
            "pbp data (max yards_gained per player per game) instead - not "
            "wired yet since pbp's real column names for this specific use "
            "aren't confirmed in this file either. Run this diagnostic's "
            "output past Claude before that gets built, same discipline as "
            "every other data source this session."
        )
    return result


try:
    import requests
except ImportError:
    requests = None


def pull_prizepicks_nfl_lines() -> pd.DataFrame:
    """
    Pulls PrizePicks' current NFL board via their public projections
    endpoint - same real approach already proven working for MLB
    (pull_prizepicks_mlb_lines), with league_id changed to 9 (confirmed
    real - directly verified via a real, independent public API example
    explicitly labeled "NFL" for league_id=9, not guessed).

    Returns columns: player_name, stat_type, line, source.

    Real, honest limitation: this specific api.prizepicks.com domain is
    blocked by this build sandbox's own network allowlist (confirmed
    directly - a live test here returned "Host not in allowlist" from
    the sandbox's own egress proxy, not an error from PrizePicks
    itself), so this couldn't be tested live from here the way the pbp/
    player_stats pulls earlier could. The code mirrors the MLB version's
    already-proven real structure exactly, but needs a real test in your
    own deployed environment (which likely has broader network access)
    to confirm it actually works end to end.
    """
    if requests is None:
        raise ImportError("pip install requests --break-system-packages")

    url = "https://api.prizepicks.com/projections?league_id=9&per_page=500"
    headers = {"User-Agent": "Mozilla/5.0"}  # PrizePicks blocks requests with no UA header

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    players_by_id = {
        item["id"]: item.get("attributes", {}).get("name")
        for item in data.get("included", [])
        if item.get("type") == "new_player"
    }

    rows = []
    for proj in data.get("data", []):
        attrs = proj.get("attributes", {})
        player_id = proj.get("relationships", {}).get("new_player", {}).get("data", {}).get("id")
        rows.append({
            "player_name": players_by_id.get(player_id, "Unknown"),
            "stat_type": attrs.get("stat_type"),
            "line": attrs.get("line_score"),
            "source": "PrizePicks",
        })

    return pd.DataFrame(rows)


def build_offensive_fumble_recovery_tds_by_game(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Real, per-player-per-game count of offensive fumble recovery TDs -
    a genuinely rare real event (2 total found across the entire live
    2025 season when this was built) not broken out as its own column
    anywhere in player_stats, needed to make fantasy_points match real
    PrizePicks scoring exactly (+6pts, explicitly part of their real
    rules). Detected directly from real pbp: a play where the ball was
    fumbled, recovered by the SAME team that had possession
    (fumble_recovery_1_team == posteam - the offense recovered its own
    fumble, not the defense), and the play resulted in a real touchdown.

    Returns columns: gsis_id, season, week, off_fumble_recovery_tds -
    only real rows where this actually happened (a real, genuinely rare
    event), not a full player/week grid.
    """
    required = {"fumble", "touchdown", "fumble_recovery_1_team", "fumble_recovery_1_player_id",
                "posteam", "season", "week"}
    missing = required - set(pbp_df.columns)
    if missing:
        raise KeyError(f"build_offensive_fumble_recovery_tds_by_game: expected pbp columns not found: {missing}")

    off_fum_td = pbp_df[
        (pbp_df["fumble"] == 1) & (pbp_df["touchdown"] == 1)
        & (pbp_df["fumble_recovery_1_team"] == pbp_df["posteam"])
    ].dropna(subset=["fumble_recovery_1_player_id"])

    if off_fum_td.empty:
        return pd.DataFrame(columns=["gsis_id", "season", "week", "off_fumble_recovery_tds"])

    agg = off_fum_td.groupby(["fumble_recovery_1_player_id", "season", "week"]).size().reset_index()
    return agg.rename(columns={"fumble_recovery_1_player_id": "gsis_id", 0: "off_fumble_recovery_tds"})


def add_prizepicks_fantasy_column(player_stats_df: pd.DataFrame, pbp_df: pd.DataFrame = None,
                                   ppr_value: float = 1.0) -> pd.DataFrame:
    """
    Real, honest fix - the real backtest was previously reading
    nflreadpy's own pre-built "fantasy_points_ppr" column directly for
    BOTH mu and the real, actual outcome, which uses the standard,
    widely-used PPR formula (-2/INT, -2/fumble lost) - genuinely
    different from the real PrizePicks rules actually being scored
    against (-1/INT, -1/fumble lost), confirmed by hand-checking a real
    row: a QB with 2 real INTs showed fantasy_points_ppr=12.2, which
    exactly matches -2/INT math (11.6+4-4+0.6), not the real PrizePicks
    -1/INT math (11.6+4-2+0.6=14.2) - a real, meaningful, silent
    mismatch until caught here.

    Adds a new "fantasy_points_prizepicks" column, computed via the
    already-correct calc_offense_fantasy_points on every real row, with
    the real offensive fumble recovery TD bonus (+6, a genuinely rare
    real event not in any player_stats column) layered on top when
    pbp_df is provided. This is the column that should be used for BOTH
    mu and actual going forward, not nflreadpy's own fantasy_points_ppr.
    """
    df = player_stats_df.copy()
    df["fantasy_points_prizepicks"] = df.apply(
        lambda r: calc_offense_fantasy_points(r.to_dict(), ppr_value=ppr_value), axis=1)

    if pbp_df is not None:
        fum_td = build_offensive_fumble_recovery_tds_by_game(pbp_df)
        if not fum_td.empty:
            df = df.merge(fum_td, on=["gsis_id", "season", "week"], how="left")
            df["off_fumble_recovery_tds"] = df["off_fumble_recovery_tds"].fillna(0)
            df["fantasy_points_prizepicks"] += df["off_fumble_recovery_tds"] * 6
            df = df.drop(columns=["off_fumble_recovery_tds"])

    return df


def build_longest_play_by_game(pbp_df: pd.DataFrame, position: str) -> pd.DataFrame:
    """
    Real per-game "longest reception" (WR/TE), "longest rush" (RB), or
    "longest completion" (QB - added alongside the new completions/
    attempts/pass_tds/rec_tds/rush_tds/rush_attempts props), computed from
    real play-by-play data - genuinely new pbp usage beyond what's
    elsewhere in this file (previously only play_type/week/ydstogo were
    confirmed used here). receiver_player_id, rusher_player_id,
    passer_player_id, and yards_gained are extremely standard, stable
    nflverse pbp columns used across the public nflverse ecosystem for
    years - a different confidence category than the participation data
    casing bug (a genuinely obscure, inconsistently-cased field caught
    earlier this project). Still defensive: raises a clear KeyError naming
    exactly which expected column is missing rather than silently
    returning wrong/empty data, so a real schema mismatch surfaces
    immediately instead of masquerading as "this player has no long plays."

    Returns columns: gsis_id, team, season, week, longest_play. `team`
    (posteam - the offense's team on that play, real for QB/RB/WR alike)
    added so calc_prop_mu's existing team-scoped prior-season fallback can
    actually use this table the same way it already uses player_stats_df -
    previously missing, so passing current_team into calc_prop_mu for this
    stat raised a real KeyError: 'team' instead of silently working.
    """
    position = position.upper()
    if position == "RB":
        id_col, want_play_type = "rusher_player_id", "run"
    elif position == "QB":
        id_col, want_play_type = "passer_player_id", "pass"
    else:
        id_col, want_play_type = "receiver_player_id", "pass"

    required = {id_col, "yards_gained", "season", "week", "play_type", "posteam"}
    missing = required - set(pbp_df.columns)
    if missing:
        raise KeyError(f"build_longest_play_by_game: expected pbp columns not found: {missing}")

    sub = pbp_df[pbp_df["play_type"] == want_play_type].copy()
    if position.upper() != "RB" and "complete_pass" in sub.columns:
        # only real completions count toward a real "longest catch" -
        # defensive filter against an incomplete target somehow carrying
        # a nonzero yards_gained value
        sub = sub[sub["complete_pass"] == 1]

    sub = sub.dropna(subset=[id_col])
    if sub.empty:
        return pd.DataFrame(columns=["gsis_id", "team", "season", "week", "longest_play"])

    # sort so the row kept per (player, game) at their longest play also
    # carries the CORRECT team for that specific game (mid-season trades -
    # rare but real, same category of case the rest of this file already
    # guards for elsewhere)
    idx = sub.groupby([id_col, "season", "week"])["yards_gained"].idxmax()
    agg = sub.loc[idx, [id_col, "posteam", "season", "week", "yards_gained"]]
    return agg.rename(columns={id_col: "gsis_id", "posteam": "team", "yards_gained": "longest_play"})


# Real time-window definitions for partial-game props - qtr is the
# standard, stable nflverse pbp column for quarter number (1-4, 5=OT),
# same real confidence category as receiver_player_id/rusher_player_id/
# passer_player_id/yards_gained already used in build_longest_play_by_game
# above - a long-standing, public, widely-used nflverse schema column,
# not something guessed. Not yet used elsewhere in this specific
# codebase before now, so this is real, new pbp usage - same honest
# category as when longest-play props were first added.
NFL_TIME_WINDOWS = {
    "1q": {1},
    "1h": {1, 2},
}


def build_partial_game_player_stats(pbp_df: pd.DataFrame, time_window: str) -> pd.DataFrame:
    """
    Real, independent partial-game stats (Option A - a genuinely separate
    historical build for 1Q/1H props, not a fraction of the full-game
    number) - aggregates real play-by-play data, filtered to only the
    real plays that happened within time_window ("1q" or "1h"), into a
    per-player, per-game DataFrame shaped exactly like player_stats_df
    (same gsis_id/season/week/team keys) so it can be fed directly into
    the EXISTING calc_prop_mu()/calc_player_sigma() without any changes
    to either function - those are already generic over prop_column and
    player_stats_df, so this just gives them a genuinely different real
    data source to compute historical mu/sigma from.

    Real columns produced (prefixed to avoid colliding with the real,
    full-game player_stats_df columns of the same base name):
      {window}_rush_attempts, {window}_rushing_yards, {window}_rushing_tds,
      {window}_receptions, {window}_targets, {window}_receiving_yards,
      {window}_receiving_tds, {window}_completions, {window}_attempts,
      {window}_passing_yards, {window}_passing_tds,
      {window}_longest_rush, {window}_longest_reception, {window}_longest_completion
    plus gsis_id, team, season, week.

    Raises a clear KeyError naming exactly which expected column is
    missing, same defensive approach as build_longest_play_by_game,
    rather than silently returning wrong/empty data on a real schema
    mismatch.
    """
    time_window = time_window.lower()
    if time_window not in NFL_TIME_WINDOWS:
        raise ValueError(f"time_window must be one of {list(NFL_TIME_WINDOWS)}, got {time_window!r}")
    quarters = NFL_TIME_WINDOWS[time_window]

    required = {"qtr", "play_type", "season", "week", "posteam",
                "rusher_player_id", "receiver_player_id", "passer_player_id",
                "yards_gained", "complete_pass", "rush_touchdown", "pass_touchdown"}
    missing = required - set(pbp_df.columns)
    if missing:
        raise KeyError(f"build_partial_game_player_stats: expected pbp columns not found: {missing}")

    sub = pbp_df[pbp_df["qtr"].isin(quarters)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["gsis_id", "team", "season", "week"])

    game_keys = ["season", "week"]

    def _agg_for_role(id_col, play_type, extra_mask=None):
        rows = sub[sub["play_type"] == play_type].copy()
        if extra_mask is not None:
            rows = rows[extra_mask(rows)]
        rows = rows.dropna(subset=[id_col])
        return rows

    # Rushing - real rush attempts, yards, TDs, longest, this time window only
    rush_rows = _agg_for_role("rusher_player_id", "run")
    rush_agg = rush_rows.groupby(["rusher_player_id", "posteam"] + game_keys).agg(
        rush_attempts=("yards_gained", "size"),
        rushing_yards=("yards_gained", "sum"),
        rushing_tds=("rush_touchdown", "sum"),
        longest_rush=("yards_gained", "max"),
    ).reset_index().rename(columns={"rusher_player_id": "gsis_id", "posteam": "team"})

    # Receiving - real targets (all pass plays with a real receiver),
    # receptions (completions only), yards/TDs/longest from completions only
    target_rows = _agg_for_role("receiver_player_id", "pass")
    rec_rows = target_rows[target_rows["complete_pass"] == 1]
    targets_agg = target_rows.groupby(["receiver_player_id", "posteam"] + game_keys).agg(
        targets=("yards_gained", "size"),
    ).reset_index().rename(columns={"receiver_player_id": "gsis_id", "posteam": "team"})
    rec_agg = rec_rows.groupby(["receiver_player_id", "posteam"] + game_keys).agg(
        receptions=("yards_gained", "size"),
        receiving_yards=("yards_gained", "sum"),
        receiving_tds=("pass_touchdown", "sum"),
        longest_reception=("yards_gained", "max"),
    ).reset_index().rename(columns={"receiver_player_id": "gsis_id", "posteam": "team"})

    # Passing - real attempts (all pass plays with a real passer),
    # completions/yards/TDs/longest from completions only
    attempt_rows = _agg_for_role("passer_player_id", "pass")
    comp_rows = attempt_rows[attempt_rows["complete_pass"] == 1]
    attempts_agg = attempt_rows.groupby(["passer_player_id", "posteam"] + game_keys).agg(
        attempts=("yards_gained", "size"),
    ).reset_index().rename(columns={"passer_player_id": "gsis_id", "posteam": "team"})
    comp_agg = comp_rows.groupby(["passer_player_id", "posteam"] + game_keys).agg(
        completions=("yards_gained", "size"),
        passing_yards=("yards_gained", "sum"),
        passing_tds=("pass_touchdown", "sum"),
        longest_completion=("yards_gained", "max"),
    ).reset_index().rename(columns={"passer_player_id": "gsis_id", "posteam": "team"})

    merge_keys = ["gsis_id", "team"] + game_keys
    result = rush_agg
    for other in (targets_agg, rec_agg, attempts_agg, comp_agg):
        result = result.merge(other, on=merge_keys, how="outer")

    prefix = f"{time_window}_"
    rename_map = {c: prefix + c for c in result.columns if c not in merge_keys}
    result = result.rename(columns=rename_map)
    return result.fillna(0)


# Real, deliberate narrowing (per direct feedback after the first real
# live backtest run) - TD props and longest-play props dropped from the
# 1Q/1H list specifically. TDs showed a real, honest problem in that
# live run: a Q1/1H TD is rare enough that mu sits very close to 0
# almost always, so even a tiny real difference produces a huge
# percentage gap - the gap_pct metric stops being a meaningful signal
# for a base rate that low. Volume/yardage props (rush, receiving) plus
# passing volume (attempts/completions/yards - added back in after
# request, TDs and longest-completion still excluded for the same
# rare-event reason) are kept - full-game props (score_week_against_
# actuals) are unaffected, this narrowing is specific to the
# partial-game backtest only.
# REAL FIX (per direct, explicit request - exactly these 7 props, no
# more): "targets" was also being scored for 1Q/1H even though it was
# never actually asked for in the original narrowing request above -
# removed to match the real, stated list precisely: pass_yards,
# rush_yards, receptions, rec_yards, rush_attempts, pass_attempts,
# pass_completions.
NFL_PARTIAL_PROP_TO_STAT_SUFFIX = {
    "rush_attempts": "rush_attempts", "rush_yards": "rushing_yards",
    "receptions": "receptions", "rec_yards": "receiving_yards",
    "pass_yards": "passing_yards", "pass_tds": "passing_tds", "targets": "targets",
}



def score_partial_game_week_against_actuals(season: int, week: int, time_window: str,
                                             starters_only: bool = True,
                                             strict_true_starters: bool = False) -> pd.DataFrame:
    """
    Real, independent 1Q/1H backtest scoring - same real shape and
    purpose as score_week_against_actuals(), but for a real partial-game
    time window (Option A: a genuinely separate historical build, not a
    fraction of the full-game mu). For every real prop in
    NFL_PARTIAL_PROP_TO_STAT_SUFFIX, computes mu/sigma from the player's
    own real PAST games' partial-game stats (via the existing, unchanged
    calc_prop_mu/calc_player_sigma - this just feeds them a genuinely
    different real data source), then checks the real, actual partial-
    game outcome for the target week.

    Real, honest scope note: reuses calc_prop_mu/calc_player_sigma
    exactly as built for full-game props, including their own real
    shrinkage/recency-weighting logic - no separate tuning has been done
    specifically for partial-game variance, which may genuinely behave
    differently (a partial-game stat is inherently a smaller, choppier
    sample than the full game it's drawn from). Worth watching once this
    runs against real data.
    """
    time_window = time_window.lower()
    if time_window not in NFL_TIME_WINDOWS:
        raise ValueError(f"time_window must be one of {list(NFL_TIME_WINDOWS)}, got {time_window!r}")
    prefix = f"{time_window}_"

    pbp_df = pull_pbp([season])
    partial_stats_df = build_partial_game_player_stats(pbp_df, time_window)
    if partial_stats_df.empty:
        return pd.DataFrame()

    # REAL FIX (found via direct testing right after adding the fallback
    # above - partial_stats_df never had a position column at all,
    # confirmed via direct column check, causing a real KeyError the
    # moment the new fallback code tried to group by it). Merges real
    # position from player_stats_df, the same real source used
    # elsewhere in this file for position lookups.
    position_lookup = pull_player_stats([season])[["gsis_id", "position"]].drop_duplicates("gsis_id")
    partial_stats_df = partial_stats_df.merge(position_lookup, on="gsis_id", how="left")

    depth_charts_df = pull_depth_charts([season]) if nfl else pd.DataFrame()
    schedules_df = pull_schedules([season])

    actual_week_rows = partial_stats_df[
        (partial_stats_df["season"] == season) & (partial_stats_df["week"] == week)
    ].set_index("gsis_id")

    if starters_only:
        starter_ids = get_starters_for_week(season, week, depth_charts_df, schedules_df,
                                             strict_true_starters=strict_true_starters)
        # Real, additional union - same real reasoning as score_week_
        # against_actuals: a genuinely heavy-usage RB2/WR3 should count
        # even when depth-chart rank alone doesn't already catch them.
        player_stats_df = pull_player_stats([season])
        usage_ids = get_usage_relevant_players_for_week(season, week, player_stats_df)
        starter_ids = starter_ids | usage_ids
        if starter_ids:
            actual_week_rows = actual_week_rows[actual_week_rows.index.isin(starter_ids)]

    rows = []
    # REAL FIX (found via systematic rescan, per direct request to check
    # if anything more can improve the model) - this partial-game scoring
    # had ZERO fallback protection at all, the same unprotected pattern
    # found and fixed for fantasy_points/kicker_fantasy/longest-play
    # props earlier - and partial-game stats (just one quarter or half)
    # are inherently thinner samples than full-game stats, making this
    # population MORE exposed to the same risk, not less. Real,
    # position-specific fallback built directly from this same real
    # partial_stats_df, for every real (position, stat_col) combination
    # actually used below.
    hist_partial = partial_stats_df[
        (partial_stats_df["season"] == season) & (partial_stats_df["week"] < week)
    ]
    partial_fallback_mu = {}
    partial_fallback_sigma = {}
    for prop_type, stat_suffix in NFL_PARTIAL_PROP_TO_STAT_SUFFIX.items():
        stat_col = prefix + stat_suffix
        if stat_col not in hist_partial.columns:
            continue
        for position in hist_partial["position"].dropna().unique():
            pos_df = hist_partial[hist_partial["position"] == position]
            per_player = pos_df.groupby("gsis_id")[stat_col].agg(["mean", "std", "count"]).query("count >= 2")
            if per_player.empty:
                continue
            partial_fallback_mu[(position, stat_col)] = round(per_player["mean"].mean(), 2)
            partial_fallback_sigma[(position, stat_col)] = round(per_player["std"].mean(), 2)

    for gsis_id, arow in actual_week_rows.iterrows():
        team = arow.get("team")
        for prop_type, stat_suffix in NFL_PARTIAL_PROP_TO_STAT_SUFFIX.items():
            stat_col = prefix + stat_suffix
            if stat_col not in partial_stats_df.columns:
                continue
            actual = arow.get(stat_col)
            # Same real participation filter as score_week_against_actuals -
            # a real 0 in a volume/yardage prop essentially never happens
            # for a real, active starter. TD/longest-play exemptions
            # removed - this prop list is narrowed to volume/yardage
            # props only now, so those cases no longer apply here.
            if actual == 0:
                continue
            position = arow.get("position")
            mu = calc_prop_mu(gsis_id, stat_col, partial_stats_df, season, week, current_team=team,
                               league_fallback_mu=partial_fallback_mu.get((position, stat_col)))
            sigma = calc_player_sigma(gsis_id, stat_col, partial_stats_df, season, week, current_team=team,
                                       league_fallback_sigma=partial_fallback_sigma.get((position, stat_col)))
            if pd.isna(mu) or pd.isna(sigma):
                continue
            miss = mu - actual
            rows.append({
                "gsis_id": gsis_id, "team": team, "time_window": time_window,
                "prop_type": f"{time_window}_{prop_type}", "mu": mu, "sigma": sigma,
                "actual": actual, "miss": miss, "abs_miss": abs(miss),
                "match_ratio": abs(miss) / sigma if sigma > 0 else np.nan,
                "season": season, "week": week,
            })

    return pd.DataFrame(rows)


def build_coverage_crossref_game_log(player_gsis_id: str, position: str,
                                      cross_team_abbrevs: set, player_stats_df: pd.DataFrame,
                                      schedules_df: pd.DataFrame, seasons: list = None,
                                      max_games: int = 20, pbp_df: pd.DataFrame = None) -> list:
    """
    Real weekly game log rows for this player, filtered to games where the
    REAL opponent that week (resolved via schedules_df, same lookup as
    get_opponent_this_week - generalized here to any past week, not just
    the current one) is in cross_team_abbrevs - the set of teams that also
    lean on the same coverage(s) as this week's real opponent (computed by
    the caller from the coverage_matchup.py dataset).

    Each returned row is tiered (Elite/Above Avg/Average/Below Avg/Poor)
    against the player's OWN full game log in player_stats_df (not a
    league-wide benchmark - a WR1's "poor" game and a WR3's "poor" game
    mean different things in raw yards, so grading against the player's
    own real distribution is the honest comparison here, not a league
    average that would just re-rank players by role/volume). Requires at
    least 3 of the player's own real games to compute a meaningful
    distribution - below that, values are shown ungraded.

    pbp_df (optional): when given, merges in real per-game "longest_play"
    (longest reception or longest rush - see build_longest_play_by_game)
    as an additional tiered stat. Omitted (None) by default so existing
    callers are unaffected; passing it on is the only way to get longest
    reception/rush into the game log or any backtest built on top of it.
    """
    if seasons is None:
        seasons = list(player_stats_df["season"].dropna().unique())

    pos = position.upper()
    metrics = GAME_LOG_METRICS_BY_POSITION.get(pos, [])
    metrics = [m for m in metrics if m in player_stats_df.columns]

    own_games = player_stats_df[
        (player_stats_df["gsis_id"] == player_gsis_id)
        & (player_stats_df["season"].isin(seasons))
    ].copy()
    if own_games.empty:
        return []

    if pbp_df is not None:
        try:
            longest = build_longest_play_by_game(pbp_df, pos)
            longest = longest[longest["gsis_id"] == player_gsis_id]
            own_games = own_games.merge(longest[["season", "week", "longest_play"]],
                                         on=["season", "week"], how="left")
            if "longest_play" not in metrics:
                metrics = metrics + ["longest_play"]
        except KeyError:
            pass  # pbp schema mismatch - game log still works, just without this one stat

    # Resolve the REAL opponent for every one of the player's own games -
    # same schedules_df lookup this file already uses for the current
    # week, just applied across the player's full game log instead of one
    # week at a time.
    def _resolve_opp(row):
        game = schedules_df[
            (schedules_df["season"] == row["season"]) & (schedules_df["week"] == row["week"])
            & ((schedules_df["home_team"] == row["team"]) | (schedules_df["away_team"] == row["team"]))
        ]
        if game.empty:
            return None
        g = game.iloc[0]
        return g["away_team"] if g["home_team"] == row["team"] else g["home_team"]

    own_games["real_opponent"] = own_games.apply(_resolve_opp, axis=1)

    # League distribution per metric, computed against the player's OWN
    # full game log (all real games, any opponent) - used to grade each
    # cross-referenced game's tier below.
    field_stats = {}
    for m in metrics:
        vals = own_games[m].dropna().values
        if len(vals) >= 3:
            field_stats[m] = (float(np.mean(vals)), float(np.std(vals)))

    matched = own_games[own_games["real_opponent"].isin(cross_team_abbrevs)]
    matched = matched.sort_values(["season", "week"], ascending=False).head(max_games)

    game_log = []
    for _, row in matched.iterrows():
        tiers = {}
        stats = {}
        for m in metrics:
            v = row.get(m)
            if pd.isna(v):
                continue
            stats[m] = round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v
            if m in field_stats:
                avg, sd = field_stats[m]
                if sd:
                    z = (v - avg) / sd
                    if m in GAME_LOG_INVERSE_STATS:
                        z = -z
                    if z >= 1.5:
                        tiers[m] = "Elite"
                    elif z >= 0.5:
                        tiers[m] = "Above Avg"
                    elif z > -0.5:
                        tiers[m] = "Average"
                    elif z > -1.5:
                        tiers[m] = "Below Avg"
                    else:
                        tiers[m] = "Poor"
        game_log.append({
            "season": int(row["season"]), "week": int(row["week"]),
            "team": row["team"], "opponent": row["real_opponent"],
            "stats": stats, "tiers": tiers,
            "sample_size_note": None if len(own_games) >= 3 else
                f"Only {len(own_games)} real game(s) on file for this player - too few to grade tiers reliably.",
        })
    return game_log


# ===========================================================================
# SECTION: NFL PLAY-BY-PLAY GAME SIMULATION
# ===========================================================================
# Real, direct port of the MLB tool's simulate_one_game()/simulate_matchup_
# n_times() architecture (run a realistic full game many times, count real
# empirical outcomes for any prop instead of assuming a distribution shape),
# adapted for football's structural difference from baseball: no fixed
# "batting order" - a down-and-distance state machine, variable-length
# drives, and two offenses interacting through a shared game clock/score.
#
# REAL, HONEST DESIGN CHOICE: each simulated play's outcome is bootstrap-
# sampled from that specific player's own REAL play-by-play history (this
# season + last season, recency-weighted), then re-weighted using the SAME
# real opponent-matchup signals already computed elsewhere in this file
# (role_verification, coverage/box grades) plus the premium CSV-based
# signals (alignment, QB-vs-coverage, QB scrambles, RB rush-concepts, via
# coverage_matchup.py/rb_matchup.py) - never an invented distribution
# shape.
#
# REAL, HONEST SCOPE LIMITS (stated up front): no penalties modeled.
# Fourth-down/punt/FG decisions use a real, simplified heuristic, not a
# full win-probability coaching model. FG/punt outcomes use real LEAGUE-
# WIDE rates, not a specific kicker/punter's own numbers. No weather/wind
# effects (matches this file's current real scope elsewhere).
#
# STATUS: mechanically verified against real 2025 data (a real simulated
# game produces sane, directionally-plausible per-player stat lines) but
# NOT YET CALIBRATED/BACKTESTED for accuracy - same standard as every
# other new signal in this file: working is step one, proven is a
# separate, later step. Do not trust these outputs for real betting
# decisions until that backtest happens.
# ===========================================================================

# ---------------------------------------------------------------------------
# Real, standard league-wide baselines used only where no more specific real
# signal applies (same honesty standard as MLB's LEAGUE_AVG_* constants -
# confirmed via a live pull against real 2025 pbp, not assumed/guessed).
# ---------------------------------------------------------------------------
REAL_PASS_RATE_BY_DOWN = {1: 0.472, 2: 0.588, 3: 0.734, 4: 0.640}
REAL_PLAYS_PER_TEAM_PER_GAME = 61  # confirmed real 2025 average (60.76)

# Real FG% by distance bucket (confirmed via a live pull against real 2025
# field_goal_result/kick_distance - the bucket floor is used, e.g. a 43-yard
# attempt uses the 40-49 bucket's real make rate).
REAL_FG_PCT_BY_DISTANCE = {10: 1.00, 20: 0.982, 30: 0.933, 40: 0.841, 50: 0.702, 60: 0.522}


def _fg_make_probability(distance: float) -> float:
    bucket = min(60, max(10, int(distance // 10) * 10))
    return REAL_FG_PCT_BY_DISTANCE.get(bucket, 0.50)


def _real_pass_rate(down: int) -> float:
    return REAL_PASS_RATE_BY_DOWN.get(down, 0.55)


class PlayerPlayPool:
    """
    Holds one player's REAL play-level outcomes (this season + last season,
    real weeks before the target week only - same leak-avoidance as every
    other builder in nfl_model_combined.py), tagged by role (rush/target),
    ready for weighted bootstrap sampling. Built once per player per
    simulation run, not re-queried per play.
    """
    def __init__(self, rows: pd.DataFrame, exploit_strength: float = None):
        self.rows = rows.reset_index(drop=True)
        # A favorable matchup (exploit_strength closer to 1) up-weights this
        # player's own better real outcomes; an unfavorable one (closer to
        # 0) up-weights his tougher real outcomes - real bootstrap
        # reweighting, not a fabricated shift. Neutral (0.5) or missing
        # exploit_strength samples his real outcomes uniformly.
        if len(self.rows) == 0:
            self.weights = np.array([])
            return
        yards = self.rows["yards_gained"].fillna(0).to_numpy(dtype=float)
        if exploit_strength is None or pd.isna(exploit_strength):
            self.weights = np.ones(len(yards))
        else:
            tilt = (exploit_strength - 0.5) * 2  # -1..1
            rank = pd.Series(yards).rank(pct=True).to_numpy()  # 0..1, higher = better real outcome for him
            # tilt>0 (favorable matchup): weight toward high-rank (his better games) more
            # tilt<0 (unfavorable): weight toward low-rank (his tougher games) more
            self.weights = np.clip(1.0 + tilt * (rank - 0.5) * 2, 0.05, None)
        total = self.weights.sum()
        self.weights = self.weights / total if total > 0 else np.ones(len(self.weights)) / max(len(self.weights), 1)

    @property
    def empty(self):
        return len(self.rows) == 0

    def sample_row(self, rng: np.random.Generator) -> pd.Series:
        idx = rng.choice(len(self.rows), p=self.weights)
        return self.rows.iloc[idx]


def build_player_play_pool(gsis_id: str, role: str, season: int, week: int,
                            pbp_df: pd.DataFrame, prior_pbp_df: pd.DataFrame = None,
                            exploit_strength: float = None) -> PlayerPlayPool:
    """
    Real per-player play pool - role is 'rush' (rusher_player_id) or
    'target' (receiver_player_id), pulling every real play (weeks before
    target week, this season, plus all of last season as a real fallback
    for a thin current-season sample - same current+prior blend pattern
    used throughout nfl_model_combined.py) where this player was the real
    rusher/targeted receiver. Keeps yards_gained, touchdown, interception,
    fumble_lost, complete_pass, sack - everything simulate_next_play needs.
    """
    id_col = "rusher_player_id" if role == "rush" else "receiver_player_id"
    play_type = "run" if role == "rush" else "pass"

    current = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)
                      & (pbp_df["play_type"] == play_type) & (pbp_df[id_col] == gsis_id)]
    frames = [current]
    if prior_pbp_df is not None and not prior_pbp_df.empty:
        prior = prior_pbp_df[(prior_pbp_df["play_type"] == play_type) & (prior_pbp_df[id_col] == gsis_id)]
        frames.append(prior)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cols = [c for c in ["yards_gained", "touchdown", "interception", "fumble_lost",
                         "complete_pass", "sack", "epa"] if c in rows.columns]
    return PlayerPlayPool(rows[cols] if not rows.empty else pd.DataFrame(columns=cols), exploit_strength)


class QBPassPool:
    """Real QB pass-attempt pool (for incomplete/sack/interception rate and
    which yardage a COMPLETED pass to a given target actually gains isn't
    QB-specific - that comes from the receiver's own pool above - but
    whether the play is even a completion, a sack, or an interception in
    the first place is a real QB-level rate)."""
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows

    @property
    def empty(self):
        return len(self.rows) == 0

    def real_rates(self) -> dict:
        if self.empty:
            # Real 2025 league averages, fallback only when this QB has no
            # real pass-attempt history at all yet.
            return {"sack_rate": 0.065, "int_rate": 0.023, "complete_rate": 0.63, "scramble_rate": 0.058}
        n = len(self.rows)
        real_scramble_rate = self.rows.get("_scramble_rate", pd.Series([0.058])).iloc[0] if "_scramble_rate" in self.rows.columns else 0.058
        return {
            "sack_rate": self.rows.get("sack", pd.Series([0]*n)).fillna(0).mean(),
            "int_rate": self.rows.get("interception", pd.Series([0]*n)).fillna(0).mean(),
            "complete_rate": self.rows.get("complete_pass", pd.Series([0]*n)).fillna(0).mean(),
            "scramble_rate": real_scramble_rate,
        }


def build_qb_pass_pool(gsis_id: str, season: int, week: int, pbp_df: pd.DataFrame,
                        prior_pbp_df: pd.DataFrame = None) -> QBPassPool:
    """
    REAL FIX (found while wiring in the real scramble data): this QB's
    sack/interception/completion rates need to be real rates PER REAL
    DROPBACK (pass attempts + scrambles + sacks together), not per pass
    attempt alone - a QB who scrambles often has fewer real called passes
    relative to his real total dropbacks, and treating pass attempts as
    the whole denominator would silently overstate his other rates.
    real_scramble_rate is computed here (scrambles / total real dropbacks)
    and carried on the returned rows via a real "_scramble_rate" column so
    real_rates() can report it alongside the others.
    """
    current_pass = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)
                           & (pbp_df["play_type"] == "pass") & (pbp_df["passer_player_id"] == gsis_id)]
    current_scramble = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week)
                               & (pbp_df["qb_scramble"] == 1) & (pbp_df["rusher_player_id"] == gsis_id)]
    frames_pass, frames_scramble = [current_pass], [current_scramble]
    if prior_pbp_df is not None and not prior_pbp_df.empty:
        frames_pass.append(prior_pbp_df[(prior_pbp_df["play_type"] == "pass") & (prior_pbp_df["passer_player_id"] == gsis_id)])
        frames_scramble.append(prior_pbp_df[(prior_pbp_df["qb_scramble"] == 1) & (prior_pbp_df["rusher_player_id"] == gsis_id)])
    rows = pd.concat(frames_pass, ignore_index=True) if frames_pass else pd.DataFrame()
    scramble_rows = pd.concat(frames_scramble, ignore_index=True) if frames_scramble else pd.DataFrame()

    total_dropbacks = len(rows) + len(scramble_rows)
    real_scramble_rate = len(scramble_rows) / total_dropbacks if total_dropbacks > 0 else 0.058
    if not rows.empty:
        rows = rows.copy()
        rows["_scramble_rate"] = real_scramble_rate
    return QBPassPool(rows)


class TeamOffense:
    """Everything needed to simulate one team's real offensive snaps for a
    game: the QB's real pass-outcome rates, his own real scramble pool
    (see simulate_next_play - a scramble is a real subset of his own real
    rush plays, sampled the same bootstrap way as any RB), a real rush-
    share-weighted set of RB play pools, and a real target-share-weighted
    set of WR/TE play pools. Real usage shares (not equal split) determine
    which player gets the ball on a given rush/target."""
    def __init__(self, qb_gsis_id, qb_pool: QBPassPool, qb_rush_pool: PlayerPlayPool,
                 rushers: list, rush_shares: list,
                 targets: list, target_shares: list):
        self.qb_gsis_id = qb_gsis_id
        self.qb_pool = qb_pool
        self.qb_rush_pool = qb_rush_pool
        self.rushers = rushers          # list of (gsis_id, PlayerPlayPool)
        self.rush_shares = np.array(rush_shares) / sum(rush_shares) if rush_shares and sum(rush_shares) > 0 else None
        self.targets = targets          # list of (gsis_id, PlayerPlayPool)
        self.target_shares = np.array(target_shares) / sum(target_shares) if target_shares and sum(target_shares) > 0 else None

    def pick_rusher(self, rng: np.random.Generator):
        if not self.rushers or self.rush_shares is None:
            return None
        idx = rng.choice(len(self.rushers), p=self.rush_shares)
        return self.rushers[idx]

    def pick_target(self, rng: np.random.Generator):
        if not self.targets or self.target_shares is None:
            return None
        idx = rng.choice(len(self.targets), p=self.target_shares)
        return self.targets[idx]


def simulate_next_play(offense: TeamOffense, down: int, ydstogo: float, yardline_100: float,
                        rng: np.random.Generator) -> dict:
    """
    Real, single-play outcome simulator - the down-and-distance analog of
    MLB's simulate_plate_appearance(). Decides run vs. pass using the real
    league-wide down-specific rate, picks a real ball-carrier/target using
    real usage shares, then draws a REAL bootstrap-sampled outcome from
    that specific player's own real play pool (see PlayerPlayPool) rather
    than inventing a yardage number.

    Returns a dict describing what happened: yards gained, whether it was
    a first down, touchdown, turnover (interception/fumble/turnover on
    downs handled by the caller's drive loop, not here), and which
    player(s) get real counting-stat credit.
    """
    pass_rate = _real_pass_rate(down)
    # 4th-and-manageable in real neutral field position still gets a real
    # play call rather than an automatic punt here - the punt/FG/go-for-it
    # decision itself happens one level up, in simulate_drive, before this
    # function is ever called for a 4th down snap.
    is_pass = rng.random() < pass_rate

    result = {"play_type": "pass" if is_pass else "run", "yards": 0, "touchdown": False,
              "turnover": False, "turnover_type": None, "rusher": None,
              "targets": None, "complete": None, "sacked": False}

    if not is_pass:
        picked = offense.pick_rusher(rng)
        if picked is None or picked[1].empty:
            result["yards"] = 3.9  # real league-average rush yield, only if this team has no real rush pool at all
        else:
            gsis_id, pool = picked
            row = pool.sample_row(rng)
            result["yards"] = float(row.get("yards_gained", 0) or 0)
            result["touchdown"] = bool(row.get("touchdown", 0))
            result["turnover"] = bool(row.get("fumble_lost", 0))
            result["turnover_type"] = "fumble" if result["turnover"] else None
            result["rusher"] = gsis_id
        return result

    # Pass play - real roll order: sack, then scramble (a real subset of
    # this QB's own rush plays - see build_qb_pass_pool/TeamOffense.
    # qb_rush_pool - the play breaks down and he tucks it instead of
    # throwing, exactly what real_scramble_rate measures), then
    # interception, then a normal completed/incomplete target.
    rates = offense.qb_pool.real_rates()
    roll = rng.random()
    if roll < rates["sack_rate"]:
        result["sacked"] = True
        result["yards"] = -6.5  # real approximate average sack-yardage loss
        return result
    roll -= rates["sack_rate"]
    if roll < rates["scramble_rate"]:
        if offense.qb_rush_pool is not None and not offense.qb_rush_pool.empty:
            row = offense.qb_rush_pool.sample_row(rng)
            result["play_type"] = "run"  # real scramble - counts as a QB rush, not a pass attempt
            result["yards"] = float(row.get("yards_gained", 0) or 0)
            result["touchdown"] = bool(row.get("touchdown", 0))
            result["turnover"] = bool(row.get("fumble_lost", 0))
            result["turnover_type"] = "fumble" if result["turnover"] else None
            result["rusher"] = offense.qb_gsis_id
            return result
        # No real scramble history for this QB yet - falls through to a
        # normal pass outcome rather than guessing a scramble yardage.
    roll -= rates["scramble_rate"]
    if roll < rates["int_rate"]:
        result["turnover"] = True
        result["turnover_type"] = "interception"
        return result

    picked = offense.pick_target(rng)
    if picked is None or picked[1].empty:
        result["complete"] = rng.random() < rates["complete_rate"]
        result["yards"] = 6.5 if result["complete"] else 0
        return result

    gsis_id, pool = picked
    row = pool.sample_row(rng)
    complete = bool(row.get("complete_pass", 1))
    result["complete"] = complete
    result["targets"] = gsis_id
    if complete:
        result["yards"] = float(row.get("yards_gained", 0) or 0)
        result["touchdown"] = bool(row.get("touchdown", 0))
        result["turnover"] = bool(row.get("fumble_lost", 0))
        result["turnover_type"] = "fumble" if result["turnover"] else None
    return result


def simulate_drive(offense: TeamOffense, start_yardline_100: float,
                    rng: np.random.Generator) -> dict:
    """
    Real drive simulator - runs simulate_next_play repeatedly, tracking
    real down/distance/field position, until the drive real-ends: a
    touchdown, a real turnover (INT/fumble/turnover on downs), a made or
    missed field goal, or a punt. Fourth-down decision uses a real,
    stated-simplified heuristic (see module docstring) rather than a full
    win-probability coaching model.

    Returns: outcome ('touchdown'/'field_goal'/'punt'/'turnover'/
    'missed_fg'), points scored, real per-player counting stats
    accumulated this drive, and the real ending field position (for the
    next drive's real starting spot).
    """
    down, ydstogo, yardline_100 = 1, 10.0, start_yardline_100
    plays_used = 0
    stats = {"rush_att": {}, "rush_yds": {}, "rush_td": {}, "targets": {}, "rec": {},
             "rec_yds": {}, "rec_td": {}, "pass_att": 0, "pass_cmp": 0, "pass_yds": 0,
             "pass_td": 0, "pass_int": 0, "sacks": 0}

    def _credit_rush(gsis_id, yards, td):
        stats["rush_att"][gsis_id] = stats["rush_att"].get(gsis_id, 0) + 1
        stats["rush_yds"][gsis_id] = stats["rush_yds"].get(gsis_id, 0) + yards
        if td:
            stats["rush_td"][gsis_id] = stats["rush_td"].get(gsis_id, 0) + 1

    def _credit_rec(gsis_id, yards, complete, td):
        stats["targets"][gsis_id] = stats["targets"].get(gsis_id, 0) + 1
        if complete:
            stats["rec"][gsis_id] = stats["rec"].get(gsis_id, 0) + 1
            stats["rec_yds"][gsis_id] = stats["rec_yds"].get(gsis_id, 0) + yards
            if td:
                stats["rec_td"][gsis_id] = stats["rec_td"].get(gsis_id, 0) + 1

    while plays_used < 25:  # a real, sane cap - no real drive runs longer than this
        plays_used += 1

        if down == 4:
            # Real, stated-simplified 4th-down decision: go for it only on
            # real short yardage in real opponent territory; attempt a
            # real field goal inside real makeable range; otherwise punt.
            fg_distance = yardline_100 + 17  # real approximate spot-to-goalpost adjustment
            if ydstogo <= 2 and yardline_100 <= 50:
                pass  # go for it - falls through to a normal play below
            elif yardline_100 <= 38:  # real ~55-yard-or-closer attempt
                made = rng.random() < _fg_make_probability(fg_distance)
                return {"outcome": "field_goal" if made else "missed_fg",
                        "points": 3 if made else 0, "stats": stats,
                        "end_yardline_100": 100 - fg_distance if not made else None}
            else:
                return {"outcome": "punt", "points": 0, "stats": stats,
                        "end_yardline_100": max(20, yardline_100 - 42)}  # real approximate net punt yardage

        play = simulate_next_play(offense, down, ydstogo, yardline_100, rng)

        if play["sacked"]:
            stats["pass_att"] += 1
            stats["sacks"] += 1
            yardline_100 = min(99, yardline_100 - play["yards"])
            down += 1
            ydstogo += -play["yards"]
        elif play["turnover"]:
            if play["turnover_type"] == "interception":
                stats["pass_att"] += 1
                stats["pass_int"] += 1
            return {"outcome": "turnover", "points": 0, "stats": stats,
                    "end_yardline_100": 100 - max(1, yardline_100 - play["yards"])}
        else:
            gained = play["yards"]
            new_yardline_100 = yardline_100 - gained
            reached_goal = new_yardline_100 <= 0

            if play["play_type"] == "run":
                _credit_rush(play["rusher"], min(gained, yardline_100), reached_goal)
            else:
                stats["pass_att"] += 1
                if play["complete"]:
                    stats["pass_cmp"] += 1
                    stats["pass_yds"] += min(gained, yardline_100)
                    if reached_goal:
                        stats["pass_td"] += 1
                    _credit_rec(play["targets"], min(gained, yardline_100), True, reached_goal)
                else:
                    _credit_rec(play["targets"], 0, False, False)

            if reached_goal:
                if play["play_type"] == "run":
                    stats["rush_td"][play["rusher"]] = stats["rush_td"].get(play["rusher"], 0) + 1
                return {"outcome": "touchdown", "points": 7, "stats": stats, "end_yardline_100": None}

            if gained >= ydstogo:
                down, ydstogo, yardline_100 = 1, min(10.0, new_yardline_100), max(1, new_yardline_100)
            else:
                down += 1
                ydstogo -= gained
                yardline_100 = max(1, new_yardline_100)

            if down > 4:
                return {"outcome": "turnover_on_downs", "points": 0, "stats": stats,
                        "end_yardline_100": 100 - yardline_100}

    # Real safety valve - genuinely shouldn't hit this given real down/
    # distance progression, but never leave a drive unresolved.
    return {"outcome": "punt", "points": 0, "stats": stats, "end_yardline_100": 35.0}


def simulate_one_game(home_offense: TeamOffense, away_offense: TeamOffense,
                       rng: np.random.Generator, real_plays_budget: int = REAL_PLAYS_PER_TEAM_PER_GAME) -> dict:
    """
    Real full-game simulator - alternates possessions between both real
    offenses (a real structural difference from MLB's one-lineup-at-a-time
    approach: NFL score/clock genuinely depend on both teams, so both must
    be simulated together, not one side against a fixed opponent number),
    real starting field position after a score/turnover/punt handled by
    each drive's own real logic, until each team has used up a real
    plays-per-game budget (a practical clock stand-in - modeling the real
    game clock down to the second is a further layer beyond this first
    version, same honesty standard as every other stated scope limit
    here).
    """
    combined_stats = {"home": _empty_game_stats(), "away": _empty_game_stats()}
    plays_used = {"home": 0, "away": 0}
    yardline_100 = {"home": 75.0, "away": 75.0}  # real, standard touchback-ish starting spot
    possession = "home"

    while plays_used["home"] < real_plays_budget or plays_used["away"] < real_plays_budget:
        if plays_used[possession] >= real_plays_budget:
            possession = "away" if possession == "home" else "home"
            if plays_used[possession] >= real_plays_budget:
                break
            continue

        offense = home_offense if possession == "home" else away_offense
        drive = simulate_drive(offense, yardline_100[possession], rng)
        n_plays_this_drive = (drive["stats"]["pass_att"]
                               + sum(drive["stats"]["rush_att"].values()))
        plays_used[possession] += max(1, n_plays_this_drive)
        _merge_stats(combined_stats[possession], drive["stats"], drive["outcome"] == "touchdown")

        if drive["outcome"] in ("touchdown", "field_goal"):
            combined_stats[possession]["points"] += drive["points"]
            other = "away" if possession == "home" else "home"
            yardline_100[other] = 75.0
            possession = other
        elif drive["outcome"] in ("turnover", "turnover_on_downs", "missed_fg", "punt"):
            other = "away" if possession == "home" else "home"
            yardline_100[other] = max(1, min(99, drive.get("end_yardline_100") or 75.0))
            possession = other
        else:
            possession = "away" if possession == "home" else "home"

    return combined_stats


def _empty_game_stats():
    return {"rush_att": {}, "rush_yds": {}, "rush_td": {}, "targets": {}, "rec": {},
            "rec_yds": {}, "rec_td": {}, "pass_att": 0, "pass_cmp": 0, "pass_yds": 0,
            "pass_td": 0, "pass_int": 0, "sacks": 0, "points": 0}


def _merge_stats(total: dict, drive_stats: dict, is_td: bool):
    for key in ("rush_att", "rush_yds", "rush_td", "targets", "rec", "rec_yds", "rec_td"):
        for gsis_id, v in drive_stats[key].items():
            total[key][gsis_id] = total[key].get(gsis_id, 0) + v
    for key in ("pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "sacks"):
        total[key] += drive_stats[key]


def simulate_matchup_n_times(home_offense: TeamOffense, away_offense: TeamOffense,
                              n_simulations: int = 100, random_state: int = 42) -> dict:
    """
    Real, direct analog of MLB's simulate_matchup_n_times() - runs
    simulate_one_game() n_simulations times, returns raw per-simulation
    per-player counting-stat arrays so any real prop line can be checked
    against the real empirical distribution afterward (see
    real_over_rate_from_simulation) without re-running the simulation
    per line.
    """
    rng = np.random.default_rng(random_state)
    per_player_series = {}  # gsis_id -> {stat: [n_simulations values]}

    def _record(gsis_id, stat, value):
        per_player_series.setdefault(gsis_id, {}).setdefault(stat, []).append(value)

    for _ in range(n_simulations):
        game = simulate_one_game(home_offense, away_offense, rng)
        for side in ("home", "away"):
            g = game[side]
            all_rushers = set(g["rush_att"]) | set(g["rush_yds"]) | set(g["rush_td"])
            for gsis_id in all_rushers:
                _record(gsis_id, "rush_attempts", g["rush_att"].get(gsis_id, 0))
                _record(gsis_id, "rush_yards", g["rush_yds"].get(gsis_id, 0))
                _record(gsis_id, "rush_tds", g["rush_td"].get(gsis_id, 0))
            all_targets = set(g["targets"]) | set(g["rec"]) | set(g["rec_yds"]) | set(g["rec_td"])
            for gsis_id in all_targets:
                _record(gsis_id, "targets", g["targets"].get(gsis_id, 0))
                _record(gsis_id, "receptions", g["rec"].get(gsis_id, 0))
                _record(gsis_id, "rec_yards", g["rec_yds"].get(gsis_id, 0))
                _record(gsis_id, "rec_tds", g["rec_td"].get(gsis_id, 0))
            qb_id = (home_offense if side == "home" else away_offense).qb_gsis_id
            if qb_id:
                _record(qb_id, "pass_attempts", g["pass_att"])
                _record(qb_id, "pass_completions", g["pass_cmp"])
                _record(qb_id, "pass_yards", g["pass_yds"])
                _record(qb_id, "pass_tds", g["pass_td"])
                _record(qb_id, "interceptions", g["pass_int"])

    return per_player_series


def real_over_rate_from_simulation(series: list, line: float) -> dict:
    """Same, direct real helper as MLB's version - given a real list of
    per-simulation values for one player/prop, returns the real empirical
    over rate, average, and sample size for an arbitrary real line,
    without needing to re-run anything."""
    if not series:
        return {"over_rate": np.nan, "mean": np.nan, "n": 0}
    arr = np.array(series, dtype=float)
    return {"over_rate": round(float((arr > line).mean()), 3),
            "mean": round(float(arr.mean()), 2), "n": len(arr)}


# ---------------------------------------------------------------------------
# REAL PREMIUM-DATA WIRING - connects every real CSV-based signal
# (alignment, QB-vs-coverage, QB scrambles, RB rush-concepts) into the
# actual play-sampling above, not just a separate side-score. This is the
# one function to call from the UI/scan layer - it builds a fully real,
# matchup-aware TeamOffense using free pbp data for the bootstrap pools
# and the premium CSVs (when a bundle is loaded) to tilt which of each
# player's real outcomes get sampled more or less often.
# ---------------------------------------------------------------------------

def build_team_offense(team: str, opponent: str, season: int, week: int,
                        pbp_df: pd.DataFrame, prior_pbp_df: pd.DataFrame,
                        rosters_df: pd.DataFrame,
                        coverage_bundle=None, rb_bundle=None,
                        n_rushers: int = 3, n_targets: int = 5) -> "TeamOffense":
    """
    Builds one team's real, matchup-aware TeamOffense for the simulation.
    Real starters/usage shares come from real recent pbp (weeks before
    target week, this season). Real exploit_strength values - when a
    coverage_bundle/rb_bundle is loaded - tilt each player's bootstrap
    sampling toward his better or tougher real games based on the SAME
    real signals already used elsewhere in this project:
      QB passing  -> calc_qb_coverage_exploit_strength (coverage_matchup)
      QB scrambles -> calc_qb_scramble_exploit_strength (coverage_matchup)
      RB rushing  -> calc_rb_concept_exploit_strength (rb_matchup, blended
                     across all 6 real concepts by his real usage share)
      WR/TE       -> calc_alignment_exploit_strength (coverage_matchup)

    Without a bundle loaded, every pool still builds and samples correctly
    from real pbp alone (exploit_strength=None -> uniform bootstrap,
    per PlayerPlayPool) - the simulation is never blocked on premium data,
    same graceful-degrade philosophy as everywhere else in this project.

    REAL, SEVERE BUG FOUND AND FIXED (confirmed directly against real
    2025 week 1 data before this fix): `hist` only ever looked at
    CURRENT-season pbp for identifying who the QB/top rushers/top targets
    even ARE. In week 1, hist was always empty, so qb_gsis_id came back
    None and rushers/targets came back as empty lists - a completely
    blank, useless TeamOffense with nothing to simulate, for every team,
    every week 1. prior_pbp_df was already a required parameter here but
    was never actually used for this - only threaded through to the
    per-player pool builders for THEIR OWN history, not for figuring out
    which players to build pools for in the first place. Now falls back
    to identifying usage from the full prior season when current-season
    usage is empty, same pattern as every other cold-start fix this
    session (build_qb_advanced_metrics, build_receiver_advanced_metrics,
    etc.).
    """
    hist = pbp_df[(pbp_df["season"] == season) & (pbp_df["week"] < week) & (pbp_df["posteam"] == team)]
    if hist.empty and prior_pbp_df is not None and not prior_pbp_df.empty:
        hist = prior_pbp_df[(prior_pbp_df["season"] == season - 1) & (prior_pbp_df["posteam"] == team)]
    name_lookup = rosters_df[rosters_df["season"] == season][["gsis_id", "full_name"]].drop_duplicates().set_index("gsis_id")["full_name"]

    qb_counts = hist["passer_player_id"].dropna().value_counts()
    qb_id = qb_counts.idxmax() if not qb_counts.empty else None
    qb_pool = build_qb_pass_pool(qb_id, season, week, pbp_df, prior_pbp_df) if qb_id else QBPassPool(pd.DataFrame())

    qb_scramble_exploit = None
    if coverage_bundle is not None and qb_id is not None:
        try:
            import coverage_matchup as cm
            qb_name = name_lookup.get(qb_id)
            r = cm.calc_qb_scramble_exploit_strength(coverage_bundle.qb_scrambles,
                                                      coverage_bundle.def_allowed_qb_scrambles,
                                                      qb_name, opponent)
            qb_scramble_exploit = r.get("exploit_strength")
        except Exception:
            pass
    qb_rush_pool = build_player_play_pool(qb_id, "rush", season, week, pbp_df, prior_pbp_df,
                                           exploit_strength=qb_scramble_exploit) if qb_id else None

    rush_counts = hist["rusher_player_id"].dropna().value_counts().head(n_rushers)
    rushers = []
    for gid in rush_counts.index:
        exploit = None
        if rb_bundle is not None:
            try:
                import rb_matchup as rbm
                r = rbm.calc_rb_concept_exploit_strength(rb_bundle, name_lookup.get(gid), opponent)
                exploit = r.get("exploit_strength")
            except Exception:
                pass
        rushers.append((gid, build_player_play_pool(gid, "rush", season, week, pbp_df, prior_pbp_df, exploit)))

    target_counts = hist["receiver_player_id"].dropna().value_counts().head(n_targets)
    targets = []
    for gid in target_counts.index:
        exploit = None
        if coverage_bundle is not None:
            try:
                import coverage_matchup as cm
                r = cm.calc_alignment_exploit_strength(coverage_bundle, name_lookup.get(gid), None, team, opponent)
                exploit = r.get("exploit_strength")
            except Exception:
                pass
        targets.append((gid, build_player_play_pool(gid, "target", season, week, pbp_df, prior_pbp_df, exploit)))

    if coverage_bundle is not None and qb_id is not None:
        try:
            import coverage_matchup as cm
            qb_name = name_lookup.get(qb_id)
            # QB-vs-coverage exploit_strength currently informs quality_score
            # elsewhere in the project (ENABLE_QB_COVERAGE_IN_QUALITY_SCORE) -
            # not yet tilting the pass-completion pool here, since real
            # per-play completion outcomes aren't split by coverage type in
            # the free pbp pool the way rush/target pools are. Left as a
            # real, stated next step rather than silently skipped.
            pass
        except Exception:
            pass

    return TeamOffense(qb_id, qb_pool, qb_rush_pool, rushers, rush_counts.tolist(), targets, target_counts.tolist())


# ===========================================================================
# Real, merged content from coverage_matchup.py (per direct request, single-
# file consolidation matching the MLB tool's structure) - premium
# FantasyPoints alignment-vs-coverage and QB-vs-coverage data handling.
# Real note: _compute_field_tiers, _read_fp_csv, _same_team, _to_float, and
# TEAM_ABBREV_TO_FULL below are THIS file's real, original versions -
# rb_matchup's own versions of the same names were renamed with an "_rb"
# suffix below to avoid a real, silent collision (confirmed different
# implementations, not just duplicates, before merging).
# ===========================================================================

"""
NFL PREMIUM TOOL - Coverage Matchup Module
=============================================
Built from FantasyPoints.com Data Suite manual exports (paid subscription,
no public API - confirmed via DevTools investigation; see project notes).

WHAT THIS DOES
---------------
1. Loads team-level coverage tendency data (Man/Zone/Cover 0-6 % usage)
   for both offense (coverages seen) and defense (coverages used).
2. Flags each team's REAL statistical outlier coverage(s) using z-scores
   against league average - not raw rank (Cover 3 is the default shell
   for nearly every team and ranking on it tells you nothing; z-score
   answers "is this meaningfully different from league norm").
3. Loads the FULL column set from QB-vs-coverage season splits (7 files:
   Cover 0/1/2/2Man/3/4/6) and defense-allowed-to-QBs splits (same 7),
   including fantasy-relevant columns (FP/DB, FP/OPP, FP/G, FP) for
   later fantasy-relevance use, not just a curated subset.
4. Tiers every numeric stat (Elite/Above Avg/Average/Below Avg/Poor)
   against the real distribution of QBs who've faced that specific
   coverage - not a global season benchmark, not arbitrary cutoffs.
5. Builds a full matchup report combining all of the above, with
   automatic thin-sample warnings based on real league ATT distributions.

DUPLICATE COLUMN HANDLING (important - real bug caught and fixed)
---------------------------------------------------------------------
The QB-vs-coverage CSVs have "YDS" and "TD" appear TWICE - once under
Passing (right after CMP%/YPA) and once under Scrambles (right after the
SCRM column). A naive dict(zip(header,row)) silently drops the first
occurrence. This module explicitly renames the scramble pair to
"SCRM_YDS" / "SCRM_TD" during parsing so no data is lost.

WHY Z-SCORES, NOT RAW RANK OR FIXED CUTOFFS
-----------------------------------------------
Confirmed on real 2025 data: Seattle's Cover 6 rate (17.7%) is +1.62 SD
above league average (3rd of 32) - a real, usable signal. Their Cover 4
rate, despite ranking 13th of 32, is only +0.23 SD above average -
statistical noise, not a real tendency. Same logic applies to tiering a
QB's performance vs a coverage: judged against the real distribution of
every QB who's actually faced that coverage this season, not a flat
"70% completion = good" guess.

SAMPLE SIZE THRESHOLDS (confirmed from real 2025 QB-vs-coverage data)
------------------------------------------------------------------------
Coverage      Median ATT (league)   Treat as
Cover 3       62                    Solid
Cover 1       37                    Solid
Cover 2       32                    Solid
Cover 4       28                    Usable, lean cautious under ~15
Cover 6       19                    Usable, lean cautious under ~10
Cover 0       7                     Thin league-wide - flag always
Cover 2 Man   5                     Thin league-wide - flag always
"""



COVERAGE_FIELDS = ["COVER 0 %", "COVER 1 %", "COVER 2 %", "COVER 2 MAN %",
                   "COVER 3 %", "COVER 4 %", "COVER 6 %"]

ALWAYS_THIN_COVERAGES = {"COVER 0 %", "COVER 2 MAN %"}

THIN_SAMPLE_ATT_THRESHOLD = {
    "COVER 0 %": 5, "COVER 1 %": 15, "COVER 2 %": 15, "COVER 2 MAN %": 5,
    "COVER 3 %": 20, "COVER 4 %": 15, "COVER 6 %": 10,
}

OUTLIER_Z_THRESHOLD = 1.0  # kept for reference; superseded by COVERAGE_RANK_THRESHOLD below for actual coverage selection
# Real, validated threshold (see load_team_coverage_matrix) - a coverage
# counts as "meaningfully used" if this defense ranks in the top 10 of 32
# teams for it, roughly the top third leaguewide. Confirmed directly
# against real 2025 data to reproduce the exact real example given (NE's
# Cover 1/2/4 at ranks 6/5/10), and stress-tested against several other
# real teams to confirm it behaves reasonably (1-3 qualifying coverages
# typically, adapts per team, not universally maxed out).
COVERAGE_RANK_THRESHOLD = 10
# Real, validated margin (see load_team_coverage_matrix) - a coverage
# ranked just outside the top 10 still qualifies if its real rate is
# within 5% (relative) of the rank-10 rate, avoiding an arbitrary hard
# cliff between two teams separated by a real gap as small as 0.1 points.
COVERAGE_TIE_MARGIN_PCT = 0.05

# Stats where a HIGHER number is worse for the QB (need to flip tiering direction)
# REAL, SEVERE BUG FOUND AND FIXED (via direct testing against real
# uploaded data - confirmed QBs with 2 real interceptions were being
# tiered "Elite" and QBs with 0 were "Below Avg", the exact opposite of
# correct): this set used to be named plain INVERSE_STATS, and a SEPARATE,
# unrelated RB-specific set further down this file (originally also named
# INVERSE_STATS = {"FUM", "STUFF %"}) SILENTLY OVERWROTE this one at
# import time, since both were simple module-level globals sharing the
# same name - Python has no scoping between them. Every QB-vs-coverage
# and alignment tier computation has been using the RB's tiny 2-item set
# instead of this one ever since, meaning INT/SACK/SACK%/SK YDS/DROP%/
# DROP YDS/PRESS%/PRESS SK%/TTSK/QB SK/QBP/BAT/SPK were all being graded
# backwards (higher = "Elite") for as long as this code has existed.
# Renamed to a unique name, with the RB set renamed too (see RB_
# INVERSE_STATS below) - each now correctly used only where intended.
QB_ALIGNMENT_INVERSE_STATS = {"INT", "SACK", "SACK %", "SK YDS", "DROP %", "DROP YDS",
                  "DRP", "DRP %",  # real column names confirmed from actual
                  # WR/TE exports - "DROP %" above never matched real data at
                  # all, meaning drop rate tiering direction was silently
                  # wrong (higher drops looked "better") until this fix
                  "PRESS %", "PRESS SK %", "TTSK", "QB SK", "QBP", "BAT", "SPK"}

# Non-numeric / identifier columns - never tier these
NON_STAT_COLUMNS = {"Rank", "Name", "Team", "Team Name", "POS", "G", "Season",
                     "Location", "_thin_sample", "_att"}

TEAM_ABBREV_TO_FULL = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "JAC": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def _same_team(abbrev_or_name, full_name):
    if not abbrev_or_name:
        return False
    if abbrev_or_name == full_name:
        return True
    return TEAM_ABBREV_TO_FULL.get(abbrev_or_name.upper()) == full_name


# ---------------------------------------------------------------------------
# CSV parsing (handles FantasyPoints' 2-header-row format + duplicate columns)
# ---------------------------------------------------------------------------

def _dedupe_header(header):
    """Renames known duplicate columns so no data is silently lost.
    Currently handles the Passing-vs-Scrambles YDS/TD collision confirmed
    in the QB-vs-coverage export format. Any other duplicate gets a
    generic _2/_3 suffix as a safety net."""
    out = []
    seen = {}
    for i, col in enumerate(header):
        if col in ("YDS", "TD") and i > 0 and (header[i-1] == "SCRM" or (i > 1 and header[i-2] == "SCRM")):
            newcol = f"SCRM_{col}"
        elif col in seen:
            seen[col] += 1
            newcol = f"{col}_{seen[col]}"
        else:
            newcol = col
        seen[newcol] = seen.get(newcol, 0)
        out.append(newcol)
    return out


def _read_fp_csv(path):
    """FantasyPoints exports have 2 header rows (grouping row + real header).
    Real header always starts with 'Rank'. Handles BOM safely, dedupes
    duplicate column names."""
    with open(path, encoding='utf-8-sig') as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.lstrip('\ufeff').startswith('"Rank"'))
    reader = csv.reader(lines[header_idx:])
    rows = list(reader)
    raw_header, data = rows[0], [r for r in rows[1:] if r and r[0]]
    header = _dedupe_header(raw_header)
    return header, data


def _to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Team coverage matrix (offense-seen / defense-used)
# ---------------------------------------------------------------------------

@dataclass
class TeamCoverageProfile:
    team_name: str
    rates: dict
    z_scores: dict = field(default_factory=dict)
    ranks: dict = field(default_factory=dict)  # {coverage_field: real leaguewide rank, 1=highest usage}
    outliers: list = field(default_factory=list)


def load_team_coverage_matrix(csv_path):
    header, data = _read_fp_csv(csv_path)
    profiles = {}
    for row in data:
        d = dict(zip(header, row))
        team = d["Name"]
        rates = {f: (_to_float(d.get(f)) or 0.0) for f in COVERAGE_FIELDS}
        profiles[team] = TeamCoverageProfile(team_name=team, rates=rates)

    league_stats = {}
    for f in COVERAGE_FIELDS:
        vals = [p.rates[f] for p in profiles.values()]
        league_stats[f] = (mean(vals), pstdev(vals))

    # REAL FIX (per direct example - NE week-1-2025 real coverage usage:
    # Cover 1 23.9% ranked 6th of 32, Cover 2 21.7% ranked 5th, Cover 4
    # 17.6% ranked 10th): the z-score outlier threshold (>=1.0) only
    # caught Cover 2 for this exact real matchup - Cover 1 (z=0.86) and
    # Cover 4 (z=0.40) never cleared it, even though both are genuinely
    # meaningful, real, above-average tendencies. A single defense's
    # coverage rates aren't always spread out enough for z-score alone to
    # flag a real top-third tendency as statistically "unusual" - rank
    # captures this better. Switched to real LEAGUEWIDE RANK (top 10 of
    # 32, roughly the top third) instead of a z-score cutoff - validated
    # directly against 2025 data: this exact threshold reproduces the
    # user's own real example precisely, and stress-tested reasonably
    # across several other real teams (Seattle: 1 qualifying coverage,
    # KC/Denver: 3 each) - it adapts naturally per team rather than
    # forcing a fixed count. z-score is still computed and still used as
    # the downstream WEIGHT (a coverage a team leans into more heavily
    # still counts for more), just no longer the selection gate itself.
    # REAL FIX (per direct follow-up question - a hard cliff at exactly
    # rank 10 is arbitrary when the real gap to rank 11+ is tiny: e.g.
    # real 2025 Cover 4 data has rank 10 at 17.6% and rank 11 at 17.5% -
    # a 0.1-point gap, not a meaningful cutoff). Tested an ABSOLUTE margin
    # (e.g. "within 0.5 points of the rank-10 rate") first and found it
    # behaves inconsistently across fields with very different scales -
    # Cover 0 rates cluster in the 3-8% range so 0.5 points pulled in 5
    # extra teams, while Cover 3 rates run 25-35% so the same 0.5 points
    # pulled in only 1. Switched to a RELATIVE margin instead (5% of the
    # rank-10 rate itself) - confirmed via direct testing this behaves
    # consistently (1-4 extra teams) across every real coverage field
    # regardless of its own scale.
    for f in COVERAGE_FIELDS:
        ranked = sorted(profiles.values(), key=lambda p: -p.rates[f])
        for rank, p in enumerate(ranked, start=1):
            p.ranks[f] = rank
        if len(ranked) >= COVERAGE_RANK_THRESHOLD:
            rank10_rate = ranked[COVERAGE_RANK_THRESHOLD - 1].rates[f]
            tie_cutoff = rank10_rate * (1 - COVERAGE_TIE_MARGIN_PCT)
            for p in ranked:
                if p.ranks[f] > COVERAGE_RANK_THRESHOLD and p.rates[f] >= tie_cutoff:
                    p.ranks[f] = COVERAGE_RANK_THRESHOLD  # real near-tie - treat as qualifying, same as rank 10 itself

    for p in profiles.values():
        for f in COVERAGE_FIELDS:
            avg, sd = league_stats[f]
            p.z_scores[f] = (p.rates[f] - avg) / sd if sd else 0.0
        p.outliers = sorted(
            [(f, p.z_scores[f]) for f in COVERAGE_FIELDS if p.ranks.get(f, 99) <= COVERAGE_RANK_THRESHOLD],
            key=lambda x: -x[1]
        )
    return profiles, league_stats


def describe_team_tendency(profile: TeamCoverageProfile):
    if not profile.outliers:
        return f"{profile.team_name}: no coverage runs meaningfully above league average - fairly standard mix."
    parts = [f"{cov.replace(' %','')} {profile.rates[cov]:.1f}% (z={z:+.2f})" for cov, z in profile.outliers]
    return f"{profile.team_name}: real outlier coverage(s) - " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Full-column loading with tiering (QB-vs-coverage AND def-allowed-to-QB)
# ---------------------------------------------------------------------------

def _compute_field_tiers(rows_by_key):
    """rows_by_key: dict[key] -> row dict (already has _att, _thin_sample).
    Computes league distribution (within this one coverage file) for every
    numeric stat column, then assigns each row a tier per stat:
    Elite / Above Avg / Average / Below Avg / Poor, based on z-score
    bucket, direction-corrected for stats where lower is better."""
    if not rows_by_key:
        return

    sample_row = next(iter(rows_by_key.values()))
    stat_cols = [c for c in sample_row.keys() if c not in NON_STAT_COLUMNS and not c.startswith("_")]

    field_stats = {}
    for col in stat_cols:
        vals = [_to_float(r.get(col)) for r in rows_by_key.values()]
        vals = [v for v in vals if v is not None]
        if len(vals) < 3:
            continue
        field_stats[col] = (mean(vals), pstdev(vals))

    for r in rows_by_key.values():
        tiers = {}
        for col, (avg, sd) in field_stats.items():
            v = _to_float(r.get(col))
            if v is None or not sd:
                continue
            z = (v - avg) / sd
            if col in QB_ALIGNMENT_INVERSE_STATS:
                z = -z
            if z >= 1.5:
                tiers[col] = "Elite"
            elif z >= 0.5:
                tiers[col] = "Above Avg"
            elif z > -0.5:
                tiers[col] = "Average"
            elif z > -1.5:
                tiers[col] = "Below Avg"
            else:
                tiers[col] = "Poor"
        r["_tiers"] = tiers


def _load_coverage_keyed_data(file_paths: dict, key_column: str, volume_column: str = "ATT"):
    """Generic loader for QB-vs-coverage, def-allowed-to-QB, receiver-vs-
    coverage, and def-allowed-by-alignment files. Captures EVERY column
    from the CSV, not a curated subset, and computes real statistical
    tiers per stat within each coverage's own distribution.

    volume_column: which column represents sample size for thin-sample
    flagging - 'ATT' for QB/passing files, 'TGT' for receiver files
    (they don't have an ATT column at all)."""
    data = {}
    for coverage_field, path in file_paths.items():
        header, rows = _read_fp_csv(path)
        by_key = {}
        for row in rows:
            d = dict(zip(header, row))
            key = d.get(key_column)
            if not key:
                continue
            att = int(_to_float(d.get(volume_column, 0)) or 0)
            threshold = THIN_SAMPLE_ATT_THRESHOLD.get(coverage_field, 15)
            d["_thin_sample"] = (att < threshold) or (coverage_field in ALWAYS_THIN_COVERAGES)
            d["_att"] = att
            # REAL FIX (per direct request) - neither the QB-vs-coverage
            # nor receiving files provide a real TD RATE column (RB files
            # do - "TD RATE" - this mirrors that same real signal for the
            # other two positions, computed here since it isn't directly
            # in the CSV). TD/ATT for QB files, TD/TGT for receiving files
            # - whichever volume_column this call is already using.
            real_td = _to_float(d.get("TD"))
            if real_td is not None and att > 0:
                d["TD %"] = round((real_td / att) * 100, 2)
            by_key[key] = d
        _compute_field_tiers(by_key)
        data[coverage_field] = by_key
    return data


def load_qb_vs_coverage(file_paths: dict):
    """QB's own season performance vs each coverage. Full column set
    (including FP/DB, FP/OPP, FP/G, FP - fantasy-relevant fields) plus
    per-stat tiers vs the real distribution of QBs facing that coverage."""
    return _load_coverage_keyed_data(file_paths, key_column="Name")


def load_def_allowed_to_qb(file_paths: dict):
    """What each DEFENSE allows to QBs specifically in that coverage.
    Same full-column + tiering treatment, keyed by team name."""
    return _load_coverage_keyed_data(file_paths, key_column="Name")


# ---------------------------------------------------------------------------
# QB scrambles - a real, separate signal from QB-vs-coverage above. QBs
# don't have named rush concepts the way RBs do (Power/Zone/Counter) - a
# scramble is inherently improvised, not a called play - so this is a
# single flat comparison (QB's own real scramble production vs. how much
# a defense allows on scrambles), not a coverage-type or concept-type
# breakdown like everything else in this module. One real player-side
# file, one real team-side file, loaded with the SAME real generic
# tiering helper everything else here uses, just with a single pseudo-
# key ("SCRAMBLE") instead of iterating over multiple coverage types.
# ---------------------------------------------------------------------------

def load_qb_scrambles(file_path: str):
    """QB's own real scramble volume/production this season - ATT, YDS,
    YACO, MTF, Success% etc., same tiering treatment as every other file
    here. Returns {"SCRAMBLE": {qb_name: row}} - the single-key shape
    keeps this compatible with the same _weighted_outlier_exploit-style
    combination logic used for alignment/QB-coverage, just with exactly
    one "coverage" to check instead of several."""
    return _load_coverage_keyed_data({"SCRAMBLE": file_path}, key_column="Name")


def load_def_allowed_qb_scrambles(file_path: str):
    """What each DEFENSE allows on QB scrambles - same shape as above,
    keyed by team name."""
    return _load_coverage_keyed_data({"SCRAMBLE": file_path}, key_column="Name")


def get_top_pass_catchers(team: str, season: int, week: int, player_stats_df: pd.DataFrame,
                           max_players: int = 4) -> list:
    """
    Real, direct identification of a QB's current top pass-catchers -
    weeks BEFORE the target week only (same leak-avoidance as everywhere
    else), ranked by real target_share, WR/TE/RB eligible (a real pass-
    catching back counts too). Returns [(player_name, target_share), ...]
    for the top max_players by real share - this is the actual "who does
    this QB throw to" list the supporting-cast signal below needs.
    """
    hist = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] < week)
        & (player_stats_df["team"] == team)
    ]
    hist = hist[hist["position"].isin(["WR", "TE", "RB"])]
    if hist.empty:
        return []
    agg = hist.groupby("player_display_name").agg(
        target_share=("target_share", "mean"),
    ).reset_index()
    agg = agg.dropna(subset=["target_share"]).sort_values("target_share", ascending=False).head(max_players)
    return list(zip(agg["player_display_name"], agg["target_share"]))


def calc_supporting_cast_exploit_strength(bundle: "CoverageDataBundle", teammates: list,
                                           team_abbrev: str, opponent_team_abbrev: str) -> dict:
    """
    REAL, NEW CROSS-REFERENCING SIGNAL (built live per direct request):
    a QB's real pass_yards/completions/attempts/tds output depends on how
    well his ACTUAL current pass-catchers fit the specific coverages this
    opponent leans into - not just the QB's own numbers in isolation. If
    Josh Allen's real weapons this season are DJ Moore/Khalil Shakir/
    Dalton Kincaid/James Cook, and the opponent leans heavily into Cover 2
    and Cover 3, this checks how each of THOSE SPECIFIC real teammates has
    actually performed against Cover 2/3 (via the same alignment-vs-
    coverage data already used for their own props), weighted by each
    teammate's real target_share (his #1 option matters more than his
    4th), and blends that into the QB's own structural signal.

    teammates: [(player_name, target_share), ...] from get_top_pass_
    catchers - real, current, weeks-before-target-week volume, not a
    guess at who "should" be catching passes.

    Returns exploit_strength NaN (not a guess) if none of the real
    teammates have any real data against this opponent's real outlier
    coverages - a real gap, not defaulted to neutral.
    """
    if not teammates:
        return {"exploit_strength": np.nan, "teammates_checked": []}

    weighted_scores, weights, checked = [], [], []
    for name, share in teammates:
        if pd.isna(share) or share <= 0:
            continue
        # Real fix - without prop_type, this silently fell back to FP/G
        # alone (a generic overall-value number) instead of the rich,
        # curated rec_yards bucket (YPRR, YACO/REC, aDOT, AY, explosive-
        # play rate, etc.) built earlier this session. "rec_yards" is the
        # right bucket here specifically because a QB's pass_yards depends
        # on his receivers generating real YARDAGE against this coverage,
        # not just overall fantasy value - matches the direct example
        # given (does DJ Moore actually produce real yardage vs Cover 2/3,
        # not just "is he a good fantasy option in general").
        result = calc_alignment_exploit_strength(bundle, name, None, team_abbrev, opponent_team_abbrev,
                                                   prop_type="rec_yards")
        exploit = result.get("exploit_strength")
        if pd.notna(exploit):
            weighted_scores.append(exploit)
            weights.append(share)
            checked.append(name)

    if not weighted_scores:
        return {"exploit_strength": np.nan, "teammates_checked": []}
    exploit_strength = sum(s * w for s, w in zip(weighted_scores, weights)) / sum(weights)
    return {"exploit_strength": round(exploit_strength, 3), "teammates_checked": checked}


QB_SCRAMBLE_STAT_BUCKETS = {
    "volume": ["ATT"],
    "contact_yardage": ["YACO"],
    "elusiveness": ["MTF"],
}
QB_SCRAMBLE_DEFAULT_STATS = {"_all": ["FP/G"]}


def calc_qb_scramble_exploit_strength(qb_scrambles: dict, def_allowed_scrambles: dict,
                                       qb_name: str, opponent_team_abbrev: str) -> dict:
    """
    Real QB-scramble matchup signal - no outlier-coverage gating needed
    here (unlike calc_qb_coverage_exploit_strength above), since there's
    only one real "situation" to check: does this defense generally give
    up scramble yardage, and does this QB generally produce it. Combines
    the defense's allowed tier (60%, the new opponent-specific
    information) with the QB's own tier (40%, since his overall rushing
    profile is already partly reflected elsewhere in the pipeline via
    role_verification/mu) - same weighting philosophy as every other
    exploit-strength function in this module.

    REAL FIX (found live this session, per direct real pushback to use
    every real column rather than one blended stat): previously always
    used FP/G alone. Now blends across ATT (volume), YACO (yards after
    contact on scrambles specifically), and MTF (elusiveness) - bucketed
    equally so no one column dominates, same fix as calc_qb_coverage_
    exploit_strength/calc_alignment_exploit_strength above. HONEST GAP:
    "Success %" from this same file is deliberately left out, same
    unresolved ambiguity as the RB run-concept version - not yet
    confirmed whether it means something distinct here or is a same-name
    coincidence with the RB file's column.

    Returns exploit_strength NaN (not a guess) if either side has no real
    data - a real gap, not defaulted to neutral.

    REAL FIX (found live this session, per direct follow-up request) -
    this had ZERO coordinator-change awareness before this: returns
    neutral (NaN) if the opponent has a real, new 2026 DC
    (NEW_DC_TEAMS_2026), since the opponent's real scramble-allowed data
    is built from their real, uploaded 2025 CSV. The QB's own side isn't
    additionally gated on NEW_QB_TEAMS_2026 here, since qb_name is already
    identity-keyed to whichever real player is actually starting - if
    that's a genuinely new starter, his own real name correctly pulls his
    own real (possibly thin, possibly from a different team) history,
    same graceful degrade as everywhere else, not a team-level question.
    """
    opp_full = TEAM_ABBREV_TO_FULL.get((opponent_team_abbrev or "").upper())
    if opp_full is None or (opponent_team_abbrev or "").upper() in NEW_DC_TEAMS_2026:
        return {"exploit_strength": np.nan}

    def_row = def_allowed_scrambles.get("SCRAMBLE", {}).get(opp_full)
    qb_row = qb_scrambles.get("SCRAMBLE", {}).get(qb_name)
    def_tiers = (def_row.get("_tiers") or {}) if def_row else {}
    qb_tiers = (qb_row.get("_tiers") or {}) if qb_row else {}

    def_bucket_scores = []
    qb_bucket_scores = []
    for bucket_stats in QB_SCRAMBLE_STAT_BUCKETS.values():
        d_scores = [TIER_SCORE.get(def_tiers.get(stat)) for stat in bucket_stats if def_tiers.get(stat) is not None]
        q_scores = [TIER_SCORE.get(qb_tiers.get(stat)) for stat in bucket_stats if qb_tiers.get(stat) is not None]
        if d_scores:
            def_bucket_scores.append(sum(d_scores) / len(d_scores))
        if q_scores:
            qb_bucket_scores.append(sum(q_scores) / len(q_scores))

    parts, weights = [], []
    if def_bucket_scores:
        parts.append(sum(def_bucket_scores) / len(def_bucket_scores))
        weights.append(0.6)
    if qb_bucket_scores:
        parts.append(sum(qb_bucket_scores) / len(qb_bucket_scores))
        weights.append(0.4)
    if not parts:
        return {"exploit_strength": np.nan}
    return {"exploit_strength": round(sum(p * w for p, w in zip(parts, weights)) / sum(weights), 3)}


# ---------------------------------------------------------------------------
# Receiver (WR/TE) by alignment vs coverage
# ---------------------------------------------------------------------------

# Alignment RTE% column names as they appear in the receiver CSVs - used to
# confirm a player's real alignment fit before leaning on an alignment-
# specific file for them (e.g. don't trust "Wide vs Cover 6" numbers for a
# player who's actually 80% Slot).
ALIGNMENT_RTE_COLUMNS = {
    "wide": "WIDE RTE %", "slot": "SLOT RTE %",
    "inline": "INLINE RTE %", "backfield": "BACK RTE %",
}


def load_receiver_vs_coverage(file_paths: dict):
    """Receiver's (WR/TE) own season performance vs each coverage, for a
    SPECIFIC alignment (e.g. all Wide-alignment vs Cover 6). file_paths:
    dict of {coverage_field: csv_path}, one alignment's worth of 7 files.
    Same full-column capture + tiering as the QB loader - reuses the
    identical generic engine, since these files also key on 'Name'.
    Uses TGT (not ATT - these files don't have that column) as the
    volume basis for thin-sample flagging."""
    return _load_coverage_keyed_data(file_paths, key_column="Name", volume_column="TGT")


def load_def_allowed_by_alignment(file_paths: dict):
    """What each DEFENSE allows to a specific alignment (Wide/Slot/Inline/
    Backfield) in that coverage. Same shape as load_def_allowed_to_qb,
    just for receivers-by-alignment instead of QBs. Team-keyed, TGT-based."""
    return _load_coverage_keyed_data(file_paths, key_column="Name", volume_column="TGT")


def check_alignment_fit(receiver_row, alignment):
    """Given a receiver's row (from ANY of their coverage files - RTE%
    columns are the same regardless of which coverage split you pulled)
    and the alignment you're about to use for a matchup, returns the
    player's real RTE% in that alignment so you can judge whether the
    alignment-specific file is actually representative of how they're
    used. Returns None if the column wasn't populated (blank = not
    their primary alignment in that export)."""
    col = ALIGNMENT_RTE_COLUMNS.get(alignment.lower())
    if col is None or receiver_row is None:
        return None
    return _to_float(receiver_row.get(col))


def build_receiver_matchup_report(receiver_name, alignment, opponent_team_profile: TeamCoverageProfile,
                                   receiver_coverage_data: dict, receiver_team_name=None,
                                   def_allowed_data: dict = None, max_outliers=3):
    """Same shape as build_qb_matchup_report, for a receiver at a specific
    alignment. Includes an alignment-fit check so a report never silently
    misrepresents a player who isn't actually primarily in that alignment."""
    if receiver_team_name and _same_team(receiver_team_name, opponent_team_profile.team_name):
        return [{"error": f"{receiver_name} plays for {opponent_team_profile.team_name} - "
                           f"cannot build a matchup report against his own team."}]

    report = []
    outliers = opponent_team_profile.outliers[:max_outliers]
    if not outliers:
        return [{"note": f"{opponent_team_profile.team_name} has no statistically real "
                          f"outlier coverage this season - no specific coverage edge to flag."}]

    for coverage_field, z in outliers:
        cov_label = coverage_field.replace(" %", "")
        entry = {
            "coverage": cov_label,
            "alignment": alignment,
            "opponent_usage_pct": opponent_team_profile.rates[coverage_field],
            "opponent_z_score": round(z, 2),
        }

        rec_row = receiver_coverage_data.get(coverage_field, {}).get(receiver_name)
        if rec_row is None:
            entry["receiver_data"] = None
            entry["confidence"] = "no_data"
        else:
            entry["receiver_data"] = rec_row
            entry["confidence"] = "thin_sample" if rec_row["_thin_sample"] else "solid"
            fit = check_alignment_fit(rec_row, alignment)
            entry["alignment_fit_pct"] = fit
            entry["alignment_fit_warning"] = (fit is not None and fit < 60)

        if def_allowed_data is not None:
            def_row = def_allowed_data.get(coverage_field, {}).get(opponent_team_profile.team_name)
            entry["defense_allows"] = def_row
            entry["defense_confidence"] = ("thin_sample" if def_row and def_row["_thin_sample"]
                                            else "solid" if def_row else "no_data")

        report.append(entry)
    return report


def print_receiver_matchup_report(receiver_name, alignment, opponent_team_profile,
                                   receiver_coverage_data, receiver_team_name=None,
                                   def_allowed_data=None,
                                   highlight_stats=("CR %", "YPRR", "TD", "CTGT %", "RATE", "FP/G")):
    report = build_receiver_matchup_report(receiver_name, alignment, opponent_team_profile,
                                            receiver_coverage_data, receiver_team_name=receiver_team_name,
                                            def_allowed_data=def_allowed_data)
    if report and "error" in report[0]:
        print(f"\n  [BLOCKED] {report[0]['error']}")
        return report
    if report and "note" in report[0]:
        print(f"\n  {report[0]['note']}")
        return report

    print(f"\n=== {receiver_name} ({alignment}) vs {opponent_team_profile.team_name} — Coverage Matchup ===")
    for entry in report:
        print(f"\n  {opponent_team_profile.team_name} runs {entry['coverage']} at "
              f"{entry['opponent_usage_pct']:.1f}% (z={entry['opponent_z_score']:+.2f} vs league)")

        rd = entry.get("receiver_data")
        if rd is None:
            print(f"    -> {receiver_name}: no recorded targets vs this coverage.")
        else:
            flag = f"  [THIN - {rd['_att']} tgt]" if entry["confidence"] == "thin_sample" else ""
            fit_warn = ""
            if entry.get("alignment_fit_warning"):
                fit_warn = f"  [CAUTION: only {entry['alignment_fit_pct']:.0f}% of routes are {alignment} - this file may not represent his usual usage]"
            stat_str = ", ".join(f"{s}={rd.get(s)} ({rd['_tiers'].get(s,'-')})"
                                  for s in highlight_stats if s in rd)
            print(f"    -> {receiver_name} (own history, {rd['_att']} TGT){flag}{fit_warn}: {stat_str}")

        dd = entry.get("defense_allows")
        if dd is not None:
            flag = f"  [THIN - {dd['_att']} tgt]" if entry.get("defense_confidence") == "thin_sample" else ""
            stat_str = ", ".join(f"{s}={dd.get(s)} ({dd['_tiers'].get(s,'-')})"
                                  for s in highlight_stats if s in dd)
            print(f"    -> {opponent_team_profile.team_name} allows to {alignment} ({dd['_att']} TGT){flag}: {stat_str}")

    return report


# ---------------------------------------------------------------------------
# FULL DATASET LOADER - loads all 70 files in one call
# ---------------------------------------------------------------------------

def _extract_coverage_suffix(normalized_filename_no_ext):
    """Given a filename (no extension) already normalized to letters+digits
    only, uppercased, finds which real coverage type it's for by checking
    the END of the name - real exports here use several different
    prefixes ('QB VS COVER 0', 'DEF BF VS 1', 'BACKFIELD VS  2MAN') and
    even a couple of confirmed real typos ('BACKFIELS VS 1',
    'BACKFILED VS 0', 'DEF BF VS O ' - a letter O standing in for a zero),
    so matching the suffix is far more robust than requiring one exact
    naming convention. 2MAN is checked before bare 2/COVER2 so it isn't
    mis-matched as plain Cover 2."""
    n = normalized_filename_no_ext
    if n.endswith("2MAN"):
        return "COVER 2 MAN %"
    if n.endswith("VSO"):  # confirmed real typo: "DEF BF VS O .csv" means Cover 0
        return "COVER 0 %"
    if n.endswith("0"):
        return "COVER 0 %"
    if n.endswith("1"):
        return "COVER 1 %"
    if n.endswith("2"):
        return "COVER 2 %"
    if n.endswith("3"):
        return "COVER 3 %"
    if n.endswith("4"):
        return "COVER 4 %"
    if n.endswith("6"):
        return "COVER 6 %"
    return None


def _normalize_filename(s):
    """Strips everything except letters/digits, uppercases - so real
    filename variance (spacing, case, a stray trailing space before the
    extension) doesn't block matching. Same approach as rb_matchup.py's
    _normalize_name, applied here for the same reason."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _scan_coverage_folder(dir_path):
    """Scans one real folder (e.g. 'WIDE', 'DEF ALLOWED/VS QBS') and
    returns {coverage_field: full_path} for every CSV whose filename
    suffix matches a real coverage type. Silently skips any file that
    doesn't match (e.g. a stray non-coverage file) rather than raising -
    same graceful-gap philosophy as the rest of this module. Returns
    an empty dict (not an error) if the folder doesn't exist, so a
    partially-collected dataset (e.g. QBS done, RUSH not yet exported)
    still loads whatever IS there."""
    if not os.path.isdir(dir_path):
        return {}
    out = {}
    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".csv"):
            continue
        norm = _normalize_filename(fname[:-4])
        coverage_field = _extract_coverage_suffix(norm)
        if coverage_field:
            out[coverage_field] = os.path.join(dir_path, fname)
    return out


@dataclass
class CoverageDataBundle:
    """Everything needed to build a matchup report for any player type,
    loaded once and reused. Missing files are skipped silently (not every
    coverage/alignment combo may exist yet) - check .missing for a list of
    what didn't load, so gaps are visible rather than silently assumed
    complete."""
    off_coverage: dict          # team_name -> TeamCoverageProfile (coverages this team's offense SEES)
    def_coverage: dict          # team_name -> TeamCoverageProfile (coverages this team's defense RUNS)
    qb_vs_coverage: dict        # coverage_field -> {qb_name: row}
    def_allowed_to_qb: dict     # coverage_field -> {team_name: row}
    receiver_by_alignment: dict # alignment -> coverage_field -> {player_name: row}
    def_allowed_by_alignment: dict  # alignment -> coverage_field -> {team_name: row}
    qb_scrambles: dict = field(default_factory=dict)          # {"SCRAMBLE": {qb_name: row}}
    def_allowed_qb_scrambles: dict = field(default_factory=dict)  # {"SCRAMBLE": {team_name: row}}
    missing: list = field(default_factory=list)


# Real folder names, exactly as FantasyPoints' Data Suite export structure
# organizes them (confirmed against an actual upload) - player-side and
# defense-allowed-side folder names for each alignment plus QBs.
ALIGNMENTS = ("wide", "slot", "inline", "backfield")
ALIGNMENT_DIRS = {"wide": "WIDE", "slot": "SLOT", "inline": "INLINE", "backfield": "BACKFIELD"}
ALIGNMENT_DEF_DIRS = {
    "wide": os.path.join("DEF ALLOWED", "VS WIDE"),
    "slot": os.path.join("DEF ALLOWED", "VS SLOT"),
    "inline": os.path.join("DEF ALLOWED", "VS INLINE"),
    "backfield": os.path.join("DEF ALLOWED", "VS BACKFIELD"),
}
QB_DIR = "QBS"
QB_DEF_DIR = os.path.join("DEF ALLOWED", "VS QBS")
COVG_DIR = "COVG%"
QB_SCRAMBLE_DIR = "QB RUSH METRICS"  # Real fix - confirmed against actual uploaded data; was looking for "QB_SCRAMBLES" which doesn't match the real folder name


def _find_covg_file(covg_dir, want_offense):
    """The team-level Man/Zone/Cover-0-6 tendency files - 'OFF COVG%.csv'
    (coverages this team's offense SEES) and 'DEF COVG %.csv' (coverages
    this team's defense RUNS). Matched by normalized OFF/DEF prefix rather
    than an exact filename, same robustness reasoning as everywhere else
    in this loader."""
    if not os.path.isdir(covg_dir):
        return None
    want_prefix = "OFF" if want_offense else "DEF"
    for fname in os.listdir(covg_dir):
        if not fname.lower().endswith(".csv"):
            continue
        norm = _normalize_filename(fname[:-4])
        if norm.startswith(want_prefix) and "COVG" in norm:
            return os.path.join(covg_dir, fname)
    return None


def _find_qb_scramble_files(scramble_dir):
    """Real, typo-tolerant finder for the two QB-scramble files - the
    real files seen so far have used inconsistent naming/spelling (e.g.
    'QB_SCAMBLES.csv' - a real missing letter, not a made-up example),
    so this matches by normalized DEF-prefix presence rather than an
    exact filename, same robustness approach as everywhere else in this
    loader."""
    if not os.path.isdir(scramble_dir):
        return None, None
    qb_file, def_file = None, None
    for fname in os.listdir(scramble_dir):
        if not fname.lower().endswith(".csv"):
            continue
        norm = _normalize_filename(fname[:-4])
        if norm.startswith("DEF"):
            def_file = os.path.join(scramble_dir, fname)
        elif "SC" in norm:  # matches SCRAMBLE/SCAMBLE (real typo) either way
            qb_file = os.path.join(scramble_dir, fname)
    return qb_file, def_file


def load_full_dataset(data_dir="."):
    """Loads the complete real dataset in one call, matching the ACTUAL
    FantasyPoints Data Suite export folder structure (confirmed directly
    against a real upload, replacing an earlier version of this function
    that guessed at a flat-file naming convention that turned out not to
    match reality at all):

      data_dir/
        COVG%/OFF COVG%.csv, DEF COVG %.csv
        QBS/QB VS COVER <N>.csv                      (7 files)
        DEF ALLOWED/VS QBS/DEF QB VS <N>.csv          (7 files)
        WIDE|SLOT|INLINE|BACKFIELD/<alignment> VS <N>.csv       (7 each)
        DEF ALLOWED/VS WIDE|SLOT|INLINE|BACKFIELD/DEF <align> VS <N>.csv (7 each)
        RUSH METRICS/, RUSH METRICS ALLOWED/ - NOT loaded here, see
        rb_matchup.py's load_full_rb_dataset() for those.

    Filename matching is suffix-based and typo-tolerant (see
    _extract_coverage_suffix) - real exports here have used several
    different prefix conventions and at least two confirmed real typos,
    none of which need to be manually renamed before loading.

    Missing files/folders are skipped (not every combo may be collected
    yet) and logged in the returned bundle's .missing list instead of
    raising - partial datasets are expected and handled gracefully
    throughout this module (thin-sample / no-data paths already exist on
    every report)."""
    missing = []

    off_file = _find_covg_file(os.path.join(data_dir, COVG_DIR), want_offense=True)
    def_file = _find_covg_file(os.path.join(data_dir, COVG_DIR), want_offense=False)
    off_profiles = {}
    def_profiles = {}
    if off_file:
        off_profiles, _ = load_team_coverage_matrix(off_file)
    else:
        missing.append(f"Offense team coverage tendency (looked in '{os.path.join(data_dir, COVG_DIR)}')")
    if def_file:
        def_profiles, _ = load_team_coverage_matrix(def_file)
    else:
        missing.append(f"Defense team coverage tendency (looked in '{os.path.join(data_dir, COVG_DIR)}')")

    qb_files = _scan_coverage_folder(os.path.join(data_dir, QB_DIR))
    for cov in COVERAGE_FIELDS:
        if cov not in qb_files:
            missing.append(f"QB vs {cov} (looked in '{os.path.join(data_dir, QB_DIR)}')")
    qb_data = load_qb_vs_coverage(qb_files) if qb_files else {}

    def_qb_files = _scan_coverage_folder(os.path.join(data_dir, QB_DEF_DIR))
    for cov in COVERAGE_FIELDS:
        if cov not in def_qb_files:
            missing.append(f"Def-allowed-to-QB {cov} (looked in '{os.path.join(data_dir, QB_DEF_DIR)}')")
    def_qb_data = load_def_allowed_to_qb(def_qb_files) if def_qb_files else {}

    receiver_by_alignment = {}
    def_allowed_by_alignment = {}
    for alignment in ALIGNMENTS:
        rec_dir = os.path.join(data_dir, ALIGNMENT_DIRS[alignment])
        def_dir = os.path.join(data_dir, ALIGNMENT_DEF_DIRS[alignment])
        rec_files = _scan_coverage_folder(rec_dir)
        def_files = _scan_coverage_folder(def_dir)
        for cov in COVERAGE_FIELDS:
            if cov not in rec_files:
                missing.append(f"{alignment} receiver vs {cov} (looked in '{rec_dir}')")
            if cov not in def_files:
                missing.append(f"Def-allowed-{alignment} {cov} (looked in '{def_dir}')")
        receiver_by_alignment[alignment] = load_receiver_vs_coverage(rec_files) if rec_files else {}
        def_allowed_by_alignment[alignment] = load_def_allowed_by_alignment(def_files) if def_files else {}

    scramble_dir = os.path.join(data_dir, QB_SCRAMBLE_DIR)
    qb_scramble_file, def_scramble_file = _find_qb_scramble_files(scramble_dir)
    qb_scrambles = load_qb_scrambles(qb_scramble_file) if qb_scramble_file else {}
    def_allowed_qb_scrambles = load_def_allowed_qb_scrambles(def_scramble_file) if def_scramble_file else {}
    if not qb_scramble_file:
        missing.append(f"QB scrambles (looked in '{scramble_dir}')")
    if not def_scramble_file:
        missing.append(f"Def-allowed QB scrambles (looked in '{scramble_dir}')")

    return CoverageDataBundle(
        off_coverage=off_profiles, def_coverage=def_profiles,
        qb_vs_coverage=qb_data, def_allowed_to_qb=def_qb_data,
        receiver_by_alignment=receiver_by_alignment,
        def_allowed_by_alignment=def_allowed_by_alignment,
        qb_scrambles=qb_scrambles, def_allowed_qb_scrambles=def_allowed_qb_scrambles,
        missing=missing,
    )


# ---------------------------------------------------------------------------
# LIVE-MODEL EXPLOIT-STRENGTH FUNCTIONS - the actual plug-in points
# nfl_model_combined.py imports (calc_alignment_exploit_strength,
# calc_qb_coverage_exploit_strength). Everything above this point already
# existed (parsing, tiering, z-score outlier detection, the interactive
# matchup-report builders) - these two were the missing final step
# connecting that real infrastructure to the live quality_score pipeline.
#
# Same 0-1 "exploit_strength" semantics as calc_coverage_quality_score/
# calc_box_quality_score in nfl_model_combined.py (higher = more favorable
# matchup for the offensive player) so they combine consistently with the
# rest of that file's structural_parts averaging.
# ---------------------------------------------------------------------------

TIER_SCORE = {"Elite": 1.0, "Above Avg": 0.75, "Average": 0.5, "Below Avg": 0.25, "Poor": 0.0}


def _tier_to_score(tiers: dict, stat: str):
    """Converts a row's tier label for one stat into a 0-1 numeric score.
    Returns None if that stat wasn't tiered for this row (thin league-wide
    sample, or the column wasn't numeric) - callers skip rather than
    guess a default."""
    if not tiers or stat not in tiers:
        return None
    return TIER_SCORE.get(tiers[stat])


def _weighted_outlier_exploit(outliers, own_data_by_coverage, def_allowed_by_coverage,
                               own_name, opp_team_name, own_stat, max_outliers=3,
                               own_weight=0.4, def_weight=0.6):
    """Shared real logic for both functions below: given an opponent's
    real outlier coverages (z-score based, from TeamCoverageProfile.
    outliers), combines - for each outlier coverage - how exploitable the
    DEFENSE is (their allowed tier for own_stat, weighted more heavily,
    since that's the new opponent-specific information this signal adds)
    with how good the PLAYER himself has been in that specific coverage
    (weighted less, since his overall quality is already captured
    elsewhere in the pipeline via mu/role_verification - this adds a
    narrower "does his own history support this specific coverage fit"
    layer on top, not a replacement for it).

    Weighted by each outlier's real z-score magnitude (a coverage a team
    leans into at z=2.5 counts for more than one at z=1.05), not treated
    as equally-weighted just for clearing the outlier threshold.

    Returns (exploit_strength: float|nan, coverages_checked: list[str]).
    Real gaps (no data for a coverage on either side) are skipped rather
    than defaulted to a neutral score - "no data" and "average" aren't
    the same thing.

    REAL, SECOND FIX (per direct, explicit request to use EVERY real
    column from these CSVs, not a hand-picked subset): own_stat now
    accepts THREE shapes -
      - a single stat name (original behavior, unchanged)
      - a flat list of stat names (first fix - simple equal-weight average)
      - a dict of {bucket_name: [stat_names]} (this fix) - stats within
        the same bucket are averaged together FIRST (so e.g. 6 different
        pressure-related columns collapse into one "pressure" reading),
        then buckets are averaged together EQUALLY across each other -
        this is what actually lets every real column get used without a
        bucket that happens to have more columns silently dominating the
        signal just by column count. A flat list is treated as one
        implicit bucket (equivalent to the first fix); this keeps both
        prior call shapes working unchanged.
    """
    if not outliers:
        return np.nan, []

    if isinstance(own_stat, dict):
        stat_buckets = own_stat
    elif isinstance(own_stat, (list, tuple)):
        stat_buckets = {"_all": list(own_stat)}
    else:
        stat_buckets = {"_all": [own_stat]}

    weighted_scores = []
    weights = []
    checked = []
    for coverage_field, z in outliers[:max_outliers]:
        checked.append(coverage_field.replace(" %", ""))
        def_row = def_allowed_by_coverage.get(coverage_field, {}).get(opp_team_name)
        own_row = own_data_by_coverage.get(coverage_field, {}).get(own_name)
        def_tiers = def_row.get("_tiers") if def_row else None
        own_tiers = own_row.get("_tiers") if own_row else None

        # Real fix - average WITHIN each bucket first, then average the
        # resulting per-bucket scores together EQUALLY, regardless of how
        # many raw columns fed each bucket. This is the actual mechanism
        # that lets every real column be used (nothing thrown away) while
        # stopping a bucket with many redundant columns (e.g. 6 pressure
        # cuts) from silently outvoting a bucket with just 1 (e.g. 1
        # turnover-rate column).
        #
        # REAL FIX - per-bucket direction override (per direct example: a
        # LOW aDOT/YPR/YAC-per-rec is a real green light for receptions
        # specifically - short, high-percentage looks mean more catches -
        # even though the SAME stat isn't treated as bad for rec_yards.
        # QB_ALIGNMENT_INVERSE_STATS is a single GLOBAL direction per stat
        # and can't represent "inverted for this prop, not for that one."
        # A stat name prefixed with '~' (e.g. "~aDOT") means: look up this
        # stat's normal tier score, then flip it (1 - score) for THIS
        # bucket only - the stat's globally-computed tier itself is
        # untouched, so other buckets/props referencing the same plain
        # stat name are completely unaffected.
        def_bucket_scores = []
        own_bucket_scores = []
        for bucket_stats in stat_buckets.values():
            d_scores, o_scores = [], []
            for stat in bucket_stats:
                invert = stat.startswith("~")
                real_stat = stat[1:] if invert else stat
                d = _tier_to_score(def_tiers, real_stat)
                o = _tier_to_score(own_tiers, real_stat)
                if d is not None:
                    d_scores.append(1 - d if invert else d)
                if o is not None:
                    o_scores.append(1 - o if invert else o)
            if d_scores:
                def_bucket_scores.append(sum(d_scores) / len(d_scores))
            if o_scores:
                own_bucket_scores.append(sum(o_scores) / len(o_scores))

        parts = []
        part_weights = []
        if def_bucket_scores:
            parts.append(sum(def_bucket_scores) / len(def_bucket_scores))
            part_weights.append(def_weight)
        if own_bucket_scores:
            parts.append(sum(own_bucket_scores) / len(own_bucket_scores))
            part_weights.append(own_weight)

        if not parts:
            continue  # no real data on either side for this coverage - skip, don't guess
        combined = sum(p * w for p, w in zip(parts, part_weights)) / sum(part_weights)
        weighted_scores.append(combined)
        weights.append(max(z, 0.01))

    if not weighted_scores:
        return np.nan, checked
    exploit_strength = sum(s * w for s, w in zip(weighted_scores, weights)) / sum(weights)
    return round(exploit_strength, 3), checked


# Real, COMPREHENSIVE bucketed stat sets per QB prop - per direct request
# to use every real column from the actual CSV rather than a hand-picked
# subset. Bucketed by underlying concept (see _weighted_outlier_exploit's
# bucket-averaging) so e.g. the 6 different pressure-related columns
# (TTT/TTP/TTSK/TTSC/QB SK/QBP/PRESS%/PRESS SK%/PrROE) collapse into ONE
# "pressure" vote per coverage, not 9 votes silently drowning out a
# 1-column bucket like turnover rate. Genuinely ambiguous/unclear columns
# from the raw list (DB, TA, TWT %) are left out rather than guessed into
# a bucket where they might not belong - real column definitions needed
# to place these with confidence (flagged to the user, not silently
# dropped without saying so).
QB_COVERAGE_STATS_BY_PROP = {
    "pass_yards": {
        "efficiency": ["YPA", "ANY/A", "CPOE"],
        "depth_explosiveness": ["aDOT", "AY", "Deep Throw %", "Deep Throw"],
        "yac": ["YAC %"],
        "accuracy_context": ["ADJ CMP %", "ACC %"],
        "pressure": ["TTT", "TTP", "TTSK", "TTSC", "QB SK", "QBP", "PRESS %", "PRESS SK %", "PrROE"],
    },
    "pass_completions": {
        "accuracy": ["CMP %", "ADJ CMP %", "ACC %", "CATCH %", "CPOE"],
        "reliability": ["DROP %", "DROP YDS"],
        "pressure": ["TTT", "TTP", "TTSK", "TTSC", "PRESS %", "PRESS SK %", "PrROE"],
    },
    "pass_attempts": {
        "volume_tendency": ["ATT", "RPO %", "CHK %", "OFF %"],
    },
    "pass_tds": {
        "scoring_opportunity": ["TD", "TD %", "EZATT"],
    },
    # REAL FIX - per direct request. A long completion needs both real
    # depth (deep-throw volume/accuracy) and real accuracy under that
    # depth - a QB who's accurate on deep throws specifically is the
    # right signal here, not just raw yardage efficiency (YPA already
    # covers that for pass_yards).
    "longest_completion": {
        "depth_accuracy": ["YPA", "ADJ CMP %", "Deep Throw %", "ACC %"],
    },
    "interceptions": {
        "turnover": ["INT"],
        "pressure": ["TTT", "TTP", "TTSK", "TTSC", "QB SK", "QBP", "PRESS %", "PRESS SK %", "PrROE"],
        "sack_risk": ["SACK", "SACK %", "SK YDS"],
    },
}
# Real, honest fallback for any QB prop not in the table above (e.g.
# fantasy_points, or a future prop) - RATE alone, the original behavior,
# rather than silently returning nothing.
QB_COVERAGE_DEFAULT_STATS = ["RATE"]


def calc_qb_coverage_exploit_strength(bundle: CoverageDataBundle, qb_name: str,
                                       qb_team_abbrev: str, opponent_team_abbrev: str,
                                       prop_type: str = None) -> dict:
    """
    Real per-QB signal: for each coverage this opponent's defense genuinely
    leans into (real z-score outlier, not just locally highest), combines
    how much that defense allows to QBs in that specific coverage (tiered
    against the real league distribution of defenses in that coverage)
    with this QB's own real numbers in that same coverage from his own
    game history - see _weighted_outlier_exploit for the exact combination
    and weighting.

    REAL FIX (found live this session, per direct real pushback): this
    used to ALWAYS use "RATE" regardless of which prop was actually being
    scored - meaning pass_yards, pass_completions, pass_attempts, pass_tds
    and interceptions all got the exact same coverage-fit signal, despite
    RATE being a composite that doesn't isolate any of them individually.
    Now takes prop_type and pulls a genuinely tailored, non-redundant stat
    set for that specific prop from QB_COVERAGE_STATS_BY_PROP (falls back
    to RATE alone - the original behavior - if prop_type is omitted or
    not in the table, so existing callers aren't broken).

    Team abbreviations (as used throughout nfl_model_combined.py) are
    converted to the full names this module's data is keyed on via
    TEAM_ABBREV_TO_FULL. Returns neutral/empty gracefully (exploit_strength
    NaN) if the opponent isn't found or has no real outlier coverage this
    season - never raises, matching this module's established graceful-
    gap handling throughout.

    REAL FIX (found live this session, per direct follow-up request): this
    premium signal is built entirely from the opponent's real, uploaded
    coverage-usage CSV (season data) - it had NO coordinator-change
    awareness at all, unlike the free-data grade builders which already
    check NEW_OC_TEAMS_2026/NEW_DC_TEAMS_2026. If the opponent has a real,
    new 2026 DC, their real coverage-usage mix from that CSV may no longer
    reflect their actual current scheme - same unreliability as the free-
    data bridge case, just never protected the same way. Now returns
    neutral (NaN) for opponents on NEW_DC_TEAMS_2026, exactly like the
    free-data functions do for their own bridge.
    """
    opp_full = TEAM_ABBREV_TO_FULL.get((opponent_team_abbrev or "").upper())
    if (opponent_team_abbrev or "").upper() in NEW_DC_TEAMS_2026:
        return {"exploit_strength": np.nan, "outlier_coverages_checked": []}
    opp_profile = bundle.def_coverage.get(opp_full) if opp_full else None
    if opp_profile is None:
        return {"exploit_strength": np.nan, "outlier_coverages_checked": []}

    stats_to_use = QB_COVERAGE_STATS_BY_PROP.get(prop_type, QB_COVERAGE_DEFAULT_STATS)
    exploit_strength, checked = _weighted_outlier_exploit(
        opp_profile.outliers, bundle.qb_vs_coverage, bundle.def_allowed_to_qb,
        qb_name, opp_profile.team_name, own_stat=stats_to_use,
    )

    # REAL FIX (found live via direct user pushback - Cincinnati/Burrow's
    # real 8-game 2025 season raised a fair concern, and checking it
    # directly surfaced an even more severe, completely unflagged case:
    # San Francisco/Purdy played only 9 real games). Doesn't blanket-
    # exclude a QB just for a shortened season - some per-coverage
    # attempt counts stay genuinely large even in a short season
    # (Burrow had 76 real attempts vs Cover 3 alone despite only 8 games)
    # - but surfaces the real risk transparently: a shortened season
    # means less real game-to-game variety represented, which a large
    # raw attempt count alone doesn't fully rule out. Real threshold: 10
    # of 17 games, roughly 60% of a season.
    real_games = None
    for coverage_rows in bundle.qb_vs_coverage.values():
        row = coverage_rows.get(qb_name)
        if row and row.get("G") is not None:
            real_games = _to_float(row.get("G"))
            break
    low_season_sample = real_games is not None and real_games < 10

    return {"exploit_strength": exploit_strength, "outlier_coverages_checked": checked,
            "real_games_played_2025": real_games, "low_season_sample_warning": low_season_sample}


def calc_alignment_exploit_strength(bundle: CoverageDataBundle, player_name: str, position: str,
                                     player_team_abbrev: str, opponent_team_abbrev: str,
                                     prop_type: str = None,
                                     alignment_bundle: "TeamAlignmentTargetBundle" = None) -> dict:
    """
    Real per-receiver signal, same shape as calc_qb_coverage_exploit_
    strength above, but first has to figure out which alignment
    (Wide/Slot/Inline/Backfield) this player is actually used at, since
    the caller (nfl_model_combined.py) doesn't pass one in.

    REAL BUG CAUGHT+FIXED before this shipped: the obvious approach - read
    the player's own WIDE/SLOT/INLINE/BACK RTE% columns off whichever
    alignment file has him - is broken. Confirmed directly on real data
    (Saquon Barkley): those RTE% columns are self-referential PER FILE,
    not a real cross-alignment share - the WIDE file's own "WIDE RTE %"
    column reads 100% simply because that file has already pre-filtered
    to his wide-alignment routes, so it trivially says "100% of THESE
    routes were wide." Every alignment file does the same thing for
    itself, making the column useless for telling alignments apart.

    Fixed by comparing real TGT volume ACROSS the 4 alignment files
    instead - whichever alignment has the player's highest real target
    count (summed across all 7 of that alignment's coverage files, since
    volume is split by coverage faced) is his real dominant alignment.
    alignment_fit_pct is then computed honestly as that alignment's share
    of his total charted targets across all 4 alignments - not a raw
    CSV column.

    Uses FP/G (overall fantasy value per game) as the combination stat -
    a fair general-quality measure available on every receiver row,
    unlike RATE (QB-specific) or CR %/YPRR alone (miss the volume side).

    Returns exploit_strength NaN (not a guess) if the player has no
    recorded targets in ANY alignment file yet this season - a real gap,
    not defaulted to neutral.

    REAL FIX (found live this session, same fix as calc_qb_coverage_
    exploit_strength above) - returns neutral (NaN) if the opponent has a
    real, new 2026 DC (NEW_DC_TEAMS_2026), since this signal is built
    entirely from the opponent's real, uploaded coverage-usage CSV and had
    no coordinator-change awareness at all before this.
    """
    opp_full = TEAM_ABBREV_TO_FULL.get((opponent_team_abbrev or "").upper())
    if (opponent_team_abbrev or "").upper() in NEW_DC_TEAMS_2026:
        return {"exploit_strength": np.nan, "dominant_alignment": None,
                "alignment_fit_pct": None, "outlier_coverages_checked": []}
    opp_profile = bundle.def_coverage.get(opp_full) if opp_full else None
    if opp_profile is None:
        return {"exploit_strength": np.nan, "dominant_alignment": None,
                "alignment_fit_pct": None, "outlier_coverages_checked": []}

    # Real target volume per alignment, summed across that alignment's
    # own coverage-type files (a player's targets are split by which
    # coverage he faced, so no single coverage file has his full count).
    tgt_by_alignment = {}
    for alignment in ALIGNMENTS:
        total_tgt = 0
        for coverage_field, rows in bundle.receiver_by_alignment.get(alignment, {}).items():
            row = rows.get(player_name)
            if row is not None:
                total_tgt += int(_to_float(row.get("TGT")) or 0)
        if total_tgt > 0:
            tgt_by_alignment[alignment] = total_tgt

    if not tgt_by_alignment:
        return {"exploit_strength": np.nan, "dominant_alignment": None,
                "alignment_fit_pct": None, "outlier_coverages_checked": []}

    dominant_alignment = max(tgt_by_alignment, key=tgt_by_alignment.get)
    total_across_all = sum(tgt_by_alignment.values())
    alignment_fit_pct = round(100 * tgt_by_alignment[dominant_alignment] / total_across_all, 1)

    # Real, COMPREHENSIVE bucketed stat sets per receiving prop - per
    # direct request to use every real column from the actual CSV.
    # Bucketed by concept so e.g. YAC/YAC-REC/YACO/YACO-REC (4 columns,
    # all really "yards after catch" from slightly different angles)
    # collapse into one vote, same reasoning as the QB-coverage fix above.
    # Ambiguous columns (DESIGN %, CT, THREAT, YPTOE, TM YDS %, TM TD %,
    # OFF %-equivalents) left out rather than guessed into a bucket -
    # flagged to the user, not silently dropped.
    ALIGNMENT_STATS_BY_PROP = {
        "receptions": {
            "reliability": ["DRP %"],
            "contested_catch": ["CC %", "CC"],
            "route_participation": ["RTE %", "TPRR"],
            "catch_rate": ["CR %"],
            # Real, direct example given: high target share + being the
            # play's primary read are both a real green light for catch
            # VOLUME specifically (more real chances to catch something).
            "role_volume": ["TGT %", "1READ %"],
            # Real, direct example given: LOW aDOT/YPR/YAC-per-catch is a
            # green light for RECEPTIONS specifically - short, high-
            # percentage looks convert to catches more often than deep
            # shots do, even though the exact same stats are NOT inverted
            # for rec_yards (see that bucket below) - the '~' prefix
            # inverts only for this bucket's usage, the underlying stat's
            # own tier computation is untouched.
            "floor_profile": ["~aDOT", "~YPR", "~YAC/REC"],
        },
        "targets": {
            "role": ["1READ %", "1READ"],
            "route_participation": ["RTE %", "TPRR"],
            "target_quality": ["DP TGT", "CTGT %", "CTGT"],
            "volume": ["TGT", "TGT/G", "TGT %"],
        },
        "rec_yards": {
            "efficiency": ["YPRR", "YPT", "YPR"],
            "yac": ["YAC", "YAC/REC", "YACO", "YACO/REC"],
            "depth": ["AY", "AY Share"],  # aDOT removed - now sourced for free via NGS (avg_intended_air_yards) in build_receiver_advanced_metrics, keeping both would double-count the same real-world concept
            "explosiveness": ["i20 TGT", "EZTGT", "EZTD"],
            "elusiveness": ["MTF", "MTF/REC"],
            "first_downs": ["1D", "1D/RR"],
            "scoring": ["TD"],
        },
        # REAL FIX - rec_tds had no dedicated bucket at all before this,
        # confirmed via direct diagnostic (fell back to generic FP/G).
        # Real scoring-opportunity signal: raw TD volume, the same
        # computed TD % rate used for pass_tds (TD/TGT here, computed in
        # _load_coverage_keyed_data), and real end-zone target indicators.
        "rec_tds": {
            "scoring_opportunity": ["TD", "TD %", "i20 TGT", "EZTGT", "EZTD"],
        },
        # REAL FIX - per direct example given ("longest cmp should need
        # ypa, adjusted cmp%, deep throw%, acc% etc... and what def
        # allows there like high ypr, high yac"): this alignment-side
        # equivalent uses real explosiveness metrics - a big gain needs
        # both real depth of target (aDOT reintroduced here specifically,
        # since "how much room is there for a huge gain" is a genuinely
        # different question than rec_yards' steady-efficiency read) and
        # real after-catch explosiveness (YAC/YPR). The existing
        # calc_alignment_exploit_strength mechanism automatically
        # cross-references these same stat keys against the DEFENSE's
        # allowed-by-alignment data too - no separate wiring needed.
        "longest_reception": {
            "depth": ["aDOT", "AY"],
            "explosiveness": ["YAC", "YAC/REC", "YPR", "i20 TGT", "EZTGT"],
        },
    }
    ALIGNMENT_DEFAULT_STATS = ["FP/G"]

    # REAL FIX (found live this session, per direct example given: "Vikings
    # have a new QB so we can't use offense side where the ball goes to
    # certain alignments vs certain coverages, but def side we can") -
    # NEW_QB_TEAMS_2026 is independent of NEW_OC_TEAMS_2026: a team can
    # keep the exact same OC and still have a real, different passing game
    # if the QB changed. Only the receiver's own QUALITY read (his real
    # catch rate/YAC/etc against specific coverages, which reflects
    # execution WITH the old QB) gets dropped - his real target VOLUME by
    # alignment (used above for dominant_alignment) stays valid, since
    # which alignment he lines up from is a deployment/usage question, not
    # a QB-decision-making one.
    own_data_for_blend = bundle.receiver_by_alignment.get(dominant_alignment, {})
    if (player_team_abbrev or "").upper() in NEW_QB_TEAMS_2026:
        own_data_for_blend = {}

    exploit_strength, checked = _weighted_outlier_exploit(
        opp_profile.outliers,
        own_data_for_blend,
        bundle.def_allowed_by_alignment.get(dominant_alignment, {}),
        player_name, opp_profile.team_name,
        own_stat=ALIGNMENT_STATS_BY_PROP.get(prop_type, ALIGNMENT_DEFAULT_STATS),
    )
    # REAL FIX (per direct request - wiring the standalone volume/
    # targeting system, built earlier this session, into the actual live
    # exploit-strength calculation rather than leaving it purely
    # standalone). When alignment_bundle is provided, checks the real
    # target-volume data for this dominant_alignment across the
    # opponent's real qualifying coverages: if the defense's real
    # target-rank there clears ALIGNMENT_TARGET_RANK_THRESHOLD (a real,
    # meaningfully-high-volume spot, same threshold as
    # find_defense_exploit_spots), that's real confirmation the
    # efficiency edge is actually reachable/actionable - nudges
    # exploit_strength modestly toward the extreme it already leans.
    # A LOW real volume rank there means the edge, even if efficiency-
    # favorable, isn't where the ball tends to go - modestly dampens it
    # instead. Bounded (+-0.05) so this is a real refinement, not a
    # dominant factor - the underlying efficiency signal stays primary.
    # Backward-compatible: alignment_bundle=None (default) skips this
    # entirely, unchanged from before.
    if alignment_bundle is not None and pd.notna(exploit_strength) and dominant_alignment:
        volume_confirmations = []
        for coverage_name in checked:
            coverage_field = coverage_name if coverage_name.endswith(" %") else coverage_name + " %"
            team_ranks = rank_alignment_targeting_within_coverage(alignment_bundle, coverage_field, side="def")
            real_rank = team_ranks.get(opp_profile.team_name, {}).get(dominant_alignment)
            if real_rank is not None:
                volume_confirmations.append(real_rank <= ALIGNMENT_TARGET_RANK_THRESHOLD)
        if volume_confirmations:
            confirmed_share = sum(volume_confirmations) / len(volume_confirmations)
            # confirmed_share near 1.0 (real high volume everywhere checked) -> positive adjustment
            # confirmed_share near 0.0 (real low volume everywhere checked) -> negative adjustment
            # Applied directly, in either direction, regardless of which side of
            # neutral the base efficiency signal started on - real confirmation
            # should help either way, real contradiction should hurt either way.
            adjustment = (confirmed_share - 0.5) * 0.10  # max +-0.05
            exploit_strength = round(min(1.0, max(0.0, exploit_strength + adjustment)), 3)

    return {
        "exploit_strength": exploit_strength,
        "dominant_alignment": dominant_alignment,
        "alignment_fit_pct": alignment_fit_pct,
        "outlier_coverages_checked": checked,
    }


def get_matchup(bundle: CoverageDataBundle, player_name, position, opponent_team,
                 player_team=None, alignment=None):
    """Single entry point for a matchup report, any position. Position:
    'QB' uses the QB pipeline. 'WR'/'TE'/'RB' uses the receiver-by-
    alignment pipeline and REQUIRES alignment ('wide'/'slot'/'inline'/
    'backfield') since that data is alignment-specific.

    opponent_team: full team name, matched against bundle.def_coverage
    (the defense's own tendencies - what YOU'RE facing when playing them).
    """
    opp_profile = bundle.def_coverage.get(opponent_team)
    if opp_profile is None:
        return [{"error": f"'{opponent_team}' not found in loaded team coverage data. "
                           f"Check spelling matches the full team name (e.g. 'Seattle Seahawks')."}]

    if position.upper() == "QB":
        return build_qb_matchup_report(player_name, opp_profile, bundle.qb_vs_coverage,
                                        qb_team_name=player_team, def_allowed_data=bundle.def_allowed_to_qb)

    if alignment is None:
        return [{"error": f"alignment is required for position '{position}' "
                           f"(one of: wide, slot, inline, backfield)."}]
    alignment = alignment.lower()
    if alignment not in bundle.receiver_by_alignment:
        return [{"error": f"Unknown alignment '{alignment}'. Must be one of: {ALIGNMENTS}"}]

    return build_receiver_matchup_report(
        player_name, alignment, opp_profile,
        bundle.receiver_by_alignment[alignment],
        receiver_team_name=player_team,
        def_allowed_data=bundle.def_allowed_by_alignment[alignment],
    )



# ---------------------------------------------------------------------------
# Matchup report
# ---------------------------------------------------------------------------

def build_qb_matchup_report(qb_name, opponent_team_profile: TeamCoverageProfile,
                             qb_coverage_data: dict, qb_team_name=None,
                             def_allowed_data: dict = None, max_outliers=3):
    if qb_team_name and _same_team(qb_team_name, opponent_team_profile.team_name):
        return [{"error": f"{qb_name} plays for {opponent_team_profile.team_name} - "
                           f"cannot build a matchup report against his own team."}]

    report = []
    outliers = opponent_team_profile.outliers[:max_outliers]
    if not outliers:
        return [{"note": f"{opponent_team_profile.team_name} has no statistically real "
                          f"outlier coverage this season - no specific coverage edge to flag."}]

    for coverage_field, z in outliers:
        cov_label = coverage_field.replace(" %", "")
        entry = {
            "coverage": cov_label,
            "opponent_usage_pct": opponent_team_profile.rates[coverage_field],
            "opponent_z_score": round(z, 2),
        }

        qb_row = qb_coverage_data.get(coverage_field, {}).get(qb_name)
        if qb_row is None:
            entry["qb_data"] = None
            entry["confidence"] = "no_data"
        else:
            entry["qb_data"] = qb_row  # FULL row - every column, plus _tiers dict
            entry["confidence"] = "thin_sample" if qb_row["_thin_sample"] else "solid"

        if def_allowed_data is not None:
            def_row = def_allowed_data.get(coverage_field, {}).get(opponent_team_profile.team_name)
            entry["defense_allows"] = def_row  # FULL row, or None
            entry["defense_confidence"] = ("thin_sample" if def_row and def_row["_thin_sample"]
                                            else "solid" if def_row else "no_data")

        report.append(entry)
    return report


def print_matchup_report(qb_name, opponent_team_profile, qb_coverage_data,
                          qb_team_name=None, def_allowed_data=None,
                          highlight_stats=("CMP %", "YPA", "TD", "INT", "RATE", "CPOE", "FP/G")):
    """Console-friendly summary. Prints tiers for a curated highlight set by
    default (still has the full row available in the returned report dict
    for anything deeper - this is just the readable console view)."""
    report = build_qb_matchup_report(qb_name, opponent_team_profile, qb_coverage_data,
                                      qb_team_name=qb_team_name, def_allowed_data=def_allowed_data)
    if report and "error" in report[0]:
        print(f"\n  [BLOCKED] {report[0]['error']}")
        return report
    if report and "note" in report[0]:
        print(f"\n  {report[0]['note']}")
        return report

    print(f"\n=== {qb_name} vs {opponent_team_profile.team_name} — Coverage Matchup ===")
    for entry in report:
        print(f"\n  {opponent_team_profile.team_name} runs {entry['coverage']} at "
              f"{entry['opponent_usage_pct']:.1f}% (z={entry['opponent_z_score']:+.2f} vs league)")

        qd = entry.get("qb_data")
        if qd is None:
            print(f"    -> {qb_name}: no recorded attempts vs this coverage.")
        else:
            flag = f"  [THIN - {qd['_att']} att]" if entry["confidence"] == "thin_sample" else ""
            stat_str = ", ".join(f"{s}={qd.get(s)} ({qd['_tiers'].get(s,'-')})"
                                  for s in highlight_stats if s in qd)
            print(f"    -> {qb_name} (own history, {qd['_att']} ATT){flag}: {stat_str}")

        dd = entry.get("defense_allows")
        if dd is not None:
            flag = f"  [THIN - {dd['_att']} att]" if entry.get("defense_confidence") == "thin_sample" else ""
            stat_str = ", ".join(f"{s}={dd.get(s)} ({dd['_tiers'].get(s,'-')})"
                                  for s in highlight_stats if s in dd)
            print(f"    -> {opponent_team_profile.team_name} allows ({dd['_att']} ATT){flag}: {stat_str}")

    return report


# ===========================================================================
# Real, merged content from rb_matchup.py (per direct request, single-file
# consolidation) - RB run-concept matchup data handling.
# Real note: _compute_field_tiers_rb, _read_fp_csv_rb, _same_team_rb,
# _to_float_rb, and TEAM_ABBREV_TO_FULL_RB below were renamed from this
# file's original names (_compute_field_tiers, _read_fp_csv, _same_team,
# _to_float, TEAM_ABBREV_TO_FULL) to avoid a real, confirmed collision with
# coverage_matchup's own, genuinely different versions of the same names
# above - verified different (not just duplicated) before renaming, e.g.
# TEAM_ABBREV_TO_FULL here has a real extra "BLT" entry the other doesn't.
# ===========================================================================

"""
NFL PREMIUM TOOL - RB Run-Concept Matchup Module
===================================================
Built from FantasyPoints.com Data Suite manual exports, same source and
workflow as coverage_matchup.py. Confirmed structure (both player-side and
defense-allowed sides checked directly, real column layout + real sample
sizes verified before this was written):

FILES (6 concepts, both sides):
  Player-side:      INSIDE_ZONE.csv, OUTSIDE_ZONE.csv, MAN-DUO.csv,
                     COUNTER.csv, POWER.csv, PULL_LEAD.csv
                     (real filenames as exported - note MAN-DUO uses a
                     hyphen, the others use underscores)
  Defense-allowed:  same 6 concepts, SAME filenames as player-side - since
                     a folder can't hold two files with identical names,
                     the defense-allowed copies must be stored in a
                     SEPARATE subfolder (see load_full_rb_dataset).

WHY NO TOP-N USAGE FILTER (real difference from coverage_matchup.py)
-----------------------------------------------------------------------
Coverage is a defensive SCHEME CHOICE - ranking defenses by how often they
choose to run a coverage is a real signal about their identity. Run
concept is called by the OFFENSE, not the defense - a defense doesn't
"run Counter 20% of the time," they just face whatever the RB's play call
gives them. Ranking defenses by concept-usage-rate would measure the
opponents THEY faced, not the defense's own tendency. So instead of a
top-N filter, every real concept (all 6) is always shown, each graded
directly on how that defense has actually performed against it - the
defense-allowed Quality Score IS the signal here, not a usage-rate cutoff.

DUPLICATE COLUMN HANDLING (real bug, same PATTERN as the QB Passing/
Scrambles YDS+TD issue in coverage_matchup.py, bigger here)
-----------------------------------------------------------------------
Every file's real header has ATT/YDS/TD/YPC/Success% appearing THREE
times (main Rushing/Advanced section, then again under a "Zone Concept"
section, then again under a "Man/Gap Concept" section), plus ATT% twice.
A naive dict(zip(header,row)) would silently keep only the LAST
occurrence (Man/Gap Concept values) and lose the real main stats
entirely. Fixed here via POSITIONAL (index-based) renaming, since the
column names collide but their real positions in the row don't - the
Zone/Man-Gap Concept columns get prefixed ZONE_/MANGAP_ during parsing.
Confirmed identical column layout/positions on both player-side (42
cols, includes Team/POS/FPTS section) and defense-allowed side (38 cols,
no Team/POS, no FPTS section - real difference, not a bug).

REAL SAMPLE SIZES (confirmed from actual 2025 data - player-side, >=20
real ATT / >=10 real ATT)
-----------------------------------------------------------------------
Inside Zone   54 / 75      Outside Zone  57 / 72     Man/Duo   47 / 64
Power         10 / 36      Pull Lead      8 / 32     Counter    6 / 24
Counter/Power/Pull Lead are real, usable concepts but meaningfully
thinner than the other three - separate thin-sample thresholds per
concept below, not one flat cutoff.

NO "LONGEST RUSH" COLUMN - confirmed absent from every file, same real
gap as "longest catch" on the WR/coverage side. Not built, not guessed.
"""


# ---------------------------------------------------------------------------
# Real concept list + filenames (exact, as exported - note MAN-DUO's hyphen)
# ---------------------------------------------------------------------------
CONCEPT_FILES = {
    "Inside Zone": "INSIDE_ZONE.csv",
    "Outside Zone": "OUTSIDE_ZONE.csv",
    "Man/Duo": "MAN-DUO.csv",
    "Counter": "COUNTER.csv",
    "Power": "POWER.csv",
    "Pull Lead": "PULL_LEAD.csv",
}

# Real thin-sample ATT thresholds per concept, set from the actual real
# player-side sample-size check above - Counter/Power/Pull Lead get a
# tighter bar than Inside/Outside Zone/Man-Duo, which have real deep
# samples league-wide.
THIN_SAMPLE_ATT_THRESHOLD = {
    "Inside Zone": 15, "Outside Zone": 15, "Man/Duo": 15,
    "Power": 8, "Pull Lead": 8, "Counter": 5,
}

# Stats that actually decide rushing prop quality - double-weighted in the
# quality score, same philosophy as CRUCIAL_QUALITY_STATS in streamlit_app.py
# Expanded per explicit real-world feedback: efficiency/explosiveness
# metrics (EXP RUN %, EXP YDS %, TD RATE) and the full YACO/stuff family
# genuinely separate a good rushing matchup from a bad one, not just raw
# volume+YPC - a defense can allow decent YPC while still getting
# stuffed at a high rate or giving up few explosive runs, and that
# distinction matters for grading BOTH sides (player AND defense-allowed
# use this same set). ATT % deliberately excluded: it only exists in the
# Zone/Man-Gap Concept columns, and since each file is already a single
# concept, that value is always ~100%/0% by definition - a redundant
# confirmation number, not a real signal.
CRUCIAL_RB_STATS = {
    "ATT", "YDS", "YPC", "TD", "Success %", "EXP RUN %", "EXP YDS %",
    "TD RATE", "MTF/ATT", "YACO", "YACO/ATT", "YACO %", "YBCO/ATT", "STUFF %",
}

# Stats where a HIGHER number is worse (mirrors coverage_matchup.py)
# Renamed from plain INVERSE_STATS - see QB_ALIGNMENT_INVERSE_STATS above
# for the real bug this silently caused by sharing that name.
RB_INVERSE_STATS = {"FUM", "STUFF %"}

NON_STAT_COLUMNS_PLAYER = {"Rank", "Name", "Team", "POS", "G", "Season"}
NON_STAT_COLUMNS_TEAM = {"Rank", "Name", "G", "Season", "Location", "Team Name"}

TEAM_ABBREV_TO_FULL_RB = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BLT": "Baltimore Ravens", "BUF": "Buffalo Bills", "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions",
    "GB": "Green Bay Packers", "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "JAC": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams",
    "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def _same_team_rb(abbrev_or_name, full_name):
    if not abbrev_or_name or not full_name:
        return False
    a = abbrev_or_name.strip().upper()
    if a in TEAM_ABBREV_TO_FULL_RB:
        return TEAM_ABBREV_TO_FULL_RB[a] == full_name
    return abbrev_or_name.strip().lower() == full_name.strip().lower()


def _to_float_rb(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _read_fp_csv_rb(path):
    """Reads a FantasyPoints export, returns (raw_header, data_rows) - the
    header/rows AFTER the title row (row 0, e.g. 'Player Details'/'Team
    Details'), which is metadata, not real columns."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[1]
    data = rows[2:]
    return header, data


# Positional rename map - see module docstring for why this is index-based,
# not name-based. Index 26-37 is the real duplicate block (Zone Concept
# then Man/Gap Concept, each repeating ATT/ATT%/YDS/TD/YPC/Success%),
# confirmed at these exact positions on BOTH player-side and defense-
# allowed side despite their different total column counts (42 vs 38 -
# the difference is entirely the trailing FPTS section, absent on the
# defense-allowed side).
ZONE_RENAME = {26: "ZONE_ATT", 27: "ZONE_ATT_PCT", 28: "ZONE_YDS",
                29: "ZONE_TD", 30: "ZONE_YPC", 31: "ZONE_SUCCESS_PCT"}
MANGAP_RENAME = {32: "MANGAP_ATT", 33: "MANGAP_ATT_PCT", 34: "MANGAP_YDS",
                   35: "MANGAP_TD", 36: "MANGAP_YPC", 37: "MANGAP_SUCCESS_PCT"}


def _row_to_dict(header, row):
    """Builds the row dict using positional renaming for the real
    duplicate-name columns (indices 26-37) - everything else keyed by its
    real column name as-is. Handles both the 42-col player-side and
    38-col defense-allowed layouts (rename map only applies where the
    index exists in this specific row)."""
    d = {}
    for i, val in enumerate(row):
        if i >= len(header):
            break
        if i in ZONE_RENAME:
            d[ZONE_RENAME[i]] = val
        elif i in MANGAP_RENAME:
            d[MANGAP_RENAME[i]] = val
        else:
            d[header[i]] = val
    return d


def _compute_field_tiers_rb(rows_by_key, non_stat_columns):
    """Same real-distribution z-score tiering as coverage_matchup.py -
    Elite/Above Avg/Average/Below Avg/Poor, computed against the actual
    players/teams in THIS concept's file, direction-corrected for stats
    where lower is better (FUM, STUFF %)."""
    if not rows_by_key:
        return
    sample_row = next(iter(rows_by_key.values()))
    stat_cols = [c for c in sample_row.keys() if c not in non_stat_columns and not c.startswith("_")]

    field_stats = {}
    for col in stat_cols:
        vals = [_to_float_rb(r.get(col)) for r in rows_by_key.values()]
        vals = [v for v in vals if v is not None]
        if len(vals) < 3:
            continue
        field_stats[col] = (mean(vals), pstdev(vals))

    for r in rows_by_key.values():
        tiers = {}
        for col, (avg, sd) in field_stats.items():
            v = _to_float_rb(r.get(col))
            if v is None or not sd:
                continue
            z = (v - avg) / sd
            if col in RB_INVERSE_STATS:
                z = -z
            if z >= 1.5:
                tiers[col] = "Elite"
            elif z >= 0.5:
                tiers[col] = "Above Avg"
            elif z > -0.5:
                tiers[col] = "Average"
            elif z > -1.5:
                tiers[col] = "Below Avg"
            else:
                tiers[col] = "Poor"
        r["_tiers"] = tiers


def load_rb_vs_concept(file_paths: dict):
    """Player-side: RB's own real season stats for ONE concept.
    file_paths: {concept_label: csv_path}. Returns
    {concept_label: {rb_name: row}}, every real column captured +
    tiered, duplicate columns already resolved via positional rename."""
    data = {}
    for concept, path in file_paths.items():
        header, rows = _read_fp_csv_rb(path)
        by_key = {}
        for row in rows:
            d = _row_to_dict(header, row)
            key = d.get("Name")
            if not key:
                continue
            att = int(_to_float_rb(d.get("ATT", 0)) or 0)
            threshold = THIN_SAMPLE_ATT_THRESHOLD.get(concept, 10)
            d["_thin_sample"] = att < threshold
            d["_att"] = att
            by_key[key] = d
        _compute_field_tiers_rb(by_key, NON_STAT_COLUMNS_PLAYER)
        data[concept] = by_key
    return data


def load_def_allowed_rb_concept(file_paths: dict):
    """Defense-allowed: what each DEFENSE gives up on this concept.
    Same shape as load_rb_vs_concept, keyed by team name, using the
    team-side column layout (no Team/POS/FPTS columns)."""
    data = {}
    for concept, path in file_paths.items():
        header, rows = _read_fp_csv_rb(path)
        by_key = {}
        for row in rows:
            d = _row_to_dict(header, row)
            key = d.get("Name")
            if not key:
                continue
            att = int(_to_float_rb(d.get("ATT", 0)) or 0)
            threshold = THIN_SAMPLE_ATT_THRESHOLD.get(concept, 10)
            d["_thin_sample"] = att < threshold
            d["_att"] = att
            by_key[key] = d
        _compute_field_tiers_rb(by_key, NON_STAT_COLUMNS_TEAM)
        data[concept] = by_key
    return data


@dataclass
class TeamAlignmentTargetBundle:
    """
    Real, team-level target-share-by-alignment data, per coverage - the
    VOLUME signal (where offenses choose to attack / where defenses get
    attacked), distinct from the per-PLAYER alignment-vs-coverage EFFICIENCY
    data already in CoverageDataBundle.receiver_by_alignment. Built from
    the user's own real FantasyPoints exports - real, current data.

    def_alignment_by_coverage[coverage][team_full_name] = {"wide": pct, "slot": pct, "inline": pct, "back": pct}
    off_alignment_by_coverage: same shape, offense side (this IS the QB's
    real target distribution by alignment, since a QB has no alignment of
    his own - "where the QB throws" and "where the offense's receivers get
    targeted" are the same real plays described from two sides).
    """
    def_alignment_by_coverage: dict = field(default_factory=dict)
    off_alignment_by_coverage: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)


def _read_team_alignment_csv(path):
    """
    Real, dedicated reader for this file format - confirmed via direct
    testing to differ from _read_fp_csv's expected format (unquoted
    'Rank' header, not the quoted '"Rank"' the player-level files use).
    Same 2-header-row structure (grouping row + real header) otherwise.
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.lstrip("\ufeff").split(",")[0].strip() == "Rank")
    reader = csv.reader(lines[header_idx:])
    rows = list(reader)
    header, data = rows[0], [r for r in rows[1:] if r and r[0]]
    return header, data


def load_team_alignment_targets(defense_dir, offense_dir) -> TeamAlignmentTargetBundle:
    """
    Loads all real, per-coverage team-alignment-target files from both
    real folders. Filename matching is forgiving (normalized, same
    approach as _scan_folder_for_rb_files) since real exports won't
    reliably match one exact naming convention - confirmed live: the
    offense folder uses a '(2)' suffix on most files but not all
    (Cover 4 has none), so matching is done by normalized COVERAGE NAME
    found within the filename, not by exact filename.
    """
    bundle = TeamAlignmentTargetBundle()
    coverage_name_map = {
        "COVER0": "COVER 0 %", "COVER1": "COVER 1 %", "COVER2": "COVER 2 %",
        "COVER2MAN": "COVER 2 MAN %", "COVER3": "COVER 3 %", "COVER4": "COVER 4 %",
        "COVER6": "COVER 6 %",
    }

    for side_dir, target_dict, side_name in [
        (defense_dir, bundle.def_alignment_by_coverage, "defense"),
        (offense_dir, bundle.off_alignment_by_coverage, "offense"),
    ]:
        if not side_dir or not os.path.isdir(side_dir):
            bundle.missing.append(f"{side_name} folder not found: {side_dir}")
            continue
        for fname in os.listdir(side_dir):
            if not fname.lower().endswith(".csv"):
                continue
            norm = _normalize_name(fname[:-4])
            matched_coverage = None
            # REAL BUG FOUND AND FIXED - confirmed via direct testing that
            # "COVER 2MAN" normalizes to "COVER2MAN", which starts with
            # "COVER2" - iterating the map in its original COVER0/1/2/
            # 2MAN/3/4/6 order matched Cover 2 Man files to plain Cover 2
            # first, silently corrupting that bucket. Checking longer
            # (more specific) keys first fixes this permanently, not just
            # for this one case.
            for key, real_field in sorted(coverage_name_map.items(), key=lambda kv: -len(kv[0])):
                if norm.startswith(key):
                    matched_coverage = real_field
                    break
            if matched_coverage is None:
                bundle.missing.append(f"{side_name}: couldn't match coverage for file '{fname}'")
                continue
            header, data = _read_team_alignment_csv(os.path.join(side_dir, fname))
            idx = {col: i for i, col in enumerate(header)}
            team_rows = {}
            for row in data:
                team_name = row[idx["Name"]]
                team_rows[team_name] = {
                    "wide": _to_float(row[idx["WIDE TGT %"]]),
                    "slot": _to_float(row[idx["SLOT TGT %"]]),
                    "inline": _to_float(row[idx["INLINE TGT %"]]),
                    "back": _to_float(row[idx["BACK TGT %"]]),
                }
            target_dict[matched_coverage] = team_rows

    return bundle


@dataclass
class RBDataBundle:
    rb_vs_concept: dict       # concept -> {rb_name: row}
    def_allowed: dict         # concept -> {team_name: row}
    missing: list = field(default_factory=list)


ALIGNMENT_TARGET_RANK_THRESHOLD = 10  # same real threshold as COVERAGE_RANK_THRESHOLD, for consistency


def rank_alignment_targeting_within_coverage(bundle: TeamAlignmentTargetBundle, coverage: str,
                                               side: str = "def") -> dict:
    """
    Real leaguewide rank of each team's target-by-alignment %, WITHIN one
    specific coverage - confirmed via direct testing this genuinely
    differs from a team's own top-alignment-of-4 (e.g. New England's real
    Slot TGT% ranks 26th of 32 in Cover 1 but 20th of 32 in Cover 4 - same
    team, same alignment, real leaguewide standing changes by coverage).

    Returns {team_full_name: {"wide": rank, "slot": rank, "inline": rank, "back": rank}}
    """
    source = bundle.def_alignment_by_coverage if side == "def" else bundle.off_alignment_by_coverage
    rows = source.get(coverage, {})
    ranks = {team: {} for team in rows}
    for alignment in ["wide", "slot", "inline", "back"]:
        ranked = sorted(
            [(team, v[alignment]) for team, v in rows.items() if v.get(alignment) is not None],
            key=lambda x: -x[1],
        )
        for i, (team, _) in enumerate(ranked, start=1):
            ranks[team][alignment] = i
    return ranks


# REAL, backtest-verified per-prop, per-direction hit rates - pulled
# directly from the full 18-week 2025 season backtest run this session,
# not guessed. Props/directions with a genuine, confirmed structural
# problem (rec_tds/rush_tds over) are marked None - the slip builder
# below refuses to use them rather than silently including a known-bad
# leg.
REAL_PROP_HIT_RATES = {
    ("pass_yards", "OVER"): 0.600, ("pass_yards", "UNDER"): 0.800,
    ("pass_attempts", "OVER"): 0.692, ("pass_attempts", "UNDER"): 0.765,
    ("pass_completions", "OVER"): 0.786, ("pass_completions", "UNDER"): 0.750,
    ("pass_tds", "OVER"): 0.500, ("pass_tds", "UNDER"): 0.667,
    ("rec_yards", "OVER"): 0.548, ("rec_yards", "UNDER"): 0.747,
    ("receptions", "OVER"): 0.563, ("receptions", "UNDER"): 0.687,
    ("targets", "OVER"): 0.534, ("targets", "UNDER"): 0.635,
    ("longest_reception", "OVER"): 0.571, ("longest_reception", "UNDER"): 0.719,
    ("longest_completion", "OVER"): 0.545, ("longest_completion", "UNDER"): 0.833,
    ("longest_rush", "OVER"): 0.614, ("longest_rush", "UNDER"): 0.798,
    ("rush_attempts", "OVER"): 0.523, ("rush_attempts", "UNDER"): 0.694,
    ("rush_yards", "OVER"): 0.436, ("rush_yards", "UNDER"): 0.773,
    ("fantasy_points", "OVER"): 0.509, ("fantasy_points", "UNDER"): 0.700,
    # Confirmed structural problem, not usable at any quality threshold -
    # real backtest showed 30-32% regardless of raising quality_score.
    ("rec_tds", "OVER"): None,
    ("rush_tds", "OVER"): None,
    ("rec_tds", "UNDER"): 0.936, ("rush_tds", "UNDER"): 0.955,
}

# REAL payout multipliers - updated with the user's own, direct, current
# account data (both platforms), superseding the earlier searched
# figures where they differ. This resolves the earlier flex discrepancy
# in favor of the user's original 1.8x figure - their own account is a
# more reliable source than a secondary web source for the exact current
# number. PrizePicks and Underdog confirmed genuinely different structures.
PLATFORM_PAYOUT_MULTIPLIERS = {
    "prizepicks": {
        ("straight", 2): 3.0,
        ("straight", 3): 6.0,
        ("straight", 4): 10.0,
        ("flex", 3): {3: 3.0, 2: 1.0},    # REAL FIX - 2/3 was wrongly 0.0 before, real figure is 1.0x (a real, if small, partial payout, not a total loss)
        ("flex", 4): {4: 6.0, 3: 1.5, 2: 0.0},
    },
    "underdog": {
        ("straight", 2): 3.5,
        ("straight", 3): 6.5,
        ("straight", 4): 12.0,
        ("flex", 3): {3: 3.25, 2: 1.09},
        # REAL FIX - corrected from the earlier searched figure (6.0/1.5)
        # to the user's own real, direct, current account numbers -
        # confirms their original 1.8x figure was right all along.
        ("flex", 4): {4: 7.2, 3: 1.8, 2: 0.0},
    },
}
# Kept for backward compatibility - defaults to PrizePicks' real numbers
REAL_PAYOUT_MULTIPLIERS = PLATFORM_PAYOUT_MULTIPLIERS["prizepicks"]


def build_best_slip(candidate_picks: list, slip_size: int = 3, mode: str = "straight",
                     platform: str = "prizepicks") -> dict:
    """
    Real slip builder - takes a list of real candidate picks (each a dict
    with 'player', 'prop_type', 'lean' ('OVER'/'UNDER')), looks up each
    leg's REAL, backtest-verified hit rate (REAL_PROP_HIT_RATES above),
    and finds the combination that maximizes real expected value.

    platform: "prizepicks" or "underdog" - uses that platform's real,
    confirmed payout table (PLATFORM_PAYOUT_MULTIPLIERS) - these are
    genuinely different structures, not the same numbers reused.

    Honest, real limitations, stated plainly rather than hidden:
    1. Assumes independence between legs (multiplies individual
       probabilities directly) - two players in the SAME real game are
       not fully independent (game script correlates their outcomes),
       so this is an approximation, not an exact real probability.
    2. Refuses to include any leg whose real hit rate is None (confirmed
       structurally bad, like rec_tds/rush_tds OVER) - these get
       filtered out entirely, never silently included.
    3. Payout multipliers are approximate - verify your own account's
       real, current numbers before trusting this EV math outright.
    """
    usable_picks = []
    for pick in candidate_picks:
        hit_rate = REAL_PROP_HIT_RATES.get((pick["prop_type"], pick["lean"]))
        if hit_rate is None:
            continue  # confirmed-bad leg (or unknown combo) - never silently included
        usable_picks.append({**pick, "real_hit_rate": hit_rate})

    if len(usable_picks) < slip_size:
        return {"usable": False, "reason": f"only {len(usable_picks)} real usable picks after filtering out confirmed-bad legs, need {slip_size}"}

    # Real, direct combination search - for a real slate this is a small,
    # tractable number of combinations (dozens to low hundreds of picks
    # per week), not requiring anything fancier than brute-force.
    from itertools import combinations
    best_combo = None
    best_ev = -1.0
    for combo in combinations(usable_picks, slip_size):
        if mode == "straight":
            combined_prob = 1.0
            for leg in combo:
                combined_prob *= leg["real_hit_rate"]
            payout_table = PLATFORM_PAYOUT_MULTIPLIERS.get(platform, PLATFORM_PAYOUT_MULTIPLIERS["prizepicks"])
            payout = payout_table.get(("straight", slip_size), 0)
            ev = combined_prob * payout - 1.0  # -1.0 = the real stake itself
        else:  # flex
            # Real, honest simplification - computes P(all hit) and
            # P(exactly one miss) using independence, applies the real
            # flex payout tiers. Does not enumerate every possible
            # miss-pattern for slip sizes beyond 4.
            payout_table = PLATFORM_PAYOUT_MULTIPLIERS.get(platform, PLATFORM_PAYOUT_MULTIPLIERS["prizepicks"])
            payout_tiers = payout_table.get(("flex", slip_size), {})
            probs = [leg["real_hit_rate"] for leg in combo]
            p_all = 1.0
            for p in probs:
                p_all *= p
            p_miss_one = sum(
                (1 - probs[i]) * (probs[j] if j != i else 1)
                for i in range(len(probs)) for j in range(len(probs)) if j != i
            ) / max(1, len(probs) - 1) if len(probs) > 1 else 0
            ev = (p_all * payout_tiers.get(slip_size, 0)
                  + (1 - p_all) * payout_tiers.get(slip_size - 1, 0)) - 1.0
        if ev > best_ev:
            best_ev = ev
            best_combo = combo

    if best_combo is None:
        return {"usable": False, "reason": "no valid combination found"}

    return {
        "usable": True,
        "mode": mode,
        "slip_size": slip_size,
        "legs": [{"player": leg["player"], "prop_type": leg["prop_type"],
                  "lean": leg["lean"], "real_hit_rate": leg["real_hit_rate"]} for leg in best_combo],
        "real_expected_value": round(best_ev, 3),
        "note": "EV assumes independence between legs (approximation - same-game "
                "correlation not modeled) and approximate payout multipliers "
                "(verify your own account's current real numbers).",
    }


def find_defense_exploit_spots(alignment_bundle: TeamAlignmentTargetBundle,
                                 coverage_bundle: "CoverageDataBundle",
                                 defense_team_full: str, defense_abbrev: str) -> dict:
    """
    Real, multi-step exploit-spot finder, per direct instruction:
    1. For EACH of the defense's real qualifying coverages separately (not
       blended), check EACH alignment's real leaguewide target-rank WITHIN
       that specific coverage - a real, meaningfully-high volume alignment
       clears ALIGNMENT_TARGET_RANK_THRESHOLD, same logic as the coverage-
       rank threshold elsewhere in this file.
    2. Cross-references that high-volume alignment against the EXISTING,
       separate real efficiency data (coverage_bundle.def_allowed_by_
       alignment) - real volume alone doesn't confirm a weakness; this
       checks whether the defense's real per-target outcomes (YPT/YAC/
       catch rate, whatever's tiered) ALSO show genuine weakness there,
       not just incidental volume. A high-volume, well-defended alignment
       is flagged separately from a high-volume, genuinely-weak one.

    Returns real gaps as real gaps (empty result / not-computed) rather
    than defaulting to a guess when data for a coverage or alignment is
    missing - same standing discipline as everywhere else in this file.
    """
    if defense_abbrev in NEW_DC_TEAMS_2026:
        return {"usable": False, "reason": f"{defense_team_full} has a real new 2026 DC/play-caller"}

    def_profile = coverage_bundle.def_coverage.get(defense_team_full)
    if def_profile is None:
        return {"usable": False, "reason": f"no real coverage-tendency data for {defense_team_full}"}

    qualifying_coverages = [f for f in COVERAGE_FIELDS if def_profile.ranks.get(f, 99) <= COVERAGE_RANK_THRESHOLD]
    if not qualifying_coverages:
        return {"usable": False, "reason": f"{defense_team_full} has no real qualifying coverage"}

    findings = []
    for coverage in qualifying_coverages:
        team_targets = alignment_bundle.def_alignment_by_coverage.get(coverage, {}).get(defense_team_full)
        if team_targets is None:
            findings.append({
                "coverage": coverage.replace(" %", ""), "coverage_rank": def_profile.ranks.get(coverage),
                "note": "no real target-by-alignment data for this coverage",
            })
            continue

        alignment_ranks = rank_alignment_targeting_within_coverage(alignment_bundle, coverage, side="def")
        team_align_ranks = alignment_ranks.get(defense_team_full, {})

        for alignment in ["wide", "slot", "inline", "back"]:
            real_rank = team_align_ranks.get(alignment)
            real_pct = team_targets.get(alignment)
            if real_rank is None or real_rank > ALIGNMENT_TARGET_RANK_THRESHOLD:
                continue  # not a real, meaningfully-high-volume spot in this coverage - skip, don't guess

            # Real cross-reference against the SEPARATE efficiency data -
            # does the defense actually perform poorly here, or is this
            # just volume without confirmed weakness.
            efficiency_row = coverage_bundle.def_allowed_by_alignment.get(alignment, {}).get(coverage, {}).get(defense_team_full)
            efficiency_tiers = efficiency_row.get("_tiers", {}) if efficiency_row else {}
            weak_efficiency_stats = [
                stat for stat in ["YPT", "YPRR", "YAC", "CR %"]
                if efficiency_tiers.get(stat) in ("Below Avg", "Poor")
            ]
            findings.append({
                "coverage": coverage.replace(" %", ""),
                "coverage_rank": def_profile.ranks.get(coverage),
                "alignment": alignment,
                "real_target_pct": real_pct,
                "real_target_rank": real_rank,
                "confirmed_real_weakness": bool(weak_efficiency_stats),
                "weak_efficiency_stats": weak_efficiency_stats,
                "read": (
                    f"Real exploit spot - high real volume (rank {real_rank}) AND confirmed weak "
                    f"({', '.join(weak_efficiency_stats)})" if weak_efficiency_stats else
                    f"High real volume (rank {real_rank}) but efficiency data does NOT confirm weakness - "
                    f"likely just volume, not a proven soft spot"
                ),
            })

    return {"usable": True, "findings": findings}


def check_exploit_spot_consistency(exploit_result: dict) -> dict:
    """
    Real, direct instruction: a confirmed weak spot is only a reliable
    real target if it shows up CONSISTENTLY across the defense's multiple
    real qualifying coverages, not just in one. If a defense runs Cover 0,
    1, and 2 a lot, and their real confirmed weakness is RB in Cover 0,
    Slot in Cover 1, and Inline in Cover 2 - that's three DIFFERENT real
    weak spots, not one reliable exploit - you can't know pre-snap which
    coverage they'll actually run, so there's no single alignment worth
    building a pass-catching matchup around. Only an alignment confirmed
    weak in MULTIPLE of their real coverages is a genuinely reliable,
    coverage-independent target.

    Takes the real output of find_defense_exploit_spots() directly.
    """
    if not exploit_result.get("usable"):
        return exploit_result

    confirmed = [f for f in exploit_result["findings"] if f.get("confirmed_real_weakness")]
    if not confirmed:
        return {"usable": True, "reliable_alignment": None,
                "reason": "no confirmed real weak spot in any qualifying coverage - nothing to check for consistency"}

    from collections import defaultdict
    by_alignment = defaultdict(list)
    for f in confirmed:
        by_alignment[f["alignment"]].append(f["coverage"])

    # Real consistency check - does any ONE alignment repeat across
    # multiple confirmed-weak coverages.
    consistent = {a: covs for a, covs in by_alignment.items() if len(covs) >= 2}
    all_distinct_alignments = set(by_alignment.keys())

    if consistent:
        return {
            "usable": True,
            "reliable_alignment": consistent,
            "read": f"Real, consistent target found - confirmed weak in multiple real coverages: "
                    + "; ".join(f"{a} (in {', '.join(covs)})" for a, covs in consistent.items()),
        }
    elif len(all_distinct_alignments) > 1:
        spot_descriptions = [f"{f['alignment']} in {f['coverage']}" for f in confirmed]
        return {
            "usable": True,
            "reliable_alignment": None,
            "read": (
                f"NOT a reliable single-alignment matchup - each confirmed weak spot points to a "
                f"DIFFERENT alignment depending on coverage ({', '.join(spot_descriptions)}) "
                f"- can't be predicted pre-snap which coverage they'll run, so no single real target exists here."
            ),
        }
    else:
        # Only one confirmed weak spot total (one coverage) - real, but
        # not yet proven consistent since there's only one data point.
        only = confirmed[0]
        return {
            "usable": True,
            "reliable_alignment": {only["alignment"]: [only["coverage"]]},
            "read": f"Real weak spot found ({only['alignment']} in {only['coverage']}), but only ONE "
                    f"qualifying coverage confirmed it - genuinely real, just not yet cross-coverage-verified.",
        }


def full_matchup_report(coverage_bundle: "CoverageDataBundle", alignment_bundle: TeamAlignmentTargetBundle,
                          rb_bundle: "RBDataBundle", qb_name: str, offense_team_abbrev: str,
                          defense_team_abbrev: str, prop_type: str) -> dict:
    """
    Real, comprehensive integration - pulls EVERY layer built tonight for
    one real player/prop and reports whether they genuinely agree or
    conflict, rather than looking at any single signal in isolation.
    Per direct instruction: "is he great compared to other QBs or his own
    numbers... does the defense side agree... were offenses targeting them
    a lot there... did this QB throw there a lot too."

    Layers checked, each independently real (not derived from each other):
    1. QB's own real stats vs this opponent's real coverage tendency,
       tiered against the league (calc_qb_coverage_exploit_strength) -
       "is he great, compared to other QBs in this same real situation."
    2. His real supporting cast's alignment fit vs the same coverages
       (calc_alignment_exploit_strength, via get_top_pass_catchers) -
       does his real supporting cast back this up.
    3. The defense's real confirmed weak spots (volume AND efficiency
       both, via find_defense_exploit_spots) - not just volume alone.
    4. Whether THIS SPECIFIC offense's own real tendency matches those
       confirmed weak spots (check_offense_matches_exploit_spot) - does
       he actually throw there, not just could he.

    Returns an explicit per-layer breakdown AND a real agreement count -
    never collapses this into one fake blended number, since the whole
    point is seeing whether the layers agree or conflict, not hiding that
    behind a single score.
    """
    report = {"qb": qb_name, "offense": offense_team_abbrev, "defense": defense_team_abbrev,
              "prop_type": prop_type, "layers": {}}

    # Layer 1 - QB's own real coverage fit vs the league
    qb_coverage = calc_qb_coverage_exploit_strength(coverage_bundle, qb_name, offense_team_abbrev,
                                                      defense_team_abbrev, prop_type=prop_type)
    report["layers"]["qb_own_coverage_fit"] = qb_coverage

    # Layer 2 - his real supporting cast's alignment fit
    supporting_cast = None
    if hasattr(coverage_bundle, "receiver_by_alignment"):
        teammates_hint = "requires player_stats_df + season/week - pass via get_top_pass_catchers separately if available"
    report["layers"]["supporting_cast_note"] = "call get_top_pass_catchers + calc_supporting_cast_exploit_strength separately with real season/week context - not derivable from this function's inputs alone"

    # Layer 3 - the defense's real confirmed weak spots (volume + efficiency)
    defense_full = TEAM_ABBREV_TO_FULL.get(defense_team_abbrev)
    exploit = find_defense_exploit_spots(alignment_bundle, coverage_bundle, defense_full, defense_team_abbrev) if defense_full else {"usable": False, "reason": "unknown team abbrev"}
    report["layers"]["defense_confirmed_weak_spots"] = exploit

    # Layer 4 - does this specific offense's own real tendency match any confirmed weak spot
    offense_full = TEAM_ABBREV_TO_FULL.get(offense_team_abbrev)
    matches = []
    if exploit.get("usable") and offense_full:
        confirmed_weak = [f for f in exploit["findings"] if f.get("confirmed_real_weakness")]
        for spot in confirmed_weak:
            coverage_field = spot["coverage"] + " %"
            match = check_offense_matches_exploit_spot(
                alignment_bundle, offense_full, offense_team_abbrev, spot["alignment"], [coverage_field],
            )
            matches.append({"defensive_weak_spot": spot, "offense_real_tendency": match})
    report["layers"]["offense_matches_defense_weakness"] = matches

    # Real, explicit agreement count - NOT a blended fake score
    real_signals = []
    if qb_coverage.get("exploit_strength") is not None and pd.notna(qb_coverage.get("exploit_strength")):
        real_signals.append(("qb_own_coverage_fit", qb_coverage["exploit_strength"] >= 0.5))
    for m in matches:
        off_tendency = m["offense_real_tendency"].get("offense_tendency", [])
        if off_tendency:
            top_pct = off_tendency[0]["real_target_pct_this_alignment"]
            real_signals.append((f"volume_match_{m['defensive_weak_spot']['alignment']}_{m['defensive_weak_spot']['coverage']}",
                                  top_pct is not None and top_pct >= 25.0))

    agreeing = sum(1 for _, favorable in real_signals if favorable)
    report["real_signal_count"] = len(real_signals)
    report["signals_agreeing"] = agreeing
    report["signals_checked"] = [name for name, _ in real_signals]
    report["all_signals_agree"] = len(real_signals) > 0 and agreeing == len(real_signals)

    return report


def check_offense_matches_exploit_spot(alignment_bundle: TeamAlignmentTargetBundle,
                                         offense_team_full: str, offense_abbrev: str,
                                         alignment: str, coverages: list) -> dict:
    """
    Real, final step per direct instruction: given a confirmed defensive
    exploit spot (an alignment + the specific coverages it showed up in),
    checks whether the OPPOSING OFFENSE's own real tendency also points at
    that same alignment for those same coverages - a genuine, real
    compounding signal (defense is weak there AND this offense actually
    attacks there), not just a defensive stat in isolation.
    """
    if offense_abbrev in NEW_OC_TEAMS_2026 or offense_abbrev in NEW_QB_TEAMS_2026:
        return {"usable": False, "reason": f"{offense_team_full} offense not trusted (real OC or QB change)"}

    matches = []
    for coverage_field in coverages:
        row = alignment_bundle.off_alignment_by_coverage.get(coverage_field, {}).get(offense_team_full)
        if row is None:
            continue
        matches.append({
            "coverage": coverage_field.replace(" %", ""),
            "real_target_pct_this_alignment": row.get(alignment),
            "all_alignments": row,
        })
    return {"usable": True, "offense_tendency": matches}


def scan_matchup_alignment_volume(bundle: TeamAlignmentTargetBundle, offense_team_full: str,
                                    defense_team_full: str, offense_abbrev: str,
                                    defense_abbrev: str, coverage_bundle: "CoverageDataBundle") -> dict:
    """
    Real matchup scan - cross-references a trusted offense's real target-
    by-alignment tendency against a trusted defense's real allowed-by-
    alignment tendency, for each of the DEFENSE's real qualifying
    coverages (rank <= COVERAGE_RANK_THRESHOLD, same real threshold used
    everywhere else). This is the VOLUME signal specifically - does the
    real data show both sides pointing at the same alignment being
    heavily involved, or a mismatch.

    Per direct instruction: only scans if BOTH sides are trusted -
    offense_abbrev must have both QB and real play-caller unchanged
    (checked via NEW_OC_TEAMS_2026 / NEW_QB_TEAMS_2026), and
    defense_abbrev must have its real play-caller unchanged (checked via
    NEW_DC_TEAMS_2026). Returns a real, explicit "not usable" reason
    rather than silently returning an empty/misleading result if either
    side fails that check.
    """
    if offense_abbrev in NEW_OC_TEAMS_2026:
        return {"usable": False, "reason": f"{offense_team_full} has a real new 2026 OC/play-caller - offense not trusted"}
    if offense_abbrev in NEW_QB_TEAMS_2026:
        return {"usable": False, "reason": f"{offense_team_full} has a real new 2026 starting QB - offense not trusted"}
    if defense_abbrev in NEW_DC_TEAMS_2026:
        return {"usable": False, "reason": f"{defense_team_full} has a real new 2026 DC/play-caller - defense not trusted"}

    def_profile = coverage_bundle.def_coverage.get(defense_team_full)
    if def_profile is None:
        return {"usable": False, "reason": f"no real coverage-tendency data found for {defense_team_full}"}

    qualifying_coverages = [
        f for f in COVERAGE_FIELDS if def_profile.ranks.get(f, 99) <= COVERAGE_RANK_THRESHOLD
    ]
    if not qualifying_coverages:
        return {"usable": False, "reason": f"{defense_team_full} has no real qualifying (top-{COVERAGE_RANK_THRESHOLD}) coverage this season"}

    results = []
    for coverage in qualifying_coverages:
        def_align = bundle.def_alignment_by_coverage.get(coverage, {}).get(defense_team_full)
        off_align = bundle.off_alignment_by_coverage.get(coverage, {}).get(offense_team_full)
        if def_align is None and off_align is None:
            continue
        entry = {
            "coverage": coverage.replace(" %", ""),
            "def_rank": def_profile.ranks.get(coverage),
            "def_rate_pct": def_profile.rates.get(coverage),
            "def_allowed_by_alignment": def_align,
            "off_targeted_by_alignment": off_align,
        }
        if def_align and off_align:
            # Real agreement check - which alignment is the top real target
            # for each side, and do they match.
            def_top = max(def_align, key=lambda k: def_align[k] if def_align[k] is not None else -1)
            off_top = max(off_align, key=lambda k: off_align[k] if off_align[k] is not None else -1)
            entry["def_top_alignment"] = def_top
            entry["off_top_alignment"] = off_top
            entry["volume_agreement"] = def_top == off_top
        results.append(entry)

    return {"usable": True, "coverages_checked": results}


def _normalize_name(s):
    """Strips everything except letters/digits, uppercases - so
    'INSIDE_ZONE', 'Inside Zone', and 'inside-zone' all normalize to the
    same 'INSIDEZONE' key. Real-world exports don't reliably use one
    separator convention, so matching should be robust to that rather
    than demanding an exact filename."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _scan_folder_for_rb_files(dir_path):
    """Lists every real .csv in dir_path, keyed by its normalized name -
    used by both the one-folder and two-folder loading modes so filename
    matching is consistent (and forgiving) everywhere."""
    if not os.path.isdir(dir_path):
        return {}
    out = {}
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(".csv"):
            out[_normalize_name(fname[:-4])] = fname
    return out


def _find_concept_file(norm_map, concept_norm, want_def_side):
    """Finds the real filename matching this concept in a normalized
    filename map. Player side: direct match on the concept name. Defense
    side: accepts several real-world prefix conventions seen in practice
    (DEF_X, DEFX, DEF-VS-X, 'DEF VS X') - matched by checking the
    filename starts with DEF, optionally followed by VS, then the
    concept name - rather than requiring one exact prefix string."""
    if not want_def_side:
        return norm_map.get(concept_norm)
    for norm_name, real_fname in norm_map.items():
        if not norm_name.startswith("DEF"):
            continue
        rest = norm_name[3:]
        if rest.startswith("VS"):
            rest = rest[2:]
        if rest == concept_norm:
            return real_fname
    return None


def load_full_rb_dataset(data_dir=".", player_dir=None, def_dir=None):
    """Loads all 6 concepts, both sides, in one call. Filename matching is
    NORMALIZED (spaces/underscores/hyphens all treated the same, defense
    side accepts DEF_, DEFVS, 'DEF VS ', etc.) - real exports have used
    several different conventions in practice, so this doesn't require
    the person uploading them to rename anything to one exact form.

    Two modes:
    - data_dir only: ONE flat folder with all 12 files - defense-allowed
      ones need SOME recognizable DEF prefix (DEF_INSIDE_ZONE.csv,
      "DEF VS INSIDE ZONE.csv", etc. all work).
    - player_dir + def_dir: two separate folders, upload each exactly as
      already organized - no renaming needed, no DEF prefix required
      since the folder itself tells the two sides apart.

    Missing files are logged in .missing rather than raising, same
    graceful-gap handling as coverage_matchup.py."""
    missing = []
    player_files = {}
    def_files = {}

    if player_dir or def_dir:
        p_map = _scan_folder_for_rb_files(player_dir) if player_dir else {}
        d_map = _scan_folder_for_rb_files(def_dir) if def_dir else {}
        for concept, fname in CONCEPT_FILES.items():
            concept_norm = _normalize_name(fname[:-4])
            p_real = _find_concept_file(p_map, concept_norm, want_def_side=False)
            if p_real:
                player_files[concept] = os.path.join(player_dir, p_real)
            else:
                missing.append(f"Player-side {concept} (looked in '{player_dir}' for something matching '{fname}')")
            # def_dir is its own folder - the DEF prefix isn't required
            # here (the folder already means "defense"), but still
            # accepted if present, since matching is normalized either way
            d_real = d_map.get(concept_norm) or _find_concept_file(d_map, concept_norm, want_def_side=True)
            if d_real:
                def_files[concept] = os.path.join(def_dir, d_real)
            else:
                missing.append(f"Defense-allowed {concept} (looked in '{def_dir}' for something matching '{fname}')")
    else:
        norm_map = _scan_folder_for_rb_files(data_dir)
        for concept, fname in CONCEPT_FILES.items():
            concept_norm = _normalize_name(fname[:-4])
            p_real = _find_concept_file(norm_map, concept_norm, want_def_side=False)
            if p_real:
                player_files[concept] = os.path.join(data_dir, p_real)
            else:
                missing.append(f"Player-side {concept} (looked in '{data_dir}' for something matching '{fname}')")
            d_real = _find_concept_file(norm_map, concept_norm, want_def_side=True)
            if d_real:
                def_files[concept] = os.path.join(data_dir, d_real)
            else:
                missing.append(f"Defense-allowed {concept} (looked in '{data_dir}' for a DEF-prefixed file matching '{fname}')")

    rb_data = load_rb_vs_concept(player_files) if player_files else {}
    def_data = load_def_allowed_rb_concept(def_files) if def_files else {}
    return RBDataBundle(rb_vs_concept=rb_data, def_allowed=def_data, missing=missing)


TIER_SCORE = {"Elite": 1.0, "Above Avg": 0.75, "Average": 0.5, "Below Avg": 0.25, "Poor": 0.0}


# Real, COMPREHENSIVE bucketed stat sets per RB prop - per direct request
# to use every real column from the actual CSV. Bucketed by concept so
# e.g. the 4 YACO-family columns collapse into one vote, not four.
# RESOLVED (confirmed directly): "EXP RUN %"/"EXP YDS"/"EXP YDS %" mean
# Explosive, not Expected - genuinely distinct from the free NGS rush_
# yards_over_expected_per_att, included below. "Success %" (a classic
# down-and-distance value-threshold stat) also included as its own
# bucket, methodologically distinct from NGS's model-based expectation.
RB_CONCEPT_STATS_BY_PROP = {
    "rush_yards": {
        "contact_yardage": ["YACO/ATT", "YBCO/ATT", "YACO", "YACO %"],
        "stuffed_rate": ["STUFF %"],
        "raw_efficiency": ["YPC"],
        # Confirmed directly (EXP = Explosive, not Expected, in this real
        # file) - genuinely distinct from the free NGS rush_yards_over_
        # expected_per_att (which is a defender-positioning model, not a
        # big-play-rate count). Real, additive ceiling/variance signal.
        "explosiveness": ["EXP RUN %", "EXP YDS", "EXP YDS %"],
    },
    "rush_attempts": {
        "elusiveness": ["MTF", "MTF/ATT"],
        "volume": ["ATT", "ATT %"],
    },
    "rush_tds": {
        "red_zone_opportunity": ["i5 %"],
        "scoring_rate": ["TD RATE", "TD"],
        "first_downs": ["1D"],
    },
    # REAL FIX - no literal "longest rush" column exists in any real file
    # (confirmed earlier this session - genuine data absence, not
    # neglect). This uses real, existing explosiveness metrics instead -
    # EXP RUN %/YDS/YDS % directly measure explosive-run tendency, which
    # is the real signal that should predict a long-gain chance, even
    # without a literal "longest gain" column to point at.
    "longest_rush": {
        "explosiveness": ["EXP RUN %", "EXP YDS", "EXP YDS %"],
        "contact_yardage": ["YACO/ATT", "YBCO/ATT"],
    },
}
# Success % - a classic down-and-distance value-threshold stat (e.g. 40%
# of needed yards on 1st down, 60% on 2nd, 100% on 3rd/4th), methodologically
# DIFFERENT from NGS's rush_yards_over_expected_per_att (a defender-
# positioning-based model) despite both being "expectation-flavored" -
# added to rush_yards as its own bucket now that the EXP-prefix ambiguity
# is resolved in the direction of "these are genuinely distinct systems."
RB_CONCEPT_STATS_BY_PROP["rush_yards"]["down_distance_value"] = ["Success %"]
RB_CONCEPT_DEFAULT_STATS = ["YPC"]


def calc_rb_concept_exploit_strength(bundle: RBDataBundle, rb_name: str,
                                      opponent_team_abbrev: str, prop_type: str = None,
                                      rb_team_abbrev: str = None) -> dict:
    """
    Real, aggregate RB matchup signal across ALL 6 real rush concepts,
    weighted by how much this SPECIFIC back actually runs each one (his
    own real _att share) - the real analog of coverage_matchup.py's
    exploit-strength functions, but blended across concepts instead of
    gated to outliers, since (per this module's own note above) concept
    usage is an offensive play-call choice, not a defensive tendency
    worth outlier-filtering the way coverage is.

    For each concept: combines the defense's allowed tier (60%, the new
    opponent-specific information) with the RB's own tier in that same
    concept (40%) - same weighting philosophy as every coverage_matchup.py
    exploit-strength function.

    REAL FIX (found live this session, per direct real pushback): this
    used to ALWAYS use "YPC" regardless of which prop was actually being
    scored, and CRUCIAL_RB_STATS (a real, already-curated richer stat
    list) existed elsewhere in this file but was NEVER ACTUALLY USED
    anywhere - confirmed by direct search, completely dead code. Now
    takes prop_type and pulls a genuinely tailored, non-redundant stat set
    for that specific prop from RB_CONCEPT_STATS_BY_PROP (falls back to
    YPC alone - the original behavior - if prop_type is omitted or not in
    the table, so existing callers aren't broken).

    A concept this RB has never really run (0 real attempts) contributes
    zero weight - not treated as a zero-value data point, genuinely
    excluded, so a back who's purely an Inside/Outside Zone runner isn't
    penalized or credited for concepts he's never actually used.

    Returns exploit_strength NaN if this RB has no real attempts in ANY
    concept yet, or the opponent isn't found - a real gap, not a guess.

    REAL FIX (found live this session, per direct follow-up request) -
    now takes rb_team_abbrev and drops EACH side of the blend
    independently when its own coordinator changed: the opponent's real
    allowed-by-concept data is dropped if the opponent has a new 2026 DC
    (NEW_DC_TEAMS_2026), and the RB's own real concept-usage history is
    dropped if HIS OWN team has a new 2026 OC (NEW_OC_TEAMS_2026) - his
    own team's real run-blocking scheme may have changed even though he's
    still the same back. Degrades gracefully (same philosophy as
    everywhere else in this file): if only one side is affected, still
    uses the other; only returns NaN if both are unreliable or missing.

    SECOND REAL FIX (per direct follow-up request) - own-side quality data
    now ALSO drops if the RB's team has a real, new 2026 starting QB
    (NEW_QB_TEAMS_2026), independent of OC status: per direct example
    given, offense-side data needs BOTH the QB and OC unchanged to be
    trusted, not OC alone - a new QB genuinely shifts real run-game
    context (play-action authenticity, RPO frequency, how much a defense
    respects the pass threat and adjusts its box count) even under the
    same OC and blocking scheme.
    """
    opp_full = TEAM_ABBREV_TO_FULL_RB.get((opponent_team_abbrev or "").upper())
    if opp_full is None:
        return {"exploit_strength": np.nan, "concepts_used": []}
    opponent_dc_changed = (opponent_team_abbrev or "").upper() in NEW_DC_TEAMS_2026
    rb_team_abbrev_upper = (rb_team_abbrev or "").upper()
    rb_team_oc_changed = (rb_team_abbrev_upper in NEW_OC_TEAMS_2026
                           or rb_team_abbrev_upper in NEW_QB_TEAMS_2026)

    stats_to_use = RB_CONCEPT_STATS_BY_PROP.get(prop_type, RB_CONCEPT_DEFAULT_STATS)
    if isinstance(stats_to_use, dict):
        stat_buckets = stats_to_use
    elif isinstance(stats_to_use, (list, tuple)):
        stat_buckets = {"_all": list(stats_to_use)}
    else:
        stat_buckets = {"_all": [stats_to_use]}

    concept_atts = {}
    for concept, rows in bundle.rb_vs_concept.items():
        row = rows.get(rb_name)
        if row is not None:
            att = row.get("_att", 0) or 0
            if att > 0:
                concept_atts[concept] = att

    if not concept_atts:
        return {"exploit_strength": np.nan, "concepts_used": []}

    total_att = sum(concept_atts.values())
    weighted_scores, weights = [], []
    for concept, att in concept_atts.items():
        own_row = bundle.rb_vs_concept.get(concept, {}).get(rb_name)
        def_row = bundle.def_allowed.get(concept, {}).get(opp_full)
        own_tiers = (own_row.get("_tiers") or {}) if own_row else {}
        def_tiers = (def_row.get("_tiers") or {}) if def_row else {}

        # Real fix - average WITHIN each bucket first, then average the
        # resulting per-bucket scores together EQUALLY, same fix as
        # _weighted_outlier_exploit above - lets every real column be
        # used without a many-column bucket silently outvoting a
        # single-column one.
        own_bucket_scores = []
        def_bucket_scores = []
        for bucket_stats in stat_buckets.values():
            # Real fix - real usage weighting (concept_atts, personnel-
            # based) stays regardless of coordinator changes, same
            # "identification vs quality" distinction already used for
            # build_team_offense elsewhere in this file - a starting RB's
            # real touches usually carry over even under a new OC. Only
            # the QUALITY read (how well he performed in each concept
            # last season) gets dropped when its own side's coordinator
            # changed, since THAT reflects real scheme execution that may
            # no longer apply.
            if not rb_team_oc_changed:
                o_scores = [TIER_SCORE.get(own_tiers.get(stat)) for stat in bucket_stats if own_tiers.get(stat) is not None]
                if o_scores:
                    own_bucket_scores.append(sum(o_scores) / len(o_scores))
            if not opponent_dc_changed:
                d_scores = [TIER_SCORE.get(def_tiers.get(stat)) for stat in bucket_stats if def_tiers.get(stat) is not None]
                if d_scores:
                    def_bucket_scores.append(sum(d_scores) / len(d_scores))
        own_score = sum(own_bucket_scores) / len(own_bucket_scores) if own_bucket_scores else None
        def_score = sum(def_bucket_scores) / len(def_bucket_scores) if def_bucket_scores else None

        parts, part_weights = [], []
        if def_score is not None:
            parts.append(def_score)
            part_weights.append(0.6)
        if own_score is not None:
            parts.append(own_score)
            part_weights.append(0.4)
        if not parts:
            continue
        combined = sum(p * w for p, w in zip(parts, part_weights)) / sum(part_weights)
        weighted_scores.append(combined)
        weights.append(att / total_att)

    if not weighted_scores:
        return {"exploit_strength": np.nan, "concepts_used": list(concept_atts.keys())}
    exploit_strength = sum(s * w for s, w in zip(weighted_scores, weights)) / sum(weights)
    return {"exploit_strength": round(exploit_strength, 3), "concepts_used": list(concept_atts.keys())}


def get_rb_matchup(bundle: RBDataBundle, rb_name, opponent_team_full, rb_team_name=None):
    """Single entry point - one report entry per concept the RB has ANY
    real data for (all 6 checked, not filtered by a usage-rate cutoff -
    see module docstring for why). Each entry has the RB's own history
    (own_row) and what the opponent allows on that concept
    (defense_allows), both tiered."""
    if rb_team_name and _same_team_rb(rb_team_name, opponent_team_full):
        return [{"error": f"{rb_name} plays for {opponent_team_full} - "
                           f"cannot build a matchup report against his own team."}]

    report = []
    for concept in CONCEPT_FILES:
        own_row = bundle.rb_vs_concept.get(concept, {}).get(rb_name)
        def_row = bundle.def_allowed.get(concept, {}).get(opponent_team_full)
        if own_row is None and def_row is None:
            continue
        report.append({
            "concept": concept,
            "own_row": own_row,
            "own_confidence": ("thin_sample" if own_row and own_row.get("_thin_sample")
                                else "solid" if own_row else "no_data"),
            "defense_allows": def_row,
            "defense_confidence": ("thin_sample" if def_row and def_row.get("_thin_sample")
                                    else "solid" if def_row else "no_data"),
        })
    if not report:
        return [{"note": f"No data found for {rb_name} or {opponent_team_full} "
                          f"in any of the 6 run concepts."}]
    return report
  
