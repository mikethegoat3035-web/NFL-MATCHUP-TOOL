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

try:
    from coverage_matchup import (
        calc_alignment_exploit_strength, calc_qb_coverage_exploit_strength,
        TEAM_ABBREV_TO_FULL as ALIGNMENT_TEAM_MAP,
    )
except ImportError:
    # Premium alignment module not present in this deploy - both signals
    # degrade to fully absent (their flags are also off by default, so
    # this only matters if someone flips a flag on without the file
    # actually being deployed alongside this one).
    calc_alignment_exploit_strength = None
    calc_qb_coverage_exploit_strength = None
    ALIGNMENT_TEAM_MAP = {}

try:
    from rb_matchup import calc_run_concept_exploit_strength
except ImportError:
    # Same graceful-absence treatment as the alignment import above -
    # ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE is off by default.
    calc_run_concept_exploit_strength = None


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
    df = nfl.load_injuries(seasons=years)
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


def build_player_coverage_efficiency(player_gsis_id: str, role: str, season: int,
                                      participation_df: pd.DataFrame, pbp_df: pd.DataFrame,
                                      min_plays_per_bucket: int = 8, current_team: str = None,
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

    REVIVED (min_plays_per_bucket back to 8, was raised to 14): confirmed
    via a real 2025 backtest's full raw export that at 14, this fired on
    0 of 2,521 eligible rows all season - requiring 14+ real plays against
    BOTH man AND zone specifically for one player is too strict a bar to
    ever clear, even with the cross-season top-up above. The 14 threshold
    was raised speculatively to fix adjustment_direction_accuracy (stuck
    at 47%) - but a later isolated test proved that number was being
    driven ENTIRELY by the box-count adjustment (833 of 833 non-trivial
    adjustments were box, zero were coverage), so tightening THIS
    function's threshold was never actually addressing the real cause.

    REAL ROOT CAUSE FOUND (the actual reason for the 0% fire rate, not the
    threshold at all): the man/zone bucket matching below used to compare
    against hardcoded TITLE-CASE strings ("Man"/"Zone"), while the
    CONFIRMED-WORKING build_coverage_profile() function (whose real output
    - opp_man_pct/opp_zone_pct - has shown correct real percentages in
    every backtest export all session) pivots dynamically on whatever raw
    values actually exist, with no hardcoded casing assumption at all -
    strong indirect evidence the real column's values don't match "Man"/
    "Zone" exactly. This means the 0% fire rate was NEVER actually about
    sample size - even the ORIGINAL threshold of 8 (before any of tonight's
    tuning) would have produced 0% for this same reason. Fixed to match
    case-insensitively (.str.lower() == "man"/"zone") so it can't silently
    fail on a casing assumption again, whatever the real casing turns out
    to be.

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
        return len(plays_df[plays_df["defense_man_zone_type"].str.lower() == coverage_type])

    # REAL VALUES CONFIRMED via live diagnostic (diagnose_participation_data(),
    # run against real 2025 week 8 data): defense_man_zone_type's actual real
    # values are "MAN_COVERAGE" / "ZONE_COVERAGE" (with underscore) - NOT
    # "man"/"zone" or "Man"/"Zone", both of which were guessed and both wrong
    # across two earlier fix attempts tonight (confirmed 0% fire rate either
    # way). This is the first version matched against real, directly-observed
    # data instead of an assumption.
    man_n = _bucket_count(player_plays, "man_coverage")
    zone_n = _bucket_count(player_plays, "zone_coverage")

    # top up with team-filtered prior-season plays if either bucket is short
    if (man_n < min_plays_per_bucket or zone_n < min_plays_per_bucket) \
            and current_team is not None and prior_participation_df is not None and prior_pbp_df is not None:
        prior_plays = _get_player_plays(prior_participation_df, prior_pbp_df)
        prior_plays = prior_plays[prior_plays["posteam"] == current_team]
        player_plays = pd.concat([player_plays, prior_plays])

    def _bucket_avg(coverage_type):
        bucket = player_plays[player_plays["defense_man_zone_type"].str.lower() == coverage_type]
        n = len(bucket)
        if n < min_plays_per_bucket:
            return np.nan, n
        yards_col = "receiving_yards" if role == "receiver" else "passing_yards"
        return round(bucket[yards_col].mean(), 2), n

    man_avg, man_n = _bucket_avg("man_coverage")
    zone_avg, zone_n = _bucket_avg("zone_coverage")
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
                                           min_plays_per_type: int = 8) -> dict:
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

    Returns {coverage_type: {"ypp": float, "n_plays": int}, ...} for
    types clearing min_plays_per_type, plus "overall_ypp"/"overall_plays"
    keys holding this player's real overall average across every play
    regardless of coverage type (the baseline the multiplier compares
    against).
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
    Generalizes calc_coverage_adjusted_mu's exact multiplier logic (real
    per-player split x this week's opponent tendency, capped adjustment)
    from 2 buckets (man/zone) to EVERY coverage type both sides have
    reliable real data for - the fallback the user specifically asked
    for: a defense that plays 3 real coverages this season where the
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
ENABLE_QB_COVERAGE_IN_QUALITY_SCORE = False

# RB run-concept exploit signal, sourced from rb_matchup.py's premium
# FantasyPoints dataset (calc_run_concept_exploit_strength). STAYS OFF
# until alignment AND QB coverage are each confirmed clean - thinnest
# real samples of the three (Counter/Power/Pull Lead), tested last on
# purpose. Round 3.
ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE = False

# RE-ENABLE TEST (this round's ONE change, everything else held constant):
# play-action was disabled after landing untested alongside 4 other changes
# in one round, which caused a severe quality_score tier inversion never
# individually attributed to PA specifically vs personnel vs the other
# changes. Since then: the reweighting fix, mu/sigma shrinkage fix, quality_
# score sample-size fix, and the coverage-adjustment casing/root-cause
# fixes have all been tested and landed cleanly with PA still off. Testing
# PA alone now, with personnel and both mu-adjustments still off, so any
# change in results this round can be attributed to PA specifically.
# GATED per real 2025 data: the box-count mu adjustment was found to be
# net-harmful, not just weak - direction accuracy of 47% overall, and it
# got WORSE (down to 42.5%) as the adjustment size grew, the opposite of
# what a real signal should do. It also turned out to be the SOLE driver
# of adjustment_direction_accuracy across the whole report (833 of 833
# non-trivial adjustments were box, zero were coverage - see note on
# min_plays_per_bucket below). Disabled from actually moving mu until
# re-diagnosed; still computed and shown via mu_before_box_adj for
# comparison.
ENABLE_BOX_MU_ADJUSTMENT = False
# GATED per real 2025 data (once the mechanism was finally debugged to
# actually fire - two real bugs found and fixed: build_player_coverage_
# efficiency was matching the wrong coverage-type strings, and upstream of
# that, build_coverage_profile was producing man_pct/zone_pct as
# permanently-null columns due to the same underlying value mismatch, so
# the adjustment's own gate check never once passed all session): once
# correctly firing (82.2% of eligible rows), direction accuracy came back
# at 48.7% - a coinflip, essentially identical to box's 47%. Two
# INDEPENDENT situational-split mechanisms (coverage type, box count) both
# landing at coinflip accuracy on real, CORRECTLY FUNCTIONING code is
# converging evidence about the underlying idea itself (a player's
# situational split x opponent's situational tendency, turned into an
# actual mu multiplier), not a remaining bug in either implementation.
# Disabled from moving mu for the same reason as box; still computed and
# shown via mu_before_coverage_adj for comparison.
ENABLE_COVERAGE_MU_ADJUSTMENT = False

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
                                   matchup_weight: float = 0.15) -> float:
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


def build_weekly_slate(season: int, week: int, coverage_bundle=None, rb_bundle=None,
                        team_filter: list = None) -> pd.DataFrame:
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

    # Per-game longest-play tables for the new longest_completion/
    # longest_reception/longest_rush props (see build_longest_play_by_game).
    # Built once here, current season only (weeks before target week) -
    # NOTE: unlike calc_prop_mu's own_stats path, these are NOT bridged to
    # a prior-season fallback (that would need prior_pbp_df run through
    # the same aggregation too) - an intentional scope limit for this
    # first pass, so Week 1-2 rows for these 3 props will more often come
    # back NaN (flagged low-confidence, not guessed) than the yardage
    # props do. Revisit if that proves too big a gap in practice.
    qb_longest_df = build_longest_play_by_game(pbp_history_df, "QB")
    rec_longest_df = build_longest_play_by_game(pbp_history_df, "WR")
    rush_longest_df = build_longest_play_by_game(pbp_history_df, "RB")

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
                coverage_bundle, qb.get("full_name"), team, opponent,
            )
        qb_coverage_exploit_for_scoring = qb_coverage_info.get("exploit_strength") if ENABLE_QB_COVERAGE_IN_QUALITY_SCORE else np.nan

        structural_parts = [v for v in [coverage_info.get("exploit_strength"), pa_exploit_for_scoring,
                                         qb_coverage_exploit_for_scoring] if pd.notna(v)]
        combined_structural_exploit = (sum(structural_parts) / len(structural_parts)) if structural_parts else np.nan

        # ACTUAL mu adjustment (not just a quality_score side signal) using
        # this QB's own real man/zone efficiency split from their play
        # history, weighted by this specific opponent's man/zone tendency.
        # GATED per ENABLE_COVERAGE_MU_ADJUSTMENT (see flag note above) -
        # still computed below when the gate check passes, so mu_before_
        # coverage_adj stays available for comparison, just not applied.
        adjusted_mu = mu
        if ENABLE_COVERAGE_MU_ADJUSTMENT and pd.notna(mu) and opp_coverage_row:
            man_pct = opp_coverage_row.get("man_pct")
            zone_pct = opp_coverage_row.get("zone_pct")
            if pd.notna(man_pct) and pd.notna(zone_pct):
                coverage_eff = build_player_coverage_efficiency(
                    gsis_id, "passer", season, participation_df, pbp_history_df,
                    current_team=team, prior_participation_df=prior_participation_df,
                    prior_pbp_df=prior_pbp_df,
                )
                adjusted_mu = calc_coverage_adjusted_mu(mu, coverage_eff, man_pct, zone_pct)

        # NEW, SEPARATE full-coverage-type version - GATED per
        # ENABLE_FULL_COVERAGE_MU_ADJUSTMENT, off by default pending its
        # own live test. See rec_yards block for the full rationale.
        full_coverage_weight_used = 0.0
        if ENABLE_FULL_COVERAGE_MU_ADJUSTMENT and pd.notna(mu) and opp_coverage_row:
            player_full_coverage_eff = build_player_full_coverage_efficiency(
                gsis_id, "passer", participation_df, pbp_history_df,
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
        role_trend = build_role_trend(gsis_id, "attempts", ngs_pass_df, "player_gsis_id", season, week)
        role_score = calc_role_verification_score(role_trend)
        blended_exploit = calc_blended_matchup_strength(
            combined_structural_exploit, grade_exploit, role_score
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
                sib_blended = calc_blended_matchup_strength(combined_structural_exploit, sib_grade_exploit, role_score)
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

        # Longest completion - own-history-only (see qb_longest_df note
        # above: no prior-season bridge yet), so min_games gates it more
        # often than the other props for thin-sample QBs.
        longest_mu = calc_prop_mu(gsis_id, "longest_play", qb_longest_df, season, week, current_team=None)
        longest_sigma = calc_player_sigma(gsis_id, "longest_play", qb_longest_df, season, week, current_team=None)
        rows.append({
            "gsis_id": gsis_id, "player_display_name": qb.get("full_name"),
            "team": team, "position": "QB", "prop_type": "longest_completion",
            "matchup": team_to_matchup.get(team),
            "mu": longest_mu, "sigma": longest_sigma, "opponent": opponent,
            "quality_score": quality_score,
            "data_confidence": confidence_info["data_confidence"],
            "games_sampled_current": confidence_info["games_sampled_current"],
        })

    # --- Rushing props ---
    rush_pool = week_rosters[week_rosters["position"].isin(["RB", "QB"])]
    for _, rb in rush_pool.iterrows():
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

            # ACTUAL mu adjustment using this RB's own real light-vs-stacked
            # box yards-per-carry split, weighted by this week's opponent's
            # stacked-box rate - run-game equivalent of the QB/WR coverage
            # adjustment above. GATED per ENABLE_BOX_MU_ADJUSTMENT (see flag
            # note above) - still computed below so mu_before_box_adj stays
            # available for comparison, just not applied to mu itself.
            adjusted_rush_mu = mu
            if ENABLE_BOX_MU_ADJUSTMENT and opp_box_row and pd.notna(box_info.get("box_stack_pct")):
                box_eff = build_player_rush_box_efficiency(
                    gsis_id, season, ftn_df, pbp_history_df,
                    current_team=rb_team, prior_ftn_df=prior_ftn_df, prior_pbp_df=prior_pbp_df,
                )
                adjusted_rush_mu = calc_box_adjusted_mu(mu, box_eff, box_info.get("box_stack_pct"))

            # Run-concept exploit signal - premium data, only computed when
            # a bundle was actually passed in AND the flag is on (see
            # calc_run_concept_exploit_strength in rb_matchup.py for the
            # real logic). GATED same as every other premium/isolated
            # signal - off by default pending its own live test. Position
            # check mirrors rush_pool's own RB/QB filter (QBs rarely have
            # FantasyPoints run-concept rows, so this will naturally
            # degrade to NaN for most QB rush_yards rows).
            run_concept_info = {"exploit_strength": np.nan, "concepts_checked": []}
            if (ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE and rb_bundle is not None
                    and calc_run_concept_exploit_strength is not None and rb_opponent is not None):
                run_concept_info = calc_run_concept_exploit_strength(
                    rb_bundle, rb.get("full_name"), rb_team, rb_opponent,
                )
            run_concept_exploit_for_scoring = run_concept_info.get("exploit_strength") if ENABLE_RUN_CONCEPT_IN_QUALITY_SCORE else np.nan

            rb_confidence_info = get_data_confidence(gsis_id, player_stats_df, season, week, current_team=rb_team)
            own_grades = get_player_grades(gsis_id, rb_metrics)
            def_grades = get_defense_grades(rb_opponent, def_metrics)

            grade_exploit = calc_grade_matchup_strength({**own_grades, **def_grades}, "rush_yards")
            role_trend = build_role_trend(gsis_id, "rush_attempts", ngs_rush_df, "player_gsis_id", season, week)
            role_score = calc_role_verification_score(role_trend)
            structural_parts = [v for v in [box_info.get("exploit_strength"), run_concept_exploit_for_scoring]
                                 if pd.notna(v)]
            combined_rush_structural = (sum(structural_parts) / len(structural_parts)) if structural_parts else np.nan
            blended_exploit = calc_blended_matchup_strength(
                combined_rush_structural, grade_exploit, role_score
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
                    sib_blended = calc_blended_matchup_strength(combined_rush_structural, sib_grade_exploit, role_score)
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

            longest_rush_mu = calc_prop_mu(gsis_id, "longest_play", rush_longest_df, season, week, current_team=None)
            longest_rush_sigma = calc_player_sigma(gsis_id, "longest_play", rush_longest_df, season, week, current_team=None)
            rows.append({
                "gsis_id": gsis_id, "player_display_name": rb.get("full_name"),
                "team": rb.get("team"), "position": position, "prop_type": "longest_rush",
                "matchup": team_to_matchup.get(rb_team),
                "mu": longest_rush_mu, "sigma": longest_rush_sigma, "opponent": rb_opponent,
                "quality_score": rush_quality_score,
                "data_confidence": rb_confidence_info["data_confidence"],
                "games_sampled_current": rb_confidence_info["games_sampled_current"],
            })

    # --- Receiving props ---
    rec_pool = week_rosters[week_rosters["position"].isin(["WR", "TE", "RB"])]
    for _, wr in rec_pool.iterrows():
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
                    coverage_bundle, wr.get("full_name"), position, team, opponent,
                )
            alignment_exploit_for_scoring = alignment_info.get("exploit_strength") if ENABLE_ALIGNMENT_IN_QUALITY_SCORE else np.nan

            structural_parts = [v for v in [coverage_info.get("exploit_strength"), personnel_exploit_for_scoring,
                                             alignment_exploit_for_scoring] if pd.notna(v)]
            combined_structural_exploit = (sum(structural_parts) / len(structural_parts)) if structural_parts else np.nan

            # ACTUAL mu adjustment using this receiver's own real man/zone
            # efficiency split, weighted by this specific opponent's tendency.
            # GATED per ENABLE_COVERAGE_MU_ADJUSTMENT (see flag note above).
            adjusted_mu = mu
            man_pct = opp_coverage_row.get("man_pct") if opp_coverage_row else None
            zone_pct = opp_coverage_row.get("zone_pct") if opp_coverage_row else None
            if ENABLE_COVERAGE_MU_ADJUSTMENT and pd.notna(man_pct) and pd.notna(zone_pct):
                coverage_eff = build_player_coverage_efficiency(
                    gsis_id, "receiver", season, participation_df, pbp_history_df,
                    current_team=team, prior_participation_df=prior_participation_df,
                    prior_pbp_df=prior_pbp_df,
                )
                adjusted_mu = calc_coverage_adjusted_mu(mu, coverage_eff, man_pct, zone_pct)

            # NEW, SEPARATE full-coverage-type version - GATED per
            # ENABLE_FULL_COVERAGE_MU_ADJUSTMENT, off by default pending
            # its own live test. Applied on top of adjusted_mu (which is
            # just `mu` unchanged while the man/zone version stays off) so
            # this can be tested in isolation regardless of that flag's state.
            full_coverage_weight_used = 0.0
            if ENABLE_FULL_COVERAGE_MU_ADJUSTMENT and opp_coverage_row:
                player_full_coverage_eff = build_player_full_coverage_efficiency(
                    gsis_id, "receiver", participation_df, pbp_history_df,
                )
                full_cov_result = calc_full_coverage_adjusted_mu(adjusted_mu, player_full_coverage_eff, opp_coverage_row)
                adjusted_mu = full_cov_result["adjusted_mu"]
                full_coverage_weight_used = full_cov_result["coverage_weight_used"]

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
                    sib_blended = calc_blended_matchup_strength(combined_structural_exploit, sib_grade_exploit, role_score)
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

            longest_rec_mu = calc_prop_mu(gsis_id, "longest_play", rec_longest_df, season, week, current_team=None)
            longest_rec_sigma = calc_player_sigma(gsis_id, "longest_play", rec_longest_df, season, week, current_team=None)
            rows.append({
                "gsis_id": gsis_id, "player_display_name": wr.get("full_name"),
                "team": team, "position": position, "prop_type": "longest_reception",
                "matchup": team_to_matchup.get(team),
                "mu": longest_rec_mu, "sigma": longest_rec_sigma, "opponent": opponent,
                "quality_score": quality_score,
                "data_confidence": rec_confidence_info["data_confidence"],
                "games_sampled_current": rec_confidence_info["games_sampled_current"],
            })

    # --- Fantasy points (offense: QB, RB, WR, TE) ---
    offense_positions = ["QB", "RB", "WR", "TE"]
    fantasy_pool_roster = week_rosters[week_rosters["position"].isin(offense_positions)]
    for _, pr in fantasy_pool_roster.iterrows():
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


def score_week_against_actuals(season: int, week: int, starters_only: bool = True, coverage_bundle=None, rb_bundle=None) -> pd.DataFrame:
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
    slate_df = build_weekly_slate(season, week, coverage_bundle=coverage_bundle, rb_bundle=rb_bundle)
    player_stats_df = pull_player_stats([season])
    depth_charts_df = pull_depth_charts([season]) if nfl else pd.DataFrame()
    schedules_df = pull_schedules([season])
    pbp_df = pull_pbp([season])

    actual_week = player_stats_df[
        (player_stats_df["season"] == season) & (player_stats_df["week"] == week)
    ].set_index("gsis_id")

    prop_to_stat_column = {
        "pass_yards": "passing_yards",
        "rush_yards": "rushing_yards",
        "rec_yards": "receiving_yards",
        "fantasy_points": "fantasy_points_ppr",
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

    Returns columns: gsis_id, season, week, longest_play.
    """
    position = position.upper()
    if position == "RB":
        id_col, want_play_type = "rusher_player_id", "run"
    elif position == "QB":
        id_col, want_play_type = "passer_player_id", "pass"
    else:
        id_col, want_play_type = "receiver_player_id", "pass"

    required = {id_col, "yards_gained", "season", "week", "play_type"}
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
        return pd.DataFrame(columns=["gsis_id", "season", "week", "longest_play"])

    agg = sub.groupby([id_col, "season", "week"])["yards_gained"].max().reset_index()
    return agg.rename(columns={id_col: "gsis_id", "yards_gained": "longest_play"})


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
