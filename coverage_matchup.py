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

import csv
import os
import re
from dataclasses import dataclass, field
from statistics import mean, pstdev

import numpy as np

COVERAGE_FIELDS = ["COVER 0 %", "COVER 1 %", "COVER 2 %", "COVER 2 MAN %",
                   "COVER 3 %", "COVER 4 %", "COVER 6 %"]

ALWAYS_THIN_COVERAGES = {"COVER 0 %", "COVER 2 MAN %"}

THIN_SAMPLE_ATT_THRESHOLD = {
    "COVER 0 %": 5, "COVER 1 %": 15, "COVER 2 %": 15, "COVER 2 MAN %": 5,
    "COVER 3 %": 20, "COVER 4 %": 15, "COVER 6 %": 10,
}

OUTLIER_Z_THRESHOLD = 1.0

# Stats where a HIGHER number is worse for the QB (need to flip tiering direction)
INVERSE_STATS = {"INT", "SACK", "SACK %", "SK YDS", "DROP %", "DROP YDS",
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

    for p in profiles.values():
        for f in COVERAGE_FIELDS:
            avg, sd = league_stats[f]
            p.z_scores[f] = (p.rates[f] - avg) / sd if sd else 0.0
        p.outliers = sorted(
            [(f, z) for f, z in p.z_scores.items() if z >= OUTLIER_Z_THRESHOLD],
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
            if col in INVERSE_STATS:
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

    return CoverageDataBundle(
        off_coverage=off_profiles, def_coverage=def_profiles,
        qb_vs_coverage=qb_data, def_allowed_to_qb=def_qb_data,
        receiver_by_alignment=receiver_by_alignment,
        def_allowed_by_alignment=def_allowed_by_alignment,
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
    the same thing."""
    if not outliers:
        return np.nan, []

    weighted_scores = []
    weights = []
    checked = []
    for coverage_field, z in outliers[:max_outliers]:
        checked.append(coverage_field.replace(" %", ""))
        parts = []
        part_weights = []

        def_row = def_allowed_by_coverage.get(coverage_field, {}).get(opp_team_name)
        def_score = _tier_to_score(def_row.get("_tiers") if def_row else None, own_stat)
        if def_score is not None:
            parts.append(def_score)
            part_weights.append(def_weight)

        own_row = own_data_by_coverage.get(coverage_field, {}).get(own_name)
        own_score = _tier_to_score(own_row.get("_tiers") if own_row else None, own_stat)
        if own_score is not None:
            parts.append(own_score)
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


def calc_qb_coverage_exploit_strength(bundle: CoverageDataBundle, qb_name: str,
                                       qb_team_abbrev: str, opponent_team_abbrev: str) -> dict:
    """
    Real per-QB signal: for each coverage this opponent's defense genuinely
    leans into (real z-score outlier, not just locally highest), combines
    how much that defense allows to QBs in that specific coverage (RATE
    allowed, tiered against the real league distribution of defenses in
    that coverage) with this QB's own real RATE in that same coverage from
    his own game history - see _weighted_outlier_exploit for the exact
    combination and weighting.

    Team abbreviations (as used throughout nfl_model_combined.py) are
    converted to the full names this module's data is keyed on via
    TEAM_ABBREV_TO_FULL. Returns neutral/empty gracefully (exploit_strength
    NaN) if the opponent isn't found or has no real outlier coverage this
    season - never raises, matching this module's established graceful-
    gap handling throughout.
    """
    opp_full = TEAM_ABBREV_TO_FULL.get((opponent_team_abbrev or "").upper())
    opp_profile = bundle.def_coverage.get(opp_full) if opp_full else None
    if opp_profile is None:
        return {"exploit_strength": np.nan, "outlier_coverages_checked": []}

    exploit_strength, checked = _weighted_outlier_exploit(
        opp_profile.outliers, bundle.qb_vs_coverage, bundle.def_allowed_to_qb,
        qb_name, opp_profile.team_name, own_stat="RATE",
    )
    return {"exploit_strength": exploit_strength, "outlier_coverages_checked": checked}


def calc_alignment_exploit_strength(bundle: CoverageDataBundle, player_name: str, position: str,
                                     player_team_abbrev: str, opponent_team_abbrev: str) -> dict:
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
    """
    opp_full = TEAM_ABBREV_TO_FULL.get((opponent_team_abbrev or "").upper())
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

    exploit_strength, checked = _weighted_outlier_exploit(
        opp_profile.outliers,
        bundle.receiver_by_alignment.get(dominant_alignment, {}),
        bundle.def_allowed_by_alignment.get(dominant_alignment, {}),
        player_name, opp_profile.team_name, own_stat="FP/G",
    )
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
