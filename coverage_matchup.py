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
from dataclasses import dataclass, field
from statistics import mean, pstdev

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


def _load_coverage_keyed_data(file_paths: dict, key_column: str):
    """Generic loader for both QB-vs-coverage (key_column='Name') and
    def-allowed-to-QB (key_column='Name', team rows) files. Captures
    EVERY column from the CSV, not a curated subset, and computes real
    statistical tiers per stat within each coverage's own distribution."""
    data = {}
    for coverage_field, path in file_paths.items():
        header, rows = _read_fp_csv(path)
        by_key = {}
        for row in rows:
            d = dict(zip(header, row))
            key = d.get(key_column)
            if not key:
                continue
            att = int(_to_float(d.get("ATT", 0)) or 0)
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
