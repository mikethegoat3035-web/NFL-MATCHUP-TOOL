"""
NFL PREMIUM TOOL - Coverage Matchup Module
=============================================
Built from FantasyPoints.com Data Suite manual exports (paid subscription,
no public API - see project notes on why this is manual-export-based).

WHAT THIS DOES
---------------
1. Loads team-level coverage tendency data (Man/Zone/Cover 0-6 % usage)
   for both offense (coverages seen) and defense (coverages used).
2. Flags each team's REAL statistical outlier coverage(s) using z-scores
   against league average - not raw rank, not "top coverage" (which is
   almost always Cover 3 for everyone and tells you nothing).
3. Loads QB-vs-coverage season splits (7 files: Cover 0/1/2/2Man/3/4/6),
   one row per QB per coverage, aggregated across every defense that
   showed that QB that coverage this season.
4. Builds a matchup report: given a QB + an opponent defense, auto-finds
   the opponent's real outlier coverage(s) and pulls the QB's own
   performance vs that specific coverage - with an automatic thin-sample
   warning built in, based on real league-wide ATT distributions per
   coverage (not a guessed threshold).

WHY Z-SCORES, NOT RAW RANK
----------------------------
Every team runs a lot of Cover 3 - it's the default shell league-wide.
"16th highest Cover 3 rate" tells you nothing about that team's real
defensive identity. A z-score answers the actual question: is this
team's usage of this coverage meaningfully different from the league,
or just normal variance? Confirmed on real 2025 data: Seattle's Cover 6
rate (17.7%) is +1.62 SD above league average (3rd of 32) - a real,
usable signal. Their Cover 4 rate, despite ranking 13th of 32, is only
+0.23 SD above average - statistical noise, not a real tendency.

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

# Confirmed thin-sample coverages league-wide (median ATT < 10 in real 2025 data).
# These get flagged as low-confidence regardless of any individual QB's count.
ALWAYS_THIN_COVERAGES = {"COVER 0 %", "COVER 2 MAN %"}

# Minimum attempts for a QB's own vs-coverage number to be called "solid"
# rather than "thin" - calibrated per coverage from real league medians above.
THIN_SAMPLE_ATT_THRESHOLD = {
    "COVER 0 %": 5,
    "COVER 1 %": 15,
    "COVER 2 %": 15,
    "COVER 2 MAN %": 5,
    "COVER 3 %": 20,
    "COVER 4 %": 15,
    "COVER 6 %": 10,
}

# z-score cutoff for calling a team's coverage rate a real "outlier" /
# defensive identity, vs just normal team-to-team variance.
OUTLIER_Z_THRESHOLD = 1.0

# QB files store team as an abbreviation (e.g. "JAX", "LA"); team coverage
# matrix files store full names (e.g. "Jacksonville Jaguars"). This map is
# REQUIRED for the same-team safety guard to work - an exact string
# comparison between "JAX" and "Jacksonville Jaguars" silently fails and
# does NOT block the nonsense case. Extend this if a mismatch surfaces.
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
    """True if abbrev_or_name refers to the same team as full_name.
    Handles both a raw abbreviation ("JAX") and an already-full name
    being passed in."""
    if not abbrev_or_name:
        return False
    if abbrev_or_name == full_name:
        return True
    mapped = TEAM_ABBREV_TO_FULL.get(abbrev_or_name.upper())
    return mapped == full_name


# ---------------------------------------------------------------------------
# Team coverage matrix (offense-seen / defense-used)
# ---------------------------------------------------------------------------

@dataclass
class TeamCoverageProfile:
    team_name: str
    rates: dict  # coverage field -> float %
    z_scores: dict = field(default_factory=dict)
    outliers: list = field(default_factory=list)  # [(coverage, z), ...] sorted desc


def _read_fp_csv(path):
    """FantasyPoints exports have 2 header rows (grouping row + real header).
    Real header always starts with 'Rank'. Handles BOM safely."""
    with open(path, encoding='utf-8-sig') as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.lstrip('\ufeff').startswith('"Rank"'))
    reader = csv.reader(lines[header_idx:])
    rows = list(reader)
    header, data = rows[0], [r for r in rows[1:] if r and r[0]]
    return header, data


def load_team_coverage_matrix(csv_path):
    """Loads a DEF_COVG or OFF_COVG style export (one row per team, full
    Man/Zone/Cover 0-6 % breakdown). Returns dict[team_name] -> TeamCoverageProfile
    with z-scores and real outliers already computed against this file's
    own league average."""
    header, data = _read_fp_csv(csv_path)
    name_idx = header.index("Name")

    # collect raw rates per team
    profiles = {}
    for row in data:
        d = dict(zip(header, row))
        team = d["Name"]
        rates = {}
        for f in COVERAGE_FIELDS:
            try:
                rates[f] = float(d[f])
            except (ValueError, KeyError):
                rates[f] = 0.0
        profiles[team] = TeamCoverageProfile(team_name=team, rates=rates)

    # league stats per coverage field
    league_stats = {}
    for f in COVERAGE_FIELDS:
        vals = [p.rates[f] for p in profiles.values()]
        league_stats[f] = (mean(vals), pstdev(vals))

    # z-scores + outlier flagging
    for p in profiles.values():
        for f in COVERAGE_FIELDS:
            avg, sd = league_stats[f]
            z = (p.rates[f] - avg) / sd if sd else 0.0
            p.z_scores[f] = z
        p.outliers = sorted(
            [(f, z) for f, z in p.z_scores.items() if z >= OUTLIER_Z_THRESHOLD],
            key=lambda x: -x[1]
        )

    return profiles, league_stats


def describe_team_tendency(profile: TeamCoverageProfile):
    """Human-readable one-liner for a team's real coverage identity."""
    if not profile.outliers:
        return f"{profile.team_name}: no coverage runs meaningfully above league average - plays a fairly standard mix."
    parts = []
    for cov, z in profile.outliers:
        pct = profile.rates[cov]
        cov_label = cov.replace(" %", "")
        parts.append(f"{cov_label} {pct:.1f}% (z={z:+.2f})")
    return f"{profile.team_name}: real outlier coverage(s) - " + ", ".join(parts)


# ---------------------------------------------------------------------------
# QB vs coverage (player-side season splits)
# ---------------------------------------------------------------------------

# Map filename-friendly coverage key -> the field name used in team matrix
COVERAGE_FILE_MAP = {
    "VS_COVER_0.csv": "COVER 0 %",
    "VS_COVER_1.csv": "COVER 1 %",
    "VS_COVER_2.csv": "COVER 2 %",
    "VS_COVER_2MAN.csv": "COVER 2 MAN %",
    "VS_COVER_3.csv": "COVER 3 %",
    "VS_COVER_4.csv": "COVER 4 %",
    "VS_COVER_6.csv": "COVER 6 %",
}


def load_qb_vs_coverage(file_paths: dict):
    """file_paths: dict of {coverage_field: csv_path}, e.g.
    {"COVER 6 %": "/path/VS_COVER_6.csv", ...}

    Returns dict[coverage_field][qb_name] -> stat dict (raw row as dict),
    plus a computed 'thin_sample' bool per QB per coverage."""
    data = {}
    for coverage_field, path in file_paths.items():
        header, rows = _read_fp_csv(path)
        by_qb = {}
        for row in rows:
            d = dict(zip(header, row))
            name = d.get("Name")
            if not name:
                continue
            try:
                att = int(d.get("ATT", 0) or 0)
            except ValueError:
                att = 0
            threshold = THIN_SAMPLE_ATT_THRESHOLD.get(coverage_field, 15)
            d["_thin_sample"] = (att < threshold) or (coverage_field in ALWAYS_THIN_COVERAGES)
            d["_att"] = att
            by_qb[name] = d
        data[coverage_field] = by_qb
    return data


# ---------------------------------------------------------------------------
# Matchup report - the actual weekly-use function
# ---------------------------------------------------------------------------

def load_def_allowed_to_qb(file_paths: dict):
    """file_paths: dict of {coverage_field: csv_path} for the DEFENSE-ALLOWED
    side (team rows, e.g. def_allowed_cover6.csv - what each defense gives up
    to QBs specifically when running that coverage).

    Returns dict[coverage_field][team_name] -> stat dict, same shape as
    load_qb_vs_coverage but keyed by team name instead of QB name, and
    sample-size flagged the same way using the QB-side thresholds (this
    data is passing-attempts-allowed, same units)."""
    data = {}
    for coverage_field, path in file_paths.items():
        header, rows = _read_fp_csv(path)
        by_team = {}
        for row in rows:
            d = dict(zip(header, row))
            name = d.get("Name")
            if not name:
                continue
            try:
                att = int(d.get("ATT", 0) or 0)
            except ValueError:
                att = 0
            threshold = THIN_SAMPLE_ATT_THRESHOLD.get(coverage_field, 15)
            d["_thin_sample"] = (att < threshold) or (coverage_field in ALWAYS_THIN_COVERAGES)
            d["_att"] = att
            by_team[name] = d
        data[coverage_field] = by_team
    return data


def build_qb_matchup_report(qb_name, opponent_team_profile: TeamCoverageProfile,
                             qb_coverage_data: dict, qb_team_name=None,
                             def_allowed_data: dict = None,
                             min_outliers=1, max_outliers=3):
    """Given a QB and the opponent's TeamCoverageProfile, auto-detects the
    opponent's real outlier coverage(s) and pulls the QB's own season
    performance against each - with thin-sample flags already applied.

    If def_allowed_data is provided, also attaches what the opponent
    defense itself allows to QBs in that coverage - giving both sides
    (QB's own history vs the coverage, and defense's own history allowing
    it) in one report instead of one-sided.

    qb_team_name: pass the QB's own team to guard against building a
    nonsense report for a QB facing his own team.

    Returns a list of dicts, one per relevant coverage, ready to print
    or feed into a UI table.
    """
    if qb_team_name and _same_team(qb_team_name, opponent_team_profile.team_name):
        return [{
            "error": f"{qb_name} plays for {opponent_team_profile.team_name} - "
                     f"cannot build a matchup report against his own team."
        }]

    report = []
    outliers = opponent_team_profile.outliers[:max_outliers]

    if not outliers:
        return [{
            "note": f"{opponent_team_profile.team_name} has no statistically real "
                    f"outlier coverage this season - no specific coverage edge to flag "
                    f"for this matchup."
        }]

    for coverage_field, z in outliers:
        cov_label = coverage_field.replace(" %", "")
        team_pct = opponent_team_profile.rates[coverage_field]
        qb_row = qb_coverage_data.get(coverage_field, {}).get(qb_name)

        entry = {
            "coverage": cov_label,
            "opponent_usage_pct": team_pct,
            "opponent_z_score": round(z, 2),
        }

        if qb_row is None:
            entry["qb_data"] = "No recorded attempts vs this coverage this season."
            entry["confidence"] = "no_data"
        else:
            entry["qb_data"] = {
                "ATT": qb_row["_att"],
                "CMP%": qb_row.get("CMP %"),
                "YPA": qb_row.get("YPA"),
                "TD": qb_row.get("TD"),
                "INT": qb_row.get("INT"),
                "RATE": qb_row.get("RATE"),
                "CPOE": qb_row.get("CPOE"),
            }
            entry["confidence"] = "thin_sample" if qb_row["_thin_sample"] else "solid"

        # Attach defense-allowed side, if provided
        if def_allowed_data is not None:
            def_row = def_allowed_data.get(coverage_field, {}).get(opponent_team_profile.team_name)
            if def_row is None:
                entry["defense_allows"] = "No recorded data for this coverage."
            else:
                entry["defense_allows"] = {
                    "ATT": def_row["_att"],
                    "CMP%": def_row.get("CMP %"),
                    "YPA": def_row.get("YPA"),
                    "TD": def_row.get("TD"),
                    "INT": def_row.get("INT"),
                    "RATE": def_row.get("RATE"),
                }
                entry["defense_confidence"] = "thin_sample" if def_row["_thin_sample"] else "solid"

        report.append(entry)

    return report


def print_matchup_report(qb_name, opponent_team_profile, qb_coverage_data,
                          qb_team_name=None, def_allowed_data=None):
    """Console-friendly readable version for quick manual checks."""
    report = build_qb_matchup_report(qb_name, opponent_team_profile, qb_coverage_data,
                                      qb_team_name=qb_team_name, def_allowed_data=def_allowed_data)

    if report and "error" in report[0]:
        print(f"\n  [BLOCKED] {report[0]['error']}")
        return

    print(f"\n=== {qb_name} vs {opponent_team_profile.team_name} — Coverage Matchup ===")
    for entry in report:
        if "note" in entry:
            print(f"  {entry['note']}")
            continue
        print(f"\n  {opponent_team_profile.team_name} runs {entry['coverage']} at "
              f"{entry['opponent_usage_pct']:.1f}% (z={entry['opponent_z_score']:+.2f} vs league)")
        if entry["confidence"] == "no_data":
            print(f"    -> {qb_name}: {entry['qb_data']}")
        else:
            qd = entry["qb_data"]
            flag = "  [THIN - " + str(qd["ATT"]) + " att]" if entry["confidence"] == "thin_sample" else ""
            print(f"    -> {qb_name} vs {entry['coverage']} (own history): "
                  f"{qd['ATT']} ATT, {qd['CMP%']}% CMP, {qd['YPA']} YPA, "
                  f"{qd['TD']} TD, {qd['INT']} INT, {qd['RATE']} rating{flag}")
        if "defense_allows" in entry and isinstance(entry["defense_allows"], dict):
            dd = entry["defense_allows"]
            flag = "  [THIN - " + str(dd["ATT"]) + " att]" if entry.get("defense_confidence") == "thin_sample" else ""
            print(f"    -> {opponent_team_profile.team_name} allows vs {entry['coverage']}: "
                  f"{dd['ATT']} ATT, {dd['CMP%']}% CMP, {dd['YPA']} YPA, "
                  f"{dd['TD']} TD, {dd['INT']} INT, {dd['RATE']} rating{flag}")
