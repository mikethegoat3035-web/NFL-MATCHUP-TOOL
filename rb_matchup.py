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

import csv
import os
import re
from dataclasses import dataclass, field
from statistics import mean, pstdev

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
INVERSE_STATS = {"FUM", "STUFF %"}

NON_STAT_COLUMNS_PLAYER = {"Rank", "Name", "Team", "POS", "G", "Season"}
NON_STAT_COLUMNS_TEAM = {"Rank", "Name", "G", "Season", "Location", "Team Name"}

TEAM_ABBREV_TO_FULL = {
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


def _same_team(abbrev_or_name, full_name):
    if not abbrev_or_name or not full_name:
        return False
    a = abbrev_or_name.strip().upper()
    if a in TEAM_ABBREV_TO_FULL:
        return TEAM_ABBREV_TO_FULL[a] == full_name
    return abbrev_or_name.strip().lower() == full_name.strip().lower()


def _to_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _read_fp_csv(path):
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


def _compute_field_tiers(rows_by_key, non_stat_columns):
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


def load_rb_vs_concept(file_paths: dict):
    """Player-side: RB's own real season stats for ONE concept.
    file_paths: {concept_label: csv_path}. Returns
    {concept_label: {rb_name: row}}, every real column captured +
    tiered, duplicate columns already resolved via positional rename."""
    data = {}
    for concept, path in file_paths.items():
        header, rows = _read_fp_csv(path)
        by_key = {}
        for row in rows:
            d = _row_to_dict(header, row)
            key = d.get("Name")
            if not key:
                continue
            att = int(_to_float(d.get("ATT", 0)) or 0)
            threshold = THIN_SAMPLE_ATT_THRESHOLD.get(concept, 10)
            d["_thin_sample"] = att < threshold
            d["_att"] = att
            by_key[key] = d
        _compute_field_tiers(by_key, NON_STAT_COLUMNS_PLAYER)
        data[concept] = by_key
    return data


def load_def_allowed_rb_concept(file_paths: dict):
    """Defense-allowed: what each DEFENSE gives up on this concept.
    Same shape as load_rb_vs_concept, keyed by team name, using the
    team-side column layout (no Team/POS/FPTS columns)."""
    data = {}
    for concept, path in file_paths.items():
        header, rows = _read_fp_csv(path)
        by_key = {}
        for row in rows:
            d = _row_to_dict(header, row)
            key = d.get("Name")
            if not key:
                continue
            att = int(_to_float(d.get("ATT", 0)) or 0)
            threshold = THIN_SAMPLE_ATT_THRESHOLD.get(concept, 10)
            d["_thin_sample"] = att < threshold
            d["_att"] = att
            by_key[key] = d
        _compute_field_tiers(by_key, NON_STAT_COLUMNS_TEAM)
        data[concept] = by_key
    return data


@dataclass
class RBDataBundle:
    rb_vs_concept: dict       # concept -> {rb_name: row}
    def_allowed: dict         # concept -> {team_name: row}
    missing: list = field(default_factory=list)


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


def get_rb_matchup(bundle: RBDataBundle, rb_name, opponent_team_full, rb_team_name=None):
    """Single entry point - one report entry per concept the RB has ANY
    real data for (all 6 checked, not filtered by a usage-rate cutoff -
    see module docstring for why). Each entry has the RB's own history
    (own_row) and what the opponent allows on that concept
    (defense_allows), both tiered."""
    if rb_team_name and _same_team(rb_team_name, opponent_team_full):
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
