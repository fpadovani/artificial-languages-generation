"""
Look up WALS word-order features for a set of natural languages and convert
them to the 0/1 switch encoding used by src/export_language_stats.py, so they
can be matched against the artificial grammars in work/grammar/permuted_samples/.

Missing WALS values are resolved to a single concrete 0/1 by falling back to
the majority value among all WALS languages sharing the same genus; if the
genus has no data (or is tied), fall back to family; if that also has no
data (or is tied), fall back to macroarea. The fallback level actually used
is recorded per feature so the imputation is auditable.

An 8th switch, VPComp (clausal-complement-order: whether a finite complement
clause sits before or after the matrix verb that governs it), is added on
top of the 6 core WALS features. WALS has no feature for this at all; it's
sourced from Grambank's GB135 ("Do clausal objects usually occur in the same
position as nominal objects?"), which for every one of these 16 languages
present in Grambank returns "yes" -- so VPComp_switch is set equal to
OV_switch for all of them. Urdu isn't in Grambank; its value is extended by
analogy to Hindi (same "Hindi-Urdu" syntactic system). Using this grammar's
existing VP_Comp construction with this switch requires its own tag,
decoupled from OV's tag -- see basic-grammar.gr's VP_Comp_* rules (tag 8).

Usage:
    python src/grammar-general/01_get_wals_switches.py
    python src/grammar-general/01_get_wals_switches.py --wals_csv work/language.csv --out work/wals_switches.csv
"""
import argparse
import pandas as pd

# WALS "Name" column value to use for each natural language in the experiment.
# Most languages match their common name exactly; a few need disambiguation
# because WALS lists several dialects/varieties under similar names.
LANGUAGES = {
    "Danish": "Danish",
    "Swedish": "Swedish",
    "Italian": "Italian",
    "Chinese": "Mandarin",
    "Japanese": "Japanese",
    "Indonesian": "Indonesian",
    "Hindi": "Hindi",
    "Urdu": "Urdu",
    "English": "English",
    "Dutch": "Dutch",
    "Turkish": "Turkish",
    "Swahili": "Swahili",
    "Russian": "Russian",
    "Icelandic": "Icelandic",
    "Hebrew": "Hebrew (Modern)",
    "Basque": "Basque",
    "Norwegian": "Norwegian",
    "Tamil": "Tamil",
    "Arabic": "Arabic (Modern Standard)",
}

# Same column renames and value->switch maps as src/export_language_stats.py
COLUMN_RENAME = {
    "51A Position of Case Affixes": "Case",
    "82A Order of Subject and Verb": "SV",
    "83A Order of Object and Verb": "OV",
    "85A Order of Adposition and Noun Phrase": "PP",
    "87A Order of Adjective and Noun": "Adj",
    "90A Order of Relative Clause and Noun": "Rel",
}

SWITCH_MAPS = {
    "Case": {"1 Case suffixes": 0, "2 Case prefixes": 1},
    "SV": {"1 SV": 0, "2 VS": 1},
    "OV": {"1 OV": 0, "2 VO": 1},
    "PP": {"2 Prepositions": 1, "1 Postpositions": 0},
    "Adj": {"2 Noun-Adjective": 1, "1 Adjective-Noun": 0},
    "Rel": {"1 Noun-Relative clause": 1, "2 Relative clause-Noun": 0},
}

# Ordered to match each feature's actual position in the 8-character lang_id
# string (position 2, the complementizer switch, has no WALS feature mapped
# to it in this codebase and is skipped here).
TARGET_COLS = ["SV", "OV", "PP", "Adj", "Rel", "Case"]

# Position of each switch within the 8-character lang_id string used by
# permute_sentences.py / export_language_stats.py. Position 2 has no WALS
# feature mapped to it in this codebase (complementizer order) and is always
# left as a literal "2". Position 7 (VPComp) is resolved separately below,
# not from a WALS column -- see resolve_vpcomp().
SWITCH_POSITION = {"SV": 0, "OV": 1, "PP": 3, "Adj": 4, "Rel": 5, "Case": 6, "VPComp": 7}

FALLBACK_LEVELS = ["genus", "family", "macroarea"]

# Manually verified exceptions where GB135 does NOT hold -- i.e. this
# language's clausal-complement position is known (by hand, not from GB135)
# to differ from its nominal-object position, so VPComp_switch must NOT be
# tied to OV_switch here. Hindi and Urdu are both verb-final (OV_switch=0)
# but place their complement clause AFTER the verb via a postposed
# "ki"-style complementizer -- a well-documented Indo-Aryan pattern (the
# same thing independently holds for sibling languages Marathi and Eastern
# Punjabi: "VComp via ki", GB135=No). Add further entries here only after
# the same kind of manual, language-specific verification -- this dict
# exists precisely because the OV-tie default is known to be wrong for the
# languages listed in it.
VPCOMP_OVERRIDES = {
    "Hindi": 1,
    "Urdu": 1,
}

VPCOMP_LABELS = {1: "verb before complement clause", 0: "verb after complement clause"}


def resolve_vpcomp(our_name, ov_switch):
    """VPComp_switch defaults to OV_switch (GB135: clausal objects usually
    occur in the same position as nominal objects), except for the manually
    verified exceptions in VPCOMP_OVERRIDES, where GB135 is known not to
    hold for this specific language. Not derived from a WALS column, so has
    no 'raw' value or genus/family/macroarea fallback the way the other 6
    features do. The recorded source is simply a plain-language description
    of what the resulting value means, not where it came from."""
    value = VPCOMP_OVERRIDES.get(our_name, ov_switch)
    return value, VPCOMP_LABELS[value]


def majority_value(df, col, level, group_value):
    """Majority 0/1 switch value for `col` among all WALS rows sharing
    `group_value` at the given `level` (genus/family/macroarea).
    Returns (value, support) or (None, support) if no data or a tie."""
    if pd.isna(group_value):
        return None, {}
    subset = df[df[level] == group_value]
    mapped = subset[col].map(SWITCH_MAPS[col]).dropna()
    if len(mapped) == 0:
        return None, {}
    counts = mapped.value_counts()
    support = counts.to_dict()
    if len(counts) == 1 or counts.iloc[0] != counts.iloc[1]:
        return int(counts.index[0]), support
    return None, support  # tie at this level


def resolve_switch(df, row, col):
    """Return (value, source, support) for one feature of one language.
    source is 'wals' if directly observed, or the fallback level used."""
    raw = row[col]
    direct = SWITCH_MAPS[col].get(raw)
    if direct is not None:
        return direct, "wals", {}

    for level in FALLBACK_LEVELS:
        value, support = majority_value(df, col, level, row[level])
        if value is not None:
            return value, level, support

    return None, "unresolved", {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wals_csv", default="work/language.csv")
    parser.add_argument("--out", default="work/wals_switches.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.wals_csv)
    df = df.rename(columns=COLUMN_RENAME)

    rows = []
    for our_name, wals_name in LANGUAGES.items():
        match = df[df["Name"] == wals_name]
        if len(match) == 0:
            raise ValueError(f"No WALS entry found for {our_name!r} ({wals_name!r})")
        if len(match) > 1:
            raise ValueError(f"Multiple WALS entries found for {our_name!r} ({wals_name!r}): "
                              f"{match['Name'].tolist()}")
        row = match.iloc[0]

        record = {
            "language": our_name,
            "wals_name": wals_name,
            "wals_code": row.get("wals_code"),
            "genus": row.get("genus"),
            "family": row.get("family"),
            "macroarea": row.get("macroarea"),
        }
        lang_id = ["2"] * 8
        for col in TARGET_COLS:
            value, source, support = resolve_switch(df, row, col)
            record[f"{col}_raw"] = row[col]
            record[f"{col}_switch"] = value
            record[f"{col}_source"] = source
            if value is None:
                print(f"WARNING: could not resolve {col} for {our_name} "
                      f"(no data at genus/family/macroarea level, or ties throughout)")
            else:
                lang_id[SWITCH_POSITION[col]] = str(value)

        vpcomp_value, vpcomp_source = resolve_vpcomp(our_name, record["OV_switch"])
        record["VPComp_switch"] = vpcomp_value
        record["VPComp_source"] = vpcomp_source
        lang_id[SWITCH_POSITION["VPComp"]] = str(vpcomp_value)

        record["lang_id"] = "".join(lang_id)
        rows.append(record)

    out_df = pd.DataFrame(rows)
    ordered_cols = ["language", "wals_name", "wals_code", "genus", "family", "macroarea"]
    for col in TARGET_COLS:
        ordered_cols += [f"{col}_raw", f"{col}_switch", f"{col}_source"]
    ordered_cols += ["VPComp_switch", "VPComp_source", "lang_id"]
    out_df = out_df[ordered_cols]

    out_df.to_csv(args.out, index=False)
    print(out_df.to_string(index=False))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
