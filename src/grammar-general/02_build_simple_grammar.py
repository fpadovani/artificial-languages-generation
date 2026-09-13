"""
Create work/grammar/basic-grammar-simple.gr, a copy of work/grammar/basic-grammar.gr
with every terminal lexicon entry replaced by a systematic, transparent
category+index label (e.g. Noun_S's 162 nonce words -> noun_s1..noun_s162),
in place of the original arbitrary nonce vocabulary (e.g. "amackist").

Rationale (per discussion with professors): the abstract multi-language
pipeline (src/grammar-general/01_get_wals_switches.py, src/grammar-general/04_generate_target_corpora.py) only
ever needs terminal categories to be distinguishable placeholders, not
plausible-looking fake words -- a transparent, indexed label is simpler to
audit and makes the underlying category obvious in any generated sentence.

Renaming scheme: lowercase(category name) + 1-based index, in original file
order, for every category with more than one entry. Categories with exactly
one entry (CC, Comp, Rel) get the bare lowercase name with no index, per
explicit instruction. Subj/Obj are NOT in the renaming list given and are
left untouched (still "sub"/"ob").

Only the terminal (single-token RHS) lines are touched; all structural
rules, including the switch tags, are copied through unchanged -- this
does not affect grammar structure or switch permutation, only which
literal string each terminal category emits. The original weight (column 1)
of each terminal line is preserved as-is, so e.g. Zipfian lexicon weights
set on basic-grammar.gr (see src/grammar-general/03_apply_zipfian_weights.py) survive
re-deriving basic-grammar-simple.gr from it.

Usage:
    python src/grammar-general/02_build_simple_grammar.py
"""
import argparse
import pathlib

# count > 1 -> indexed (category1, category2, ...); count == 1 -> bare name.
CATEGORIES = [
    "Noun_S", "Noun_P", "Adj",
    "Verb_Comp_Past_S", "Verb_Comp_Past_P",
    "IVerb_Past_S", "IVerb_Past_P",
    "TVerb_Past_S", "TVerb_Past_P",
    "Verb_Comp_Pres_S", "Verb_Comp_Pres_P",
    "IVerb_Pres_S", "IVerb_Pres_P",
    "TVerb_Pres_S", "TVerb_Pres_P",
    "Prep", "CC", "Comp", "Rel",
    "Pronoun_S", "Pronoun_P",
]


def find_terminal_lines(lines, lhs):
    indices = []
    for i, line in enumerate(lines):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and fields[1] == lhs and " " not in fields[2].strip():
            indices.append(i)
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_grammar", default="work/grammar/basic-grammar.gr")
    parser.add_argument("--out_grammar", default="work/grammar/basic-grammar-simple.gr")
    args = parser.parse_args()

    base_path = pathlib.Path(args.base_grammar)
    lines = base_path.open().readlines()

    print(f"{'category':20s} {'count':>6s}  label(s)")
    for cat in CATEGORIES:
        idx = find_terminal_lines(lines, cat)
        n = len(idx)
        base_label = cat.lower()
        for rank, i in enumerate(idx, start=1):
            weight = lines[i].rstrip("\n").split("\t")[0]
            label = base_label if n == 1 else f"{base_label}{rank}"
            lines[i] = f"{weight}\t{cat}\t{label}\n"
        sample = base_label if n == 1 else f"{base_label}1..{base_label}{n}"
        print(f"{cat:20s} {n:6d}  {sample}")

    out_path = pathlib.Path(args.out_grammar)
    out_path.open("w").writelines(lines)
    print(f"\nWritten: {out_path}")
    print("Left unchanged (not in renaming list): Subj (sub), Obj (ob)")


if __name__ == "__main__":
    main()
