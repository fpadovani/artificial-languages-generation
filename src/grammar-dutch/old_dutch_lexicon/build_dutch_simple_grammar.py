"""
ARCHIVED (see old_dutch_lexicon/README.md) -- this was the one-time bridge
script that produced today's active starting grammar
(basic-grammar-nld-simple-uniform.gr) from the real-Dutch-word grammar,
work/grammar/basic-grammar-nld.gr. Already run; kept for provenance, not
part of the active pipeline.

Create basic-grammar-nld-simple.gr, a copy of
work/grammar/basic-grammar-nld.gr with every terminal lexicon entry
replaced by a systematic, transparent category+index label (e.g. Noun_S's
162 Dutch words -> noun_s1..noun_s162), the same treatment
src/grammar-general/02_build_simple_grammar.py applied to the abstract base grammar.

Renaming scheme: lowercase(category name) + 1-based index (original file
order), for every category with more than one entry; bare lowercase name
(no index) for categories with exactly one entry (CC, Comp, Rel, and the
4 Aux terminals). Exception: Aux_Zijn_S/P and Aux_Hebben_S/P get the
explicit aux1_s/aux1_p/aux2_s/aux2_p labels (1=zijn, 2=hebben) rather than
their lowercased category name, per explicit instruction -- this also has
the effect of not revealing which aux is which from the label alone.

Only true lexical leaf categories are touched (explicit list below) --
NOT single-child structural rules like `NP_S -> Noun_S` or `VP_S ->
VP_Past_S`, which look the same (single-token RHS, no space) but reference
another nonterminal rather than a literal word.

Usage:
    python src/grammar-dutch/old_dutch_lexicon/build_dutch_simple_grammar.py
"""
import argparse
import pathlib

# category -> None (indexed lowercase(category) + rank) or explicit label
# prefix list (1:1 with original file order) for categories needing a
# non-derived name.
LEXICAL_CATEGORIES = [
    "Noun_S", "Noun_P", "Adj",
    "Verb_Comp_Past_S", "Verb_Comp_Past_P",
    "IVerb_Past_S", "IVerb_Past_P",
    "TVerb_Past_S", "TVerb_Past_P",
    "Verb_Comp_Pres_S", "Verb_Comp_Pres_P",
    "IVerb_Pres_S", "IVerb_Pres_P",
    "TVerb_Pres_S", "TVerb_Pres_P",
    "Prep", "CC", "Comp", "Rel",
    "Pronoun_Subj_S", "Pronoun_Subj_P", "Pronoun_Obj_S", "Pronoun_Obj_P",
    "IVerb_Ptcp_Zijn", "IVerb_Ptcp_Hebben", "TVerb_Ptcp", "VerbComp_Ptcp",
    "Aux_Zijn_S", "Aux_Zijn_P", "Aux_Hebben_S", "Aux_Hebben_P",
]

# Explicit overrides: category -> fixed base label (no index, count must be 1).
EXPLICIT_LABELS = {
    "Aux_Zijn_S": "aux1_s",
    "Aux_Zijn_P": "aux1_p",
    "Aux_Hebben_S": "aux2_s",
    "Aux_Hebben_P": "aux2_p",
}


def find_terminal_lines(lines, lhs):
    indices = []
    for i, line in enumerate(lines):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and fields[1] == lhs and " " not in fields[2].strip():
            indices.append(i)
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_grammar", default="work/grammar/basic-grammar-nld.gr")
    parser.add_argument("--out_grammar", default="work/grammar/basic-grammar-nld-simple-uniform.gr")
    args = parser.parse_args()

    base_path = pathlib.Path(args.base_grammar)
    lines = base_path.open().readlines()

    print(f"{'category':20s} {'count':>6s}  label(s)")
    for cat in LEXICAL_CATEGORIES:
        idx = find_terminal_lines(lines, cat)
        n = len(idx)
        if cat in EXPLICIT_LABELS:
            if n != 1:
                raise ValueError(f"{cat} has {n} entries, expected exactly 1 for an explicit-label override")
            base_label = EXPLICIT_LABELS[cat]
        else:
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


if __name__ == "__main__":
    main()
