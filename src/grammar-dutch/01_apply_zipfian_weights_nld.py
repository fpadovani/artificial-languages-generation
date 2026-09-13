"""
Step 1 of the grammar-dutch active pipeline: assign Zipfian PCFG weights
(weight(rank) = 1/rank) to the open-class lexical categories of a Dutch
simple-lexicon grammar (pass one via --in_grammar, e.g.
work/grammar/basic-grammar-nld-simple-uniform.gr), mirroring
src/grammar-general/03_apply_zipfian_weights.py's --mode weights for the
abstract grammar but accounting for this grammar's extra perfect-tense
participle categories.

Lemma groups (weight shared across every inflected form of the same lemma,
one shuffled 1/rank weight list per group):
  - Noun_S + Noun_P                                            (162 lemmas)
  - Adj                                                         (42, standalone)
  - TVerb_Past_S/P + TVerb_Pres_S/P + TVerb_Ptcp                (113 lemmas)
  - Verb_Comp_Past_S/P + Verb_Comp_Pres_S/P + VerbComp_Ptcp     (22 lemmas)
TVerb_Ptcp and VerbComp_Ptcp fold in directly by file-order index:
old_dutch_lexicon/build_dutch_perfect.py wrote them by iterating the same
lemma list, in the same order, as the finite forms, with no reordering or
splitting.

IVerb is handled specially, per explicit instruction: the 113 IVerb lemmas
form ONE ranking, but not a free shuffle across all 113 -- the 7 lemmas
that select "zijn" (ZIJN_VERBS, matching old_dutch_lexicon/build_dutch_perfect.py) are
deliberately given the top 7 ranks (highest weights), and the 106
"hebben"-selecting lemmas fill ranks 8-113. This over-represents the
zijn-class as a whole, despite it being the numerical minority. Within each
bucket, which specific lemma gets which specific rank is shuffled, same as
everywhere else in this pipeline.

That per-lemma weight is then applied identically across ALL FIVE of a
lemma's forms: IVerb_Past_S, IVerb_Past_P, IVerb_Pres_S, IVerb_Pres_P, and
its participle (IVerb_Ptcp_Zijn or IVerb_Ptcp_Hebben, whichever it belongs
to). The participle categories are independently renumbered subsets of the
same 113 lemmas (not the same file-order index as the finite forms -- see
conversation record for why naive index-matching would silently link
unrelated verbs), so the correspondence is reconstructed via the actual
lemma strings still in work/grammar/dutch_verb_lexicon.csv /
dutch_perfect_lexicon.csv, not via numeric suffix.

By default this reads and writes the same --in_grammar path in place; pass
--out_grammar to write elsewhere instead, leaving the input untouched --
e.g. to apply Zipfian weights to a heavy-object variant without also
overwriting the plain grammar it was derived from.

Usage:
    python src/grammar-dutch/01_apply_zipfian_weights_nld.py \\
        --in_grammar work/grammar/basic-grammar-nld-simple-uniform.gr \\
        --out_grammar work/grammar/basic-grammar-nld-simple-zipf.gr
    python src/grammar-dutch/01_apply_zipfian_weights_nld.py \\
        --in_grammar work/grammar/basic-grammar-nld-simple-heavy-uniform.gr \\
        --out_grammar work/grammar/basic-grammar-nld-simple-heavy-zipf.gr
"""
import argparse
import csv
import pathlib
import random

RANDOM_SEED = 1

# Must match src/grammar-dutch/old_dutch_lexicon/build_dutch_perfect.py exactly.
ZIJN_VERBS = {"blijven", "verdwijnen", "overlijden", "verschijnen", "lukken", "slagen", "mislukken"}

SIMPLE_LEMMA_GROUPS = [
    ["Noun_S", "Noun_P"],
    ["Adj"],
    ["TVerb_Past_S", "TVerb_Past_P", "TVerb_Pres_S", "TVerb_Pres_P", "TVerb_Ptcp"],
    ["Verb_Comp_Past_S", "Verb_Comp_Past_P", "Verb_Comp_Pres_S", "Verb_Comp_Pres_P", "VerbComp_Ptcp"],
]


def find_terminal_lines(lines, lhs):
    indices = []
    for i, line in enumerate(lines):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and fields[1] == lhs and " " not in fields[2].strip():
            indices.append(i)
    return indices


def set_weight(lines, i, w):
    fields = lines[i].rstrip("\n").split("\t")
    fields[0] = f"{w:.6f}"
    lines[i] = "\t".join(fields) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_grammar", "--grammar", dest="in_grammar", required=True,
                         help="e.g. work/grammar/basic-grammar-nld-simple-uniform.gr -- no default: the old "
                             "default (basic-grammar-nld.gr) is now archived, see old_dutch_lexicon/")
    parser.add_argument("--out_grammar", default=None,
                         help="Defaults to --in_grammar (in-place), matching the original behavior")
    parser.add_argument("--verb_lexicon_csv", default="work/grammar/dutch_verb_lexicon.csv")
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    in_path = pathlib.Path(args.in_grammar)
    out_path = pathlib.Path(args.out_grammar) if args.out_grammar else in_path
    lines = in_path.open().readlines()

    print(f"{'group':70s} {'count':>6s}  top-3 weights")
    for group in SIMPLE_LEMMA_GROUPS:
        idx_by_cat = {cat: find_terminal_lines(lines, cat) for cat in group}
        counts = {cat: len(idx) for cat, idx in idx_by_cat.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Lemma group {group} has mismatched counts: {counts}")
        n = next(iter(counts.values()))

        weights = [1.0 / rank for rank in range(1, n + 1)]
        rng.shuffle(weights)

        for cat in group:
            for i, w in zip(idx_by_cat[cat], weights):
                set_weight(lines, i, w)

        print(f"{'+'.join(group):70s} {n:6d}  {sorted(weights, reverse=True)[:3]}")

    # --- IVerb: one 113-lemma ranking, zijn-verbs forced into ranks 1-7 ---
    with open(args.verb_lexicon_csv) as f:
        iverb_lemmas = [row["lemma"] for row in csv.DictReader(f) if row["category"] == "IVerb"]

    zijn_lemmas = [l for l in iverb_lemmas if l in ZIJN_VERBS]
    hebben_lemmas = [l for l in iverb_lemmas if l not in ZIJN_VERBS]
    if len(zijn_lemmas) != len(ZIJN_VERBS):
        raise ValueError(f"Expected {len(ZIJN_VERBS)} zijn-verbs in {args.verb_lexicon_csv}, found {len(zijn_lemmas)}")

    n_total = len(iverb_lemmas)
    zijn_ranks = list(range(1, len(zijn_lemmas) + 1))
    hebben_ranks = list(range(len(zijn_lemmas) + 1, n_total + 1))
    rng.shuffle(zijn_lemmas)     # which zijn-verb gets which of ranks 1..7
    rng.shuffle(hebben_lemmas)   # which hebben-verb gets which of ranks 8..113

    lemma_to_weight = {}
    for lemma, rank in zip(zijn_lemmas, zijn_ranks):
        lemma_to_weight[lemma] = 1.0 / rank
    for lemma, rank in zip(hebben_lemmas, hebben_ranks):
        lemma_to_weight[lemma] = 1.0 / rank

    # Apply to the 4 finite-form categories via file-order index (same order as iverb_lemmas).
    for cat in ["IVerb_Past_S", "IVerb_Past_P", "IVerb_Pres_S", "IVerb_Pres_P"]:
        idx = find_terminal_lines(lines, cat)
        if len(idx) != n_total:
            raise ValueError(f"{cat} has {len(idx)} entries, expected {n_total}")
        for i, lemma in zip(idx, iverb_lemmas):
            set_weight(lines, i, lemma_to_weight[lemma])
    print(f"{'IVerb_Past_S/P + IVerb_Pres_S/P (zijn-verbs forced to top 7 ranks)':70s} {n_total:6d}  "
          f"{sorted(lemma_to_weight.values(), reverse=True)[:3]}")

    # Apply to the participle categories via lemma cross-reference (NOT file-order index).
    zijn_idx = find_terminal_lines(lines, "IVerb_Ptcp_Zijn")
    hebben_idx = find_terminal_lines(lines, "IVerb_Ptcp_Hebben")
    with open(args.verb_lexicon_csv.replace("dutch_verb_lexicon.csv", "dutch_perfect_lexicon.csv")) as f:
        ptcp_rows = [row for row in csv.DictReader(f) if row["category"] == "IVerb"]
    ptcp_zijn_lemmas = [r["lemma"] for r in ptcp_rows if r["aux"] == "zijn"]
    ptcp_hebben_lemmas = [r["lemma"] for r in ptcp_rows if r["aux"] == "hebben"]
    if len(zijn_idx) != len(ptcp_zijn_lemmas) or len(hebben_idx) != len(ptcp_hebben_lemmas):
        raise ValueError(f"IVerb_Ptcp count mismatch: zijn {len(zijn_idx)} vs {len(ptcp_zijn_lemmas)}, "
                          f"hebben {len(hebben_idx)} vs {len(ptcp_hebben_lemmas)}")

    for i, lemma in zip(zijn_idx, ptcp_zijn_lemmas):
        set_weight(lines, i, lemma_to_weight[lemma])
    for i, lemma in zip(hebben_idx, ptcp_hebben_lemmas):
        set_weight(lines, i, lemma_to_weight[lemma])
    print(f"{'IVerb_Ptcp_Zijn (linked to same lemma as finite forms)':70s} {len(zijn_idx):6d}  "
          f"{sorted([lemma_to_weight[l] for l in ptcp_zijn_lemmas], reverse=True)[:3]}")
    print(f"{'IVerb_Ptcp_Hebben (linked to same lemma as finite forms)':70s} {len(hebben_idx):6d}  "
          f"{sorted([lemma_to_weight[l] for l in ptcp_hebben_lemmas], reverse=True)[:3]}")

    out_path.open("w").writelines(lines)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
