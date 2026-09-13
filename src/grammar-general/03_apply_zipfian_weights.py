"""
Step 3 of the grammar-general pipeline (after 01_get_wals_switches.py,
02_build_simple_grammar.py): turn a uniform-weight simple grammar into a
Zipfian-weight one, and (by default) untie the VP_Comp switch so clausal-
complement order can vary independently of object-verb order.

This merges what used to be three separate scripts -- add_zipfian_lexicon_
weights.py, untie_vpcomp_switch.py, and add_zipfian_weights_and_untie_vpcomp.py
(a thin wrapper chaining the other two) -- into one, via --mode. No logic
changed in the merge; apply_zipfian_weights() and untie_vpcomp() below are
copied verbatim from the two original scripts.

--mode weights: only apply_zipfian_weights() (see its docstring below).
--mode untie:   only untie_vpcomp() (see its docstring below).
--mode both:    both, in that order (default -- this is what
                add_zipfian_weights_and_untie_vpcomp.py used to do, e.g. to
                derive work/grammar/basic-grammar-simple-zipf.gr from
                work/grammar/basic-grammar-simple-uniform.gr).

=== apply_zipfian_weights() ===
Assign Zipfian PCFG weights (weight(rank) = 1/rank) to the open-class
lexical categories, replacing their current uniform weight of 1.

Scope, matching the convention in src/word-order-preferences/artificial_languages/
generation/base_grammar.gr (checked directly): Zipfian weights are applied
only to open-class content categories (nouns, adjectives, verbs) -- closed
function-word classes (Prep, CC, Comp, Rel, Subj, Obj, Pronoun_S/P) stay at
uniform weight 1 there too, even when they have many entries (e.g. their
Prep has 30 entries, still all weight 1).

Critical fix from the first version of this script: weights are shared
across inflected forms of the same lemma, not assigned independently per
category. Noun_S and Noun_P are the same 162 lemmas (confirmed against the
original nonce lexicon -- e.g. "amackist"/"amackists" are positionally
paired), so noun_s7 and noun_p7 must have the same weight, not two
unrelated random ranks. Likewise the 4 tense/number forms of each IVerb,
TVerb, and Verb_Comp lemma. LEMMA_GROUPS below lists which categories share
a lemma set; for each group, ONE shuffled 1/rank weight list is generated
and applied at the same lemma index across every category in the group.
Categories not in a multi-member group (Adj) get their own independent
shuffle, since there's no inflected sibling to keep in sync with.

Rank-to-lemma assignment is shuffled (not tied to file order), matching the
reference grammar: their weight values follow 1/rank, but the specific
word/lemma that gets rank 1 vs rank 2 etc. isn't in alphabetical/file
order. This avoids a label-frequency confound where a token's own
identifier (e.g. a numeric suffix, in a tokenizer that treats digits
specially) could leak its sampling frequency. A fixed random seed keeps
this reproducible.

No code changes are needed elsewhere: PCFG.load_rules() in
src/artificial-langs/sample_sentences.py already normalizes whatever raw
weight is in column 1 by dividing by the sum for that left-hand side, so
writing 1/rank values here is immediately usable.

--zipf_exponent (default 1.0, classic weight(rank) = 1/rank) controls how
concentrated the distribution is; e.g. 0.5 gives weight(rank) = 1/rank^0.5,
a shallower curve with a smaller max/min ratio within each lemma group.

=== untie_vpcomp() ===
Give VP_Comp_* its own switch tag (8), decoupled from the shared OV tag (2)
it previously used.

Why: VP_Comp_Past_S -> S_Comp Verb_Comp_Past_S (and its Pres_S/Past_P/Pres_P
siblings) was tagged "2" -- the same tag as the ordinary object-verb order
rule (VP_Past_S -> NP_Obj TVerb_Past_S). Since a single switch bit reverses
one bracket, this forced object order and clausal-complement order to move
together: no language could have plain objects before their verb but a
complement clause after its verb (or vice versa). See
01_get_wals_switches.py, whose lang_id has an 8th character (VPComp,
position 7, sourced from Grambank GB135) specifically to encode these
independently. Retagging VP_Comp_* to 8 is what makes that 8th lang_id
character actually do anything -- without this, flip_as_needed would just
ignore bit 7 entirely, since no bracket in the grammar is tagged "8".

Confirmed empirically earlier (see conversation record): the same base
sentence run through the tied (shared tag 2) vs. untied (split tags 2/8)
versions of this grammar produces 2 reachable object/complement-order
combinations tied, 4 untied -- with the 2 new combinations being exactly
the ones the shared tag couldn't produce.

By default this reads and writes the same --in_grammar path in place; pass
--out_grammar to write elsewhere instead, leaving the input untouched.

If starting from a grammar without the simplified lexicon, re-derive it
with 02_build_simple_grammar.py afterward (it preserves the weight column
rather than overwriting it -- see that script).

Usage:
    python src/grammar-general/03_apply_zipfian_weights.py \\
        --in_grammar work/grammar/basic-grammar-simple-uniform.gr \\
        --out_grammar work/grammar/basic-grammar-simple-zipf.gr
    python src/grammar-general/03_apply_zipfian_weights.py --zipf_exponent 0.5 \\
        --in_grammar work/grammar/basic-grammar-simple-uniform.gr \\
        --out_grammar work/grammar/basic-grammar-simple-zipf-softer.gr \\
        --mode weights   # weights only, e.g. re-running a Zipf-exponent sweep
                          # on a grammar that's already untied
"""
import argparse
import pathlib
import random

# Categories that share the same underlying lemma set (same count, same
# file-order lemma index), grouped so they get identical per-lemma weights.
LEMMA_GROUPS = [
    ["Noun_S", "Noun_P"],
    ["Adj"],
    ["IVerb_Past_S", "IVerb_Past_P", "IVerb_Pres_S", "IVerb_Pres_P"],
    ["TVerb_Past_S", "TVerb_Past_P", "TVerb_Pres_S", "TVerb_Pres_P"],
    ["Verb_Comp_Past_S", "Verb_Comp_Past_P", "Verb_Comp_Pres_S", "Verb_Comp_Pres_P"],
]

RANDOM_SEED = 1

TARGET_RULES = {
    ("VP_Comp_Pres_S", "S_Comp Verb_Comp_Pres_S"),
    ("VP_Comp_Past_S", "S_Comp Verb_Comp_Past_S"),
    ("VP_Comp_Pres_P", "S_Comp Verb_Comp_Pres_P"),
    ("VP_Comp_Past_P", "S_Comp Verb_Comp_Past_P"),
}


def find_terminal_lines(lines, lhs):
    indices = []
    for i, line in enumerate(lines):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and fields[1] == lhs and " " not in fields[2].strip():
            indices.append(i)
    return indices


def apply_zipfian_weights(lines, zipf_exponent=1.0):
    """Mutate lines in place, replacing each LEMMA_GROUPS category's terminal
    weights with a shared shuffled 1/rank^zipf_exponent Zipfian weight per
    lemma. Returns a list of (group_label, n, top3_weights) for reporting."""
    rng = random.Random(RANDOM_SEED)
    report = []
    for group in LEMMA_GROUPS:
        idx_by_cat = {cat: find_terminal_lines(lines, cat) for cat in group}
        counts = {cat: len(idx) for cat, idx in idx_by_cat.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Lemma group {group} has mismatched counts: {counts} -- "
                              f"can't share weights across a misaligned lemma set")
        n = next(iter(counts.values()))

        weights = [1.0 / (rank ** zipf_exponent) for rank in range(1, n + 1)]
        rng.shuffle(weights)  # ONE shuffle per group, shared by every category in it

        for cat in group:
            for i, w in zip(idx_by_cat[cat], weights):
                fields = lines[i].rstrip("\n").split("\t")
                fields[0] = f"{w:.6f}"
                lines[i] = "\t".join(fields) + "\n"

        report.append(("+".join(group), n, sorted(weights, reverse=True)[:3]))
    return report


def untie_vpcomp(lines):
    """Return a new list of lines with VP_Comp_* rules retagged 2 -> 8."""
    n_retagged = 0
    new_lines = []
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) == 4 and (fields[1], fields[2]) in TARGET_RULES:
            if fields[3] != "2":
                raise ValueError(f"Expected tag 2 on {fields[1]} -> {fields[2]}, found {fields[3]!r}")
            fields[3] = "8"
            new_lines.append("\t".join(fields) + "\n")
            n_retagged += 1
        else:
            new_lines.append(line)

    if n_retagged != len(TARGET_RULES):
        raise ValueError(f"Expected to retag {len(TARGET_RULES)} rules, retagged {n_retagged}")
    return new_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in_grammar", "--grammar", dest="in_grammar", default="work/grammar/basic-grammar.gr")
    parser.add_argument("--out_grammar", default=None,
                         help="Defaults to --in_grammar (in-place)")
    parser.add_argument("--mode", choices=["weights", "untie", "both"], default="both",
                         help="weights = apply_zipfian_weights only; untie = untie_vpcomp only; "
                              "both = both in that order (default)")
    parser.add_argument("--zipf_exponent", type=float, default=1.0,
                         help="weight(rank) = 1/rank^zipf_exponent; 1.0 = classic Zipf, lower = flatter "
                              "(only used when --mode is weights or both)")
    args = parser.parse_args()

    in_path = pathlib.Path(args.in_grammar)
    out_path = pathlib.Path(args.out_grammar) if args.out_grammar else in_path
    lines = in_path.open().readlines()

    if args.mode in ("weights", "both"):
        print(f"{'group':45s} {'count':>6s}  top-3 weights (shuffled onto random lemmas, shared across the group)")
        for label, n, top3 in apply_zipfian_weights(lines, args.zipf_exponent):
            print(f"{label:45s} {n:6d}  {top3}")

    if args.mode in ("untie", "both"):
        lines = untie_vpcomp(lines)
        print(f"Retagged {len(TARGET_RULES)} VP_Comp_* rules from switch tag 2 to tag 8")

    out_path.open("w").writelines(lines)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
