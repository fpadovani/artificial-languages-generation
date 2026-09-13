"""
Create a new, separate version of the Dutch grammar
(work/grammar/basic-grammar-nld-v2.gr, copied from basic-grammar-nld.gr)
that adds a 5th VP_Perf_S/VP_Perf_P alternative letting ordinary transitive
verbs (TVerb_Ptcp) take the zijn auxiliary with an accusative object, e.g.
"hij is het boek gelezen":

    VP_Perf_S -> Aux_Zijn_S NP_Obj TVerb_Ptcp
    VP_Perf_P -> Aux_Zijn_P NP_Obj TVerb_Ptcp

This is a deliberate choice, not a linguistic claim: BLiMP-NL's
argument_structure paradigm confirms no genuinely accusative-transitive
Dutch verb actually selects zijn (the only zijn-taking 2-argument verbs are
the NOM-DAT psych class -- bevallen, opvallen, overkomen, etc. -- which take
a dative experiencer, not an accusative object, and aren't in the TVerb
pool). This rule reuses the existing 113-lemma TVerb_Ptcp pool as-is, so it
will generate some sentences that aren't grammatical standard Dutch, on
purpose, to further balance the hebben/zijn ratio in the corpus.

Original basic-grammar-nld.gr is left untouched -- this is saved as a
separate grammar version so the two can be compared / the original remains
available.

Usage:
    python src/grammar-dutch/old_dutch_lexicon/add_zijn_transitive_variant.py
"""
import argparse
import pathlib
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_grammar", default="work/grammar/basic-grammar-nld.gr")
    parser.add_argument("--out_grammar", default="work/grammar/basic-grammar-nld-v2.gr")
    args = parser.parse_args()

    base_path = pathlib.Path(args.base_grammar)
    out_path = pathlib.Path(args.out_grammar)
    shutil.copy(base_path, out_path)

    lines = out_path.open().readlines()

    if any("Aux_Zijn_S NP_Obj TVerb_Ptcp" in line for line in lines):
        raise ValueError(f"{out_path} already has the zijn+TVerb rule -- refusing to duplicate it.")

    def find_line(lhs, rhs):
        for i, line in enumerate(lines):
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3 and fields[1] == lhs and fields[2].strip() == rhs:
                return i
        raise ValueError(f"Could not find rule {lhs} -> {rhs}")

    # Insert right after the existing hebben+TVerb perfect rules, so the two
    # transitive-perfect alternatives stay grouped together.
    idx_s = find_line("VP_Perf_S", "Aux_Hebben_S NP_Obj TVerb_Ptcp")
    lines.insert(idx_s + 1, "1\tVP_Perf_S\tAux_Zijn_S NP_Obj TVerb_Ptcp\n")

    idx_p = find_line("VP_Perf_P", "Aux_Hebben_P NP_Obj TVerb_Ptcp")
    lines.insert(idx_p + 1, "1\tVP_Perf_P\tAux_Zijn_P NP_Obj TVerb_Ptcp\n")

    out_path.open("w").writelines(lines)
    print(f"Written: {out_path}")
    print("VP_Perf_S now has 5 alternatives (each weight 1 -> 20% probability):")
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and fields[1] == "VP_Perf_S":
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()