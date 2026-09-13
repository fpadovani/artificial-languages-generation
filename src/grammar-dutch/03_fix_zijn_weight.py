"""
Raise the structural weight of VP_Perf_S/P's zijn-auxiliary branch
(Aux_Zijn_S/P IVerb_Ptcp_Zijn) relative to its 3 hebben-auxiliary sibling
branches, without touching the input file.

Why this is needed: VP_Perf_S has 4 sibling alternatives --
    Aux_Zijn_S   IVerb_Ptcp_Zijn              (zijn -- unaccusative intransitive)
    Aux_Hebben_S IVerb_Ptcp_Hebben             (hebben -- unergative intransitive)
    Aux_Hebben_S NP_Obj[_Heavy] TVerb_Ptcp     (hebben -- transitive)
    Aux_Hebben_S VerbComp_Ptcp S_Comp          (hebben -- clause-embedding)
-- all at the grammar's default weight of 1, so a PCFG samples zijn only
25% of the time (1 of 4 equally-weighted branches), no matter how the
IVerb lexicon itself is weighted (src/grammar-dutch/01_apply_zipfian_weights_nld.py
only reweights terminals, never these structural sibling rules). Confirmed
empirically: 12,926 zijn vs 39,103 hebben tokens (24.8%) in a generated
100k-sentence Dutch corpus.

This script raises ONLY the Aux_Zijn_S/Aux_Zijn_P branch weight (leaving
the other 6 VP_Perf_S/P lines at 1), so P(zijn) = new_weight / (new_weight + 3).
Default new_weight=3 gives a 50/50 zijn/hebben split (3 / (3+3) = 0.5).

Usage:
    python src/grammar-dutch/03_fix_zijn_weight.py \\
        --in_grammar work/grammar/basic-grammar-nld-simple-zipf.gr \\
        --out_grammar work/grammar/basic-grammar-nld-simple-zipf-fix-zijn.gr
    python src/grammar-dutch/03_fix_zijn_weight.py --zijn_weight 1.286 ...  # ~30% zijn instead of 50%
"""
import argparse
import pathlib

TARGET_RULES = {
    ("VP_Perf_S", "Aux_Zijn_S IVerb_Ptcp_Zijn"),
    ("VP_Perf_P", "Aux_Zijn_P IVerb_Ptcp_Zijn"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in_grammar", required=True)
    parser.add_argument("--out_grammar", required=True)
    parser.add_argument("--zijn_weight", type=float, default=3.0,
                         help="New weight for the Aux_Zijn_S/P branch (siblings stay at 1); "
                              "default 3 gives a 50/50 zijn/hebben split")
    args = parser.parse_args()

    in_path = pathlib.Path(args.in_grammar)
    out_path = pathlib.Path(args.out_grammar)
    lines = in_path.open().readlines()

    n_changed = 0
    new_lines = []
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and (fields[1], fields[2]) in TARGET_RULES:
            if fields[0] != "1":
                raise ValueError(f"Expected weight 1 on {fields[1]} -> {fields[2]}, found {fields[0]!r}")
            fields[0] = f"{args.zijn_weight:g}"
            new_lines.append("\t".join(fields) + "\n")
            n_changed += 1
        else:
            new_lines.append(line)

    if n_changed != len(TARGET_RULES):
        raise ValueError(f"Expected to change {len(TARGET_RULES)} rules, changed {n_changed}")

    out_path.open("w").writelines(new_lines)
    p_zijn = args.zijn_weight / (args.zijn_weight + 3)
    print(f"Set Aux_Zijn_S/P branch weight to {args.zijn_weight:g} (siblings stay at 1)")
    print(f"-> P(zijn) = {args.zijn_weight:g}/({args.zijn_weight:g}+3) = {p_zijn:.1%}, P(hebben) = {1 - p_zijn:.1%}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()