"""
Remove the Subj/Obj case-affix marking from work/grammar/basic-grammar-nld.gr,
so subject/object NPs surface with zero marking -- matching Dutch's actual
WALS Case_switch value (0, "No case affixes or adpositional clitics" --
see work/wals_switches.csv), which the earlier nonce placeholders ("sub"/
"ob") didn't reflect.

Rather than making Subj/Obj expand to an empty string (which breaks the
PCFG loader in src/artificial-langs/sample_sentences.py -- its
`line.rstrip().split("\\t")` call strips trailing tabs along with the
newline, so a blank third field collapses a 3-field row to 2 fields and
hits the parser's `else: raise NotImplementedError` branch; confirmed by
actually hitting that crash), the `Subj`/`Obj` symbol is dropped from the
right-hand side of the rules that reference it entirely:

    NP_Subj_S -> NP_S Subj   (tag 7)   =>   NP_Subj_S -> NP_S
    NP_Subj_P -> NP_P Subj   (tag 7)   =>   NP_Subj_P -> NP_P
    NP_Obj    -> NP_S Obj    (tag 7)   =>   NP_Obj    -> NP_S
    NP_Obj    -> NP_P Obj    (tag 7)   =>   NP_Obj    -> NP_P

The tag 7 is dropped too: it controlled the order of the case affix
relative to its NP, which is meaningless once there's no affix (a
single-child expansion has no order to permute). The `Subj -> sub` /
`Obj -> ob` terminal lines (however they're currently written -- this
script matches on the left-hand side, not exact RHS content, so it's safe
to run even if those lines were already manually half-edited) are deleted;
leaving them in without any rule referencing `Subj`/`Obj` would either
crash the loader (if malformed) or, if simply left as `Subj -> sub`,
silently do nothing.

Other NP_Subj_S/NP_Obj alternatives (the Pronoun_Subj_S/Pronoun_Obj_S/P
routes added by src/restructure_dutch_pronouns.py) are untouched -- pronouns
were already zero-marking-compatible (case is lexical there, not affixal).

Usage:
    python src/remove_dutch_case_marking.py
"""
import argparse
import pathlib

TARGET_RULES = {
    ("NP_Subj_S", "NP_S Subj"): "NP_S",
    ("NP_Subj_P", "NP_P Subj"): "NP_P",
    ("NP_Obj", "NP_S Obj"): "NP_S",
    ("NP_Obj", "NP_P Obj"): "NP_P",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", default="work/grammar/basic-grammar-nld.gr")
    args = parser.parse_args()

    grammar_path = pathlib.Path(args.grammar)
    lines = grammar_path.open().readlines()

    new_lines = []
    n_rewritten = 0
    n_dropped = 0
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 2:
            lhs = fields[1]
            rhs = fields[2].strip() if len(fields) >= 3 else ""
            if (lhs, rhs) in TARGET_RULES:
                new_lines.append(f"1\t{lhs}\t{TARGET_RULES[(lhs, rhs)]}\n")
                n_rewritten += 1
                continue
            if lhs in ("Subj", "Obj"):
                n_dropped += 1
                continue  # drop now-unreachable terminal entries (any RHS state)
        new_lines.append(line)

    if n_rewritten != len(TARGET_RULES):
        raise ValueError(
            f"Expected to rewrite {len(TARGET_RULES)} rules, rewrote {n_rewritten} -- "
            f"grammar structure may not match what this script expects."
        )

    grammar_path.open("w").writelines(new_lines)
    print(f"Rewrote {n_rewritten} rules (dropped Subj/Obj symbol + tag 7)")
    print(f"Dropped {n_dropped} now-unreachable Subj/Obj terminal lines")
    print(f"Written: {grammar_path}")


if __name__ == "__main__":
    main()