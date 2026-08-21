"""
Add the Dutch perfect-tense construction (hebben/zijn + past participle) to
work/grammar/basic-grammar-nld.gr, for the argument-structure/auxiliary-
selection BLiMP-nl paradigm (e.g. "het vuur heeft/is gebrand").

Auxiliary classification: TVerb (113, all transitive) and Verb_Comp (22, all
clause-embedding) categorically select "hebben" in Dutch -- not a per-verb
judgment call. Within IVerb (113, intransitive), the classification is a
hardcoded list of the 7 standard Dutch zijn-verbs that happen to already be
in the grammar (out of a larger reference list of ~14-16 well-known
zijn-verbs; the other ~7 aren't in this grammar's IVerb selection at all):
blijven, verdwijnen, overlijden, verschijnen, lukken, slagen, mislukken.
Every other IVerb lemma gets hebben.

Participle forms come from UniMorph (work/external/unimorph_nld.txt,
tag V.PTCP;PST), same source used for the Pres/Past forms in
src/build_dutch_verb_lexicon.py. All 248 verbs already in the grammar have
full UniMorph participle coverage (no gaps).

Structural design (why a naive `VP_Perf_S -> Aux_S Participle` doesn't work):
the auxiliary and the participle must be sampled together, not
independently, or a PCFG could pair "is" with a hebben-verb's participle.
So each auxiliary class gets its own dedicated participle category and its
own VP_Perf_S alternative (same pattern as the Pronoun_Subj_S/Pronoun_Obj_S
split in src/restructure_dutch_pronouns.py):

    VP_Perf_S -> Aux_Zijn_S   IVerb_Ptcp_Zijn
    VP_Perf_S -> Aux_Hebben_S IVerb_Ptcp_Hebben
    VP_Perf_S -> Aux_Hebben_S NP_Obj TVerb_Ptcp
    VP_Perf_S -> Aux_Hebben_S VerbComp_Ptcp S_Comp

Word order (Aux-Object-Participle for TVerb; Aux-Participle-clause for
Verb_Comp) is fixed to match real Dutch V2/verb-bracket structure, not tied
to the existing 7 typological switches -- this construction is meant to be
authentically Dutch, not typologically parametrized.

Usage:
    python src/build_dutch_perfect.py
"""
import argparse
import csv
import pathlib

ZIJN_VERBS = {"blijven", "verdwijnen", "overlijden", "verschijnen", "lukken", "slagen", "mislukken"}


def load_participles(unimorph_tsv):
    ptcp = {}
    with open(unimorph_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[2] == "V.PTCP;PST":
                ptcp[parts[0]] = parts[1]
    return ptcp


def load_grammar_verbs(verb_lexicon_csv):
    verbs = {"TVerb": [], "IVerb": [], "Comp": []}
    with open(verb_lexicon_csv) as f:
        for row in csv.DictReader(f):
            verbs[row["category"]].append(row["lemma"])
    return verbs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", default="work/grammar/basic-grammar-nld.gr")
    parser.add_argument("--verb_lexicon_csv", default="work/grammar/dutch_verb_lexicon.csv")
    parser.add_argument("--unimorph_tsv", default="work/external/unimorph_nld.txt")
    parser.add_argument("--out_audit_csv", default="work/grammar/dutch_perfect_lexicon.csv")
    args = parser.parse_args()

    grammar_path = pathlib.Path(args.grammar)
    lines = grammar_path.open().readlines()

    if any("VP_Perf_S" in line for line in lines):
        raise ValueError(
            f"{grammar_path} already has VP_Perf_S entries -- this script is not "
            f"idempotent (it appends rather than replacing in place)."
        )

    participles = load_participles(args.unimorph_tsv)
    verbs = load_grammar_verbs(args.verb_lexicon_csv)

    missing = [l for cat in verbs for l in verbs[cat] if l not in participles]
    if missing:
        raise ValueError(f"No UniMorph participle found for: {missing}")

    iverb_zijn = [l for l in verbs["IVerb"] if l in ZIJN_VERBS]
    iverb_hebben = [l for l in verbs["IVerb"] if l not in ZIJN_VERBS]
    print(f"IVerb split: {len(iverb_zijn)} zijn {iverb_zijn}, {len(iverb_hebben)} hebben")
    print(f"TVerb: {len(verbs['TVerb'])} (all hebben)")
    print(f"Verb_Comp: {len(verbs['Comp'])} (all hebben)")

    # 1. Add VP_S/VP_P -> VP_Perf_S/VP_Perf_P alternatives, right after the
    #    existing VP_S -> VP_Comp_S / VP_P -> VP_Comp_P lines.
    def is_vp_comp_route(line, lhs, rhs):
        fields = line.rstrip("\n").split("\t")
        return len(fields) >= 3 and fields[1] == lhs and fields[2].strip() == rhs

    insert_s = next(i for i, l in enumerate(lines) if is_vp_comp_route(l, "VP_S", "VP_Comp_S")) + 1
    lines[insert_s:insert_s] = ["1\tVP_S\tVP_Perf_S\n"]
    insert_p = next(i for i, l in enumerate(lines) if is_vp_comp_route(l, "VP_P", "VP_Comp_P")) + 1
    lines[insert_p:insert_p] = ["1\tVP_P\tVP_Perf_P\n"]

    # 2. Append the VP_Perf_S / VP_Perf_P rules and the Aux terminals.
    new_rules = [
        "\n",
        "1\tVP_Perf_S\tAux_Zijn_S IVerb_Ptcp_Zijn\n",
        "1\tVP_Perf_S\tAux_Hebben_S IVerb_Ptcp_Hebben\n",
        "1\tVP_Perf_S\tAux_Hebben_S NP_Obj TVerb_Ptcp\n",
        "1\tVP_Perf_S\tAux_Hebben_S VerbComp_Ptcp S_Comp\n",
        "1\tVP_Perf_P\tAux_Zijn_P IVerb_Ptcp_Zijn\n",
        "1\tVP_Perf_P\tAux_Hebben_P IVerb_Ptcp_Hebben\n",
        "1\tVP_Perf_P\tAux_Hebben_P NP_Obj TVerb_Ptcp\n",
        "1\tVP_Perf_P\tAux_Hebben_P VerbComp_Ptcp S_Comp\n",
        "\n",
        "1\tAux_Hebben_S\theeft\n",
        "1\tAux_Hebben_P\thebben\n",
        "1\tAux_Zijn_S\tis\n",
        "1\tAux_Zijn_P\tzijn\n",
        "\n",
    ]
    for lemma in iverb_zijn:
        new_rules.append(f"1\tIVerb_Ptcp_Zijn\t{participles[lemma]}\n")
    for lemma in iverb_hebben:
        new_rules.append(f"1\tIVerb_Ptcp_Hebben\t{participles[lemma]}\n")
    for lemma in verbs["TVerb"]:
        new_rules.append(f"1\tTVerb_Ptcp\t{participles[lemma]}\n")
    for lemma in verbs["Comp"]:
        new_rules.append(f"1\tVerbComp_Ptcp\t{participles[lemma]}\n")

    lines.extend(new_rules)
    grammar_path.open("w").writelines(lines)
    print(f"Written: {grammar_path}")

    with open(args.out_audit_csv, "w") as f:
        f.write("category,lemma,aux,participle\n")
        for lemma in verbs["IVerb"]:
            aux = "zijn" if lemma in ZIJN_VERBS else "hebben"
            f.write(f"IVerb,{lemma},{aux},{participles[lemma]}\n")
        for lemma in verbs["TVerb"]:
            f.write(f"TVerb,{lemma},hebben,{participles[lemma]}\n")
        for lemma in verbs["Comp"]:
            f.write(f"Verb_Comp,{lemma},hebben,{participles[lemma]}\n")
    print(f"Written: {args.out_audit_csv}")


if __name__ == "__main__":
    main()
