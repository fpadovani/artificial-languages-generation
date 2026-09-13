"""
Create work/grammar/basic-grammar-nld-simple-heavy.gr (a copy of
basic-grammar-nld-simple.gr, made by this script) where the object between
a perfect-tense auxiliary and TVerb_Ptcp is systematically longer, to
increase aux-participle dependency length -- without touching NP structure
anywhere else in the grammar, and WITHOUT open self-recursion.

Design history (see conversation record): the first version duplicated
NP_S/NP_P's recursive modification rules (PP-attachment, coordination,
relative-clause) with boosted weights, self-referentially
(NP_S_Heavy -> PP_Heavy NP_S_Heavy). That caused real, observed failures:
boosting recursive weights to compete with "stop" made recursion
near-self-sustaining, and a runaway object occasionally consumed the
entire sample_sentences.py max_expansions budget (400) before the grammar
ever reached TVerb_Ptcp, producing a sentence with NO participle at all --
useless for the exact purpose this was built for.

This version instead GUARANTEES exactly one bounded modifier, with no
self-reference back into the Heavy category -- so the maximum extra depth
the heavy step itself can add is capped at one level, full stop:
    NP_S_Heavy -> Adj Noun_S                    (~2 tokens; ~18% chance
                                                  longer -- see below)
    NP_S_Heavy -> Adj Adj Noun_S                (~3 tokens, same caveat, twice)
    NP_S_Heavy -> PP_Bare NP_S_Bare  (tag 4)    (exactly 2 tokens: prep + bare noun)
    NP_S_Heavy -> VP_S Rel Noun_S    (tag 6)    (~2-5+ tokens, a real relative clause --
                                                  routes through the ORDINARY VP_S, the
                                                  same recursion this grammar has already
                                                  run successfully at 100k+ sentence scale
                                                  many times over -- not a new loop, so
                                                  raising this weight is safe, just shifts
                                                  more mass into the long tail)
Per explicit request ("highest the better, but avoid the recursion error
from before"): Rel's weight was raised back to 1 (from the earlier 0.3),
since its recursion was never the source of the crash -- that came
specifically from NP_S_Heavy calling itself. Adj Adj Noun_S was added as a
genuine extra option rather than suppressing Adj's own ~18% chance of
coordinating into "Adj CC Adj" (see conversation record) -- that
recursion is bounded/self-limiting (can only ever add more "CC Adj" pairs,
never reach VP_Perf_S or consume the expansion budget the way the object
runaway did), so its extra variance is welcome here, not a risk.
Second iteration fix: the PP alternative originally used the ORDINARY
NP_S, which -- it turns out -- already has its own
`NP_S -> VP_S Rel Noun_S` rule at weight 1 (same as a bare noun) in the
base grammar. That meant the "PP" branch was independently picking up
relative clauses roughly a third of the time on its own, regardless of
this script's Rel weight -- diagnosed empirically when dropping that
weight to 0.05 barely moved the measured mean. Fixed by giving the PP
branch a genuinely bare NP_S_Bare/PP_Bare (Noun_S only, no further
modification possible), so its length is now fully predictable, and the
Rel alternative is the only remaining source of variance, tunable on its
own.

No "just a bare noun" alternative, so every perfect-tense transitive
object gets at least one modifier. Switch tags (4=PP order, 6=Rel order)
are preserved on NP_S_Heavy/NP_P_Heavy's own rules so Dutch's actual
word-order switches still apply correctly; NP_S_Bare/PP_Bare intentionally
carry the same tag 4 (same switch, since it's still adposition order) but
have no tag-6 rule at all (no Rel option, by design).

NP_Obj_Heavy deliberately excludes the Pronoun_Obj_S/P alternative that
plain NP_Obj has, for the same reason as before -- a pronoun object is
trivially short, working against the point of this variant.

Usage:
    python src/grammar-dutch/02_add_heavy_object_nld.py
"""
import argparse
import pathlib

NEW_RULES = """
1\tNP_Obj_Heavy\tNP_S_Heavy
1\tNP_Obj_Heavy\tNP_P_Heavy

1\tNP_S_Heavy\tAdj Noun_S
1\tNP_P_Heavy\tAdj Noun_P
1\tNP_S_Heavy\tAdj Adj Noun_S
1\tNP_P_Heavy\tAdj Adj Noun_P
1\tNP_S_Heavy\tPP_Bare NP_S_Bare\t4
1\tNP_P_Heavy\tPP_Bare NP_P_Bare\t4
1\tNP_S_Heavy\tVP_S Rel Noun_S\t6
1\tNP_P_Heavy\tVP_P Rel Noun_P\t6

1\tNP_S_Bare\tNoun_S
1\tNP_P_Bare\tNoun_P
1\tPP_Bare\tNP_S_Bare Prep\t4
1\tPP_Bare\tNP_P_Bare Prep\t4
"""

REPOINT = {
    "1\tVP_Perf_S\tAux_Hebben_S NP_Obj TVerb_Ptcp\n": "1\tVP_Perf_S\tAux_Hebben_S NP_Obj_Heavy TVerb_Ptcp\n",
    "1\tVP_Perf_P\tAux_Hebben_P NP_Obj TVerb_Ptcp\n": "1\tVP_Perf_P\tAux_Hebben_P NP_Obj_Heavy TVerb_Ptcp\n",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_grammar", default="work/grammar/basic-grammar-nld-simple.gr")
    parser.add_argument("--out_grammar", default="work/grammar/basic-grammar-nld-simple-heavy.gr")
    args = parser.parse_args()

    base_path = pathlib.Path(args.base_grammar)
    out_path = pathlib.Path(args.out_grammar)
    lines = base_path.open().readlines()

    n_repointed = 0
    new_lines = []
    for line in lines:
        if line in REPOINT:
            new_lines.append(REPOINT[line])
            n_repointed += 1
        else:
            new_lines.append(line)
    if n_repointed != len(REPOINT):
        raise ValueError(f"Expected to repoint {len(REPOINT)} rules, repointed {n_repointed} -- "
                          f"has the grammar changed since this script was written?")

    new_lines.append(NEW_RULES)

    out_path.open("w").writelines(new_lines)
    print(f"Repointed {n_repointed} VP_Perf rules to use NP_Obj_Heavy")
    print(f"Appended {len([l for l in NEW_RULES.strip().splitlines() if l.strip()])} new Heavy-NP rules "
          f"(no self-recursion -- bounded to one guaranteed modifier)")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
