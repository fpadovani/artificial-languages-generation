"""
Sample sentences from the Dutch-lexicalized grammar
(work/grammar/basic-grammar-nld.gr), apply Dutch's WALS-derived word-order
switches (from work/wals_switches.csv), and write the resulting surface text
-- alongside the lexicon audit CSVs (work/grammar/dutch_lexicon.csv,
work/grammar/dutch_verb_lexicon.csv) so a reviewer can check generated
sentences against exactly the word lists that produced them.

This mirrors src/generate_target_corpora.py's approach (same
flip_as_needed permutation logic from src/artificial-langs/permute_sentences.py)
but samples fresh from the Dutch-lexicalized grammar rather than permuting
the old abstract-lexicon base sample, and defaults to a single concrete
lang_id rather than generating both complementizer variants.

With --out_jsonl set, also writes the full {"surface", "linearized_tree",
"tree"} record per sentence (same schema as generate_target_corpora.py's
target_samples/*.jsonl), for use as structured pretraining data; --out
alone (the default) only writes plain surface text, e.g. for the small
review samples used during grammar development.

Usage:
    # small review sample (plain text only)
    python src/sample_dutch_grammar.py --compl before --n 200 \\
        --out work/grammar/dutch_sample_review.txt

    # full pretraining corpus (text + structured jsonl)
    python src/sample_dutch_grammar.py --compl before --n 100000 \\
        --out work/grammar/dutch_corpus_complBefore.txt \\
        --out_jsonl work/grammar/dutch_corpus_complBefore.jsonl
"""
import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent / "artificial-langs"))
from sample_sentences import PCFG  # noqa: E402
from permute_sentences import (  # noqa: E402
    flip_as_needed,
    remove_bracketing,
    label_brackets_with_nonterminals,
    convert_sentence_to_tree,
)

N_SWITCHES = 7
COMPL_VALUE = {"before": "1", "after": "0"}


class _FlipArgs:
    n_switches = N_SWITCHES


def lang_id_to_index(lang_id: str) -> int:
    """Inverse of permute_sentences.py's grammar_name = format(i, '07b')[::-1]:
    character k of lang_id is bit k of i."""
    return sum(2 ** k for k, ch in enumerate(lang_id) if ch == "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", default="work/grammar/basic-grammar-nld.gr")
    parser.add_argument("--switches_csv", default="work/wals_switches.csv")
    parser.add_argument("--language", default="Dutch")
    parser.add_argument("--compl", choices=["before", "after"], default="before")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max_expansions", type=int, default=400)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--out", default="work/grammar/dutch_sample_review.txt")
    parser.add_argument("--out_jsonl", default=None,
                         help="If set, also write the full surface/linearized_tree/tree jsonl records")
    args = parser.parse_args()

    df = pd.read_csv(args.switches_csv, dtype={"lang_id": str})
    row = df[df["language"] == args.language]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one row for {args.language!r} in {args.switches_csv}, found {len(row)}")
    template = row.iloc[0]["lang_id"]
    assert template[2] == "2", f"expected wildcard at position 2, got {template!r}"
    concrete_id = template[:2] + COMPL_VALUE[args.compl] + template[3:]
    i = lang_id_to_index(concrete_id)
    print(f"{args.language} lang_id template={template} -> concrete (compl={args.compl})={concrete_id} (index={i})")

    grammar = PCFG(pathlib.Path(args.grammar), args.random_seed)

    out_path = pathlib.Path(args.out)
    jsonl_path = pathlib.Path(args.out_jsonl) if args.out_jsonl else None
    f_jsonl = jsonl_path.open("w") if jsonl_path else None

    with out_path.open("w") as f_txt:
        for k in range(args.n):
            bracketed = grammar.sample_sentence(args.max_expansions, bracketing=True)
            permuted = flip_as_needed(i, bracketed, _FlipArgs())
            surface = remove_bracketing(permuted)
            f_txt.write(surface + "\n")
            if f_jsonl:
                linearized_tree = label_brackets_with_nonterminals(permuted)
                tree = convert_sentence_to_tree(permuted)
                f_jsonl.write(json.dumps({
                    "surface": surface,
                    "linearized_tree": linearized_tree,
                    "tree": tree,
                }) + "\n")
            if (k + 1) % 10000 == 0:
                print(f"  {k + 1}/{args.n} sentences generated")

    if f_jsonl:
        f_jsonl.close()

    print(f"Written {args.n} sentences to {out_path}")
    if jsonl_path:
        print(f"Written {args.n} structured records to {jsonl_path}")
    print("Lexicon audit files for cross-reference:")
    print("  work/grammar/dutch_lexicon.csv       (Noun_S/Noun_P/Adj)")
    print("  work/grammar/dutch_verb_lexicon.csv  (IVerb/TVerb/Verb_Comp)")


if __name__ == "__main__":
    main()
