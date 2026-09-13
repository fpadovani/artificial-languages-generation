"""
Generate pretraining corpora for the natural languages listed in
work/wals_switches.csv, matched typologically via their resolved lang_id.

For each language, position 2 of lang_id (complementizer order, unmapped to
any WALS feature -- see src/grammar-general/01_get_wals_switches.py) is filled with both
possible values, producing two concrete 8-bit switch strings per language:
  - position 2 = "0": complementizer-after-clause  (canonical S_Comp -> S Comp)
  - position 2 = "1": complementizer-before-clause (flipped -> Comp S)
Position 7 (VPComp, clausal-complement-order) is already resolved per
language in wals_switches.csv (tied to OV_switch via Grambank GB135).

Each of the resulting IDs is used to permute the full 100,000-sentence base
sample (work/grammar/sample_basic-grammar.txt) via the exact flipping logic
in src/artificial-langs/permute_sentences.py. No train/dev/test split is
applied -- each output file is the complete permuted corpus, intended for
language model pretraining.

NOTE: the base sample must be (re)generated from the current
basic-grammar.gr before running this -- its bracket tags are baked in at
sampling time, so a stale base sample from before a grammar retag (e.g. the
VP_Comp_* tag-2->8 split, see src/grammar-general/03_apply_zipfian_weights.py) would silently
ignore the new switch.

Usage:
    python src/grammar-general/04_generate_target_corpora.py
"""
import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "artificial-langs"))
from permute_sentences import (  # noqa: E402
    flip_as_needed,
    remove_bracketing,
    label_brackets_with_nonterminals,
    convert_sentence_to_tree,
)

N_SWITCHES = 8
COMPL_LABEL = {"0": "complAfter", "1": "complBefore"}


class _FlipArgs:
    n_switches = N_SWITCHES


def lang_id_to_index(lang_id: str) -> int:
    """Inverse of permute_sentences.py's grammar_name = format(i, '07b')[::-1]:
    character k of lang_id is bit k of i."""
    return sum(2 ** k for k, ch in enumerate(lang_id) if ch == "1")


def generate_file(i: int, sentences, output_file: pathlib.Path):
    with output_file.open("w") as f:
        for s in sentences:
            permuted_s = flip_as_needed(i, s, _FlipArgs())
            surface_s = remove_bracketing(permuted_s)
            linearized_tree = label_brackets_with_nonterminals(permuted_s)
            tree = convert_sentence_to_tree(permuted_s)
            f.write(json.dumps({
                "surface": surface_s,
                "linearized_tree": linearized_tree,
                "tree": tree,
            }) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switches_csv", default="work/wals_switches.csv")
    parser.add_argument("--base_sample", default="work/grammar/sample_basic-grammar.txt")
    parser.add_argument("--out", default="work/grammar/target_samples")
    args = parser.parse_args()

    df = pd.read_csv(args.switches_csv, dtype={"lang_id": str})

    with pathlib.Path(args.base_sample).open() as f:
        sentences = f.readlines()
    print(f"Loaded {len(sentences)} base sentences from {args.base_sample}")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        template = row["lang_id"]
        assert template[2] == "2", f"expected wildcard at position 2, got {template!r}"

        for compl_value, label in COMPL_LABEL.items():
            concrete_id = template[:2] + compl_value + template[3:]
            i = lang_id_to_index(concrete_id)
            out_file = out_dir / f"{row['language']}_{label}_{concrete_id}.jsonl"
            generate_file(i, sentences, out_file)
            print(f"{row['language']:12s} {label:12s} lang_id={concrete_id}  -> {out_file}")

    print(f"\nDone: {2 * len(df)} corpora written to {out_dir}")


if __name__ == "__main__":
    main()
