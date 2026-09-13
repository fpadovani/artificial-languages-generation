"""
Lexicalize the remaining small closed-class function-word categories in
work/grammar/basic-grammar-nld.gr: Comp, Rel, Prep, CC. Subj/Obj are
deliberately left untouched (kept as the original nonce placeholders, per
project decision -- see work/grammar/dutch_lexicon.csv discussion).

These are hand-picked, not mined, since each category has only 1-4 entries
in the base grammar -- too small and too closed-class for the
wordfreq/spaCy/Alpino pipeline used for nouns/adjectives/verbs to be
worthwhile.

Word choices:
  - Comp ("S_Comp -> S Comp", tag 3): "dat" -- the standard Dutch
    complementizer introducing a finite embedded clause ("ik denk dat...").
  - Rel ("NP_S/NP_P -> VP Rel Noun", tag 6): "die" -- the Dutch relative
    pronoun for common-gender (de-word) antecedents. Dutch actually has a
    gender-conditioned alternation (die/dat) that this grammar doesn't
    track (no gender feature on Noun_S/Noun_P), so "die" is used uniformly
    as the majority-class form; flagged here as a known simplification.
  - Prep ("PP -> NP Prep" / "NP -> PP NP", tag 4): in, op, van, met --
    four common, semantically distinct Dutch prepositions (location,
    location, source/possession, instrument/accompaniment).
  - CC ("Adj -> Adj CC Adj", coordination): "en" -- the standard Dutch
    coordinator ("and").

Usage:
    python src/lexicalize_dutch_function_words.py
"""
import argparse
import pathlib

WORDS = {
    "Comp": ["dat"],
    "Rel": ["die"],
    "Prep": ["in", "op", "van", "met"],
    "CC": ["en"],
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
    parser.add_argument("--grammar", default="work/grammar/basic-grammar-nld.gr")
    args = parser.parse_args()

    grammar_path = pathlib.Path(args.grammar)
    lines = grammar_path.open().readlines()

    for lhs, words in WORDS.items():
        idx = find_terminal_lines(lines, lhs)
        if len(idx) != len(words):
            raise ValueError(
                f"Found {len(idx)} terminal {lhs} lines in {grammar_path}, "
                f"expected {len(words)} words to slot in -- mismatch."
            )
        for i, word in zip(idx, words):
            lines[i] = f"1\t{lhs}\t{word}\n"
        print(f"{lhs}: {' / '.join(words)}")

    grammar_path.open("w").writelines(lines)
    print(f"Written: {grammar_path}")


if __name__ == "__main__":
    main()
