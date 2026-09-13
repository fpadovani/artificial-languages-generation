# Archived: original real-Dutch-word lexicon pipeline

This directory holds the first Dutch grammar-building approach: mining real
Dutch words (via `wordfreq`, spaCy, and the Alpino Treebank) to populate
`basic-grammar-nld.gr`'s lexicon, rather than the transparent `noun_s1`-style
placeholder labels used everywhere else in this project. **No longer part of
the active pipeline** — kept here for provenance only.

## What's here

Note: `basic-grammar-nld.gr` itself stays in `work/grammar/` (restored there
on request), not archived in this directory -- only the scripts that built
it and the `dutch_lexicon.csv` audit file live here.

- `work/grammar/basic-grammar-nld.gr` — the real-Dutch-word grammar, built up
  by the scripts below, in this order:
  1. `build_dutch_lexicon.py` — mines nouns/adjectives (`dutch_lexicon.csv`)
  2. `build_dutch_verb_lexicon.py` — mines verbs (writes to
     `work/grammar/dutch_verb_lexicon.csv`, which is **not** archived here —
     it's still read by the active `01_apply_zipfian_weights_nld.py`)
  3. `lexicalize_dutch_function_words.py` — hand-picked closed-class words
  4. `remove_dutch_case_marking.py` — drops the Subj/Obj case-affix rules
     (Dutch has none)
  5. `restructure_dutch_pronouns.py` — gives pronouns suppletive
     subject/object forms
  6. `build_dutch_perfect.py` — adds the hebben/zijn perfect-tense
     construction (writes `work/grammar/dutch_perfect_lexicon.csv`, also
     **not** archived — still read by `01_apply_zipfian_weights_nld.py`)
  7. `add_zijn_transitive_variant.py` — **never actually run**: its declared
     output, `basic-grammar-nld-v2.gr`, doesn't exist on disk. Kept as a
     dormant, unapplied variant.
- `dutch_lexicon.csv` — the noun/adjective word list from step 1.
- `build_dutch_simple_grammar.py` — the one-time bridge script that renamed
  `basic-grammar-nld.gr`'s lexicon to the transparent `noun_s1`-style labels,
  producing `work/grammar/basic-grammar-nld-simple-uniform.gr`. **That output
  is what the active pipeline (`src/grammar-dutch/01_apply_zipfian_weights_nld.py`
  onward) starts from.** This script has already been run; it's kept here
  only so the derivation is reproducible from scratch if ever needed.

## Why archived

Once the lexicon was renamed to transparent placeholder labels (matching how
every other language in this project is handled), nothing downstream needed
real Dutch words anymore — only the category structure and, for a couple of
specific correctness questions the numeric labels can't answer (which IVerb
lemmas select "zijn", how perfect-tense participles map back to their finite
forms), the two CSVs that stayed in `work/grammar/`.
