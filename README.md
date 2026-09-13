# word-order-universals-cogLM
This repository started as a fork of [Kuribayashi et al.'s (2024) ACL codebase](https://github.com/kuribayashi4/word-order-universals-cogLM) for *Emergent Word Order Universals from Cognitively-Motivated Language Models*, and has since been substantially enriched and adjusted for our own research.

## Encoding word-order switches for our 19 languages

To build an artificial grammar for a real language (e.g. English, Dutch,
Japanese), we need to know that language's word order along 8 independent
binary "switches." Each switch is either `0` or `1`, and stringing all 8
together gives an 8-character `lang_id` (e.g. `01210101` for English) that
the grammar-permutation scripts (`src/artificial-langs/permute_sentences.py`
and the minimal-pairs generators) use directly.

| # | Switch | Meaning | `0` | `1` |
|---|---|---|---|---|
| 1 | `SV` | subject/verb order | SV | VS |
| 2 | `OV` | object/verb order | OV | VO |
| 3 | `Compl` | complementizer/clause order | *(not yet resolved from data -- see below)* | |
| 4 | `PP` | adposition/NP order | postpositions | prepositions |
| 5 | `Adj` | adjective/noun order | Adjective-Noun | Noun-Adjective |
| 6 | `Rel` | relative clause/noun order | relative clause-Noun | Noun-relative clause |
| 7 | `Case` | case-affix position | case suffixes | case prefixes |
| 8 | `VPComp` | complement clause/verb order | verb after complement clause | verb before complement clause |

**Where each value comes from:**
- Switches 1, 2, 4, 5, 6, 7 (`SV`, `OV`, `PP`, `Adj`, `Rel`, `Case`) come
  straight from **WALS** (wals.info) -- chapters **82A, 83A, 85A, 87A, 90A,
  51A** respectively. If a language has no WALS value for one of these, the
  script falls back to the majority value among WALS languages sharing the
  same **genus**, then **family**, then **macroarea** -- whichever level
  first gives a clear majority. Which source was actually used (`wals`,
  `genus`, `family`, or `macroarea`) is recorded per feature in
  `work/wals_switches.csv`, so every imputed value is auditable.
- Switch 3 (`Compl`) has **no WALS feature mapped to it in this codebase**.
  It's always left as a literal placeholder character (`"2"`, not `0`/`1`)
  in every `lang_id` -- i.e. no single value is resolved per language. This
  is intentional, not a gap: `04_generate_target_corpora.py` fills the
  wildcard with both `0` and `1` for every language, generating a
  `complAfter` *and* a `complBefore` corpus side by side (see "Generating
  the per-language corpora" below), so both complementizer orders are
  always available to test rather than picking one arbitrarily.
- **Switch 8 (`VPComp`) is one we added ourselves** -- it isn't part of the
  original fork's switch set. No single WALS feature captures
  clausal-complement/verb order cleanly, so it defaults to whatever
  `OV_switch` is, per Grambank's **GB135** ("do clausal objects usually
  occur in the same position as nominal objects?"), which holds for most
  languages -- but not all. **Hindi and Urdu are manually overridden**
  (`VPCOMP_OVERRIDES` in the script): both are verb-final (`OV_switch=0`)
  but place their complement clause *after* the verb via a postposed
  "ki"-style complementizer, so `VPComp_switch=1` despite `OV_switch=0`.
  Add further overrides there only after the same kind of manual,
  language-specific verification.

**Script to run:**
```
python src/grammar-general/01_get_wals_switches.py \
    --wals_csv work/language.csv \
    --out work/wals_switches.csv
```
This recomputes **every** language's switches from scratch (not just new
ones) and overwrites `work/wals_switches.csv`.

**To add a new language:** add one entry to the `LANGUAGES` dict at the top
of `01_get_wals_switches.py` -- `"OurName": "Exact WALS Name"` (must match
a `Name` value in `work/language.csv` exactly; if a language has several
WALS varieties, e.g. Arabic or Tamil, pick the specific one you mean) --
then rerun the command above. Double-check the printed table for any
`WARNING: could not resolve ...` lines (a feature with no data even after
every fallback level) and, separately, verify `VPComp_switch` by hand for
the new language the way we did for Hindi/Urdu above -- the OV-tie default
is only a heuristic, not a guarantee.

## Grammar variants in `work/grammar/`

Two families of hand-written PCFGs live under `work/grammar/`. Within each
family, every file shares the same rules and switch tags -- only the
weights (and, for Dutch, a couple of targeted rule tweaks) differ.

### `grammar_main/` -- the abstract cross-linguistic grammar
- `basic-grammar.gr` -- the original grammar inherited from the fork, with
  an invented, English-like lexicon (`amackist`, `bolician`, ...).
- `basic-grammar-newlex-*.gr` -- rule-for-rule identical to `basic-grammar.gr`,
  but with that lexicon replaced by abstract numbered placeholders
  (`noun_s1`, `tverb_pres_p39`, `adj10`, ...), so word choice carries no
  accidental English-like phonotactic or frequency cues. The `newlex`
  files differ only in how probability mass is distributed within each
  open-class category (`Noun_S/P`, `{I,T}Verb_{Past,Pres}_{S,P}`,
  `Verb_Comp_*`, `Adj`):
  - `basic-grammar-newlex-uniform.gr`: every item in a class is equally likely.
  - `basic-grammar-newlex-zipf-<s>.gr` (`s` in `0.35, 0.7, 0.95, 0.96, 0.97,
    0.98, 0.99, 1`): item weight is proportional to `rank^-s` (`s=1` is
    classical Zipf; lower `s` is flatter, `uniform` is the `s=0` limit).
  - Caveat: `Adj` and each verb tense/number class fold their own
    coordination rule (`Adj -> Adj CC Adj`,
    `TVerb_Pres_S -> TVerb_Pres_S CC TVerb_Pres_S`, etc.) into the *same*
    rule bucket as their own lexical items, so re-weighting the lexicon
    also changes how often those categories coordinate -- Zipf-weighted
    corpora run measurably longer per sentence than uniform ones as a
    side effect of this, not by design.

### `grammar_dutch/` -- Dutch case studies
- `basic-grammar-nld.gr` -- archived; the original real-Dutch-word grammar
  (see `old_dutch_lexicon/`), no longer the active pipeline's input.
- `basic-grammar-nld-simple-*.gr` -- abstract-lexicon Dutch grammars (same
  numbered-placeholder approach as `newlex`), varying along three
  independent axes that combine into the six files:
  - lexical weighting: `uniform` vs `zipf`, as above.
  - `heavy`: the object of the periphrastic perfect (`aux ... participle`)
    is forced to be branching (adjective(s), a PP, or a relative clause)
    rather than a bare noun or pronoun, for testing heavy-NP-shift /
    extraposition preferences. Only that one construction is affected --
    simple-tense transitives are untouched.
  - `fix-zijn`: triples the weight of the `zijn`-auxiliary (unaccusative)
    perfect relative to the three `hebben`-auxiliary perfects, since it's
    naturally sparse otherwise.

## Generating the per-language corpora

Three steps turn `basic-grammar.gr` + `work/wals_switches.csv` into one
100k-sentence corpus per language. Run them in order:

```
# 1. Sample 100k canonical-order trees from the grammar (bracketed, switch-tagged).
python src/artificial-langs/sample_sentences.py \
    -g work/grammar/grammar_main/basic-grammar.gr \
    -n 100000 -m 400 -b 1 -r 1 \
    -O work/grammar/grammar_main
# -> work/grammar/grammar_main/sample_basic-grammar.txt

# 2. Permute those trees per language (2 files each: complAfter / complBefore).
python src/grammar-general/04_generate_target_corpora.py \
    --switches_csv work/wals_switches.csv \
    --base_sample  work/grammar/grammar_main/sample_basic-grammar.txt \
    --out          work/grammar/grammar_main/target_samples_basic_grammar
# -> <language>_<complAfter|complBefore>_<lang_id>.jsonl

# 3. Extract the plain surface text (one sentence per line).
python src/extract_surface_text.py \
    --in_dir work/grammar/grammar_main/target_samples_basic_grammar
# -> matching .jsonl -> .txt
```

Step 1 samples; step 2 only reorders. Every language's corpus is the *same*
100k sentences, differing only in word order, so cross-linguistic
comparisons vary word order alone.

Switch tags (1-8) are baked into the brackets at sampling time. **After any
grammar retag, rerun step 1** or the new/moved switch is silently ignored
for every language. Clear the output dir first if switch count changed, so
stale-`lang_id` files don't linger.