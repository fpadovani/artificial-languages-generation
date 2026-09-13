"""
Mine frequent Dutch nouns (singular + plural surface forms) and adjectives,
and splice them into a Dutch-lexicalized copy of work/grammar/basic-grammar.gr.

Frequency ranking comes from `wordfreq` (nl); POS, lemma, and Number
morphology come from spaCy's `nl_core_news_sm` pipeline. Candidate words are
tagged in isolation (one token per "sentence"), which is noisier than tagging
in context but good enough at this vocabulary size for a first pass -- check
the printed lists and work/grammar/dutch_lexicon.csv before trusting them.

Noun pairing policy: candidates are grouped by lemma; a lemma is used as a
Noun_S/Noun_P pair only when spaCy found both a singular and a plural surface
form for it among the candidates ("paired"). This keeps every Noun_S/Noun_P
row in the output as the same real word in its two number forms, mirroring
the readability of the original nonce-word lists. Pairs are ranked by their
combined frequency and truncated/padded to exactly match the terminal-rule
counts found in the base grammar; if there aren't enough paired lemmas, the
script raises rather than silently filling with unpaired forms.

Only the terminal lexicon lines for Noun_S / Noun_P / Adj are replaced in the
output grammar (identified as lines whose right-hand side is a single token,
so the recursive rule `Adj -> Adj CC Adj` is left untouched); every other
rule, including the rest of the as-yet-unlexicalized categories (verbs, Prep,
Comp, Subj, Obj, CC), is copied over unchanged.

Usage:
    python src/grammar-dutch/old_dutch_lexicon/build_dutch_lexicon.py
    python src/grammar-dutch/old_dutch_lexicon/build_dutch_lexicon.py --n_candidates 30000
"""
import argparse
import pathlib

import spacy
from wordfreq import top_n_list, zipf_frequency

# Known spaCy nl_core_news_sm mis-tags at the token-in-isolation level:
# ordinals get lemmatized to their cardinal ("eerste" -> "één", "tweede" ->
# "twee"), so a bare numeral can otherwise sneak into the adjective list
# under the ordinal's frequency. Filtered by both surface text and lemma.
DUTCH_NUMERALS = {
    "een", "één", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht",
    "negen", "tien", "elf", "twaalf", "dertien", "veertien", "twintig",
    "dertig", "veertig", "vijftig", "honderd", "duizend",
}

MIN_WORD_LEN = 3

# Hand-verified fixes for spaCy nl_core_news_sm morphologizer errors found by
# manually skimming the mined pairs (it mistagged the adverb "nachts" and the
# adjective "werelds" as NOUN Plur of "nacht"/"wereld"), plus one spelling
# normalization (informal "fotos" -> "foto's"). Applied after mining, and
# recorded as source="manual_correction" in the audit CSV.
MANUAL_NOUN_CORRECTIONS = {
    "wereld": "werelden",
    "nacht": "nachten",
    "foto": "foto's",
}


def collect_candidates(nlp, n_candidates):
    words = top_n_list("nl", n_candidates)
    docs = nlp.pipe(words, batch_size=500)

    nouns = {}  # lemma -> {"Sing": (text, freq), "Plur": (text, freq)}
    adjs = {}   # lemma -> (text, freq)  -- text is always the lemma itself

    for w, doc in zip(words, docs):
        if len(doc) != 1:
            continue
        tok = doc[0]
        if not tok.is_alpha or not tok.text.islower() or len(tok.text) < MIN_WORD_LEN:
            continue
        freq = zipf_frequency(w, "nl")

        if tok.pos_ == "NOUN":
            numbers = tok.morph.get("Number")
            if len(numbers) != 1 or numbers[0] not in ("Sing", "Plur"):
                continue
            slot = nouns.setdefault(tok.lemma_, {})
            number = numbers[0]
            if number not in slot or freq > slot[number][1]:
                slot[number] = (tok.text, freq)

        elif tok.pos_ == "ADJ":
            if tok.text in DUTCH_NUMERALS or tok.lemma_ in DUTCH_NUMERALS:
                continue
            lemma = tok.lemma_.lower()
            if len(lemma) < MIN_WORD_LEN:
                continue
            if lemma not in adjs or freq > adjs[lemma][1]:
                adjs[lemma] = (lemma, freq)

    return nouns, adjs


def build_noun_pairs(nouns, n_target):
    paired = [
        (lemma, slot["Sing"], slot["Plur"])
        for lemma, slot in nouns.items()
        if "Sing" in slot and "Plur" in slot
    ]
    paired.sort(key=lambda r: r[1][1] + r[2][1], reverse=True)

    rows = []
    seen_sing, seen_plur = set(), set()
    for lemma, sing, plur in paired:
        if sing[0] in seen_sing or plur[0] in seen_plur:
            continue  # guard against two lemmas sharing a surface form
        source = "mined"
        if lemma in MANUAL_NOUN_CORRECTIONS:
            corrected = MANUAL_NOUN_CORRECTIONS[lemma]
            plur = (corrected, zipf_frequency(corrected.replace("'", ""), "nl"))
            source = "manual_correction"
        rows.append({"lemma": lemma, "sing": sing[0], "sing_freq": sing[1],
                      "plur": plur[0], "plur_freq": plur[1], "source": source})
        seen_sing.add(sing[0])
        seen_plur.add(plur[0])
        if len(rows) == n_target:
            break

    if len(rows) < n_target:
        raise ValueError(
            f"Only found {len(rows)} paired singular/plural noun lemmas, "
            f"need {n_target}. Re-run with a larger --n_candidates."
        )
    return rows


def build_adjectives(adjs, n_target):
    ranked = sorted(adjs.values(), key=lambda r: r[1], reverse=True)
    seen = set()
    rows = []
    for word, freq in ranked:
        if word in seen:
            continue
        seen.add(word)
        rows.append({"word": word, "freq": freq})
        if len(rows) == n_target:
            break

    if len(rows) < n_target:
        raise ValueError(
            f"Only found {len(rows)} adjective lemmas, need {n_target}. "
            f"Re-run with a larger --n_candidates."
        )
    return rows


def find_terminal_lines(lines, lhs):
    """Line indices whose rule is `1\\t{lhs}\\t<single-token>` (a lexicon
    entry), as opposed to a structural rule like `Adj -> Adj CC Adj` whose
    right-hand side contains spaces."""
    indices = []
    for i, line in enumerate(lines):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and fields[1] == lhs and " " not in fields[2].strip():
            indices.append(i)
    return indices


def splice_grammar(base_grammar, out_grammar, noun_rows, adj_rows):
    with base_grammar.open() as f:
        lines = f.readlines()

    noun_s_idx = find_terminal_lines(lines, "Noun_S")
    noun_p_idx = find_terminal_lines(lines, "Noun_P")
    adj_idx = find_terminal_lines(lines, "Adj")

    for kind, idx, n_target in [("Noun_S", noun_s_idx, len(noun_rows)),
                                 ("Noun_P", noun_p_idx, len(noun_rows)),
                                 ("Adj", adj_idx, len(adj_rows))]:
        if len(idx) != n_target:
            raise ValueError(
                f"Found {len(idx)} terminal {kind} lines in {base_grammar}, "
                f"expected {n_target} new words to slot in -- mismatch."
            )

    for i, row in zip(noun_s_idx, noun_rows):
        lines[i] = f"1\tNoun_S\t{row['sing']}\n"
    for i, row in zip(noun_p_idx, noun_rows):
        lines[i] = f"1\tNoun_P\t{row['plur']}\n"
    for i, row in zip(adj_idx, adj_rows):
        lines[i] = f"1\tAdj\t{row['word']}\n"

    with out_grammar.open("w") as f:
        f.writelines(lines)


def write_audit_csv(out_csv, noun_rows, adj_rows):
    with out_csv.open("w") as f:
        f.write("category,lemma,sing,sing_freq,plur,plur_freq\n")
        for row in noun_rows:
            f.write(f"Noun,{row['lemma']},{row['sing']},{row['sing_freq']},"
                    f"{row['plur']},{row['plur_freq']}\n")
        f.write("category,word,freq\n")
        for row in adj_rows:
            f.write(f"Adj,{row['word']},{row['freq']}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_candidates", type=int, default=30000,
                         help="How many top-frequency wordfreq candidates to POS-tag")
    parser.add_argument("--base_grammar", default="work/grammar/basic-grammar.gr")
    parser.add_argument("--out_grammar", default="work/grammar/basic-grammar-nld.gr")
    parser.add_argument("--out_audit_csv", default="work/grammar/dutch_lexicon.csv")
    args = parser.parse_args()

    nlp = spacy.load("nl_core_news_sm")
    nouns, adjs = collect_candidates(nlp, args.n_candidates)
    print(f"Candidates tagged: {len(nouns)} noun lemmas "
          f"({sum('Sing' in s and 'Plur' in s for s in nouns.values())} paired), "
          f"{len(adjs)} adjective lemmas")

    base_grammar = pathlib.Path(args.base_grammar)
    lines = base_grammar.open().readlines()
    n_noun_s = len(find_terminal_lines(lines, "Noun_S"))
    n_adj = len(find_terminal_lines(lines, "Adj"))
    print(f"Target counts from {base_grammar}: Noun_S/Noun_P={n_noun_s}, Adj={n_adj}")

    noun_rows = build_noun_pairs(nouns, n_noun_s)
    adj_rows = build_adjectives(adjs, n_adj)

    out_grammar = pathlib.Path(args.out_grammar)
    splice_grammar(base_grammar, out_grammar, noun_rows, adj_rows)
    print(f"Written: {out_grammar}")

    out_csv = pathlib.Path(args.out_audit_csv)
    write_audit_csv(out_csv, noun_rows, adj_rows)
    print(f"Written: {out_csv}")

    print("\nSample Noun_S/Noun_P pairs (top 10 by frequency):")
    for row in noun_rows[:10]:
        print(f"  {row['sing']:15s} / {row['plur']:15s}")
    print("\nSample Adj (top 10 by frequency):")
    for row in adj_rows[:10]:
        print(f"  {row['word']}")


if __name__ == "__main__":
    main()
