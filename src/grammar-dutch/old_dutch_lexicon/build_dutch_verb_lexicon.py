"""
Mine frequent Dutch verbs for the IVerb (intransitive), TVerb (transitive),
and Verb_Comp (clause-embedding) terminal categories in
work/grammar/basic-grammar-nld.gr, with all four subject-agreement/tense
surface forms (Pres_Sing, Pres_Plur, Past_Sing, Past_Plur) per verb.

This is a two-stage pipeline, because the two things needed -- which verbs
belong to which subcategorization class, and what their inflected surface
forms are -- come from different kinds of evidence.

Stage 1 (valency classification): `wordfreq` only gives word frequencies, no
sentence context, so it cannot say whether a verb takes a direct object.
Isolated-word POS tagging can't either. What's needed is a parsed corpus.
The Alpino Treebank (~7,136 Dutch sentences, distributed via
`nltk.download("alpino")`) is used for this: for every clause node, we look
at the verb's sibling dependents and check for an `obj1` (direct object) or
a clausal `vc` (verb complement) relation, then relemmatize the surface verb
with spaCy for a consistent lemma space (Alpino's own `root` attribute is
not a reliable infinitive -- e.g. it gives "gezegd" rather than "zeggen").
A small stoplist excludes modal/auxiliary verbs (zijn, hebben, worden,
zullen, kunnen, moeten, mogen, willen), which are functional, not lexical.

  - IVerb / TVerb candidates: lemmas whose Alpino evidence is *exclusively*
    intransitive / transitive across the whole corpus.
  - Verb_Comp candidates: lemmas attested with a clausal complement at least
    once (NOT required to be exclusive -- most complement-taking verbs like
    "denken"/"zien"/"beweren" are also used transitively or intransitively
    in other sentences; requiring exclusivity left only 3 candidates against
    a target of 22, so this category needs "can this verb take a clause",
    not "does this verb only ever take a clause").
  - Because Verb_Comp uses an inclusive rule, its pool can overlap with the
    IVerb/TVerb pools; Verb_Comp (the scarcest category) is assigned first
    and those lemmas are removed from the IVerb/TVerb pools so the three
    lexicons stay disjoint.

Stage 2 (surface forms): earlier versions of this script tried to mine the 4
(Tense, Number) forms per lemma the same way src/build_dutch_lexicon.py
mines noun singular/plural forms -- tagging isolated wordfreq candidates
with spaCy. That failed badly for verbs: Dutch present-tense plural is
spelled identically to the bare infinitive ("lopen" = "to walk" = "they
walk"), so an isolated token gives spaCy no way to disambiguate, and only
29 of 3,643 candidate lemmas ended up with all 4 forms attested (need 248).

Instead, Stage 2 uses UniMorph (https://github.com/unimorph/nld,
CC-BY-SA), an open cross-linguistic inflectional lexicon built from
Wiktionary: for each lemma it lists every attested inflected form tagged
with its full morphological feature bundle (e.g. `V;IND;PST;3;SG`), which
covers exactly the 4 forms needed and includes irregular ablaut correctly
(e.g. "denken" -> "dacht"/"dachten", "zien" -> "zag"/"zagen") without any
hand-written conjugation rules. A copy is checked into
work/external/unimorph_nld.txt (~1.9MB), the same convention as
work/language.csv for the WALS data.

The final candidate pool per category is the intersection of Stage 1's
classification and Stage 2's full-paradigm coverage, ranked by the lemma's
own wordfreq frequency.

One-time setup:
    python -c "import nltk; nltk.download('alpino')"
    (work/external/unimorph_nld.txt is already checked in; to refresh it:
     curl -sL https://raw.githubusercontent.com/unimorph/nld/master/nld \
       -o work/external/unimorph_nld.txt)

Usage:
    python src/build_dutch_verb_lexicon.py
"""
import argparse
import pathlib
import xml.etree.ElementTree as ET
from collections import defaultdict

import spacy
from wordfreq import zipf_frequency

AUX_STOPLIST = {"zijn", "hebben", "worden", "zullen", "kunnen", "moeten", "mogen", "willen"}
CLAUSE_CATS = {"smain", "ssub", "sv1", "inf", "ppart", "whq", "whsub"}

# Lemmas found by manual review (post-hoc, via a generated sample) to be rare
# verb senses whose ranking got hijacked by an unrelated, much more common
# non-verb reading of the same string -- e.g. "velen" (rare/archaic verb "to
# tolerate", as in "niet kunnen velen") was ranked as if common because
# `zipf_frequency("velen", "nl")` actually reflects the unrelated, very
# common pronoun "velen" ("many [people]", plural of "veel"). Isolated-tag
# POS-mismatch alone is NOT a reliable detector for this (most Dutch verb
# infinitives share spelling with a *related* plural noun, e.g. "boeken" =
# "books"/"to book", which is normal, not a bug) -- these are confirmed
# cases only.
HOMOGRAPH_STOPLIST = {"velen"}

UNIMORPH_TAGS = {
    "pres_sing": "V;IND;PRS;3;SG",
    "pres_plur": "V;IND;PRS;PL",
    "past_sing": "V;IND;PST;3;SG",
    "past_plur": "V;IND;PST;PL",
}


def alpino_valency_evidence(alpino_xml, nlp):
    """Per spaCy-lemma counts of clause instances where the verb has a
    direct object (obj1), a clausal complement (vc), or neither (su_only)."""
    tree = ET.parse(alpino_xml)
    verb_surfaces = set()
    raw_evidence = []  # (surface_word, has_obj1, has_vc_clause, su_only)

    for ds in tree.getroot().findall("alpino_ds"):
        top = ds.find("node")
        for node in top.iter("node"):
            if node.get("cat") not in CLAUSE_CATS:
                continue
            children = list(node)
            verb = next((c for c in children if c.get("pos") == "verb" and c.get("rel") == "hd"), None)
            if verb is None:
                continue
            word = verb.get("word")
            verb_surfaces.add(word)
            rels = [c.get("rel") for c in children]
            has_obj1 = "obj1" in rels
            has_vc_clause = False
            if "vc" in rels:
                vc_child = next(c for c in children if c.get("rel") == "vc")
                has_vc_clause = vc_child.get("cat") in ("cp", "ssub", "whsub")
            su_only = not has_obj1 and not has_vc_clause
            raw_evidence.append((word, has_obj1, has_vc_clause, su_only))

    surf_list = list(verb_surfaces)
    docs = nlp.pipe(surf_list, batch_size=500)
    surf2lemma = {w: doc[0].lemma_ for w, doc in zip(surf_list, docs) if len(doc) == 1}

    evidence = defaultdict(lambda: {"obj1": 0, "vc_clause": 0, "su_only": 0})
    for word, has_obj1, has_vc_clause, su_only in raw_evidence:
        lemma = surf2lemma.get(word)
        if lemma is None or lemma in AUX_STOPLIST:
            continue
        if has_obj1:
            evidence[lemma]["obj1"] += 1
        elif has_vc_clause:
            evidence[lemma]["vc_clause"] += 1
        elif su_only:
            evidence[lemma]["su_only"] += 1
    return evidence


def classify_valency(evidence):
    """trans/intrans: exclusive evidence only. comp: inclusive (>=1 clausal
    complement attestation), since most comp-taking verbs are ambitransitive
    in other sentences too -- see module docstring."""
    trans, intrans, comp = set(), set(), set()
    for lemma, e in evidence.items():
        if e["obj1"] > 0 and e["vc_clause"] == 0 and e["su_only"] == 0:
            trans.add(lemma)
        if e["su_only"] > 0 and e["obj1"] == 0 and e["vc_clause"] == 0:
            intrans.add(lemma)
        if e["vc_clause"] > 0:
            comp.add(lemma)
    return trans, intrans, comp


def load_unimorph_paradigms(unimorph_tsv):
    """lemma -> {pres_sing/pres_plur/past_sing/past_plur: surface_form},
    restricted to lemmas with all 4 needed tags attested."""
    wanted_tags = set(UNIMORPH_TAGS.values())
    tag2key = {v: k for k, v in UNIMORPH_TAGS.items()}

    raw = defaultdict(dict)
    with open(unimorph_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            lemma, form, tag = parts
            if tag in wanted_tags:
                raw[lemma][tag2key[tag]] = form

    return {lemma: slot for lemma, slot in raw.items() if len(slot) == 4}


def build_rows(lemmas, paradigms, n_target, category_label):
    candidates = sorted(lemmas, key=lambda l: zipf_frequency(l, "nl"), reverse=True)
    if len(candidates) < n_target:
        raise ValueError(
            f"Only found {len(candidates)} full-paradigm {category_label} lemmas, "
            f"need {n_target}."
        )
    rows = []
    for lemma in candidates[:n_target]:
        slot = paradigms[lemma]
        rows.append({"lemma": lemma, **slot})
    return rows


def find_terminal_lines(lines, lhs):
    indices = []
    for i, line in enumerate(lines):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and fields[1] == lhs and " " not in fields[2].strip():
            indices.append(i)
    return indices


def splice_grammar(base_grammar, out_grammar, verb_rows):
    with base_grammar.open() as f:
        lines = f.readlines()

    slot_map = {
        "IVerb_Pres_S": ("pres_sing", "IVerb"), "IVerb_Pres_P": ("pres_plur", "IVerb"),
        "IVerb_Past_S": ("past_sing", "IVerb"), "IVerb_Past_P": ("past_plur", "IVerb"),
        "TVerb_Pres_S": ("pres_sing", "TVerb"), "TVerb_Pres_P": ("pres_plur", "TVerb"),
        "TVerb_Past_S": ("past_sing", "TVerb"), "TVerb_Past_P": ("past_plur", "TVerb"),
        "Verb_Comp_Pres_S": ("pres_sing", "Comp"), "Verb_Comp_Pres_P": ("pres_plur", "Comp"),
        "Verb_Comp_Past_S": ("past_sing", "Comp"), "Verb_Comp_Past_P": ("past_plur", "Comp"),
    }

    for lhs, (form_key, category) in slot_map.items():
        idx = find_terminal_lines(lines, lhs)
        rows = verb_rows[category]
        if len(idx) != len(rows):
            raise ValueError(
                f"Found {len(idx)} terminal {lhs} lines in {base_grammar}, "
                f"expected {len(rows)} new words to slot in -- mismatch."
            )
        for i, row in zip(idx, rows):
            lines[i] = f"1\t{lhs}\t{row[form_key]}\n"

    with out_grammar.open("w") as f:
        f.writelines(lines)


def write_audit_csv(out_csv, verb_rows):
    with out_csv.open("w") as f:
        f.write("category,lemma,pres_sing,pres_plur,past_sing,past_plur\n")
        for category, rows in verb_rows.items():
            for row in rows:
                f.write(f"{category},{row['lemma']},{row['pres_sing']},{row['pres_plur']},"
                        f"{row['past_sing']},{row['past_plur']}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpino_xml",
                         default=str(pathlib.Path.home() / "nltk_data/corpora/alpino/alpino.xml"))
    parser.add_argument("--unimorph_tsv", default="work/external/unimorph_nld.txt")
    parser.add_argument("--base_grammar", default="work/grammar/basic-grammar-nld.gr")
    parser.add_argument("--out_grammar", default="work/grammar/basic-grammar-nld.gr")
    parser.add_argument("--out_audit_csv", default="work/grammar/dutch_verb_lexicon.csv")
    args = parser.parse_args()

    nlp = spacy.load("nl_core_news_sm")

    print("Stage 1: parsing Alpino treebank for valency evidence...")
    evidence = alpino_valency_evidence(args.alpino_xml, nlp)
    trans, intrans, comp = classify_valency(evidence)
    trans -= HOMOGRAPH_STOPLIST
    intrans -= HOMOGRAPH_STOPLIST
    comp -= HOMOGRAPH_STOPLIST
    print(f"  trans(exclusive)={len(trans)}, intrans(exclusive)={len(intrans)}, "
          f"comp(inclusive)={len(comp)}")

    print(f"Stage 2: loading full 4-form paradigms from {args.unimorph_tsv}...")
    paradigms = load_unimorph_paradigms(args.unimorph_tsv)
    print(f"  {len(paradigms)} lemmas with all 4 forms attested in UniMorph")

    base_grammar = pathlib.Path(args.base_grammar)
    lines = base_grammar.open().readlines()
    n_iverb = len(find_terminal_lines(lines, "IVerb_Pres_S"))
    n_tverb = len(find_terminal_lines(lines, "TVerb_Pres_S"))
    n_comp = len(find_terminal_lines(lines, "Verb_Comp_Pres_S"))
    print(f"Target counts from {base_grammar}: IVerb={n_iverb}, TVerb={n_tverb}, Verb_Comp={n_comp}")

    comp_candidates = comp & paradigms.keys()
    comp_rows = build_rows(comp_candidates, paradigms, n_comp, "comp-taking")
    used = {row["lemma"] for row in comp_rows}

    trans_candidates = (trans & paradigms.keys()) - used
    tverb_rows = build_rows(trans_candidates, paradigms, n_tverb, "transitive")
    used |= {row["lemma"] for row in tverb_rows}

    intrans_candidates = (intrans & paradigms.keys()) - used
    iverb_rows = build_rows(intrans_candidates, paradigms, n_iverb, "intransitive")

    verb_rows = {"TVerb": tverb_rows, "IVerb": iverb_rows, "Comp": comp_rows}

    out_grammar = pathlib.Path(args.out_grammar)
    splice_grammar(base_grammar, out_grammar, verb_rows)
    print(f"Written: {out_grammar}")

    out_csv = pathlib.Path(args.out_audit_csv)
    write_audit_csv(out_csv, verb_rows)
    print(f"Written: {out_csv}")

    for category, rows in verb_rows.items():
        print(f"\nSample {category} (top 10 by lemma frequency):")
        for row in rows[:10]:
            print(f"  {row['lemma']:12s} pres: {row['pres_sing']:10s}/{row['pres_plur']:10s}  "
                  f"past: {row['past_sing']:10s}/{row['past_plur']:10s}")


if __name__ == "__main__":
    main()