"""
Mine IN-DOMAIN subject-verb number-agreement minimal pairs directly from a
training corpus -- the complement of generate_agreement_minimal_pairs.py's
OUT-of-domain set (novel lexical combinations, guaranteed absent from
training). Here the grammatical half is a REAL sentence the model actually
trained on; the ungrammatical half is built by flipping its governing
verb's number suffix on the SAME lemma (e.g. iverb_pres_s7 -> iverb_pres_p7)
-- no sampling needed, the correct answer is already sitting in the corpus.
Comparing accuracy on this set vs. the out-of-domain one is what actually
tells you whether the model generalizes the agreement rule or is just
pattern-matching memorized material.

Scope matches test_artificial_pairs_plur.txt exactly, so the two sets are
directly comparable: the same 4 construction types (none/pp/rc/coordination)
x congruent/incongruent x transitive/intransitive, restricted to the 8 exact
FLAT token shapes the out-of-domain generator itself produces (no complement-
clause embedding, no adjectives). A corpus line is mined only if it matches
one of these shapes exactly, position by position -- this is a strict
enough match that lines with embedding/adjectives essentially never
accidentally qualify (they'd need a literal "sub"/"rel"/"cc"/"ob" marker and
noun/verb regex at every required position, which additional material makes
implausible), so no separate exclusion filter is needed. Anything more
complex is simply skipped, not miscounted -- a deliberate scoping choice for
apples-to-apples comparability with the out-of-domain set, not a limitation
of what COULD be mined (a future version could walk the full tree the way
03_enforce_dependencies.py does, to also mine constructions embedded inside
longer sentences).

Word order/case-marking assumptions below (subject NP first within its
constituent, PP's own head noun after the PP, RC's own object as attractor,
coordination's second/closer conjunct as the congruency-relevant one) are
specific to English word order (--corpus should be an
English_complBefore_01110101-style file) and to case-marked grammars (see
generate_agreement_minimal_pairs.py's --no_case_markers for why that matters).

Usage:
    python src/blimp-minimal-pairs/mine_in_domain_minimal_pairs.py \\
        --corpus work/grammar/grammar_main/target_samples_newlex_zipf_1/English_complBefore_01110101.txt \\
        --out test_artificial_pairs_zipf_indomain.jsonl \\
        --txt_out test_artificial_pairs_zipf_indomain.txt
"""
import argparse
import json
import pathlib
import random
import re

NOUN_RE = re.compile(r"^noun_([sp])\d+$")
IVERB_RE = re.compile(r"^iverb_(?:pres|past)_([sp])\d+$")
TVERB_RE = re.compile(r"^tverb_(?:pres|past)_([sp])\d+$")
PREP_RE = re.compile(r"^prep\d+$")


def noun_number(tok: str):
    m = NOUN_RE.match(tok)
    return m.group(1).upper() if m else None


def verb_number(tok: str, pattern: re.Pattern):
    m = pattern.match(tok)
    return m.group(1).upper() if m else None


def flip_number(verb_token: str) -> str:
    """Same lemma, opposite number -- e.g. iverb_pres_s7 <-> iverb_pres_p7.
    Works because every verb category's literal token embeds its number as
    a single 's'/'p' character directly before the lemma index (see
    build_simple_grammar.py's renaming convention)."""
    m = re.match(r"^(.*_)([sp])(\d+)$", verb_token)
    if not m:
        raise ValueError(f"Can't flip number on {verb_token!r}")
    prefix, num, idx = m.groups()
    return f"{prefix}{'p' if num == 's' else 's'}{idx}"


def match_shape(tokens):
    """Try each of the 8 flat construction shapes the out-of-domain
    generator produces, in turn. Returns a dict in the same schema as
    generate_agreement_minimal_pairs.py's items, or None if this line
    doesn't match any of them exactly."""
    n = len(tokens)

    # none, intransitive: N sub V
    if n == 3 and tokens[1] == "sub":
        subj_num = noun_number(tokens[0])
        v_num = verb_number(tokens[2], IVERB_RE)
        if subj_num and v_num == subj_num:
            return dict(level="none", transitive=False, subject_number=subj_num,
                        congruent=None, attractor_number=None,
                        prefix=f"{tokens[0]} sub", correct_verb=tokens[2], suffix="")

    # none, transitive: N sub V N2 ob
    elif n == 5 and tokens[1] == "sub" and tokens[4] == "ob":
        subj_num = noun_number(tokens[0])
        v_num = verb_number(tokens[2], TVERB_RE)
        if subj_num and v_num == subj_num and noun_number(tokens[3]):
            return dict(level="none", transitive=True, subject_number=subj_num,
                        congruent=None, attractor_number=None,
                        prefix=f"{tokens[0]} sub", correct_verb=tokens[2], suffix=f"{tokens[3]} ob")

    # pp, intransitive: N1(head) prep N2(attractor) sub V
    if n == 5 and PREP_RE.match(tokens[1]) and tokens[3] == "sub":
        subj_num, attr_num = noun_number(tokens[0]), noun_number(tokens[2])
        v_num = verb_number(tokens[4], IVERB_RE)
        if subj_num and attr_num and v_num == subj_num:
            return dict(level="pp", transitive=False, subject_number=subj_num,
                        congruent=(attr_num == subj_num), attractor_number=attr_num,
                        prefix=f"{tokens[0]} {tokens[1]} {tokens[2]} sub",
                        correct_verb=tokens[4], suffix="")

    # pp, transitive: N1 prep N2 sub V N3 ob
    elif n == 7 and PREP_RE.match(tokens[1]) and tokens[3] == "sub" and tokens[6] == "ob":
        subj_num, attr_num = noun_number(tokens[0]), noun_number(tokens[2])
        v_num = verb_number(tokens[4], TVERB_RE)
        if subj_num and attr_num and v_num == subj_num and noun_number(tokens[5]):
            return dict(level="pp", transitive=True, subject_number=subj_num,
                        congruent=(attr_num == subj_num), attractor_number=attr_num,
                        prefix=f"{tokens[0]} {tokens[1]} {tokens[2]} sub",
                        correct_verb=tokens[4], suffix=f"{tokens[5]} ob")

    # rc, intransitive: N1(head) rel V_rc(TVerb, agrees w/ head) N2(attractor=RC's obj) ob sub V
    if n == 7 and tokens[1] == "rel" and tokens[4] == "ob" and tokens[5] == "sub":
        subj_num, attr_num = noun_number(tokens[0]), noun_number(tokens[3])
        rc_v_num = verb_number(tokens[2], TVERB_RE)
        v_num = verb_number(tokens[6], IVERB_RE)
        if subj_num and attr_num and rc_v_num == subj_num and v_num == subj_num:
            return dict(level="rc", transitive=False, subject_number=subj_num,
                        congruent=(attr_num == subj_num), attractor_number=attr_num,
                        prefix=f"{tokens[0]} rel {tokens[2]} {tokens[3]} ob sub",
                        correct_verb=tokens[6], suffix="")

    # rc, transitive: N1 rel V_rc N2 ob sub V N3 ob
    elif n == 9 and tokens[1] == "rel" and tokens[4] == "ob" and tokens[5] == "sub" and tokens[8] == "ob":
        subj_num, attr_num = noun_number(tokens[0]), noun_number(tokens[3])
        rc_v_num = verb_number(tokens[2], TVERB_RE)
        v_num = verb_number(tokens[6], TVERB_RE)
        if subj_num and attr_num and rc_v_num == subj_num and v_num == subj_num and noun_number(tokens[7]):
            return dict(level="rc", transitive=True, subject_number=subj_num,
                        congruent=(attr_num == subj_num), attractor_number=attr_num,
                        prefix=f"{tokens[0]} rel {tokens[2]} {tokens[3]} ob sub",
                        correct_verb=tokens[6], suffix=f"{tokens[7]} ob")

    # coordination, intransitive: N1 cc N2(closer conjunct) sub V  (V always plural)
    if n == 5 and tokens[1] == "cc" and tokens[3] == "sub":
        n1, n2 = noun_number(tokens[0]), noun_number(tokens[2])
        v_num = verb_number(tokens[4], IVERB_RE)
        if n1 and n2 and v_num == "P":
            return dict(level="coordination", transitive=False, subject_number="P",
                        congruent=(n2 == "P"), attractor_number=None,
                        prefix=f"{tokens[0]} cc {tokens[2]} sub", correct_verb=tokens[4], suffix="")

    # coordination, transitive: N1 cc N2 sub V N3 ob  (V always plural)
    elif n == 7 and tokens[1] == "cc" and tokens[3] == "sub" and tokens[6] == "ob":
        n1, n2 = noun_number(tokens[0]), noun_number(tokens[2])
        v_num = verb_number(tokens[4], TVERB_RE)
        if n1 and n2 and v_num == "P" and noun_number(tokens[5]):
            return dict(level="coordination", transitive=True, subject_number="P",
                        congruent=(n2 == "P"), attractor_number=None,
                        prefix=f"{tokens[0]} cc {tokens[2]} sub", correct_verb=tokens[4], suffix=f"{tokens[5]} ob")

    return None


CONDITIONS = (
    ("none", ["S", "P"], False),
    ("pp", ["S", "P"], True),
    ("rc", ["S", "P"], True),
    ("coordination", ["P"], True),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="the training corpus to mine sentences from")
    parser.add_argument("--n_per_condition", type=int, default=50,
                         help="target count per (level, transitive, congruent, subject_number) cell -- "
                              "matches generate_agreement_minimal_pairs.py's default, so the two sets "
                              "come out the same target size (1200 records) if enough natural matches exist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--txt_out", default=None)
    args = parser.parse_args()

    # bucket every match by its full condition cell first, so sampling can
    # be balanced the same way the out-of-domain set is, rather than just
    # taking the first N lines (which would skew toward whatever's most
    # frequent early in a Zipfian-sampled file).
    buckets = {}
    n_scanned = 0
    with pathlib.Path(args.corpus).open() as f:
        for line in f:
            n_scanned += 1
            tokens = line.rstrip("\n").split(" ")
            if tokens and tokens[-1] == ".":
                tokens = tokens[:-1]
            rec = match_shape(tokens)
            if rec is None:
                continue
            key = (rec["level"], rec["transitive"], rec["congruent"], rec["subject_number"])
            buckets.setdefault(key, []).append(rec)

    print(f"Scanned {n_scanned} corpus lines.")
    print(f"{'level':12s} {'trans':5s} {'congruent':9s} {'subj':4s}  {'found':>6s}  {'kept':>6s}")

    rng = random.Random(args.seed)
    items = []
    for level, subject_numbers, has_congruency in CONDITIONS:
        for transitive in (False, True):
            for congruent in ((True, False) if has_congruency else (None,)):
                for subject_number in subject_numbers:
                    key = (level, transitive, congruent, subject_number)
                    pool = buckets.get(key, [])
                    n_take = min(args.n_per_condition, len(pool))
                    chosen = rng.sample(pool, n_take) if pool else []
                    print(f"{level:12s} {str(transitive):5s} {str(congruent):9s} {subject_number:4s}  "
                          f"{len(pool):6d}  {n_take:6d}")
                    for rec in chosen:
                        rec["wrong_verb"] = flip_number(rec["correct_verb"])
                        rec["pair_id"] = None  # naturally-occurring, not synthetically paired
                        rec["id"] = len(items)
                        items.append(rec)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"\nWrote {len(items)} in-domain minimal pairs to {out_path}")

    if args.txt_out:
        txt_path = pathlib.Path(args.txt_out)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with txt_path.open("w") as f:
            for it in items:
                for key in ("correct_verb", "wrong_verb"):
                    toks = ([it["prefix"]] if it["prefix"] else []) + [it[key]]
                    if it["suffix"]:
                        toks.append(it["suffix"])
                    f.write(" ".join(toks) + " .\n")
        print(f"Wrote plain-text sentences to {txt_path}")


if __name__ == "__main__":
    main()
