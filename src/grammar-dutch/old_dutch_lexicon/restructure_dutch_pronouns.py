"""
Dutch-specific structural change to work/grammar/basic-grammar-nld.gr: give
personal pronouns their own subject/object forms, instead of routing them
through the generic NP_S/NP_P -> case-affix pathway used for common nouns.

Why this is needed (not just a lexicon fill-in): NP_S currently has a single
role-agnostic `NP_S -> Pronoun_S` expansion, shared by both
`NP_Subj_S -> NP_S Subj` and `NP_Obj -> NP_S Obj` (tag 7, the WALS Case-affix
switch). A single Pronoun_S lexicon entry would therefore appear unchanged
in both subject and object position, with only the Subj/Obj affix bolted on
-- but Dutch pronouns are suppletive for case (hij/hem, zij/haar), not
affixed, so no affix-based lexicon choice can produce that alternation.

This is intentionally Dutch-only, not applied to the shared
work/grammar/basic-grammar.gr template: it's a structural choice specific to
how this particular language's pronoun paradigm works, not a change to the
cross-linguistic base grammar other language-specific grammars are derived
from.

Changes made to basic-grammar-nld.gr:
  1. Remove `NP_S -> Pronoun_S` / `NP_P -> Pronoun_P` and their old nonce
     Pronoun_S/Pronoun_P terminal entries (this pathway is fully replaced
     for pronouns; common nouns keep using NP_S/NP_P + Subj/Obj untouched).
  2. Add role-specific routing rules, untagged (no switch -- case here is
     lexical suppletion, not an orderable affix):
         NP_Subj_S -> Pronoun_Subj_S
         NP_Subj_P -> Pronoun_Subj_P
         NP_Obj    -> Pronoun_Obj_S
         NP_Obj    -> Pronoun_Obj_P
  3. Add terminal lexicon entries for the 4 new categories: the standard
     Dutch personal pronoun paradigm (6 singular persons, 3 plural persons),
     full citation forms (not clitic reductions like 'k/je/ze). The
     3rd-plural object form is the prescriptive "hen" (vs. "hun" for
     indirect objects, vs. colloquial "ze" for both) -- flagged here for a
     native-speaker check, same as other lexicon choices in this pipeline.

Usage:
    python src/restructure_dutch_pronouns.py
"""
import argparse
import pathlib

# (subject form, object form) per person, full citation forms.
SINGULAR_PRONOUNS = [
    ("ik", "mij"),      # 1sg
    ("jij", "jou"),     # 2sg informal
    ("u", "u"),         # 2sg formal
    ("hij", "hem"),     # 3sg masc
    ("zij", "haar"),    # 3sg fem
    ("het", "het"),     # 3sg neut
]
PLURAL_PRONOUNS = [
    ("wij", "ons"),      # 1pl
    ("jullie", "jullie"),  # 2pl
    ("zij", "hen"),      # 3pl -- "hen" is the prescriptive direct-object
                          # form (vs. "hun" indirect, "ze" colloquial for
                          # both); worth a native-speaker check.
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", default="work/grammar/basic-grammar-nld.gr")
    args = parser.parse_args()

    grammar_path = pathlib.Path(args.grammar)
    lines = grammar_path.open().readlines()

    if any("Pronoun_Subj_S" in line for line in lines):
        raise ValueError(
            f"{grammar_path} already has Pronoun_Subj_S entries -- this script "
            f"is not idempotent (it appends rather than replacing in place). "
            f"Re-generate basic-grammar-nld.gr from basic-grammar.gr first if "
            f"you need to re-run this."
        )

    # 1. Drop the old routing rules and their terminal entries.
    def is_old_pronoun_line(line):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            return False
        lhs, rhs = fields[1], fields[2].strip()
        return (lhs, rhs) in {("NP_S", "Pronoun_S"), ("NP_P", "Pronoun_P")} or \
            lhs in ("Pronoun_S", "Pronoun_P")

    new_lines = [line for line in lines if not is_old_pronoun_line(line)]
    removed = len(lines) - len(new_lines)
    print(f"Removed {removed} lines (old NP_S/NP_P -> Pronoun_S/Pronoun_P routing + nonce entries)")

    # 2. Insert the new role-specific routing rules right after the existing
    #    NP_Subj_S / NP_Subj_P / NP_Obj rules, so related rules stay grouped.
    routing_rules = [
        "1\tNP_Subj_S\tPronoun_Subj_S\n",
        "1\tNP_Subj_P\tPronoun_Subj_P\n",
        "1\tNP_Obj\tPronoun_Obj_S\n",
        "1\tNP_Obj\tPronoun_Obj_P\n",
    ]
    insert_at = None
    for i, line in enumerate(new_lines):
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3 and fields[1] == "NP_Obj" and fields[2].strip() == "NP_P Obj":
            insert_at = i + 1
    if insert_at is None:
        raise ValueError("Could not find 'NP_Obj -> NP_P Obj' to insert pronoun routing rules after")
    new_lines[insert_at:insert_at] = routing_rules
    print(f"Inserted {len(routing_rules)} new routing rules after line {insert_at}")

    # 3. Append terminal lexicon entries for the 4 new pre-terminal categories.
    lexicon_lines = ["\n"]
    for subj, _ in SINGULAR_PRONOUNS:
        lexicon_lines.append(f"1\tPronoun_Subj_S\t{subj}\n")
    for _, obj in SINGULAR_PRONOUNS:
        lexicon_lines.append(f"1\tPronoun_Obj_S\t{obj}\n")
    for subj, _ in PLURAL_PRONOUNS:
        lexicon_lines.append(f"1\tPronoun_Subj_P\t{subj}\n")
    for _, obj in PLURAL_PRONOUNS:
        lexicon_lines.append(f"1\tPronoun_Obj_P\t{obj}\n")
    new_lines.extend(lexicon_lines)

    grammar_path.open("w").writelines(new_lines)
    print(f"Written: {grammar_path}")
    print(f"  Pronoun_Subj_S: {len(SINGULAR_PRONOUNS)} entries")
    print(f"  Pronoun_Obj_S:  {len(SINGULAR_PRONOUNS)} entries")
    print(f"  Pronoun_Subj_P: {len(PLURAL_PRONOUNS)} entries")
    print(f"  Pronoun_Obj_P:  {len(PLURAL_PRONOUNS)} entries")


if __name__ == "__main__":
    main()