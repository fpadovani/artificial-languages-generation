"""
Measure the number of intervening surface tokens between a Dutch perfect-
tense auxiliary (is/zijn/heeft/hebben) and its participle, using the
labeled bracketed tree (linearized_tree field) so aux and participle are
paired unambiguously by actual tree structure -- not by heuristics on flat
surface text, which risks mis-pairing in sentences with multiple perfect
constructions (e.g. nested clausal complements, each with their own aux).

For each VP_Perf_S/VP_Perf_P node, its direct children are exactly one of:
    Aux_*_S/P  IVerb_Ptcp_Zijn/Hebben                (0 tokens between: adjacent)
    Aux_*_S/P  NP_Obj  TVerb_Ptcp                     (N tokens: NP_Obj's yield)
    Aux_*_S/P  VerbComp_Ptcp  S_Comp                  (0 tokens: adjacent;
                                                        S_Comp comes AFTER
                                                        the participle, not
                                                        between it and aux)
so "intervening tokens" is just the number of surface words yielded by
whatever sits between the Aux child and the Ptcp child in that node's
child list (empty for the two adjacent shapes).

Usage:
    python src/grammar-dutch/checks/check_aux_ptcp_distance.py work/grammar/target_dutch/dutch_simple_corpus_complBefore.jsonl
"""
import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict

import numpy as np

PTCP_LABELS = {"TVerb_Ptcp", "IVerb_Ptcp_Zijn", "IVerb_Ptcp_Hebben", "VerbComp_Ptcp"}


def parse_tree(linearized: str):
    """(LABEL tok tok (LABEL2 tok )LABEL2 )LABEL  ->  (label, [children]),
    where each child is either a str (terminal) or another (label, children) node."""
    tokens = linearized.split(" ")
    stack = []
    root_children = []
    cur_label = None
    cur_children = root_children
    frame_stack = []
    for tok in tokens:
        if tok.startswith("("):
            label = tok[1:]
            frame_stack.append((cur_label, cur_children))
            cur_label = label
            cur_children = []
        elif tok.startswith(")"):
            label = tok[1:]
            assert label == cur_label, f"mismatched close: {label} != {cur_label}"
            node = (cur_label, cur_children)
            cur_label, cur_children = frame_stack.pop()
            cur_children.append(node)
        else:
            cur_children.append(tok)
    return root_children[0]  # (ROOT, [...])


def yield_terminals(node):
    label, children = node
    words = []
    for c in children:
        if isinstance(c, str):
            words.append(c)
        else:
            words.extend(yield_terminals(c))
    return words


def find_perf_nodes(node, out):
    label, children = node
    if label in ("VP_Perf_S", "VP_Perf_P"):
        out.append(node)
    for c in children:
        if not isinstance(c, str):
            find_perf_nodes(c, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to a .jsonl file with a 'linearized_tree' field per line")
    args = parser.parse_args()

    path = pathlib.Path(args.input)
    distances = []
    by_construction = defaultdict(list)

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tree = parse_tree(json.loads(line)["linearized_tree"])
            perf_nodes = []
            find_perf_nodes(tree, perf_nodes)
            for node in perf_nodes:
                _, children = node
                labels = [c[0] for c in children if not isinstance(c, str)]
                aux_idx = next(i for i, l in enumerate(labels) if l.startswith("Aux_"))
                ptcp_idx = next(i for i, l in enumerate(labels) if l in PTCP_LABELS)
                between = children[aux_idx + 1:ptcp_idx] if ptcp_idx > aux_idx else children[ptcp_idx + 1:aux_idx]
                n_between = sum(len(yield_terminals(c)) for c in between if not isinstance(c, str))
                distances.append(n_between)
                by_construction[labels[ptcp_idx]].append(n_between)

    distances = np.array(distances)
    print(f"Total perfect-tense (aux+participle) instances found: {len(distances)}\n")
    print(f"Overall intervening-token distance: mean={distances.mean():.3f}  median={np.median(distances):.1f}  "
          f"max={distances.max()}  (0 = aux and participle directly adjacent)")
    print(f"Distribution: {dict(sorted(Counter(distances.tolist()).items())[:10])} ...\n")

    print("By construction (participle category):")
    for label, ds in sorted(by_construction.items()):
        ds = np.array(ds)
        print(f"  {label:20s} n={len(ds):6d}  mean={ds.mean():.3f}  median={np.median(ds):.1f}  "
              f"max={ds.max()}  nonzero={np.mean(ds > 0) * 100:.1f}%")


if __name__ == "__main__":
    main()
