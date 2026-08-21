"""
Extract the plain surface-text sentences from each generated .jsonl corpus
in work/grammar/target_samples/, writing a matching .txt file (one sentence
per line) alongside it -- for use as plain-text pretraining data.

Usage:
    python src/extract_surface_text.py
"""
import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_dir", default="work/grammar/target_samples")
    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    jsonl_files = sorted(in_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {in_dir}")

    for jsonl_file in jsonl_files:
        txt_file = jsonl_file.with_suffix(".txt")
        with jsonl_file.open() as f_in, txt_file.open("w") as f_out:
            for line in f_in:
                surface = json.loads(line)["surface"]
                f_out.write(surface + "\n")
        print(f"{jsonl_file.name} -> {txt_file.name}")

    print(f"\nDone: {len(jsonl_files)} .txt files written to {in_dir}")


if __name__ == "__main__":
    main()