import argparse
import json
import os


def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_label(item):
    return item.get("relation") or item.get("label") or item.get("rel") or "no_relation"


def main():
    parser = argparse.ArgumentParser(description="Prepare helper files for ChemProt runs.")
    parser.add_argument("--input", required=True, help="Path to ChemProt train/test json or jsonl.")
    parser.add_argument("--relations-output", required=True, help="Output path for relation dictionary json.")
    parser.add_argument("--labels-output", required=True, help="Output path for label list json.")
    args = parser.parse_args()

    data = read_data(args.input)
    labels = [str(get_label(item)) for item in data]
    relations = {label: idx for idx, label in enumerate(sorted(set(labels)))}

    write_json(args.relations_output, relations)
    write_json(args.labels_output, labels)
    print(f"Saved {len(relations)} relations and {len(labels)} labels.")


if __name__ == "__main__":
    main()
