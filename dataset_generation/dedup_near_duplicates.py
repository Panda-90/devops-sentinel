import json
from difflib import SequenceMatcher

INPUT_PATH = "data/synthetic/devops_synthetic.jsonl"
OUTPUT_PATH = "data/synthetic/devops_synthetic_deduped.jsonl"
SIMILARITY_THRESHOLD = 0.80


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    items = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Loaded {len(items)} total items.")

    kept = []
    removed_count = 0

    for item in items:
        instruction = item["instruction"]
        is_duplicate = False

        for kept_item in kept:
            # Only compare within the same category — faster, and cross-category
            # matches are irrelevant anyway.
            if kept_item.get("category") != item.get("category"):
                continue
            if similarity(instruction, kept_item["instruction"]) >= SIMILARITY_THRESHOLD:
                is_duplicate = True
                break

        if is_duplicate:
            removed_count += 1
        else:
            kept.append(item)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Kept: {len(kept)}")
    print(f"Removed as near-duplicates: {removed_count}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()