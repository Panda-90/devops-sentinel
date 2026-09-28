import json
import random
from collections import defaultdict

INPUT_PATH = "data/synthetic/devops_synthetic_final.jsonl"
TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
TEST_PATH = "data/processed/test.jsonl"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# remaining 0.10 goes to test

RANDOM_SEED = 42  # fixed seed = reproducible split every time we run this

random.seed(RANDOM_SEED)


def main():
    by_category = defaultdict(list)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                by_category[item.get("category", "UNKNOWN")].append(item)

    train_set, val_set, test_set = [], [], []

    print("=== Split breakdown per category ===")
    for category, items in by_category.items():
        random.shuffle(items)  # shuffle within category before splitting
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # test gets whatever remains, to avoid rounding losses

        train_part = items[:n_train]
        val_part = items[n_train:n_train + n_val]
        test_part = items[n_train + n_val:]

        train_set.extend(train_part)
        val_set.extend(val_part)
        test_set.extend(test_part)

        print(f"{category}: total={n}, train={len(train_part)}, val={len(val_part)}, test={len(test_part)}")

    # Shuffle the combined sets too, so categories aren't grouped together in the file
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_PATH, train_set)
    write_jsonl(VAL_PATH, val_set)
    write_jsonl(TEST_PATH, test_set)

    print(f"\nTotal: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    print(f"Saved to {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")


if __name__ == "__main__":
    main()