import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

with open("data/synthetic/devops_synthetic_final.jsonl", "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f if line.strip()]

print("=== Category balance ===")
counts = Counter(item.get("category", "UNKNOWN") for item in items)
for cat, n in counts.most_common():
    print(f"{cat}: {n}")

print(f"\nTotal: {len(items)}")

print("\n=== Sample check: shortest 3 responses (often lowest quality) ===")
sorted_by_length = sorted(items, key=lambda x: len(x["response"]))
for item in sorted_by_length[:3]:
    print(f"\nQ: {item['instruction']}")
    print(f"A: {item['response']}")
    