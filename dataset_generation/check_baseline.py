import json
import random

with open("data/processed/baseline_results.jsonl", "r", encoding="utf-8") as f:
    results = [json.loads(line) for line in f]

error_count = sum(1 for r in results if r["baseline_model_answer"].startswith("ERROR"))
print(f"Total: {len(results)}, Errors: {error_count}")

print("\n=== Random sample ===")
sample = random.sample(results, 2)
for r in sample:
    print(f"\nCategory: {r['category']}")
    print(f"Q: {r['instruction']}")
    print(f"\n[Reference answer - our dataset]:\n{r['reference_answer'][:300]}")
    print(f"\n[Baseline model answer]:\n{r['baseline_model_answer'][:300]}")
    print("\n" + "-"*80)