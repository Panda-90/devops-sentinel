import json
import time
import requests

TEST_PATH = "data/processed/test.jsonl"
OUTPUT_PATH = "data/processed/baseline_results.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def query_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    test_items = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_items.append(json.loads(line))

    print(f"Loaded {len(test_items)} test questions.")

    results = []
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
        for i, item in enumerate(test_items, 1):
            print(f"[{i}/{len(test_items)}] {item['instruction'][:60]}...")

            try:
                model_answer = query_ollama(item["instruction"])
            except Exception as e:
                print(f"  Error: {e}")
                model_answer = f"ERROR: {e}"

            record = {
                "category": item.get("category", "UNKNOWN"),
                "instruction": item["instruction"],
                "reference_answer": item["response"],
                "baseline_model_answer": model_answer,
            }
            results.append(record)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(0.5)  # small pause between local calls, easy on CPU

    print(f"\nDone. Saved {len(results)} baseline results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()