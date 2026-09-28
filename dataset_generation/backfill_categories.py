import json

INPUT_PATH = "data/synthetic/devops_synthetic_deduped.jsonl"
OUTPUT_PATH = "data/synthetic/devops_synthetic_final.jsonl"

CATEGORY_KEYWORDS = {
    "Kubernetes": ["kubectl", "kubernetes", "k8s", "pod", "helm", "namespace", "ingress", "deployment", "statefulset", "node"],
    "Docker": ["docker", "dockerfile", "container", "image"],
    "Linux": ["linux", "bash", "shell", "systemd", "chmod", "grep", "awk", "sed", "disk", "cpu", "process", "port", "systemctl", "journalctl"],
    "Git/GitHub": ["git ", "github", "commit", "branch", "merge", "pull request", "rebase", "clone", "reflog"],
    "CI/CD": ["ci/cd", "jenkins", "github actions", "pipeline", "terraform", "deploy", "build stage", "gitlab-ci"],
}


def classify(instruction: str, response: str):
    text = (instruction + " " + response).lower()
    scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in CATEGORY_KEYWORDS.items()}
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return None
    return best_cat


def main():
    items = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    backfilled = 0
    unresolved = []

    for item in items:
        if not item.get("category"):
            guessed = classify(item["instruction"], item["response"])
            if guessed:
                item["category"] = guessed
                backfilled += 1
            else:
                unresolved.append(item)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Backfilled: {backfilled}")
    print(f"Still unresolved: {len(unresolved)}")
    for item in unresolved:
        print(f"  - {item['instruction'][:70]}")


if __name__ == "__main__":
    main()