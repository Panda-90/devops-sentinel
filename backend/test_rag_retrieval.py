import requests
import numpy as np
import faiss

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "all-minilm"

docs = [
    "Kubernetes CrashLoopBackOff troubleshooting: Check pod events and logs to identify application crashes.",
    "Docker container exits immediately: Ensure the main process stays in the foreground and doesn't exit.",
    "Linux disk full troubleshooting: Use df -h and du -sh to find large files filling up the filesystem.",
    "Git merge conflict resolution: Use git status, edit conflicting files to resolve markers, and commit.",
    "GitHub Actions workflow failure: Check workflow logs for failed steps, lint errors, or missing secrets.",
    "Kubernetes ImagePullBackOff: Verify the image name, tag, and registry credentials in the pod spec."
]

def get_embedding(text):
    payload = {"model": MODEL_NAME, "prompt": text}
    res = requests.post(OLLAMA_URL, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()
    return data["embedding"]

embeddings = []
for doc in docs:
    emb = get_embedding(doc)
    assert len(emb) == 384, f"Expected dimension 384, got {len(emb)}"
    embeddings.append(emb)

# Convert to float32 numpy array
embeddings_np = np.array(embeddings, dtype=np.float32)

# Normalize vectors for cosine similarity (L2 norm)
faiss.normalize_L2(embeddings_np)

# Create an Inner Product (IP) index. IP on normalized vectors is cosine similarity.
dimension = 384
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_np)

query = "Kubernetes pod is repeatedly restarting and showing CrashLoopBackOff"
query_emb = get_embedding(query)
assert len(query_emb) == 384, f"Expected dimension 384, got {len(query_emb)}"
query_emb_np = np.array([query_emb], dtype=np.float32)

# Normalize query vector
faiss.normalize_L2(query_emb_np)

# Search top 3
k = 3
distances, indices = index.search(query_emb_np, k)

print("query:", query)
for rank in range(k):
    idx = indices[0][rank]
    score = distances[0][rank]
    print(f"rank: {rank + 1}, score: {score:.4f}, doc: {docs[idx]}")
