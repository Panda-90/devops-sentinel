import json
import os
import requests
import chromadb

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL_NAME = "all-minilm"

DEFAULT_JSONL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "synthetic", "devops_synthetic_final.jsonl"
)

CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "chromadb"
)


def load_docs(jsonl_path: str) -> list[dict]:
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instruction = record.get("instruction", "")
            response = record.get("response", "")
            category = record.get("category", "unknown")
            text = f"{instruction}\n{response}"
            docs.append({
                "category": category,
                "text": text
            })
    return docs


def chunk_text(text: str, max_chars: int = 600, overlap: int = 100) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    step = max_chars - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + max_chars]
        chunks.append(chunk)
        if i + max_chars >= len(text):
            break
    return chunks


def get_embedding(text: str) -> list[float]:
    payload = {"model": EMBED_MODEL_NAME, "prompt": text}
    res = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()
    return data["embedding"]


class RAGRetriever:
    def __init__(self, jsonl_path: str = DEFAULT_JSONL_PATH):
        self.docs = load_docs(jsonl_path)
        print(f"Loaded {len(self.docs)} documents from {jsonl_path}")
        
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name="devops_knowledge")
        
        if self.collection.count() == 0:
            self._build_index()
        else:
            print(f"ChromaDB already initialized with {self.collection.count()} chunks. Skipping embedding.")

    def _build_index(self):
        print("Building ChromaDB index...")
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for doc_idx, doc in enumerate(self.docs):
            chunks = chunk_text(doc["text"])
            for chunk_idx, chunk in enumerate(chunks):
                emb = get_embedding(chunk)
                
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                ids.append(chunk_id)
                documents.append(chunk)
                embeddings.append(emb)
                metadatas.append({
                    "doc_id": str(doc_idx),
                    "chunk_index": chunk_idx,
                    "category": doc["category"]
                })
        
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            
        print(f"Stored {len(ids)} chunks from {len(self.docs)} documents in ChromaDB.")

    def retrieve(self, query: str, k: int = 2) -> list[str]:
        query_emb = get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k * 3
        )
        
        retrieved_docs = []
        seen_docs = set()
        
        if not results['metadatas'] or not results['metadatas'][0]:
            return []
            
        for metadata in results['metadatas'][0]:
            doc_id_str = metadata.get("doc_id")
            if doc_id_str is not None:
                doc_idx = int(doc_id_str)
                if doc_idx not in seen_docs:
                    seen_docs.add(doc_idx)
                    retrieved_docs.append(self.docs[doc_idx]["text"])
                    if len(retrieved_docs) == k:
                        break
                        
        return retrieved_docs
