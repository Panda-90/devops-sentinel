# CONCERNS

## Architecture & Implementation
- **RAG Integration**: RAG retrieval is implemented as a standalone test script (`backend/test_rag_retrieval.py`) using FAISS and Ollama embeddings but is completely missing from the main FastAPI application (`backend/main.py`). The `/chat` endpoint currently forwards requests directly to the LLM without context retrieval.
- **Model Fine-Tuning Setup**: The repository contains `Modelfile.txt` and dataset generation scripts, but there is no pipeline or script dedicated to the actual fine-tuning process of the model (e.g., using LoRA, unsloth, etc.).

## Technical Debt & Code Quality
- **Automated Testing**: There is no formal test suite (e.g., Pytest). Testing is restricted to ad-hoc scripts.
- **Hardcoded Configuration**: Many file paths and service URLs (like `OLLAMA_URL = "http://localhost:11434/api/generate"`) are hardcoded directly into the Python scripts instead of being configured via environment variables.
- **Dataset Pipeline Error Handling**: The dataset generation has rudimentary retries, but a failure on one chunk could cause data loss or require manual intervention to resume cleanly.

## Security
- Potential risk if `.env` gets committed. The `.gitignore` is present, but this should be strictly verified.
