# INTEGRATIONS

## External APIs
- **Google Gemini API**: Used via `google-genai` SDK for synthetic dataset generation (`generate_dataset.py`).
- **Groq API**: Used via `groq` SDK for synthetic dataset generation (`generate_dataset.py`) to diversify generated data.

## Local Services
- **Ollama API**: Local instance running on `http://localhost:11434`. Used for inference in backend (`/api/generate`) and for generating embeddings (`/api/embeddings`) in the RAG retrieval test script.
