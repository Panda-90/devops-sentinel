# STACK

## Core
- **Language**: Python 3
- **Framework**: FastAPI (Backend)
- **Local LLM**: Ollama

## Dependencies
- **API & Web**: `fastapi`, `requests`, `httpx`
- **Machine Learning / Data**: `numpy`, `faiss` (used for local vector retrieval)
- **LLM SDKs**: `google-genai`, `groq`
- **Utilities**: `python-dotenv`, `pydantic`

## Runtimes & Models
- **Model format**: GGUF (`devops-sentinel-q4_k_m.gguf`)
- **Evaluation Model**: `qwen2.5:3b` (via Ollama)
