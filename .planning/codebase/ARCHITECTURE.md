# ARCHITECTURE

## System Overview
The project is a local AI assistant tailored for DevOps, named "Devops Sentinel". The architecture consists of a synthetic dataset generation pipeline and a FastAPI backend serving a locally hosted LLM via Ollama.

## Components
1. **Dataset Generation Pipeline** (`dataset_generation/`):
   - Uses external LLMs (Gemini, Groq) to generate synthetic instruction-response pairs for DevOps scenarios.
   - Cleans, deduplicates (`dedup_near_duplicates.py`), and splits (`split_dataset.py`) the data into train/test sets.
   - Performs baseline evaluation (`run_baseline_eval.py`) using a local model.
2. **Backend API** (`backend/`):
   - A FastAPI application exposing a `/chat` endpoint.
   - Forwards user prompts to a local Ollama instance running the custom `devops-sentinel` model.
3. **Retrieval-Augmented Generation (RAG)**:
   - Experimental implementation in `backend/test_rag_retrieval.py`.
   - Uses Ollama embeddings (`all-minilm`) and FAISS for vector search. (Not yet integrated into the main API).

## Data Flow
- **Generation**: Prompt -> Gemini/Groq -> JSONL file.
- **Inference**: User Request -> FastAPI `/chat` -> Ollama `generate` API -> User Response.
