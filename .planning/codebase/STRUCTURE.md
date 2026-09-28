# STRUCTURE

## Directory Layout
- `backend/`: Contains the FastAPI application and backend scripts.
  - `main.py`: Entry point for the FastAPI server.
  - `test_rag_retrieval.py`: Prototype script for RAG.
- `data/`: Storage for datasets.
  - `raw/`, `processed/`, `synthetic/`: Subdirectories for different stages of the data pipeline.
- `dataset_generation/`: Pipeline scripts for generating and processing synthetic data.
  - Scripts for dataset generation (`generate_dataset.py`), deduplication (`dedup_near_duplicates.py`), splitting, and baseline evaluation.
- `models/`: Destination directory for GGUF model files.
- `Modelfile.txt`: Ollama configuration for the `devops-sentinel` model.
- `requirements.txt`: Python dependency list.
