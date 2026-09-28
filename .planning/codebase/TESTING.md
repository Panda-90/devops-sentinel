# TESTING

## Framework
- No standard testing framework (like `pytest` or `unittest`) is currently configured.

## Test Scripts
Testing is primarily done via ad-hoc standalone scripts:
- `backend/test_rag_retrieval.py`: A manual script to test embedding generation and FAISS retrieval.
- `dataset_generation/test_gemini.py` & `dataset_generation/test_groq.py`: Scripts to verify connectivity and generation from external LLM providers.
- `dataset_generation/run_baseline_eval.py`: Evaluates the baseline model against the test dataset and outputs results to a JSONL file.

## Coverage
- Test coverage is unknown/minimal, as there is no automated test suite for the FastAPI application or the dataset processing utilities.
