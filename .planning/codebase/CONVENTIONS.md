# CONVENTIONS

## Code Style
- **Python**: Standard Python conventions. No explicit linting (e.g., `flake8` or `black` configuration) is evident in the root.
- **Data Formats**: Uses `jsonl` (JSON Lines) extensively for dataset storage (each line is a valid JSON object representing an instruction-response pair).

## Configuration
- Environment variables are managed using `.env` via `python-dotenv` (e.g., for API keys like `GEMINI_API_KEY`, `GROQ_API_KEY`).
- Hardcoded constants used for paths (e.g., `OUTPUT_PATH = "data/synthetic/devops_synthetic.jsonl"`) and local service URLs (e.g., `OLLAMA_URL = "http://localhost:11434/api/generate"`).

## Error Handling
- The FastAPI backend catches `requests.exceptions` and maps them to standard HTTP exceptions (504, 503, 500).
- The dataset generation scripts implement simple retry loops with exponential backoff and error logging for API calls.
