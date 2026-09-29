from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from .rag import RAGRetriever

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "devops-sentinel"

retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    try:
        retriever = RAGRetriever()
        print("RAG retriever initialized.")
    except Exception as e:
        print(f"Warning: Failed to initialize RAG retriever: {e}")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    context = ""
    if retriever:
        try:
            docs = retriever.retrieve(req.message, k=2)
            if docs:
                context = "Relevant documentation:\n" + "\n".join(f"- {d}" for d in docs) + "\n\n"
        except Exception as e:
            print(f"Warning: Retrieval failed: {e}")

    prompt = req.message
    if context:
        prompt = f"{context}User question: {req.message}"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama API timeout")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama API is unavailable")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = res.json()
    if "response" not in data:
        raise HTTPException(status_code=500, detail="Invalid response from Ollama API")

    return ChatResponse(response=data["response"])
