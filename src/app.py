from fastapi import FastAPI
from pydantic import BaseModel
import ollama


app = FastAPI(title="LLM API")

class ChatRequest(BaseModel):
    message: str
    model: str = "llama3.2:1b"

class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        response = ollama.chat(
            model=req.model,
            messages=[{"role": "user", "content": req.message}],
        )
        return ChatResponse(response=response["message"]["content"])
    except Exception as e:
        return ChatResponse(response=f"Error: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="[IP_ADDRESS]", port=8000)