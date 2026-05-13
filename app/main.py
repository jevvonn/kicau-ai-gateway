from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router

app = FastAPI(
    title="Kicau AI Gateway",
    description="FastAPI gateway for OpenAI (LLM, embeddings, images), "
                "ChromaDB-backed RAG, and Laravel/Supabase image storage.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root() -> dict:
    return {"service": "kicau-ai-gateway", "docs": "/docs"}
