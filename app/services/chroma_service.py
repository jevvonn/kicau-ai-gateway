import uuid
import chromadb
from app.config import get_settings


class ChromaService:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
        )

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict | None = None,
    ) -> list[str]:
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [dict(metadata or {}, chunk_index=i) for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    def query(self, embedding: list[float], top_k: int = 4) -> list[str]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        docs = result.get("documents") or [[]]
        return docs[0]

    def count(self) -> int:
        return self.collection.count()
