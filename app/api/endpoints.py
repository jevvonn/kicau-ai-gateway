import uuid
from typing import List, Literal

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from app.services.openai_service import OpenAIService
from app.services.storage_service import StorageService
from app.services.rag_service import (
    FinalOutput,
    chat_with_rag,
    generate_story as rag_generate_story,
    get_vectorstore,
)
from app.utils.file_handler import extract_text, chunk_text

router = APIRouter()


NilaiMoral = Literal["Menabung", "Kejujuran", "Bijak", "Berbagi"]
StoryIdea = Literal["Sekolah", "Fantasi", "Belanja", "Jelajah"]


class StoryRequest(BaseModel):
    prompt: str = Field(description="Custom story prompt dari user (Bahasa Indonesia)")
    nilai_moral: NilaiMoral = Field(default="Menabung")
    story_idea: StoryIdea = Field(default="Fantasi")


class ImageRequest(BaseModel):
    prompts: List[str] = Field(description="Daftar prompt gambar, di-generate sekuensial")
    bucket: str = Field(default="generated-images")
    image_size: str = Field(default="1280x720")


class ImageResponse(BaseModel):
    image_urls: List[str]


ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        description="Riwayat percakapan termasuk system prompt dari Laravel"
    )


class ChatResponse(BaseModel):
    messages: List[ChatMessage]


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/story", response_model=FinalOutput)
def generate_story(req: StoryRequest) -> FinalOutput:
    try:
        return rag_generate_story(
            prompt=req.prompt,
            nilai_moral=req.nilai_moral,
            story_idea=req.story_idea,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {e}")


@router.post("/image", response_model=ImageResponse)
async def generate_images(req: ImageRequest) -> ImageResponse:
    if not req.prompts:
        raise HTTPException(status_code=400, detail="prompts must not be empty")

    openai_svc = OpenAIService()
    storage_svc = StorageService()

    batch_id = uuid.uuid4().hex[:8]
    previous_b64: str | None = None
    image_urls: List[str] = []

    for idx, prompt in enumerate(req.prompts):
        try:
            b64 = openai_svc.generate_image_b64(
                prompt=prompt,
                size=req.image_size,
                input_image=previous_b64,
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Image generation failed at index {idx}: {e}",
            )

        previous_b64 = b64

        try:
            upload = await storage_svc.upload_base64_image(
                b64_image=b64,
                bucket=req.bucket,
                filename=f"{batch_id}_{idx}.png",
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Image upload failed at index {idx}: {e}",
            )

        image_urls.append(upload.get("image_url") or "")

    return ImageResponse(image_urls=image_urls)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    if not any(m.role == "user" for m in req.messages):
        raise HTTPException(status_code=400, detail="messages must contain a user message")

    try:
        reply = chat_with_rag([m.model_dump() for m in req.messages])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}")

    return ChatResponse(messages=[ChatMessage(role="assistant", content=reply)])


@router.post("/rag/upload")
async def upload_rag_document(
    file: UploadFile = File(...),
    source: str = Form(default=""),
) -> dict:
    content = await file.read()
    try:
        text = extract_text(file.filename or "", content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text in file")

    vectorstore = get_vectorstore()
    metadatas = [
        {"filename": file.filename, "source": source, "chunk_index": i}
        for i in range(len(chunks))
    ]
    ids = vectorstore.add_texts(texts=chunks, metadatas=metadatas)
    return {
        "filename": file.filename,
        "chunks_indexed": len(ids),
        "ids": ids,
    }


@router.get("/rag/stats")
def rag_stats() -> dict:
    vs = get_vectorstore()
    return {"documents": vs._collection.count()}
