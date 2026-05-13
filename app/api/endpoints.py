from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from app.services.openai_service import OpenAIService
from app.services.storage_service import StorageService
from app.services.rag_service import (
    FinalOutput,
    generate_story as rag_generate_story,
    get_vectorstore,
)
from app.utils.file_handler import extract_text, chunk_text

router = APIRouter()


class StoryRequest(BaseModel):
    tema: str = Field(default="Fantasi")
    nilai_moral: str = Field(default="Menabung, Berbagi")
    fokus_finansial: str = Field(default="Menabung di bank")


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    size: str = "1024x1024"
    quality: str = "standard"
    upload: bool = True
    filename: str | None = None


class ImageResponse(BaseModel):
    b64: str
    storage: dict | None = None


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/story", response_model=FinalOutput)
def generate_story(req: StoryRequest) -> FinalOutput:
    try:
        return rag_generate_story(
            tema=req.tema,
            nilai_moral=req.nilai_moral,
            fokus_finansial=req.fokus_finansial,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {e}")


@router.post("/image", response_model=ImageResponse)
async def generate_image(req: ImageRequest) -> ImageResponse:
    openai_svc = OpenAIService()
    b64 = openai_svc.generate_image_b64(
        prompt=req.prompt, size=req.size, quality=req.quality
    )

    storage_result: dict | None = None
    if req.upload:
        storage_svc = StorageService()
        try:
            storage_result = await storage_svc.upload_base64_image(
                b64_image=b64,
                filename=req.filename or "generated.png",
                extra_fields={"prompt": req.prompt},
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Image generated but upload failed: {e}",
            )

    return ImageResponse(b64=b64, storage=storage_result)


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
