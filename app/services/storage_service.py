import base64
import httpx
from app.config import get_settings


class StorageService:
    """Uploads generated images (base64) to the Laravel storage endpoint
    using multipart/form-data so Laravel can persist them in Supabase."""

    def __init__(self) -> None:
        settings = get_settings()
        self.url = settings.laravel_storage_url
        self.token = settings.laravel_storage_token

    async def upload_base64_image(
        self,
        b64_image: str,
        filename: str = "generated.png",
        content_type: str = "image/png",
        extra_fields: dict | None = None,
    ) -> dict:
        if not self.url:
            raise RuntimeError("LARAVEL_STORAGE_URL is not configured")

        binary = base64.b64decode(b64_image)
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        files = {"file": (filename, binary, content_type)}
        data = extra_fields or {}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.url,
                headers=headers,
                files=files,
                data=data,
            )
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError:
                return {"status": "ok", "raw": resp.text}
