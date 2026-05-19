import base64
import httpx
from app.config import get_settings


class StorageService:
    """Uploads generated images (base64) to the Laravel storage endpoint
    using multipart/form-data."""

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.laravel_api_url.rstrip("/")

    @property
    def upload_url(self) -> str:
        return f"{self.base_url}/api/storage/upload"

    def absolute_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        return f"{self.base_url}{path_or_url if path_or_url.startswith('/') else '/' + path_or_url}"

    async def upload_base64_image(
        self,
        b64_image: str,
        bucket: str,
        filename: str = "generated.png",
        content_type: str = "image/png",
    ) -> dict:
        if not self.base_url:
            raise RuntimeError("LARAVEL_API_URL is not configured")

        binary = base64.b64decode(b64_image)
        files = {"file": (filename, binary, content_type)}
        data = {"bucket": bucket}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self.upload_url, files=files, data=data)
            resp.raise_for_status()
            try:
                payload = resp.json()
            except ValueError:
                return {"status": "ok", "raw": resp.text}

        if isinstance(payload, dict) and payload.get("url"):
            payload["image_url"] = self.absolute_url(payload["url"])
        return payload
