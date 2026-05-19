import requests
from openai import OpenAI
from app.config import get_settings


class OpenAIService:
    """LLM chat uses the main OpenAI endpoint (OPENAI_API_*) via the OpenAI SDK.
    Image generation hits OPENAI_IMAGE_BASE_URL directly via HTTP (Azure-style)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.llm_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base_url,
        )
        self.image_url = settings.openai_image_base_url
        self.image_api_key = settings.openai_image_api_key
        self.llm_model = settings.openai_llm_model
        self.image_model = settings.openai_image_model

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        resp = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def generate_image_b64(
        self,
        prompt: str,
        size: str = "1280x720",
        quality: str = "low",
        output_format: str = "png",
        output_compression: int = 100,
        input_image: str | None = None,
    ) -> str:
        if not self.image_url:
            raise RuntimeError("OPENAI_IMAGE_BASE_URL is not configured")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.image_api_key}",
        }

        width, height = size.split("x")

        payload = {
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "n": 1,
            "model": self.image_model,
        }
        if input_image:
            payload["input_image"] = input_image

        response = requests.post(
            self.image_url, headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()
        return response.json()["data"][0]["b64_json"]
