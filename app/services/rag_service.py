from typing import List
from functools import lru_cache

import chromadb
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import get_settings


class StoryItem(BaseModel):
    order_index: int
    narrative: str = Field(description="Narasi cerita dalam Bahasa Indonesia")
    image_prompt: str = Field(
        description="Prompt detail untuk generator gambar (DALL-E/Midjourney)"
    )


class FinalOutput(BaseModel):
    title: str = Field(description="Judul cerita anak")
    story: List[StoryItem]
    moral_message: str
    key_points: List[str]


STORY_PROMPT = ChatPromptTemplate.from_template("""
Anda adalah ahli literasi finansial. Gunakan konteks buku panduan di bawah ini untuk membuat cerita.

KONTEKS BUKU:
{context}

PERTANYAAN USER:
{question}

Patuhi struktur JSON yang diminta: 7 section story, moral_message, dan key_points dari buku.
""")


def build_user_prompt(tema: str, nilai_moral: str, fokus_finansial: str) -> str:
    return f"""Kamu adalah AI storyteller untuk anak usia 7 tahun.
Gunakan informasi dari dokumen berikut sebagai dasar utama:
1. Panduan Implementasi Pendidikan Literasi Finansial (OJK)
2. Dongeng Profil Pelajar Pancasila (nilai moral: kejujuran, gotong royong, dll)

=== TUJUAN ===
Buat cerita anak yang:
- Menyenangkan, imajinatif, dan mudah dipahami
- Mengandung literasi finansial berbasis konteks dokumen
- Mengajarkan kebiasaan baik secara natural (tidak menggurui)

=== INPUT DINAMIS ===
Gunakan parameter berikut (jika tidak ada, pilih default yang paling relevan):

Tema Cerita:
{tema}

Nilai Moral:
{nilai_moral}

Fokus Literasi Finansial:
{fokus_finansial}

=== ATURAN CERITA ===
- Buat 7 section (alur berurutan) tanpa nomor, cukup narasi
- Gunakan karakter hewan lucu (tupai, kelinci, burung, rusa kecil, dll)
- Latar harus konsisten: hutan cerah, hangat, penuh warna
- Karakter harus konsisten di semua section
- Gunakan konflik sederhana + solusi berbasis literasi finansial
- Sisipkan konsep dari RAG seperti:
  - kebiasaan menabung
  - konsep bank (disederhanakan, misalnya "tempat menyimpan makanan dengan aman")
  - gotong royong / berbagi
  - membuat pilihan bijak

=== ATURAN PROMPT GAMBAR ===
WAJIB dalam Bahasa Inggris
WAJIB konsisten antar section

Gunakan template style ini:
- children's storybook illustration
- cute animal characters
- bright colorful forest
- warm lighting
- soft shading
- expressive faces
- consistent characters
- magical, cozy atmosphere
- 16:9 aspect ratio

Tambahkan deskripsi aktivitas sesuai scene.

Contoh:
"cute squirrel and rabbit saving acorns together in a bright colorful forest, warm sunlight, children's storybook illustration, soft lighting, expressive faces, magical cozy forest, 16:9"

=== OUTPUT TAMBAHAN ===

Pesan Moral:
[Tulis singkat, ramah anak]

Insight Literasi Finansial (berdasarkan dokumen RAG):
- Menabung:
- Kebutuhan vs Keinginan:
- Perencanaan:
- Berbagi / Gotong Royong:

=== CATATAN PENTING ===
- Gunakan informasi dari dokumen RAG sebagai referensi utama
- Jangan menyebutkan sumber dokumen secara eksplisit
- Gunakan bahasa anak-anak, bukan bahasa formal
- Pastikan cerita engaging, tidak seperti buku pelajaran"""


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        base_url=s.openai_api_base_url,
        api_key=s.openai_api_key,
        default_headers={"User-Agent": "MyCustomClient/1.0"},
        model=s.openai_embedding_model,
        dimensions=s.openai_embedding_dimensions,
    )


@lru_cache
def get_vectorstore() -> Chroma:
    s = get_settings()
    client = chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)
    return Chroma(
        client=client,
        collection_name=s.chroma_collection,
        embedding_function=get_embeddings(),
    )


@lru_cache
def get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        base_url=s.openai_api_base_url,
        api_key=s.openai_api_key,
        default_headers={"User-Agent": "MyCustomClient/1.0"},
        temperature=0.7,
        model=s.openai_llm_model,
    )


def build_rag_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 7})
    structured_llm = get_llm().with_structured_output(FinalOutput)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | STORY_PROMPT
        | structured_llm
    )


def generate_story(tema: str, nilai_moral: str, fokus_finansial: str) -> FinalOutput:
    chain = build_rag_chain()
    user_prompt = build_user_prompt(tema, nilai_moral, fokus_finansial)
    return chain.invoke(user_prompt)
