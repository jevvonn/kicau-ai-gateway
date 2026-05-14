from typing import List, Optional
from functools import lru_cache

import chromadb
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import get_settings


class Choice(BaseModel):
    text: str = Field(description="Teks pilihan jawaban untuk anak")
    is_correct: bool = Field(
        description="True jika pilihan ini sesuai nilai moral/literasi finansial yang baik, False jika tidak"
    )


class Question(BaseModel):
    prompt: str = Field(description="Pertanyaan pilihan untuk anak terkait situasi di narasi")
    choices: List[Choice] = Field(description="Tepat 2 pilihan: satu benar dan satu salah")


class StoryItem(BaseModel):
    order_index: int
    narrative: str = Field(
        description="Narasi cerita dalam Bahasa Indonesia, maksimal 2 kalimat"
    )
    image_prompt: str = Field(
        description="Prompt detail untuk generator gambar (DALL-E/Midjourney)"
    )
    question: Optional[Question] = Field(
        default=None,
        description="Opsional. Diisi hanya pada section yang menghadirkan pilihan moral/finansial untuk gamifikasi.",
    )


class FinalOutput(BaseModel):
    title: str = Field(description="Judul cerita anak")
    story: List[StoryItem] = Field(description="6 sampai 7 section cerita berurutan")
    moral_message: str
    key_points: List[str]


STORY_PROMPT = ChatPromptTemplate.from_template("""
Anda adalah ahli literasi finansial. Gunakan konteks buku panduan di bawah ini untuk membuat cerita.

KONTEKS BUKU:
{context}

PERTANYAAN USER:
{question}

Patuhi struktur JSON yang diminta: title, story (6-7 section dengan narrative maks 2 kalimat, sebagian section memiliki question berisi 2 choices dengan is_correct true/false), moral_message, dan key_points dari buku.
""")


def build_user_prompt(tema: str, nilai_moral: str, fokus_finansial: str) -> str:
    return f"""Kamu adalah AI storyteller untuk anak usia 7-10 tahun.
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
- Buat 6 sampai 7 section (alur berurutan) tanpa nomor, cukup narasi
- Setiap narrative WAJIB maksimal 2 kalimat. Singkat, asik, dan natural seperti mendongeng ke anak 7-10 tahun
- Gunakan karakter hewan lucu yang BERVARIASI (rubah, babi, kelinci, tupai, burung hantu, rusa, beruang, musang, berang-berang, dll). Setiap karakter WAJIB punya nama (contoh: Kimo si rubah, Poki si babi, Lulu si kelinci)
- Karakter dan latar harus konsisten di semua section
- Latar konsisten: hutan cerah, hangat, penuh warna
- Bangun konflik sederhana + solusi berbasis literasi finansial / nilai moral
- Sisipkan konsep dari RAG seperti:
  - kebiasaan menabung
  - konsep bank (disederhanakan, misalnya "tempat menyimpan makanan dengan aman")
  - kebutuhan vs keinginan
  - gotong royong / berbagi
  - kejujuran dan membuat pilihan bijak

=== ATURAN PERTANYAAN GAMIFIKASI ===
- Di antara section, sisipkan 2 sampai 3 section yang memuat field "question" untuk gamifikasi (di Laravel)
- Section dengan question: narasi diakhiri dengan situasi/dilema, lalu pertanyaan singkat "Apa yang harus dilakukan ...?"
- Setiap question berisi TEPAT 2 choices:
  - 1 pilihan BENAR (is_correct: true) -> mencerminkan nilai moral & literasi finansial yang baik
  - 1 pilihan SALAH (is_correct: false) -> pilihan yang buruk/egois/tidak bijak, tapi tetap masuk akal sebagai godaan
- Bahasa pilihan natural untuk anak, jangan menggurui
- Section setelah question lanjut menceritakan keputusan baik si tokoh (cerita tetap berakhir positif), bukan mengulang pertanyaan
- Section yang tidak memuat question: field "question" dibiarkan null/tidak diisi

Contoh struktur (ilustratif):
- Section 1: pembuka, perkenalan tokoh (tanpa question)
- Section 2: muncul konflik/pilihan -> ADA question
- Section 3: lanjutan keputusan baik (tanpa question)
- Section 4: konflik kedua -> ADA question
- Section 5: lanjutan (tanpa question)
- Section 6: penutup yang menyenangkan + pesan moral natural (tanpa question)

=== ATURAN PROMPT GAMBAR ===
WAJIB dalam Bahasa Inggris.

PENTING — Cara pakai gambar:
Gambar di-generate berurutan dengan FLUX.2-pro. Hasil gambar section sebelumnya akan dikirim sebagai `input_image` (reference) ke section berikutnya. Artinya:
- Section pertama (order_index 0) = ANCHOR. Ia menetapkan style, palet warna, karakter, dan latar. WAJIB sangat detail.
- Section setelahnya = LANJUTAN. Reference image sudah membawa style + karakter, jadi prompt tidak perlu mengulang seluruh style tag dan deskripsi panjang. Cukup arahkan model untuk mempertahankan apa yang ada di reference dan beri instruksi PERUBAHAN scene.

ATURAN KARAKTER (rancang sebelum menulis):
Tetapkan "character sheet" untuk SETIAP tokoh sekali saja:
- Nama (cth: "Kimo")
- Spesies (cth: "fox")
- Warna bulu/kulit spesifik (cth: "orange fur with white belly")
- Ciri khas tetap (cth: "green scarf", "round glasses", "tiny red backpack")
Karakter sheet ini TIDAK BOLEH berubah antar section (warna, pakaian, ciri tetap sama).

=== STRUKTUR image_prompt PER SECTION ===

A. Section pertama (order_index = 0) — ANCHOR PROMPT (panjang & detail):
Wajib memuat:
1. Style tags lengkap: "children's storybook illustration, cute animal characters, bright colorful forest, warm lighting, soft shading, expressive faces, magical cozy atmosphere, 16:9"
2. Deskripsi LENGKAP setiap karakter yang muncul di section ini, format:
   "<Name> the <species> with <warna> and <ciri khas>"
   Contoh: "Kimo the orange fox with white belly and green scarf, Lulu the white rabbit with pink inner ears and blue overalls"
3. Aktivitas / ekspresi / komposisi scene.
4. Detail latar (mis. "inside a hollow oak tree at golden hour, sunlight through leaves").

B. Section ke-2 dan seterusnya — REFERENCE-BASED PROMPT (fokus pada perubahan, TAPI karakter tetap dideskripsikan ulang):
Format yang disarankan:
"Same art style, color palette, and forest setting as the reference image. <Setiap karakter yang muncul disebut LENGKAP: 'Name the species with <warna> and <ciri khas>'>. <Aktivitas & ekspresi baru>. <Perudahan latar atau properti baru>. 16:9."

Aturan tambahan untuk section lanjutan:
- JANGAN mengulang style tags panjang (children's storybook illustration, soft shading, dst) — reference image sudah membawanya.
- WAJIB ulangi deskripsi LENGKAP setiap karakter yang muncul di section itu, persis sama dengan character sheet (spesies, warna, pakaian, ciri khas). Menyebut hanya nama (mis. "Kimo and Lulu") TIDAK CUKUP — FLUX akan mudah drift.
- Konsistensi deskripsi WAJIB sama di setiap section. Jangan ubah urutan kata atau atribut karakter antar section.
- JIKA ada karakter BARU yang belum pernah muncul, deskripsikan LENGKAP karakter baru tersebut juga (sesuai character sheet).
- Selalu mulai dengan frasa "Same art style, color palette, and forest setting as the reference image" agar model mengikat ke input_image untuk style dan latar.
- Tetap akhiri dengan "16:9".

=== CONTOH ===

Contoh ANCHOR (section 0) — BENAR:
"children's storybook illustration, cute animal characters, bright colorful forest, warm lighting, soft shading, expressive faces, magical cozy atmosphere. Kimo the orange fox with white belly and green scarf and Lulu the white rabbit with pink inner ears and blue overalls standing together near a hollow oak tree, smiling and waving, soft golden sunlight filtering through colorful leaves, mossy ground with tiny flowers, 16:9"

Contoh section lanjutan (karakter sama) — BENAR:
"Same art style, color palette, and forest setting as the reference image. Kimo the orange fox with white belly and green scarf and Lulu the white rabbit with pink inner ears and blue overalls sitting on a tree stump counting acorns in a small wooden bowl, curious and focused expressions, scattered acorns on the moss, 16:9"

Contoh section lanjutan dengan KARAKTER BARU — BENAR:
"Same art style, color palette, and forest setting as the reference image. Kimo the orange fox with white belly and green scarf stands beside a new character Pak Boro the brown bear shopkeeper with a beige apron and a friendly smile, behind a small wooden market stall with baskets of berries, warm afternoon light, 16:9"

Contoh SALAH:
- "Kimo and Lulu collecting acorns, 16:9" — karakter hanya disebut nama, tidak ada deskripsi spesies/warna/pakaian. FLUX akan drift dan menggambar ulang karakter dengan tampilan berbeda.
- "Same characters, ... Kimo and Lulu sharing a snack ..." — kata "Same characters" saja TIDAK CUKUP; tetap wajib mengulang deskripsi visual setiap karakter.
- Mengulang seluruh style tags ("children's storybook illustration, cute animal characters, soft shading, ...") di section lanjutan — ini sudah dibawa reference image, cukup hapus.

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
        reasoning_effort="medium",
        model=s.openai_llm_model,
    )


def build_rag_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 10})
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
