"""Clara Study Assistant: a personal, terminal-based AI chat assistant.

Live AI mode uses the Google Gemini Interactions API. If GEMINI_API_KEY is
not configured, the program starts in a limited local demo mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge.md"
DEFAULT_MODEL = "gemini-3.8-flash"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE entries from .env without extra dependencies."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_knowledge(path: Path = KNOWLEDGE_FILE) -> str:
    """Read the user's custom knowledge file."""
    if not path.exists():
        return "Belum ada data khusus. Tambahkan informasi di data/knowledge.md."
    return path.read_text(encoding="utf-8").strip()


def offline_reply(message: str, knowledge: str) -> str:
    """Return a small rule-based reply for offline demonstration only."""
    text = message.casefold().strip()

    if any(word in text for word in ("siapa saya", "tentang saya", "profil saya", "data saya", "profil")):
        return "Berikut data yang dimuat dari data/knowledge.md:\n\n" + knowledge

    if any(word in text for word in ("halo", "hai", "hello", "assalamualaikum", "selamat pagi", "selamat siang", "selamat malam")):
        return (
            "Halo! Saya Clara Study Assistant. Saya bisa membantu belajar dan tugas. "
            "Saat ini saya berada dalam mode DEMO LOKAL. Ketik /help untuk melihat perintah."
        )

    if "virtual environment" in text or "venv" in text or "lingkungan virtual" in text:
        return (
            "Virtual environment adalah lingkungan Python terpisah untuk satu proyek. "
            "Tujuannya agar paket dan versinya tidak bercampur dengan proyek lain. "
            "Di Windows, buat dengan `py -m venv .venv`, lalu aktifkan dengan "
            "`.\\.venv\\Scripts\\Activate.ps1`."
        )

    if "python" in text and ("ai" in text or "kecerdasan buatan" in text):
        return (
            "Python untuk AI tetap menggunakan bahasa Python, tetapi latihan dan proyeknya "
            "diarahkan ke kebutuhan AI: mengolah data, memakai pustaka, berkomunikasi dengan "
            "API/model, serta membangun aplikasi nyata. Ini jawaban demo berbasis aturan, "
            "bukan jawaban dari model generatif."
        )

    if any(word in text for word in ("bantu tugas", "tugas saya", "belajar", "belajar apa", "rencana belajar")):
        return (
            "Saya dapat membantu menjelaskan konsep, menyusun kerangka tugas, memberi contoh "
            "kode, dan membuat rencana belajar. Dalam mode DEMO LOKAL, respons saya terbatas. "
            "Tambahkan GEMINI_API_KEY di file .env untuk mengaktifkan jawaban AI generatif."
        )

    if text in {"/help", "help", "bantuan"}:
        return (
            "Perintah tersedia:\n"
            "- /help : menampilkan bantuan\n"
            "- /profil : membaca data pribadi dari knowledge base\n"
            "- /reset : memulai ulang konteks percakapan AI\n"
            "- /keluar : menutup program\n\n"
            "Mode DEMO LOKAL hanya merespons beberapa topik. Untuk percakapan generatif, "
            "siapkan GEMINI_API_KEY di file .env."
        )

    return (
        "Saya menerima pesanmu, tetapi sekarang masih di mode DEMO LOKAL dengan kemampuan "
        "terbatas. Untuk jawaban yang lebih alami dan dapat membahas topik umum, buat API key "
        "Gemini, simpan di file .env, lalu jalankan ulang program.\n\n"
        "Contoh pertanyaan offline: 'Siapa saya?', 'Apa itu virtual environment?', atau 'Python untuk AI'."
    )


class StudyAssistant:
    """Manage chat requests and keep conversational context in live AI mode."""

    def __init__(self) -> None:
        load_env_file()
        self.knowledge = load_knowledge()
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.previous_interaction_id: Optional[str] = None
        self.client = None
        self.live_mode = False

        if self.api_key:
            try:
                from google import genai  # Imported only when live mode is requested.
            except ImportError as exc:
                raise RuntimeError(
                    "Paket google-genai belum terpasang. Jalankan: pip install -r requirements.txt"
                ) from exc
            self.client = genai.Client(api_key=self.api_key)
            self.live_mode = True

        self.system_instruction = self._build_system_instruction()

    def _build_system_instruction(self) -> str:
        return (
            "Kamu adalah Clara Study Assistant, asisten belajar pribadi yang ramah dan teliti. "
            "Gunakan bahasa Indonesia kecuali pengguna meminta bahasa lain. Jelaskan secara "
            "runtut dan mudah dipahami, beri contoh praktis terutama untuk Python dan "
            "pengembangan perangkat lunak. Jangan mengarang fakta tentang pemilik data. "
            "Jika data tidak tersedia, katakan dengan jujur. Data berikut adalah konteks "
            "personal yang diberikan pengguna; gunakan hanya bila relevan.\n\n"
            "=== DATA PRIBADI / KNOWLEDGE BASE ===\n"
            f"{self.knowledge}\n"
            "=== AKHIR KNOWLEDGE BASE ===\n\n"
            "Jika membantu tugas akademik, dorong pengguna memahami isi dan menyesuaikannya "
            "dengan instruksi dosen, bukan sekadar menyalin tanpa pemeriksaan."
        )

    def reply(self, message: str) -> str:
        """Generate a reply using Gemini or the local fallback."""
        if not self.live_mode or self.client is None:
            return offline_reply(message, self.knowledge)

        kwargs = {
            "model": self.model,
            "input": message,
            "system_instruction": self.system_instruction,
            "generation_config": {"thinking_level": "low"},
        }
        if self.previous_interaction_id:
            kwargs["previous_interaction_id"] = self.previous_interaction_id

        try:
            response = self.client.interactions.create(**kwargs)
        except Exception as exc:  # Keep the CLI open and offer actionable troubleshooting.
            return (
                "Maaf, permintaan ke Gemini gagal. Periksa koneksi internet, API key, kuota, "
                "dan nama model pada file .env. Detail teknis: "
                f"{type(exc).__name__}: {exc}"
            )

        self.previous_interaction_id = getattr(response, "id", None)
        answer = getattr(response, "output_text", "")
        return answer.strip() or "Maaf, model tidak mengembalikan jawaban teks. Coba pertanyaan lain."

    def reset_context(self) -> None:
        """Start a new conversation by clearing the previous interaction ID."""
        self.previous_interaction_id = None


def main() -> None:
    try:
        assistant = StudyAssistant()
    except RuntimeError as exc:
        print(f"[KESALAHAN SETUP] {exc}")
        raise SystemExit(1) from exc

    print("=" * 62)
    print("CLARA STUDY ASSISTANT | AI Chat berbasis Python")
    print("=" * 62)
    if assistant.live_mode:
        print(f"Mode: LIVE AI | Model: {assistant.model}")
    else:
        print("Mode: DEMO LOKAL (jawaban terbatas; Gemini API belum diaktifkan)")
    print("Ketik /help untuk bantuan atau /keluar untuk menutup program.\n")

    while True:
        try:
            user_message = input("Kamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Sampai jumpa. Tetap semangat belajar!")
            break

        if not user_message:
            continue
        command = user_message.casefold()

        if command in {"/keluar", "/exit", "keluar", "exit", "quit"}:
            print("Assistant: Sampai jumpa. Tetap semangat belajar!")
            break
        if command == "/help":
            print(offline_reply("/help", assistant.knowledge) + "\n")
            continue
        if command == "/profil":
            print("Assistant: " + assistant.knowledge + "\n")
            continue
        if command == "/reset":
            assistant.reset_context()
            print("Assistant: Konteks percakapan direset. Silakan mulai topik baru.\n")
            continue

        print("Assistant: " + assistant.reply(user_message) + "\n")


if __name__ == "__main__":
    main()
