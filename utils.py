import os, io, re
from typing import List
from pypdf import PdfReader

_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)

def read_pdf_bytes_from_path(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join([p.extract_text() or "" for p in reader.pages]).strip()

def chunk_text(s: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    s = s.strip()
    if not s: return []
    chunks, start, n = [], 0, len(s)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(s[start:end])
        if end == n: break
        start = max(0, end - overlap)
    return chunks

def clean_question_remove_uris(text: str) -> str:
    txt = _URL_RE.sub(" ", text or "")
    toks = [t for t in re.split(r"\s+", txt) if not t.lower().endswith(".pdf")]
    return re.sub(r"\s+", " ", " ".join(toks)).strip()

def print_help():
    print("\n Các lệnh có sẵn:")
    print("  - 'exit' hoặc 'quit': Thoát chương trình")
    print("  - 'clear': Xóa lịch sử cuộc trò chuyện")
    print("  - 'reload': Nạp lại PDF vào hệ thống")
    print("  - 'status': Kiểm tra trạng thái VectorDB")
    print("  - 'reset': Xóa VectorDB và nạp lại từ đầu")
    print("  - 'help': Hiển thị danh sách lệnh này")
