import os
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from chromadb.config import Settings
from config import llm, emb, VECTORDB_PATH, PDF_PATH
from utils import read_pdf_bytes_from_path, extract_pdf_text, chunk_text, clean_question_remove_uris

# ===== VectorDB =====
vectordb = Chroma(
    embedding_function=emb,
    persist_directory=VECTORDB_PATH,
    client_settings=Settings(anonymized_telemetry=False, persist_directory=VECTORDB_PATH)
)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# ===== Prompt =====
PDF_READER_SYS = """Bạn là một trợ lý AI chuyên đọc tài liệu PDF về Luật Lao động Việt Nam.
Chỉ trả lời dựa trên nội dung có trong tài liệu. Nếu câu hỏi không liên quan, hãy từ chối lịch sự.
Trả lời rõ ràng, chuẩn mực, trích dẫn điều khoản hoặc mục khi có thể."""

# ===== Helper =====
def build_context_from_hits(hits, max_chars=6000):
    ctx, total = [], 0
    for i, h in enumerate(hits, 1):
        seg = f"[{i}] {h.page_content.strip()}"
        if total + len(seg) > max_chars:
            break
        ctx.append(seg)
        total += len(seg)
    return "\n\n".join(ctx)

def check_vectordb_exists():
    try:
        return len(vectordb.similarity_search("test", k=1)) > 0
    except Exception:
        return False

def get_vectordb_stats() -> Dict[str, Any]:
    try:
        col = vectordb._collection
        return {"total_documents": col.count(), "path": VECTORDB_PATH, "exists": col.count() > 0}
    except Exception as e:
        return {"total_documents": 0, "path": VECTORDB_PATH, "exists": False, "error": str(e)}

def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Không tìm thấy PDF: {PDF_PATH}")
        return False
    print(f"📖 Đang đọc: {os.path.basename(PDF_PATH)}")
    text = extract_pdf_text(read_pdf_bytes_from_path(PDF_PATH))
    if not text.strip():
        print("PDF trống hoặc không đọc được nội dung.")
        return False
    chunks = chunk_text(text)
    docs = [Document(page_content=ch, metadata={"chunk": i}) for i, ch in enumerate(chunks)]
    vectordb.add_documents(docs)
    print(f"✅ Đã nạp {len(docs)} đoạn vào VectorDB ({VECTORDB_PATH})")
    return True

def clear_vectordb():
    try:
        vectordb._collection.delete()
        print("🗑️ Đã xóa toàn bộ dữ liệu trong VectorDB")
        return True
    except Exception as e:
        print(f"Lỗi khi xóa VectorDB: {e}")
        return False

def process_pdf_question(i: Dict[str, Any]) -> str:
    message = i["message"]
    history: List[BaseMessage] = i.get("history", [])
    if not check_vectordb_exists():
        print("VectorDB trống → nạp PDF...")
        if not ingest_pdf():
            return "Xin lỗi, tôi gặp lỗi khi nạp PDF."
    query = clean_question_remove_uris(message)
    hits = retriever.invoke(query)
    if not hits:
        return "Xin lỗi, tôi không tìm thấy thông tin trong tài liệu PDF."
    context = build_context_from_hits(hits)
    msgs = [SystemMessage(content=PDF_READER_SYS)] + history[-10:]
    msgs.append(HumanMessage(content=f"Câu hỏi: {query}\n\nTài liệu:\n{context}"))
    ans = llm.invoke(msgs).content
    return ans + f"\n\n_Nguồn: {os.path.basename(PDF_PATH)}_"

# ===== Chain & Memory =====
pdf_chain = RunnableLambda(process_pdf_question)
store: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chatbot = RunnableWithMessageHistory(pdf_chain, get_history, input_messages_key="message", history_messages_key="history")
