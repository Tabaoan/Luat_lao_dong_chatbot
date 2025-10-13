# Import library and load OpenAI key
import os, re, io
from typing import Dict, Any, List
from pathlib import Path

from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv(override=True)  

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

# ========== ENV ==========
OPENAI__API_KEY = os.getenv("OPENAI__API_KEY")
OPENAI__EMBEDDING_MODEL = os.getenv("OPENAI__EMBEDDING_MODEL")
OPENAI__MODEL_NAME = os.getenv("OPENAI__MODEL_NAME")
OPENAI__TEMPERATURE = os.getenv("OPENAI__TEMPERATURE")

llm = ChatOpenAI(
    api_key=OPENAI__API_KEY,
    model_name=OPENAI__MODEL_NAME,
    temperature=float(OPENAI__TEMPERATURE) if OPENAI__TEMPERATURE else 0
)

# ===== VectorDB + PDF Processing =====

VECTORDB_PATH = r"./vectordb_storage"  
os.makedirs(VECTORDB_PATH, exist_ok=True)  

emb = OpenAIEmbeddings(api_key=OPENAI__API_KEY, model=OPENAI__EMBEDDING_MODEL)

vectordb = Chroma(
    embedding_function=emb,
    persist_directory=VECTORDB_PATH,
    client_settings=Settings(
        anonymized_telemetry=False,
        persist_directory=VECTORDB_PATH
    )
)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# ---- PDF Processing Functions ----
def read_pdf_bytes_from_path(path: str) -> bytes:
    """Đọc file PDF thành bytes"""
    with open(path, "rb") as f:
        return f.read()

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Trích xuất text từ PDF bytes"""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()

def chunk_text(s: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chia text thành các chunk nhỏ"""
    s = s.strip()
    if not s: return []
    chunks, start, n = [], 0, len(s)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(s[start:end])
        if end == n: break
        start = max(0, end - overlap)
    return chunks

# --- ĐƯỜNG DẪN PDF CỐ ĐỊNH ---
PDF_PATH = r"C:\Users\tabao\Downloads\luat_lao_dong\45_2019_QH14_333670.pdf"

# ===== System Prompt cho PDF Reader =====
PDF_READER_SYS = (
    "Bạn là một trợ lý AI chuyên đọc tài liệu PDF được cung cấp và CHỈ trả lời các câu hỏi "
    "LIÊN QUAN TRỰC TIẾP đến Luật Lao động Việt Nam.\n\n"
    "⚙️ Quy tắc đặc biệt:\n"
    "- Nếu người dùng chào hỏi hoặc đặt câu hỏi chung chung (ví dụ: 'xin chào', 'bạn có thể giúp gì cho tôi', 'bạn làm được gì', v.v.) "
    "thì hãy trả lời nguyên văn như sau:\n"
    "'Xin chào! Mình là Chatbot Cổng việc làm Việt Nam. Mình có thể giúp anh/chị tra cứu và giải thích các quy định pháp luật "
    "(luật, nghị định, thông tư...) liên quan đến lao động, việc làm và các lĩnh vực pháp lý khác. "
    "Gõ câu hỏi cụ thể hoặc mô tả tình huống nhé — mình sẽ trả lời ngắn gọn, có dẫn nguồn.'\n\n"
    "Nguyên tắc làm việc:\n"
    "1) Phạm vi: Chỉ trả lời câu hỏi về Luật Lao động Việt Nam và các quy định trong tài liệu PDF. "
    "Nếu câu hỏi không liên quan, lịch sự từ chối: "
    "\"Xin lỗi, tôi chỉ hỗ trợ nội dung liên quan đến Luật Lao động trong tài liệu này\".\n"
    "2) Nguồn thông tin: Chỉ sử dụng thông tin có trong PDF; không suy diễn hay bổ sung kiến thức bên ngoài. "
    "Nếu thông tin không có, trả lời nguyên văn: "
    "\"Thông tin này không có trong tài liệu được cung cấp\".\n"
    "3) Ngôn ngữ: Sử dụng văn phong chuẩn mực, pháp lý, rõ ràng và trung lập; tránh suy đoán hoặc diễn đạt thiếu chính xác.\n"
    "4) Trình bày:\n"
    "   - LUÔN sử dụng gạch đầu dòng (-) hoặc số thứ tự (1., 2., 3...) để liệt kê các ý chính\n"
    "   - Chia thông tin thành các phần rõ ràng, dễ đọc\n"
    "   - Mỗi ý chính nên được trình bày trên một dòng riêng\n"
    "   - Khi giải thích các điều khoản, hãy chia nhỏ thành từng mục cụ thể\n"
    "   - Nếu có thể, nêu rõ số điều, khoản, mục hoặc số trang trong PDF\n"
    "5) Bài tập & ngữ pháp (chỉ khi gắn với nội dung Luật Lao động trong tài liệu):\n"
    "   - Bài tập: giải thích chi tiết từng bước dựa trên nội dung PDF, sử dụng gạch đầu dòng cho mỗi bước.\n"
    "   - Ngữ pháp: giải thích quy tắc và đưa ví dụ trích từ phần quy định trong tài liệu.\n"
    "6) Ngữ cảnh: Sử dụng lịch sử cuộc trò chuyện để hiểu rõ câu hỏi nhưng luôn tuân thủ phạm vi trên.\n"
    "7) Trường hợp mơ hồ: Yêu cầu người dùng làm rõ để bảo đảm câu trả lời chính xác, phù hợp với tài liệu.\n\n"
    "VÍ DỤ ĐỊNH DẠNG TRẢ LỜI TỐT:\n"
    "Theo quy định tại Điều X, nghỉ phép năm được quy định như sau:\n\n"
    "- Người lao động làm việc đủ 12 tháng cho một người sử dụng lao động thì được nghỉ phép năm, hưởng nguyên lương\n"
    "- Số ngày nghỉ phép năm được quy định:\n"
    "  + 12 ngày đối với người làm việc trong điều kiện bình thường\n"
    "  + 14 ngày đối với người làm công việc nặng nhọc, độc hại, nguy hiểm\n"
    "  + 16 ngày đối với người làm công việc đặc biệt nặng nhọc, độc hại, nguy hiểm\n\n"
    "Mục tiêu: Cung cấp câu trả lời chính xác, hữu ích, có cấu trúc rõ ràng với gạch đầu dòng, dễ hiểu về Luật Lao động Việt Nam, "
    "dựa hoàn toàn trên nội dung của tài liệu PDF được cung cấp."
)


def build_context_from_hits(hits, max_chars: int = 6000) -> str:
    """Ghép các đoạn trích cho LLM, giới hạn độ dài để tránh vượt context."""
    ctx = []
    total = 0
    for idx, h in enumerate(hits, start=1):
        seg = f"[{idx}] {h.page_content.strip()}"
        if total + len(seg) > max_chars:
            break
        ctx.append(seg)
        total += len(seg)
    return "\n\n".join(ctx)

def check_vectordb_exists() -> bool:
    """Kiểm tra xem vectorDB đã có dữ liệu chưa"""
    try:
        # Thử search một từ bất kỳ để kiểm tra
        test_results = vectordb.similarity_search("test", k=1)
        return len(test_results) > 0
    except Exception:
        return False

def get_vectordb_stats() -> Dict[str, Any]:
    """Lấy thống kê về vectorDB"""
    try:
        # Lấy collection để kiểm tra số lượng documents
        collection = vectordb._collection
        count = collection.count()
        return {
            "total_documents": count,
            "path": VECTORDB_PATH,
            "exists": count > 0
        }
    except Exception as e:
        return {
            "total_documents": 0,
            "path": VECTORDB_PATH,
            "exists": False,
            "error": str(e)
        }

def ingest_pdf():
    """Đọc và đưa PDF vào VectorDB"""
    new_docs: List[Document] = []
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(PDF_PATH):
            print(f" Không tìm thấy file PDF: {PDF_PATH}")
            return False
            
        print(f"📖 Đang đọc file PDF: {os.path.basename(PDF_PATH)}")
        b = read_pdf_bytes_from_path(PDF_PATH)
        text = extract_pdf_text(b)
        
        if not text.strip():
            print(f"Không đọc được nội dung từ PDF: {PDF_PATH}")
            return False
            
        print(" Đang chia text thành các chunks...")
        chunks = chunk_text(text)
        
        print(f"Tạo {len(chunks)} documents...")
        for i, ch in enumerate(chunks):
            new_docs.append(Document(
                page_content=ch,
                metadata={"source": PDF_PATH, "chunk": i, "total_chunks": len(chunks)}
            ))
            
        if new_docs:
            print(" Đang lưu vào VectorDB...")
            vectordb.add_documents(new_docs)
            print(f"Đã nạp {len(new_docs)} đoạn từ PDF vào VectorDB")
            print(f"VectorDB được lưu tại: {VECTORDB_PATH}")
            return True
        else:
            print(f" Không có nội dung để nạp từ PDF")
            return False
            
    except Exception as e:
        print(f"Lỗi khi xử lý PDF {PDF_PATH}: {e}")
        return False

def clear_vectordb():
    """Xóa toàn bộ dữ liệu trong vectorDB"""
    try:
        # Xóa collection hiện tại
        vectordb._collection.delete()
        print("Đã xóa toàn bộ dữ liệu trong VectorDB")
        return True
    except Exception as e:
        print(f" Lỗi khi xóa VectorDB: {e}")
        return False

_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)

def clean_question_remove_uris(text: str) -> str:
    """Loại bỏ các URL và path .pdf ra khỏi câu hỏi trước khi đưa vào retriever."""
    txt = _URL_RE.sub(" ", text or "")
    toks = re.split(r"\s+", txt)
    toks = [t for t in toks if not t.lower().endswith(".pdf")]
    txt = " ".join(toks)
    return re.sub(r"\s+", " ", txt).strip()

def process_pdf_question(i: Dict[str, Any]) -> str:
    """Xử lý câu hỏi về PDF"""
    message = i["message"]
    history: List[BaseMessage] = i.get("history", [])

    # Kiểm tra và nạp PDF nếu cần
    if not check_vectordb_exists():
        print("VectorDB trống, đang nạp PDF vào hệ thống...")
        if not ingest_pdf():
            return "Xin lỗi, tôi gặp lỗi khi nạp tài liệu PDF. Vui lòng thử lại."

    # Làm sạch câu hỏi
    clean_question = clean_question_remove_uris(message)
    
    # Tìm kiếm trong vectordb
    try:
        hits = retriever.invoke(clean_question)
        if not hits:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu PDF để trả lời câu hỏi của bạn."
        
        # Xây dựng context từ các đoạn tìm được
        context = build_context_from_hits(hits, max_chars=6000)
        
        # Tạo messages để gửi cho LLM
        messages = [SystemMessage(content=PDF_READER_SYS)]
        
        # Thêm lịch sử cuộc trò chuyện
        if history:
            messages.extend(history[-10:]) 
        # Thêm câu hỏi hiện tại với context
        user_message = f"""Câu hỏi: {clean_question}

Nội dung liên quan từ tài liệu PDF:
{context}

Hãy trả lời câu hỏi dựa trên nội dung tài liệu trên."""

        messages.append(HumanMessage(content=user_message))
        
        # Gọi LLM để trả lời
        response = llm.invoke(messages).content
        
        # Thêm thông tin nguồn
        #source_info = f"\n\n_Nguồn: {os.path.basename(PDF_PATH)}_"
        
        return response #+ source_info
        
    except Exception as e:
        return f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi: {str(e)}"

# ===== Main Chain =====
pdf_chain = RunnableLambda(process_pdf_question)

# ===== Memory wrapper =====
store: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str):
    if session_id not in store: 
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chatbot = RunnableWithMessageHistory(
    pdf_chain,
    get_history,
    input_messages_key="message",
    history_messages_key="history"
)

def print_help():
    """In ra danh sách các lệnh có sẵn"""
    print("\n Các lệnh có sẵn:")
    print("  - 'exit' hoặc 'quit': Thoát chương trình")
    print("  - 'clear': Xóa lịch sử cuộc trò chuyện")
    print("  - 'reload': Nạp lại PDF vào hệ thống")
    print("  - 'status': Kiểm tra trạng thái VectorDB")
    print("  - 'reset': Xóa VectorDB và nạp lại từ đầu")
    print("  - 'help': Hiển thị danh sách lệnh này")

def handle_command(command: str, session: str) -> bool:
    """Xử lý các lệnh đặc biệt. Trả về True nếu cần tiếp tục, False nếu thoát"""
    global vectordb, retriever
    
    command = command.lower().strip()
    
    if command in {"exit", "quit"}:
        print("Tạm biệt!")
        return False
        
    elif command == "clear":
        if session in store:
            store[session].clear()
            print(" Đã xóa lịch sử cuộc trò chuyện.\n")
        return True
        
    elif command == "reload":
        print("Đang nạp lại PDF...")
        if ingest_pdf():
            print("Đã nạp lại PDF thành công!\n")
        else:
            print("Lỗi khi nạp lại PDF\n")
        return True
        
    elif command == "status":
        stats = get_vectordb_stats()
        if stats["exists"]:
            print(f"VectorDB có {stats['total_documents']} documents")
            print(f" Đường dẫn: {stats['path']}")
            print(f" File PDF: {os.path.basename(PDF_PATH)}")
        else:
            print(" VectorDB trống hoặc chưa được khởi tạo")
            if "error" in stats:
                print(f"Chi tiết: {stats['error']}")
        print()
        return True
        
    elif command == "reset":
        print("Đang xóa VectorDB hiện tại...")
        if clear_vectordb():
            print("Đang nạp lại PDF từ đầu...")
            # Tạo lại vectordb
            vectordb = Chroma(
                embedding_function=emb,
                persist_directory=VECTORDB_PATH
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})
            
            if ingest_pdf():
                print("Đã reset và nạp lại PDF thành công!\n")
            else:
                print(" Lỗi khi nạp lại PDF sau reset\n")
        else:
            print(" Lỗi khi reset VectorDB\n")
        return True
        
    elif command == "help":
        print_help()
        return True
        
    else:
        return True  

# ===== CLI Interface =====
if __name__ == "__main__":
    session = "pdf_reader_session"
    
    print("=" * 60)
    print("CHATBOT Cổng Việc Làm Việt Nam")
    print("=" * 60)
    print(f"Tài liệu: {os.path.basename(PDF_PATH)}")
    print(f"VectorDB: {VECTORDB_PATH}")
    print("Tôi chỉ trả lời các câu hỏi về Luật lao động")
    
    print_help()
    print("=" * 60)
    
    # Kiểm tra và nạp PDF nếu cần
    if check_vectordb_exists():
        stats = get_vectordb_stats()
        print(f"VectorDB đã có {stats['total_documents']} documents, sẵn sàng trả lời!\n")
    else:
        print(" Đang nạp tài liệu PDF lần đầu...")
        if ingest_pdf():
            print(" Sẵn sàng trả lời câu hỏi!\n")
        else:
            print("Không thể nạp PDF. Vui lòng kiểm tra đường dẫn file.\n")
    
    # Main conversation loop
    while True:
        try:
            message = input(" Bạn: ").strip()
            
            if not message:
                print("  Vui lòng nhập câu hỏi hoặc lệnh.\n")
                continue
            
            # Xử lý lệnh đặc biệt
            if not handle_command(message, session):
                break  
                
            # Nếu không phải lệnh đặc biệt, xử lý như câu hỏi
            if message.lower() not in ["clear", "reload", "status", "reset", "help"]:
                print(" Đang tìm kiếm trong tài liệu...")
                
                try:
                    response = chatbot.invoke(
                        {"message": message}, 
                        config={"configurable": {"session_id": session}}
                    )
                    print(f"\n Bot: {response}\n")
                    print("-" * 60 + "\n")
                except Exception as e:
                    print(f" Lỗi khi xử lý câu hỏi: {e}\n")
            
        except KeyboardInterrupt:
            print("\n Tạm biệt!")
            break
        except EOFError:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f" Lỗi không mong muốn: {e}\n")