# ===================== IMPORTS =====================
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
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage 


# ===================== ENV =====================
OPENAI__API_KEY = os.getenv("OPENAI__API_KEY")
OPENAI__EMBEDDING_MODEL = os.getenv("OPENAI__EMBEDDING_MODEL")
OPENAI__MODEL_NAME = os.getenv("OPENAI__MODEL_NAME")
OPENAI__TEMPERATURE = os.getenv("OPENAI__TEMPERATURE")

llm = ChatOpenAI(
    api_key=OPENAI__API_KEY,
    model_name=OPENAI__MODEL_NAME,
    temperature=float(OPENAI__TEMPERATURE) if OPENAI__TEMPERATURE else 0
)

# ===================== VECTORDB =====================
VECTORDB_PATH = r"./vectordb_storage"


emb = OpenAIEmbeddings(api_key=OPENAI__API_KEY, model=OPENAI__EMBEDDING_MODEL)

vectordb = None
retriever = None



# ===================== SYSTEM PROMPT =====================
PDF_READER_SYS = (
    "Bạn là một trợ lý AI pháp lý chuyên đọc hiểu và tra cứu các tài liệu được cung cấp "
    "(bao gồm: Luật, Nghị định, Quyết định, Thông tư, Văn bản hợp nhất, Quy hoạch, Danh mục khu công nghiệp, v.v.). "
    "Nhiệm vụ của bạn là trích xuất và trả lời chính xác các thông tin có trong tài liệu, "
    "đặc biệt liên quan đến Lao động, Dân sự và các Khu công nghiệp, Cụm công nghiệp tại Việt Nam.\n\n"

    
    "⚙️ QUY TẮC ĐẶC BIỆT:\n"
    "- Nếu người dùng chỉ chào hỏi hoặc đặt câu hỏi chung chung (ví dụ: 'xin chào', 'bạn làm được gì', 'giúp tôi với' ...), "
    "hãy trả lời nguyên văn như sau:\n"
    "'Xin chào! Mình là Chatbot Cổng việc làm Việt Nam. Mình có thể giúp anh/chị tra cứu và giải thích các quy định pháp luật "
    "(luật, nghị định, thông tư...) liên quan đến lao động, việc làm, dân sự và các lĩnh vực pháp lý khác. "
    "Gõ câu hỏi cụ thể hoặc mô tả tình huống nhé — mình sẽ trả lời ngắn gọn, có dẫn nguồn.'\n\n"

    "📘 NGUYÊN TẮC CHUNG KHI TRẢ LỜI:\n"
    "1) Phạm vi: Chỉ dựa vào nội dung trong các tài liệu đã được cung cấp; tuyệt đối không sử dụng hoặc suy diễn kiến thức bên ngoài.\n"
    "2) Nguồn trích dẫn: Khi có thể, chỉ ghi rõ nguồn theo quy định (ví dụ: Theo Điều X, Nghị định số Y/NĐ-CP...), "
    "nhưng không được ghi theo dạng liệt kê tài liệu như [1], [2], [3]... Không được phép sử dụng hoặc nhắc đến cụm từ như:'tài liệu PDF', 'trích từ tài liệu PDF', 'dưới đây là thông tin từ tài liệu PDF', hoặc các cụm tương tự."
    "Thay vào đó, chỉ nêu trực tiếp nội dung pháp luật, ví dụ: 'Thông tin liên quan đến Luật Việc làm quy định rằng...'.\n"
    "3) Ngôn ngữ: Sử dụng văn phong pháp lý, trung lập, rõ ràng và tôn trọng ngữ điệu hành chính.\n"
    "4) Trình bày: Ưu tiên trình bày dưới dạng danh sách (số thứ tự hoặc gạch đầu dòng) để dễ theo dõi; "
    "tuyệt đối không được sử dụng ký hiệu in đậm (** hoặc __) trong bất kỳ phần trả lời nào.\n"
    "5) Nếu thông tin không có: Trả lời rõ ràng: 'Thông tin này không có trong tài liệu được cung cấp.'\n"
    "6) Nếu câu hỏi mơ hồ: Yêu cầu người dùng làm rõ hoặc bổ sung chi tiết để trả lời chính xác hơn.\n"
 
    
"Không được phép sử dụng hoặc nhắc đến cụm từ như: " "'tài liệu PDF', 'trích từ tài liệu PDF', 'dưới đây là thông tin từ tài liệu PDF', hoặc các cụm tương tự. " 
    "Thay vào đó, chỉ nêu trực tiếp nội dung pháp luật, ví dụ: 'Thông tin liên quan đến Luật Việc làm quy định rằng...'.\n"

    "🏭 QUY ĐỊNH RIÊNG ĐỐI VỚI CÁC KHU CÔNG NGHIỆP / CỤM CÔNG NGHIỆP:\n"
    "1) Nếu người dùng hỏi 'Tỉnh/thành phố nào có bao nhiêu khu hoặc cụm công nghiệp', "
    "hãy trả lời theo định dạng sau:\n"
    "   - Số lượng khu/cụm công nghiệp trong tỉnh hoặc thành phố đó.\n"
    "   - Danh sách tên của tất cả các khu/cụm.\n\n"
    "   Ví dụ:\n"
    "   'Tỉnh Bình Dương có 29 khu công nghiệp. Bao gồm:\n"
    "   - Khu công nghiệp Sóng Thần 1\n"
    "   - Khu công nghiệp VSIP 1\n"
    "   - Khu công nghiệp Mỹ Phước 3\n"
    "   ...'\n\n"

    "2) Nếu người dùng hỏi chi tiết về một khu/cụm công nghiệp cụ thể (lần đầu tiên), hãy trình bày đầy đủ thông tin (nếu có trong tài liệu), gồm:\n"
    "   - Tên khu công nghiệp / cụm công nghiệp\n"
    "   - Địa điểm (tỉnh/thành phố, huyện/thị xã)\n"
    "   - Diện tích (ha hoặc m²)\n"
    "   - Cơ quan quản lý / chủ đầu tư\n"
    "   - Quyết định thành lập hoặc phê duyệt quy hoạch\n"
    "   - Ngành nghề hoạt động chính\n"
    "   - Tình trạng hoạt động (đang hoạt động / đang quy hoạch / đang xây dựng)\n"
    "   - Các thông tin khác liên quan (nếu có)\n\n"

    "3) Nếu người dùng tiếp tục hỏi chi tiết về các cụm hoặc khu công nghiệp (từ lần thứ hai trở đi), "
    "hãy không liệt kê lại thông tin chi tiết, mà trả lời cố định như sau:\n"
    "'Nếu bạn muốn biết thêm thông tin chi tiết về các cụm, hãy truy cập vào website https://iipmap.com/.'\n\n"

    "4) Nếu người dùng chỉ hỏi thống kê (ví dụ: 'Tỉnh Bắc Ninh có bao nhiêu cụm công nghiệp?'), "
    "hãy luôn trả lời số lượng và liệt kê tên cụm/khu theo quy định tại mục (1) ở trên.\n\n"

    "5) Nếu người dùng hỏi câu ngoài phạm vi pháp luật hoặc khu/cụm công nghiệp "
    "(ví dụ: hỏi về tuyển dụng, giá đất, đầu tư cá nhân, v.v.), "
    "hãy trả lời nguyên văn như sau:\n"
    "'Anh/chị vui lòng để lại tên và số điện thoại, chuyên gia của IIP sẽ liên hệ và giải đáp các yêu cầu của anh/chị ạ.'\n\n"
)

# ===================== VECTORDB UTILS =====================
def build_context_from_hits(hits, max_chars: int = 6000) -> str:
    """Xây dựng context từ kết quả tìm kiếm"""
    ctx = []
    total = 0
    for idx, h in enumerate(hits, start=1):
        source = h.metadata.get('source', 'unknown')
        # Thay đổi f"[{idx}] (Nguồn: {source})" thành chỉ nguồn để tối giản
        seg = f"[Nguồn: {source}, Trang: {h.metadata.get('page', '?')}]\n{h.page_content.strip()}"
        if total + len(seg) > max_chars:
            break
        ctx.append(seg)
        total += len(seg)
    return "\n\n".join(ctx)

def get_existing_sources() -> set:
    """Lấy danh sách file đã có trong VectorDB"""
    global vectordb
    
    if vectordb is None:
        return set()
    
    try:
        collection = vectordb._collection
        existing_data = collection.get()
        
        if existing_data and existing_data.get('metadatas'):
            return set(m.get('source', '') for m in existing_data['metadatas'] if m and m.get('source'))
        
        return set()
    except Exception as e:
        # print(f"⚠️ Lỗi khi lấy danh sách file: {e}")
        return set()

def check_vectordb_exists() -> bool:
    """Kiểm tra xem VectorDB có document nào không"""
    global vectordb
    
    if vectordb is None:
        return False
    
    try:
        collection = vectordb._collection
        count = collection.count()
        return count > 0
        
    except Exception as e:
        # print(f"⚠️ Lỗi khi kiểm tra VectorDB: {e}")
        return False

def get_vectordb_stats() -> Dict[str, Any]:
    """Lấy thông tin thống kê về VectorDB"""
    global vectordb
    
    if vectordb is None:
        return {"total_documents": 0, "path": VECTORDB_PATH, "exists": False}
    
    try:
        collection = vectordb._collection
        count = collection.count()
        
        # Lấy danh sách file đã nạp
        sources = get_existing_sources()
        
        return {
            "total_documents": count,
            "path": VECTORDB_PATH,
            "exists": count > 0,
            "sources": list(sources)
        }
    except Exception as e:
        return {
            "total_documents": 0,
            "path": VECTORDB_PATH,
            "exists": False,
            "error": str(e)
        }



def load_vectordb(vectordb_path=None, emb_fn=None):
    """Load VectorDB từ thư mục lưu trữ (Chỉ Đọc)"""
    global vectordb, retriever
    
    vectordb_path = vectordb_path if vectordb_path is not None else VECTORDB_PATH
    emb_fn = emb_fn if emb_fn is not None else emb

    try:
        # Load VectorDB từ thư mục
        vectordb = Chroma(
            collection_name="luat_tong_hop_v1",
            embedding_function=emb_fn,
            persist_directory=vectordb_path,
        )
        
        # Kiểm tra xem có document nào không
        if vectordb._collection.count() == 0:
            print(f"❌ VectorDB tại '{vectordb_path}' không có document nào.")
            vectordb = None
            retriever = None
            return None
            
        # Cập nhật retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 50})
        return vectordb
        
    except Exception as e:
        print(f"❌ Lỗi khi load VectorDB: {e}")
        vectordb = None
        retriever = None
        return None

# ===================== CLEANING & RETRIEVAL =====================
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)

def clean_question_remove_uris(text: str) -> str:
    """Làm sạch câu hỏi, loại bỏ URL và tên file PDF"""
    txt = _URL_RE.sub(" ", text or "")
    toks = re.split(r"\s+", txt)
    toks = [t for t in toks if not t.lower().endswith(".pdf")]
    return " ".join(toks).strip()

# Chuỗi trả lời cố định theo Quy tắc 3
FIXED_RESPONSE_Q3 = 'Nếu bạn muốn biết thêm thông tin chi tiết về các cụm, hãy truy cập vào website https://iipmap.com/.'

def is_detail_query(text: str) -> bool:
    """Kiểm tra xem câu hỏi có phải là câu hỏi chi tiết về khu/cụm công nghiệp hay không"""
    text_lower = text.lower()
    keywords = ["nêu chi tiết", "chi tiết về", "thông tin chi tiết", "cụm công nghiệp", "khu công nghiệp"]
    if any(k in text_lower for k in keywords):
        # Tránh nhầm lẫn với câu hỏi thống kê
        if "có bao nhiêu" in text_lower or "thống kê" in text_lower:
            return False
        return True
    return False

def count_previous_detail_queries(history: List[BaseMessage]) -> int:
    """Đếm số lần hỏi chi tiết về KCN/CCN đã được trả lời trước đó (lần đầu được tính là 0)"""
    count = 0
    # Lặp qua lịch sử từ tin nhắn cũ nhất đến tin nhắn gần nhất
    for i in range(len(history)):
        current_message = history[i]
        
        # Chỉ xét tin nhắn HumanMessage và tin nhắn Bot (AIMessage) liền kề
        if isinstance(current_message, HumanMessage):
            # Kiểm tra xem tin nhắn người dùng có phải là câu hỏi chi tiết không
            is_q = is_detail_query(current_message.content)
            
            # Kiểm tra câu trả lời liền kề của Bot
            if is_q and i + 1 < len(history) and isinstance(history[i+1], AIMessage):

                bot_response = history[i+1].content
                if FIXED_RESPONSE_Q3 not in bot_response:
                    count += 1
                
    return count

def process_pdf_question(i: Dict[str, Any]) -> str:
    """Xử lý câu hỏi từ người dùng"""
    global retriever
    
    message = i["message"]
    history: List[BaseMessage] = i.get("history", [])

    # ************************************************
    # BỔ SUNG LOGIC CHO QUY TẮC 3 TẠI ĐÂY
    # ************************************************
    clean_question = clean_question_remove_uris(message)
    
    if is_detail_query(clean_question):
        count_detail_queries = count_previous_detail_queries(history)

        if count_detail_queries >= 1: 
            return FIXED_RESPONSE_Q3
        
    # KIỂM TRA VECTORDB ĐÃ SẴN SÀNG CHƯA
    if retriever is None:
        return "❌ VectorDB chưa được load hoặc không có dữ liệu. Vui lòng kiểm tra lại đường dẫn lưu trữ."

    
    try:
        # Tìm kiếm trong VectorDB
        hits = retriever.invoke(clean_question)
        
        if not hits:
            # Sửa thông báo để phù hợp với việc chỉ đọc từ DB
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong dữ liệu hiện có."

        # Xây dựng context từ kết quả tìm kiếm
        context = build_context_from_hits(hits, max_chars=6000)
        
        # Tạo messages
        messages = [SystemMessage(content=PDF_READER_SYS)]

        if history:
            messages.extend(history[-10:]) 

        user_message = f"""Câu hỏi: {clean_question}

Nội dung liên quan từ tài liệu:
{context}

Hãy trả lời dựa trên các nội dung trên."""
        
        messages.append(HumanMessage(content=user_message))
        
        # Gọi LLM
        response = llm.invoke(messages).content
        

        return response

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi: {str(e)}"

# ===================== MAIN CHATBOT =====================
pdf_chain = RunnableLambda(process_pdf_question)
store: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str):
    """Lấy hoặc tạo lịch sử chat cho session"""
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
    """In hướng dẫn sử dụng"""
    print("\n" + "="*60)
    print("📚 CÁC LỆNH CÓ SẴN:")
    print("="*60)
    print(" - exit / quit  : Thoát chương trình")
    print(" - clear        : Xóa lịch sử hội thoại")
    # print(" - sync         : (Bị vô hiệu hóa - Chế độ chỉ đọc)")
    # print(" - reload       : (Bị vô hiệu hóa - Chế độ chỉ đọc)")
    print(" - status       : Kiểm tra trạng thái VectorDB")
    print(" - help         : Hiển thị hướng dẫn này")
    print("="*60 + "\n")

def handle_command(command: str, session: str) -> bool:
    """Xử lý các lệnh đặc biệt"""
    global vectordb, retriever
    cmd = command.lower().strip()

    if cmd in {"exit", "quit"}:
        print("\n👋 Tạm biệt! Hẹn gặp lại!")
        return False
    
    elif cmd == "clear":
        if session in store:
            store[session].clear()
            print("🧹 Đã xóa lịch sử hội thoại.\n")
        return True
    

    elif cmd == "status":
        stats = get_vectordb_stats()
        print("\n" + "="*60)
        print("📊 TRẠNG THÁI VECTORDB (CHẾ ĐỘ CHỈ ĐỌC)")
        print("="*60)
        if stats["exists"]:
            print(f"✅ Trạng thái: Sẵn sàng")
            print(f"📊 Tổng documents: {stats['total_documents']}")
            print(f"📂 Đường dẫn: {stats['path']}")
            print(f"📘 Các file đã nạp:")
            for src in stats.get('sources', []):
                print(f"   - {src}")
        else:
            print("❌ Trạng thái: Chưa sẵn sàng")
            print("💡 Không thể load VectorDB. Vui lòng kiểm tra lại đường dẫn và dữ liệu đã nạp.")
        print("="*60 + "\n")
        return True
    
    elif cmd == "help":
        print_help()
        return True
    
    else:
        return True

# ===================== CLI =====================
if __name__ == "__main__":
    session = "pdf_reader_session"

    print("\n" + "="*60)
    print("🤖 CHATBOT CỔNG VIỆC LÀM VIỆT NAM (CHỈ ĐỌC VECTRORDB)")
    print("="*60)
    print(f"📂 VectorDB: {VECTORDB_PATH}")
    print("🔍 Tôi hỗ trợ: Luật Lao động & Luật Dân sự Việt Nam")
    print_help()

    # KHỞI TẠO VectorDB bằng cách LOAD TỪ ĐĨA
    print("📥 Đang load VectorDB từ thư mục lưu trữ...")
    result = load_vectordb()
    
    if result is None:
        print("❌ KHÔNG THỂ LOAD VECTORDB. Vui lòng kiểm tra lại đường dẫn và dữ liệu đã nạp trước đó.")
        exit(1)

    # In thống kê sau khi load
    stats = get_vectordb_stats()
    print(f"✅ VectorDB sẵn sàng với {stats['total_documents']} documents")
    print(f"📚 Đã nạp: {', '.join(stats.get('sources', []))}\n")
    
    print("💬 Sẵn sàng trả lời câu hỏi! (Gõ 'help' để xem hướng dẫn)\n")

    # Main loop
    while True:
        try:
            message = input("👤 Bạn: ").strip()
            
            if not message:
                continue
            
            # Xử lý lệnh
            if not handle_command(message, session):
                break
            
            # Bỏ qua nếu là lệnh
            if message.lower() in ["clear", "status", "help"]: 
                continue
            
            # Xử lý câu hỏi thường
            print("🔎 Đang tìm kiếm trong tài liệu...")
            response = chatbot.invoke(
                {"message": message},
                config={"configurable": {"session_id": session}}
            )
            print(f"\n🤖 Bot: {response}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}\n")