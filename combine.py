# ===================== IMPORTS =====================
import os, re, io
from typing import Dict, Any, List
from pathlib import Path

from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_community.document_loaders import PyMuPDFLoader

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
os.makedirs(VECTORDB_PATH, exist_ok=True)

emb = OpenAIEmbeddings(api_key=OPENAI__API_KEY, model=OPENAI__EMBEDDING_MODEL)

vectordb = None
retriever = None

# ===================== PDF FOLDER =====================
PDF_FOLDER = "./data" 


def get_pdf_files_from_folder(folder_path: str) -> List[str]:
    """Lấy tất cả file PDF trong folder"""
    pdf_files = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder không tồn tại: {folder_path}")
        return pdf_files
    
    for file in os.listdir(folder_path):
        if file.lower().endswith('.pdf'):
            full_path = os.path.join(folder_path, file)
            pdf_files.append(full_path)
    
    return sorted(pdf_files)

PDF_PATHS = get_pdf_files_from_folder(PDF_FOLDER)

# ===================== COUNTER & HISTORY STORE =====================
# Lưu trữ lịch sử chat
store: Dict[str, ChatMessageHistory] = {}
# Lưu trữ bộ đếm số lần hỏi chi tiết KCN cho từng session
query_counter: Dict[str, int] = {} 


def get_history(session_id: str):
    """Lấy hoặc tạo lịch sử chat cho session"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        # Khởi tạo counter cho session mới
        query_counter[session_id] = 0
    return store[session_id]

def increment_counter(session_id: str):
    """Tăng bộ đếm cho session"""
    if session_id in query_counter:
        query_counter[session_id] += 1
    else:
        query_counter[session_id] = 1

# ===================== SYSTEM PROMPT =====================
PDF_READER_SYS = (
    "Bạn là một **trợ lý AI pháp lý** chuyên đọc hiểu và tra cứu các tài liệu PDF được cung cấp "
    "(bao gồm: Luật, Nghị định, Quyết định, Thông tư, Văn bản hợp nhất, Quy hoạch, Danh mục khu công nghiệp, v.v.). "
    "Nhiệm vụ của bạn là **trích xuất và trả lời chính xác các thông tin có trong tài liệu**, "
    "đặc biệt liên quan đến **Lao động**, **Dân sự** và **các Khu công nghiệp, Cụm công nghiệp tại Việt Nam**.\n\n"

    "⚙️ **QUY TẮC ĐẶC BIỆT:**\n"
    "- Nếu người dùng chỉ chào hỏi hoặc đặt câu hỏi chung chung (ví dụ: 'xin chào', 'bạn làm được gì', 'giúp tôi với' ...), "
    "hãy trả lời **nguyên văn** như sau:\n"
    "'Xin chào! Mình là Chatbot Cổng việc làm Việt Nam. Mình có thể giúp anh/chị tra cứu và giải thích các quy định pháp luật "
    "(luật, nghị định, thông tư...) liên quan đến lao động, việc làm, dân sự và các lĩnh vực pháp lý khác. "
    "Gõ câu hỏi cụ thể hoặc mô tả tình huống nhé — mình sẽ trả lời ngắn gọn, có dẫn nguồn.'\n\n"

    "📘 **NGUYÊN TẮC CHUNG KHI TRẢ LỜI:**\n"
    "1) **Phạm vi:** Chỉ dựa vào nội dung trong các tài liệu PDF đã được cung cấp; tuyệt đối không sử dụng hoặc suy diễn kiến thức bên ngoài.\n"
    "2) **Nguồn trích dẫn:** Khi có thể, dẫn rõ nguồn gốc (ví dụ: 'Theo Điều X, Nghị định số Y/NĐ-CP...').\n"
    "3) **Ngôn ngữ:** Sử dụng văn phong pháp lý, trung lập, rõ ràng và tôn trọng ngữ điệu hành chính.\n"
    "4) **Trình bày:** Ưu tiên trình bày dưới dạng danh sách (số thứ tự hoặc gạch đầu dòng) để dễ theo dõi.\n"
    "5) **Nếu thông tin không có:** Trả lời rõ ràng: 'Thông tin này không có trong tài liệu được cung cấp.'\n"
    "6) **Nếu câu hỏi mơ hồ:** Yêu cầu người dùng làm rõ hoặc bổ sung chi tiết để trả lời chính xác hơn.\n\n"

    "🏭 **QUY ĐỊNH RIÊNG ĐỐI VỚI CÁC KHU CÔNG NGHIỆP / CỤM CÔNG NGHIỆP:**\n"
    "1) Nếu người dùng hỏi **'Tỉnh/thành phố nào có bao nhiêu khu hoặc cụm công nghiệp'**, "
    "hãy trả lời theo **định dạng sau**:\n"
    "   - Số lượng khu/cụm công nghiệp trong tỉnh hoặc thành phố đó.\n"
    "   - Danh sách tên của tất cả các khu/cụm (chỉ tên, không nêu chi tiết khác).\n\n"
    "   Ví dụ:\n"
    "   'Tỉnh Bình Dương có 29 khu công nghiệp. Bao gồm:\n"
    "   - Khu công nghiệp Sóng Thần 1\n"
    "   - Khu công nghiệp VSIP 1\n"
    "   - Khu công nghiệp Mỹ Phước 3\n"
    "   ...'\n\n"

    "2) Nếu người dùng hỏi **chi tiết về một khu/cụm công nghiệp cụ thể (lần đầu tiên)**, hãy trình bày đầy đủ thông tin (nếu có trong tài liệu), gồm:\n"
    "   - Tên khu công nghiệp / cụm công nghiệp\n"
    "   - Địa điểm (tỉnh/thành phố, huyện/thị xã)\n"
    "   - Diện tích (ha hoặc m²)\n"
    "   - Cơ quan quản lý / chủ đầu tư\n"
    "   - Quyết định thành lập hoặc phê duyệt quy hoạch\n"
    "   - Ngành nghề hoạt động chính\n"
    "   - Tình trạng hoạt động (đang hoạt động / đang quy hoạch / đang xây dựng)\n"
    "   - Các thông tin khác liên quan (nếu có)\n\n"

    "3) Nếu người dùng **tiếp tục hỏi chi tiết** về các cụm hoặc khu công nghiệp (từ lần thứ hai trở đi), "
    "hãy **không liệt kê lại thông tin chi tiết**, mà **trả lời cố định** như sau:\n"
    "'Nếu bạn muốn biết thêm thông tin chi tiết về các cụm, hãy truy cập vào website https://iipmap.com/.'\n\n"

    "4) Nếu người dùng chỉ hỏi thống kê (ví dụ: 'Tỉnh Bắc Ninh có bao nhiêu cụm công nghiệp?'), "
    "hãy luôn trả lời số lượng và liệt kê tên cụm/khu theo quy định tại mục (1) ở trên.\n\n"

    "5) Nếu người dùng hỏi **câu ngoài phạm vi pháp luật hoặc khu/cụm công nghiệp** "
    "(ví dụ: hỏi về tuyển dụng, giá đất, đầu tư cá nhân, v.v.), "
    "hãy trả lời nguyên văn như sau:\n"
    "'Anh/chị vui lòng để lại tên và số điện thoại, chuyên gia của IIP sẽ liên hệ và giải đáp các yêu cầu của anh/chị ạ.'\n\n"
    
    "⚠️ **BỘ ĐẾM HIỆN TẠI:** {{QUERY_COUNT}} (Lưu ý: Bộ đếm này chỉ áp dụng cho Quy tắc (3) - Chi tiết KCN/CCN)."
)


# ===================== VECTORDB UTILS =====================
def build_context_from_hits(hits, max_chars: int = 6000) -> str:
    """Xây dựng context từ kết quả tìm kiếm"""
    ctx = []
    total = 0
    for idx, h in enumerate(hits, start=1):
        source = h.metadata.get('source', 'unknown')
        seg = f"[{idx}] (Nguồn: {source})\n{h.page_content.strip()}"
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
        print(f"⚠️ Lỗi khi lấy danh sách file: {e}")
        return set()

def check_vectordb_exists() -> bool:
    """Kiểm tra xem VectorDB có đủ tất cả file PDF không"""
    global vectordb
    
    if vectordb is None:
        return False
    
    try:
        collection = vectordb._collection
        count = collection.count()
        
        if count == 0:
            return False
        
        # Kiểm tra xem đã có đủ tất cả file PDF chưa
        target_files = set(os.path.basename(p) for p in PDF_PATHS)
        existing_sources = get_existing_sources()
        
        return target_files.issubset(existing_sources)
        
    except Exception as e:
        print(f"⚠️ Lỗi khi kiểm tra VectorDB: {e}")
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

# ===================== INGEST MULTIPLE PDFs (INCREMENTAL) =====================
def ingest_pdf(pdf_paths=None, vectordb_path=None, emb_fn=None, force_reload=False):
    """
    Nạp tài liệu PDF vào VectorDB (incremental update)
    """
    global vectordb, retriever

    pdf_paths = pdf_paths if pdf_paths is not None else PDF_PATHS
    vectordb_path = vectordb_path if vectordb_path is not None else VECTORDB_PATH
    emb_fn = emb_fn if emb_fn is not None else emb

    print("🚀 Bắt đầu kiểm tra và nạp tài liệu PDF...\n")

    # Nếu force reload, xóa toàn bộ và nạp lại
    if force_reload:
        print("🗑️ Chế độ force reload - Xóa toàn bộ VectorDB...")
        try:
            temp_db = Chroma(
                collection_name="luat_tong_hop_v1",
                embedding_function=emb_fn,
                persist_directory=vectordb_path,
            )
            temp_db.delete_collection()
            print("✅ Đã xóa VectorDB cũ\n")
            vectordb = None
        except Exception as e:
            print(f"ℹ️ Không có VectorDB cũ để xóa: {e}\n")

    # Khởi tạo hoặc load VectorDB
    if vectordb is None:
        try:
            # Dùng collection name cố định
            vectordb = Chroma(
                collection_name="luat_tong_hop_v1",
                embedding_function=emb_fn,
                persist_directory=vectordb_path,
            )
            print("📂 Đã khởi tạo/load VectorDB")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo VectorDB: {e}")
            return None

    # Lấy danh sách file đã có trong VectorDB
    existing_sources = get_existing_sources()
    print(f"📊 VectorDB hiện có: {len(existing_sources)} file")
    if existing_sources:
        print(f"   └─ {', '.join(sorted(existing_sources))}")
    
    # Xác định file cần nạp mới
    target_files = {os.path.basename(p): p for p in pdf_paths}
    new_files = {name: path for name, path in target_files.items() if name not in existing_sources}
    
    if not new_files:
        print(f"\n✅ Tất cả {len(target_files)} file đã có trong VectorDB!")
        print("💡 Dùng lệnh 'reload' để nạp lại toàn bộ nếu cần.\n")
        # Cập nhật k=50
        retriever = vectordb.as_retriever(search_kwargs={"k": 50}) 
        return vectordb
    
    print(f"\n📥 Cần nạp {len(new_files)} file mới:")
    for name in sorted(new_files.keys()):
        print(f"   + {name}")
    print()

    all_new_docs = []
    total_chunks = 0

    # Đọc và chunk từng file PDF mới
    for filename, path in new_files.items():
        if not os.path.exists(path):
            print(f"⚠️ Không tìm thấy file: {path}")
            continue

        print(f"📖 Đang đọc: {filename} ...")

        loader = PyMuPDFLoader(path)
        try:
            docs = loader.load()
        except Exception as e:
            print(f"❌ Lỗi khi load {filename}: {e}")
            continue

        # Gắn thông tin nguồn file
        for i, d in enumerate(docs):
            if d.metadata is None:
                d.metadata = {}
            d.metadata["source"] = filename
            d.metadata["page"] = i + 1

        # Chunk nội dung (chunk_size=500, overlap=300)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        split_docs = splitter.split_documents(docs)

        # Gắn thêm chunk index
        for i, d in enumerate(split_docs):
            d.metadata["chunk_id"] = i

        print(f"   🔹 Tạo {len(split_docs)} đoạn từ {filename}")
        all_new_docs.extend(split_docs)
        total_chunks += len(split_docs)

    if not all_new_docs:
        print("⚠️ Không có document mới nào để nạp.")
        retriever = vectordb.as_retriever(search_kwargs={"k": 50})
        return vectordb

    print(f"\n📚 Tổng cộng: {total_chunks} đoạn nội dung mới\n")

    # Thêm vào VectorDB với ID duy nhất
    print("💾 Đang thêm vào VectorDB...")
    ids = []
    for d in all_new_docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", 0)
        chunk = d.metadata.get("chunk_id", 0)
        ids.append(f"{src}_page{page}_chunk{chunk}")

    try:
        # Chia nhỏ để tránh lỗi khi batch quá lớn
        batch_size = 100
        for i in range(0, len(all_new_docs), batch_size):
            batch_docs = all_new_docs[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            vectordb.add_documents(batch_docs, ids=batch_ids)
            print(f"   ✓ Đã thêm {min(i+batch_size, len(all_new_docs))}/{len(all_new_docs)} documents")
        
        print("✅ Đã thêm toàn bộ documents mới vào VectorDB!")
    except Exception as e:
        print(f"❌ Lỗi khi thêm documents: {e}")
        return None

    # Cập nhật retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})

    # Thống kê cuối cùng
    try:
        count = vectordb._collection.count()
        final_sources = get_existing_sources()
        print(f"\n📂 Lưu tại: {vectordb_path}")
        print(f"📊 VectorDB hiện có:")
        print(f"   • Tổng documents: {count}")
        print(f"   • Tổng file: {len(final_sources)}")
        print(f"   • Danh sách: {', '.join(sorted(final_sources))}\n")
    except Exception as e:
        print(f"⚠️ Không thể lấy thống kê: {e}\n")

    return vectordb

# ===================== CLEANING & RETRIEVAL =====================
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)

# Regex để phát hiện các câu hỏi chi tiết về khu/cụm công nghiệp (dạng hỏi thông tin cụ thể)
_KCN_DETAIL_RE = re.compile(
    r"(thông tin|chi tiết|diện tích|địa điểm|về|giới thiệu|cơ quan quản lý|chủ đầu tư|quyết định thành lập|ngành nghề|tình trạng)\s+(khu công nghiệp|cụm công nghiệp)\s+([^\s]+)", 
    re.IGNORECASE | re.UNICODE
)
_KCN_STAT_RE = re.compile(
    r"(bao nhiêu|tỉnh|thành phố|có)\s+(khu công nghiệp|cụm công nghiệp|kcn|ccn)",
    re.IGNORECASE | re.UNICODE
)


def clean_question_remove_uris(text: str) -> str:
    """Làm sạch câu hỏi, loại bỏ URL và tên file PDF"""
    txt = _URL_RE.sub(" ", text or "")
    toks = re.split(r"\s+", txt)
    toks = [t for t in toks if not t.lower().endswith(".pdf")]
    return " ".join(toks).strip()

def is_kcn_detail_query(text: str) -> bool:
    """Kiểm tra xem câu hỏi có phải là hỏi chi tiết về một KCN/CCN cụ thể không"""
    # Nếu câu hỏi chứa các từ khóa hỏi chi tiết và tên KCN/CCN, VÀ không phải là câu hỏi thống kê
    return bool(_KCN_DETAIL_RE.search(text.lower())) and not bool(_KCN_STAT_RE.search(text.lower()))


def process_pdf_question(i: Dict[str, Any]) -> str:
    """
    Xử lý câu hỏi từ người dùng.
    i sẽ chứa: 'message', 'history' và 'configurable': {'session_id': '...'}
    """
    global retriever
    
    message = i["message"]
    history: List[BaseMessage] = i.get("history", [])
    
    # SỬA LỖI: Truy cập session_id từ dictionary 'configurable'
    session_id = i.get("configurable", {}).get("session_id", "pdf_reader_session") 

    # Kiểm tra VectorDB
    if not check_vectordb_exists():
        print("⚠️ VectorDB chưa sẵn sàng, đang nạp PDF vào hệ thống...")
        result = ingest_pdf()
        if result is None:
            return "Xin lỗi, tôi gặp lỗi khi nạp tài liệu PDF. Vui lòng kiểm tra lại đường dẫn file."

    clean_question = clean_question_remove_uris(message)
    
    try:
        current_count = query_counter.get(session_id, 0)
        
        # Kiểm tra và tăng bộ đếm cho câu hỏi chi tiết KCN
        if is_kcn_detail_query(clean_question):
            # Nếu bộ đếm >= 1 (đã hỏi lần đầu), LLM phải trả lời cố định theo Quy tắc (3)
            if current_count >= 1:
                increment_counter(session_id) # Tăng bộ đếm cho lần hỏi tiếp theo
                return "Nếu bạn muốn biết thêm thông tin chi tiết về các cụm, hãy truy cập vào website https://iipmap.com/."
            else:
                # Lần hỏi chi tiết đầu tiên, tăng bộ đếm
                increment_counter(session_id)
        
        # Tìm kiếm trong VectorDB
        hits = retriever.invoke(clean_question)
        
        # Xây dựng context từ kết quả tìm kiếm
        context = build_context_from_hits(hits, max_chars=6000)
        
        # Tạo messages
        # Thay thế placeholder bộ đếm trong SYSTEM PROMPT (cập nhật lại current_count sau khi increment)
        sys_prompt_with_count = PDF_READER_SYS.replace("{{QUERY_COUNT}}", str(query_counter.get(session_id, 0)))
        messages = [SystemMessage(content=sys_prompt_with_count)]

        if history:
            messages.extend(history[-10:])  # Chỉ lấy 10 tin nhắn gần nhất

        user_message = f"""Câu hỏi: {clean_question}

Nội dung liên quan từ tài liệu PDF:
{context}

Hãy trả lời dựa trên các nội dung trên. (Nếu nội dung không liên quan, hãy tuân theo các QUY TẮC đã cho.)"""
        
        messages.append(HumanMessage(content=user_message))
        
        # Gọi LLM
        response = llm.invoke(messages).content
        return response

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        # Không thay đổi bộ đếm khi có lỗi
        return f"Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi: {str(e)}"

# ===================== MAIN CHATBOT =====================
# SỬA LỖI: Loại bỏ .bind(session_id=None)
pdf_chain = RunnableLambda(process_pdf_question) 

def get_history_for_chain(session_id: str):
    """Hàm wrapper cho RunnableWithMessageHistory"""
    return get_history(session_id)

chatbot = RunnableWithMessageHistory(
    pdf_chain,
    get_history_for_chain,
    input_messages_key="message",
    history_messages_key="history"
)

def print_help():
    """In hướng dẫn sử dụng"""
    print("\n" + "="*60)
    print("📚 CÁC LỆNH CÓ SẴN:")
    print("="*60)
    print(" - exit / quit  : Thoát chương trình")
    print(" - clear        : Xóa lịch sử hội thoại (và reset bộ đếm KCN)")
    print(" - sync         : Đồng bộ file mới từ folder vào VectorDB (Kiểm tra incremental)")
    print(" - reload       : Xóa toàn bộ và nạp lại (force reload)")
    print(" - status       : Kiểm tra trạng thái VectorDB")
    print(" - help         : Hiển thị hướng dẫn này")
    print("="*60 + "\n")

def handle_command(command: str, session: str) -> bool:
    """Xử lý các lệnh đặc biệt"""
    global vectordb, retriever, query_counter
    cmd = command.lower().strip()

    if cmd in {"exit", "quit"}:
        print("\n👋 Tạm biệt! Hẹn gặp lại!")
        return False
    
    elif cmd == "clear":
        if session in store:
            store[session].clear()
            # Reset bộ đếm khi xóa lịch sử
            query_counter[session] = 0
            print("🧹 Đã xóa lịch sử hội thoại và reset bộ đếm KCN.\n")
        return True
    
    elif cmd == "reload":
        print("🔄 Đang xóa và nạp lại toàn bộ PDF...")
        ingest_pdf(force_reload=True)
        return True
    
    elif cmd == "sync":
        print("🔄 Đang đồng bộ (tự động kiểm tra file mới)...")
        ingest_pdf(force_reload=False)
        return True
    
    elif cmd == "status":
        stats = get_vectordb_stats()
        print("\n" + "="*60)
        print("📊 TRẠNG THÁI VECTORDB & BỘ ĐẾM")
        print("="*60)
        if stats["exists"]:
            print(f"✅ Trạng thái: Sẵn sàng")
            print(f"📊 Tổng documents: {stats['total_documents']}")
            print(f"📂 Đường dẫn: {stats['path']}")
            print(f"📘 Các file đã nạp: {', '.join(stats.get('sources', []))}")
        else:
            print("❌ Trạng thái: Chưa sẵn sàng")
            print("💡 Hãy đợi hệ thống nạp PDF hoặc gõ 'reload'")
            
        print(f"⏱️ Bộ đếm KCN hiện tại (Session: {session}): {query_counter.get(session, 0)}")
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
    print("🤖 CHATBOT CỔNG VIỆC LÀM VIỆT NAM")
    print("="*60)
    print(f"📁 Folder tài liệu: {PDF_FOLDER}")
    print(f"📚 Tìm thấy {len(PDF_PATHS)} file PDF:")
    
    if PDF_PATHS:
        for idx, p in enumerate(PDF_PATHS, 1):
            status = "✅" if os.path.exists(p) else "❌"
            print(f"   {idx}. {status} {os.path.basename(p)}")
    else:
        print("   ⚠️ Không tìm thấy file PDF nào trong folder!")
    
    print(f"\n📂 VectorDB: {VECTORDB_PATH}")
    print("🔍 Tôi hỗ trợ: Luật Lao động, Luật Dân sự & Thông tin KCN/CCN Việt Nam")
    print_help()

    # Khởi tạo VectorDB và History/Counter
    get_history(session) # Khởi tạo history và counter cho session
    
    if not PDF_PATHS:
        print("❌ Không có file PDF nào để xử lý. Vui lòng kiểm tra lại folder.")
        exit(1)
    
    # Kiểm tra và nạp PDF
    if not check_vectordb_exists():
        print("📥 Đang nạp PDF lần đầu tiên...")
        result = ingest_pdf()
        if result is None:
            print("❌ Không thể khởi tạo VectorDB. Vui lòng kiểm tra lại đường dẫn file PDF.")
            exit(1)
    else:
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
            if message.lower() in ["clear", "reload", "status", "help", "sync"]:
                continue
            
            # Xử lý câu hỏi thường
            print("🔎 Đang tìm kiếm trong tài liệu...")
            
            # Gán session_id vào config
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