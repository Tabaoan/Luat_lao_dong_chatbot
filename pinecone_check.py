# ===================== IMPORTS =====================
import os, re, io
from typing import Dict, Any, List
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_pinecone import Pinecone 
from pinecone import Pinecone as PineconeClient, PodSpec # ĐÚNG
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage 
from langchain_community.document_loaders import PyMuPDFLoader

# ===================== ENV =====================
OPENAI__API_KEY = os.getenv("OPENAI__API_KEY")
OPENAI__EMBEDDING_MODEL = os.getenv("OPENAI__EMBEDDING_MODEL")
OPENAI__MODEL_NAME = os.getenv("OPENAI__MODEL_NAME")
OPENAI__TEMPERATURE = os.getenv("OPENAI__TEMPERATURE")

# ⬅️ THÊM BIẾN MÔI TRƯỜNG PINECONE
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_DIM = 3072 

llm = ChatOpenAI(
    api_key=OPENAI__API_KEY,
    model_name=OPENAI__MODEL_NAME,
    temperature=float(OPENAI__TEMPERATURE) if OPENAI__TEMPERATURE else 0
)

# Khởi tạo Pinecone Client
if PINECONE_API_KEY:
    pc = PineconeClient(api_key=PINECONE_API_KEY)
else:
    pc = None
    print("❌ Lỗi: Không tìm thấy PINECONE_API_KEY. Pinecone sẽ không hoạt động.")

# ===================== VECTORDB =====================
# VECTORDB_PATH = r"./vectordb_storage" # KHÔNG DÙNG NỮA
# os.makedirs(VECTORDB_PATH, exist_ok=True) # KHÔNG DÙNG NỮA

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

# ===================== SYSTEM PROMPT (Giữ nguyên) =====================
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

# ===================== VECTORDB UTILS (Cập nhật cho Pinecone) =====================
def build_context_from_hits(hits, max_chars: int = 6000) -> str:
    """Xây dựng context từ kết quả tìm kiếm"""
    ctx = []
    total = 0
    for idx, h in enumerate(hits, start=1):
        # Pinecone retriever trả về Document
        source = h.metadata.get('source', 'unknown')
        page = h.metadata.get('page', '?')
        seg = f"[{idx}] (Nguồn: {source}, Trang: {page})\n{h.page_content.strip()}"
        if total + len(seg) > max_chars:
            break
        ctx.append(seg)
        total += len(seg)
    return "\n\n".join(ctx)

def get_existing_sources() -> set:
    """Lấy danh sách file đã có trong VectorDB (Pinecone - Giả lập vì API không hỗ trợ dễ dàng)"""
    # Trong môi trường Pinecone, việc lấy tất cả sources từ metadata không hiệu quả.
    # Ta sẽ trả về rỗng và dựa vào force_reload/kiểm tra vector count.
    return set()

def check_vectordb_exists() -> bool:
    """Kiểm tra xem Pinecone Index có tồn tại và có vectors không"""
    global pc, vectordb, retriever
    
    if pc is None or not PINECONE_INDEX_NAME:
        return False

    try:
        # Kiểm tra index có tồn tại không
        # SỬA LỖI ĐÃ ĐƯỢC THỰC HIỆN Ở ĐÂY (ĐÃ CÓ ())
        if PINECONE_INDEX_NAME not in pc.list_indexes().names(): 
            return False
            
        # Lấy thống kê
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        total_vectors = stats['total_vector_count']
        
        if total_vectors > 0:
            # Nếu đã có vectors, khởi tạo vectordb và retriever (nếu chưa)
            if vectordb is None:
                 vectordb = Pinecone(
                    index=index, 
                    embedding=emb, 
                    text_key="text"
                )
                 retriever = vectordb.as_retriever(search_kwargs={"k": 50})

            return True
            
        return False
        
    except Exception as e:
        # print(f"⚠️ Lỗi khi kiểm tra Pinecone Index: {e}")
        return False

def get_vectordb_stats() -> Dict[str, Any]:
    """Lấy thông tin thống kê về VectorDB (Pinecone)"""
    global pc
    
    # SỬA LỖI ĐÃ ĐƯỢC THỰC HIỆN Ở ĐÂY (ĐÃ CÓ ())
    if pc is None or not PINECONE_INDEX_NAME or PINECONE_INDEX_NAME not in pc.list_indexes().names():
        return {"total_documents": 0, "name": PINECONE_INDEX_NAME, "exists": False, "sources": []}
    
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        
        count = stats['total_vector_count']
        sources = ["Thông tin nguồn cần nạp lại để cập nhật."]
        
        return {
            "total_documents": count,
            "name": PINECONE_INDEX_NAME,
            "exists": count > 0,
            "sources": sources,
            "dimension": stats.get('dimension', EMBEDDING_DIM)
        }
    except Exception as e:
        return {
            "total_documents": 0,
            "name": PINECONE_INDEX_NAME,
            "exists": False,
            "error": str(e),
            "sources": []
        }

# ===================== INGEST MULTIPLE PDFs (Pinecone) =====================
def ingest_pdf(pdf_paths=None, emb_fn=None, force_reload=False):
    """
    Nạp tài liệu PDF vào VectorDB (Pinecone)
    """
    global vectordb, retriever, pc

    if pc is None:
        print("❌ Lỗi: Pinecone Client chưa được khởi tạo. Vui lòng kiểm tra PINECONE_API_KEY.")
        return None
    
    pdf_paths = pdf_paths if pdf_paths is not None else PDF_PATHS
    emb_fn = emb_fn if emb_fn is not None else emb

    print("🚀 Bắt đầu kiểm tra và nạp tài liệu PDF vào Pinecone...\n")
    
    index_name = PINECONE_INDEX_NAME
    
    # 1. Xử lý Force Reload: Xóa Index và tạo lại
    if force_reload:
        print(f"🗑️ Chế độ force reload - Xóa Index '{index_name}'...")
        # SỬA LỖI ĐÃ ĐƯỢC THỰC HIỆN Ở ĐÂY (ĐÃ CÓ ())
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            print(f"✅ Đã xóa Index '{index_name}'\n")
        else:
             print(f"ℹ️ Index '{index_name}' không tồn tại. Tiếp tục tạo mới.")
        vectordb = None
        retriever = None

    # 2. Tạo Index nếu chưa tồn tại
    # SỬA LỖI ĐÃ ĐƯỢC THỰC HIỆN Ở ĐÂY (ĐÃ CÓ ())
    if index_name not in pc.list_indexes().names():
        print(f"🛠️ Index '{index_name}' chưa tồn tại. Đang tạo Index mới...")
        
        if PINECONE_ENVIRONMENT:
             pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric='cosine',
                spec=PodSpec(environment=PINECONE_ENVIRONMENT)
             )
        else:
             print("❌ Lỗi: PINECONE_ENVIRONMENT chưa được khai báo. Không thể tạo Index.")
             return None

        print(f"✅ Đã tạo Index '{index_name}'.")

    # 3. Kết nối đến Index
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    existing_vectors = stats['total_vector_count']
    
    print(f"📊 Pinecone Index '{index_name}' hiện có: {existing_vectors} vectors.")
    
    # 4. Logic nạp: Chỉ nạp nếu force_reload=True HOẶC index chưa có vectors
    if existing_vectors > 0 and not force_reload:
        print("\n✅ Index đã có dữ liệu. Không nạp lại.")
        vectordb = Pinecone(index=index, embedding=emb_fn, text_key="text")
        retriever = vectordb.as_retriever(search_kwargs={"k": 50})
        return vectordb
    
    # Chuẩn bị nạp document
    print("\n📥 Bắt đầu đọc và chunk documents để nạp...")
    all_new_docs = []
    total_chunks = 0
    
    # Đọc và chunk tất cả file PDF
    for filename, path in {os.path.basename(p): p for p in pdf_paths}.items():
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
            if d.metadata is None: d.metadata = {}
            d.metadata["source"] = filename
            d.metadata["page"] = i + 1

        # Chunk nội dung
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        split_docs = splitter.split_documents(docs)

        # Gắn thêm chunk index
        for i, d in enumerate(split_docs):
            d.metadata["chunk_id"] = i
            
        print(f"   🔹 Tạo {len(split_docs)} đoạn từ {filename}")
        all_new_docs.extend(split_docs)
        total_chunks += len(split_docs)
        
    if not all_new_docs:
        print("⚠️ Không có document nào để nạp.")
        return None
    
    print(f"\n📚 Tổng cộng: {total_chunks} đoạn nội dung mới\n")

    # Thêm vào Pinecone
    print("💾 Đang nạp documents vào Pinecone Index...")
    
    try:
        # Sử dụng Pinecone.from_documents để nạp
        # Hàm này sẽ tự động tạo embedding và gửi batch lên Pinecone
        vectordb = Pinecone.from_documents(
            all_new_docs,
            index_name=index_name,
            embedding=emb_fn,
            text_key="text" 
        )
        print("✅ Đã nạp toàn bộ documents mới vào Pinecone!")
    except Exception as e:
        print(f"❌ Lỗi khi thêm documents vào Pinecone: {e}")
        return None

    # Cập nhật retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})

    # Thống kê cuối cùng
    stats = get_vectordb_stats()
    print(f"\n📊 Pinecone Index hiện có:")
    print(f"   • Tổng documents: {stats['total_documents']}")
    print(f"   • Tên Index: {stats['name']}\n")
    
    return vectordb

# ===================== CLEANING & RETRIEVAL (Giữ nguyên) =====================
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
FIXED_RESPONSE_Q3 = 'Nếu bạn muốn biết thêm thông tin chi tiết về các cụm, hãy truy cập vào website https://iipmap.com/.'

def clean_question_remove_uris(text: str) -> str:
    """Làm sạch câu hỏi, loại bỏ URL và tên file PDF"""
    txt = _URL_RE.sub(" ", text or "")
    toks = re.split(r"\s+", txt)
    toks = [t for t in toks if not t.lower().endswith(".pdf")]
    return " ".join(toks).strip()

def is_detail_query(text: str) -> bool:
    """Kiểm tra xem câu hỏi có phải là câu hỏi chi tiết về khu/cụm công nghiệp hay không"""
    text_lower = text.lower()
    keywords = ["nêu chi tiết", "chi tiết về", "thông tin chi tiết", "cụm công nghiệp", "khu công nghiệp"]
    if any(k in text_lower for k in keywords):
        if "có bao nhiêu" in text_lower or "thống kê" in text_lower:
            return False
        return True
    return False

def count_previous_detail_queries(history: List[BaseMessage]) -> int:
    """Đếm số lần hỏi chi tiết về KCN/CCN đã được trả lời trước đó (lần đầu được tính là 0)"""
    count = 0
    for i in range(len(history)):
        current_message = history[i]
        if isinstance(current_message, HumanMessage):
            is_q = is_detail_query(current_message.content)
            
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

    clean_question = clean_question_remove_uris(message)
    
    if is_detail_query(clean_question):
        count_detail_queries = count_previous_detail_queries(history)

        if count_detail_queries >= 1: 
            return FIXED_RESPONSE_Q3
    
    # Kiểm tra VectorDB và tự động nạp nếu cần (Chỉ chạy khi Index trống)
    if not check_vectordb_exists():
        print("⚠️ VectorDB (Pinecone) chưa sẵn sàng hoặc không có dữ liệu, đang nạp PDF...")
        result = ingest_pdf()
        if result is None:
             return "Xin lỗi, tôi gặp lỗi khi nạp tài liệu PDF vào Pinecone. Vui lòng kiểm tra API Key và Index Name."

    try:
        # Tìm kiếm trong VectorDB
        hits = retriever.invoke(clean_question)
        
        if not hits:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."

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

# ===================== MAIN CHATBOT (Giữ nguyên) =====================
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
    print(" - sync / reload: Xóa và NẠP LẠI toàn bộ PDF vào Pinecone Index") 
    print(" - status       : Kiểm tra trạng thái Pinecone Index")
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
    
    elif cmd in {"reload", "sync"}:
        print("🔄 Đang xóa và nạp lại toàn bộ PDF vào Pinecone Index...")
        ingest_pdf(force_reload=True)
        return True
    
    elif cmd == "status":
        stats = get_vectordb_stats()
        print("\n" + "="*60)
        print("📊 TRẠNG THÁI VECTORDB (PINECONE)")
        print("="*60)
        if stats["exists"]:
            print(f"✅ Trạng thái: Sẵn sàng")
            print(f"📚 Tên Index: {stats['name']}")
            print(f"📊 Tổng documents: {stats['total_documents']}")
            print(f"📏 Dimension: {stats['dimension']}")
        else:
            print("❌ Trạng thái: Chưa sẵn sàng")
            print(f"💡 Index '{PINECONE_INDEX_NAME}' không tồn tại hoặc không có documents.")
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

    # KIỂM TRA MÔI TRƯỜNG PINECONE
    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
        print("❌ LỖI: Thiếu một hoặc nhiều biến môi trường Pinecone (PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME).")
        exit(1)


    print("\n" + "="*60)
    print("🤖 CHATBOT CỔNG VIỆC LÀM VIỆT NAM (DÙNG PINECONE)")
    print("="*60)
    print(f"📁 Folder tài liệu: {PDF_FOLDER}")
    print(f"📚 Tìm thấy {len(PDF_PATHS)} file PDF.")
    
    if PDF_PATHS:
        for idx, p in enumerate(PDF_PATHS, 1):
            status = "✅" if os.path.exists(p) else "❌"
            print(f"   {idx}. {status} {os.path.basename(p)}")
    else:
        print("   ⚠️ Không tìm thấy file PDF nào trong folder!")
    
    print(f"\n☁️ Pinecone Index: {PINECONE_INDEX_NAME}")
    print("🔍 Tôi hỗ trợ: Luật Lao động & Luật Dân sự Việt Nam")
    print_help()

    # Khởi tạo VectorDB (Kết nối hoặc tạo Index)
    if not PDF_PATHS:
        print("❌ Không có file PDF nào để xử lý. Vui lòng kiểm tra lại folder.")
        exit(1)
    
    if check_vectordb_exists():
        stats = get_vectordb_stats()
        print(f"✅ Pinecone sẵn sàng: Index '{stats['name']}' với {stats['total_documents']} documents.")
    else:
        print("📥 Đang nạp PDF lần đầu tiên vào Pinecone Index...")
        result = ingest_pdf()
        if result is None:
            print("❌ Không thể khởi tạo Index. Vui lòng kiểm tra Pinecone API Key và môi trường.")
            exit(1)

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
            if message.lower() in ["clear", "reload", "sync", "status", "help"]:
                continue
            
            # Xử lý câu hỏi thường
            print("🔎 Đang tìm kiếm trong Index Pinecone...")
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