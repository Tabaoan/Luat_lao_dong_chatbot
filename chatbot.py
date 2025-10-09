from rag_pipeline import chatbot, ingest_pdf, clear_vectordb, get_vectordb_stats, check_vectordb_exists
from utils import print_help
from config import PDF_PATH, VECTORDB_PATH

def handle_command(cmd, session):
    cmd = cmd.lower().strip()
    if cmd in {"exit", "quit"}:
        print("Tạm biệt!"); return False
    elif cmd == "clear":
        print("🧹 Đã xóa lịch sử cuộc trò chuyện.\n"); return True
    elif cmd == "reload":
        print("🔄 Nạp lại PDF..."); ingest_pdf(); return True
    elif cmd == "status":
        stats = get_vectordb_stats()
        print(f"VectorDB: {stats['total_documents']} documents tại {stats['path']}\n"); return True
    elif cmd == "reset":
        print("🗑️ Reset VectorDB..."); clear_vectordb(); ingest_pdf(); return True
    elif cmd == "help":
        print_help(); return True
    else:
        return True

if __name__ == "__main__":
    session = "pdf_reader_session"
    print("=" * 60)
    print("Chatbot Luật Lao động Việt Nam")
    print("=" * 60)
    # print(f"Tài liệu: {PDF_PATH}")
    # print(f"VectorDB: {VECTORDB_PATH}\n")
    print_help(); print("=" * 60)

    if not check_vectordb_exists():
        print("Đang nạp tài liệu lần đầu...")
        ingest_pdf()

    while True:
        try:
            message = input("👤 Bạn: ").strip()
            if not message:
                continue
            if not handle_command(message, session):
                break
            if message.lower() not in ["clear", "reload", "status", "reset", "help"]:
                print("🧠 Đang xử lý câu hỏi...")
                resp = chatbot.invoke({"message": message}, config={"configurable": {"session_id": session}})
                print(f"\n🤖 Bot: {resp}\n" + "-" * 60)
        except KeyboardInterrupt:
            print("\nTạm biệt!"); break
