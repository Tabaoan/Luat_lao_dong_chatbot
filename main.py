from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from typing import Optional, Any


try:
    import app
except ImportError:
    app = None

# --- Khai báo Model cho dữ liệu đầu vào ---
# FastAPI sử dụng Pydantic để định nghĩa cấu trúc dữ liệu
class Question(BaseModel):
    """Định nghĩa cấu trúc dữ liệu JSON đầu vào."""
    question: str

# ---------------------------------------
# 1️⃣ Khởi tạo FastAPI App + bật CORS
# ---------------------------------------
# Khởi tạo ứng dụng FastAPI
app_fastapi = FastAPI(
    title="Chatbot Luật Lao động API",
    description="API cho mô hình chatbot",
    version="1.0.0"
)


origins = [
    "*", 
    
]

app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# 2️⃣ Route kiểm tra hoạt động (GET /)
# ---------------------------------------
@app_fastapi.get("/", summary="Kiểm tra trạng thái API")
async def home():
    """Route kiểm tra xem API có hoạt động không."""
    return {
        "message": "✅ Chatbot Luật Lao động API đang hoạt động.",
        "usage": "Gửi POST tới /predict với JSON { 'question': 'Câu hỏi của bạn' }"
    }

# ---------------------------------------
# 3️⃣ Route chính: /predict (POST)
# ---------------------------------------
@app_fastapi.post("/predict", summary="Dự đoán/Trả lời câu hỏi từ Chatbot")
async def predict(data: Question):
    """
    Nhận câu hỏi và trả về câu trả lời từ mô hình chatbot.
    """
    question = data.question.strip()

    if not question:
        # FastAPI tự động validate JSON theo Pydantic, nhưng kiểm tra thêm trường hợp rỗng
        raise HTTPException(status_code=400, detail="Thiếu trường 'question' trong JSON hoặc câu hỏi bị rỗng.")

    try:
        # ✅ Gọi chatbot thực tế nếu có (Giả định app.py có chứa đối tượng chatbot)
        if hasattr(app, "chatbot"):
            session = "api_session" # ID session cố định cho API call

            response = await app.chatbot.invoke(
                {"message": question},
                config={"configurable": {"session_id": session}}
            )
            
            # Giả định response là một dict hoặc object có thể lấy ra text (ví dụ LangChain/LangGraph)
            # Bạn cần điều chỉnh cách lấy response thực tế tùy thuộc vào framework chatbot
            if isinstance(response, dict) and 'output' in response:
                 answer = response['output']
            elif isinstance(response, str):
                 answer = response
            else:
                 # Trường hợp không rõ format, dùng repr() để hiển thị
                 answer = f"Lỗi: Chatbot trả về định dạng không mong muốn: {repr(response)}"


        else:
            # Nếu chưa có chatbot thật, trả về giả lập
            answer = f"(Chatbot mô phỏng) Bạn hỏi: '{question}'"

        return {"answer": answer}

    except Exception as e:
        # Trả về lỗi server 500 nếu có lỗi xảy ra trong quá trình gọi chatbot
        print(f"Lỗi trong quá trình gọi chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Chạy Uvicorn server, trỏ đến đối tượng ứng dụng app_fastapi trong module main
    uvicorn.run("main:app_fastapi", host="127.0.0.1", port=port, log_level="info", reload=True)