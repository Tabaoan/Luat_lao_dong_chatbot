# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Cho phép gọi API từ domain khác
import os

# ✅ Import chatbot gốc (đổi lại nếu file của bạn không tên là app.py)
try:
    import app  # ví dụ: nếu chatbot nằm trong file app.py
except ImportError:
    app = None

# ---------------------------------------
# 1️⃣ Khởi tạo Flask App + bật CORS
# ---------------------------------------
app_flask = Flask(__name__)
CORS(app_flask)  # 🔹 Cho phép tất cả domain gọi API này

# Nếu bạn chỉ muốn cho phép domain cụ thể:
# CORS(app_flask, origins=["https://chatbotlaodong.vn"])

# ---------------------------------------
# 2️⃣ Route kiểm tra hoạt động
# ---------------------------------------
@app_flask.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ Chatbot Luật Lao động API đang hoạt động.",
        "usage": "Gửi POST tới /predict với JSON { 'question': 'Câu hỏi của bạn' }"
    })


# ---------------------------------------
# 3️⃣ Route chính: /predict
# ---------------------------------------
@app_flask.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu từ request
        data = request.get_json(force=True)
        question = (data.get("question") or "").strip()

        if not question:
            return jsonify({"error": "Thiếu trường 'question' trong JSON."}), 400

        # ✅ Gọi chatbot thực tế nếu có
        if hasattr(app, "chatbot"):
            session = "api_session"
            response = app.chatbot.invoke(
                {"message": question},
                config={"configurable": {"session_id": session}}
            )
        else:
            # Nếu chưa có chatbot thật, trả về giả lập
            response = f"(Chatbot mô phỏng) Bạn hỏi: '{question}'"

        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# 4️⃣ Khởi động server Flask
# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app_flask.run(host="0.0.0.0", port=port)
