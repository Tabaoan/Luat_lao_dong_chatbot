from flask import Flask, request, jsonify
import os
import app  # 🔹 file chính của bạn ở trên (đổi lại nếu tên khác, ví dụ chatbot.py)

app_flask = Flask(__name__)

@app_flask.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Chatbot Luật Lao động API đang hoạt động.",
        "usage": "Gửi POST tới /predict với JSON { 'question': 'Câu hỏi của bạn' }"
    })

@app_flask.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy câu hỏi từ request
        data = request.get_json(force=True)
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Thiếu câu hỏi (question) trong JSON."}), 400

        # Xử lý câu hỏi bằng chatbot gốc
        session = "api_session"
        response = app.chatbot.invoke(
            {"message": question},
            config={"configurable": {"session_id": session}}
        )

        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app_flask.run(host="0.0.0.0", port=port)
