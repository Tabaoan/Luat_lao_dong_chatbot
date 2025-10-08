# =========================
# 1️⃣ Base image Python
# =========================
FROM python:3.11-slim

# =========================
# 2️⃣ Thiết lập thư mục làm việc
# =========================
WORKDIR /app

# =========================
# 3️⃣ Copy các file cấu hình cần thiết
# =========================
COPY pyproject.toml uv.lock* ./

# =========================
# 4️⃣ Cài đặt uv và dependencies trong hệ thống container
# =========================
RUN pip install --upgrade pip uv \
    && uv pip install --system .

# =========================
# 5️⃣ Copy toàn bộ code vào container
# =========================
COPY . .

# =========================
# 6️⃣ Expose cổng (tuỳ app của bạn)
# =========================
EXPOSE 8000

# =========================
# 7️⃣ Lệnh chạy chính
# =========================
CMD ["uv", "run", "python", "app.py"]
