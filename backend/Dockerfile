# ============================================
# BACKEND DOCKERFILE
# ============================================
# Python FastAPI uygulaması için Docker image

# Python 3.11 slim image kullanıyoruz (daha küçük boyut)
FROM python:3.11-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Önce sadece requirements.txt kopyala (cache için)
# Bu sayede bağımlılıklar değişmedikçe tekrar yüklenmez
COPY requirements.txt .

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Port 8000'i aç
EXPOSE 8000

# Uygulamayı başlat
# --host 0.0.0.0: Tüm IP'lerden erişime izin ver (Docker için gerekli)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
