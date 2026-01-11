"""
MAIN.PY - FastAPI Backend

FastAPI Nedir?
--------------
FastAPI, modern Python web framework'üdür. Flask'a benzer ama daha hızlı ve
otomatik API dokümantasyonu oluşturur.

Bu dosyada ne yapıyoruz?
------------------------
1. /chat endpoint: Kullanıcı mesajını alır, LLM'den cevap alır, veritabanına kaydeder
2. /history endpoint: Bir sohbetin geçmişini getirir
3. CORS ayarları: React frontend'in backend'e erişebilmesi için

Nasıl çalıştırılır?
-------------------
Terminal'de bu klasörde şu komutu çalıştır:
    uvicorn main:app --reload --port 8000

Sonra tarayıcıda http://localhost:8000/docs adresine git,
otomatik oluşturulan API dokümantasyonunu görebilirsin!
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid

# Kendi modüllerimizi import ediyoruz
from database import init_db, save_message, get_conversation
from agent import get_chat_response

# ============================================
# FastAPI Uygulaması Oluştur
# ============================================
app = FastAPI(
    title="Chat API",
    description="LangChain ve LiteLLM ile çalışan chat API'si",
    version="1.0.0"
)

# ============================================
# CORS Ayarları
# ============================================
# CORS (Cross-Origin Resource Sharing):
# Farklı portlarda çalışan frontend ve backend'in
# birbiriyle iletişim kurmasını sağlar.
#
# Frontend: http://localhost:5173 (Vite default)
# Backend: http://localhost:8000 (FastAPI)
#
# Bu ayar olmadan tarayıcı güvenlik nedeniyle isteği engeller.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],  # Frontend adresleri
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin ver (GET, POST, vs.)
    allow_headers=["*"],  # Tüm header'lara izin ver
)

# ============================================
# Pydantic Modelleri
# ============================================
# Pydantic, gelen verilerin doğruluğunu kontrol eder.
# Yanlış formatta veri gelirse otomatik hata döner.


class ChatRequest(BaseModel):
    """
    /chat endpoint'ine gelen istek formatı

    Örnek:
    {
        "message": "Merhaba, nasılsın?",
        "session_id": "abc-123"  (opsiyonel, yoksa yeni oluşturulur)
    }
    """
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """
    /chat endpoint'inden dönen cevap formatı

    Örnek:
    {
        "response": "Merhaba! Ben bir AI asistanıyım...",
        "session_id": "abc-123"
    }
    """
    response: str
    session_id: str


class MessageItem(BaseModel):
    """Tek bir mesajın formatı"""
    role: str
    content: str
    created_at: Optional[str] = None


class HistoryResponse(BaseModel):
    """
    /history endpoint'inden dönen cevap formatı
    """
    session_id: str
    messages: List[MessageItem]


# ============================================
# Uygulama Başlangıcı
# ============================================
@app.on_event("startup")
async def startup_event():
    """
    Uygulama başladığında çalışır.
    Veritabanını hazırlar.
    """
    init_db()
    print("Chat API başlatıldı!")


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """
    Ana sayfa - API'nin çalıştığını gösterir
    """
    return {"message": "Chat API çalışıyor!", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Kullanıcı mesajını alır ve LLM'den cevap döner.

    İşlem sırası:
    1. Session ID yoksa yeni oluştur
    2. Sohbet geçmişini veritabanından al
    3. Kullanıcı mesajını kaydet
    4. LLM'den cevap al
    5. LLM cevabını kaydet
    6. Cevabı döndür
    """
    try:
        # 1. Session ID kontrolü
        session_id = request.session_id or str(uuid.uuid4())

        # 2. Geçmiş mesajları al (sohbet bağlamı için)
        history = get_conversation(session_id)

        # 3. Kullanıcı mesajını veritabanına kaydet
        save_message(session_id, "user", request.message)

        # 4. LLM'den cevap al
        response = get_chat_response(request.message, history)

        # 5. LLM cevabını veritabanına kaydet
        save_message(session_id, "assistant", response)

        # 6. Cevabı döndür
        return ChatResponse(response=response, session_id=session_id)

    except Exception as e:
        # Hata durumunda detaylı bilgi ver
        raise HTTPException(status_code=500, detail=f"Bir hata oluştu: {str(e)}")


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """
    Belirli bir sohbetin tüm mesajlarını getirir.

    Kullanım: GET /history/abc-123
    """
    messages = get_conversation(session_id)

    # Mesajları MessageItem formatına çevir
    message_items = [
        MessageItem(
            role=msg["role"],
            content=msg["content"],
            created_at=msg.get("created_at")
        )
        for msg in messages
    ]

    return HistoryResponse(session_id=session_id, messages=message_items)


@app.get("/health")
async def health_check():
    """
    Sağlık kontrolü - API'nin çalışıp çalışmadığını kontrol eder
    """
    return {"status": "healthy"}


# ============================================
# Direkt Çalıştırma
# ============================================
if __name__ == "__main__":
    import uvicorn
    # Bu dosyayı direkt çalıştırınca uvicorn'u başlat
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
