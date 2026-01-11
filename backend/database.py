"""
DATABASE.PY - PostgreSQL ile Chat Geçmişi Yönetimi

PostgreSQL Nedir?
-----------------
PostgreSQL, güçlü ve güvenilir bir açık kaynak veritabanıdır.
Docker'da ayrı bir container olarak çalışır.

Bu dosyada ne yapıyoruz?
------------------------
1. PostgreSQL'e bağlanıyoruz (DATABASE_URL environment variable ile)
2. conversations tablosunu oluşturuyoruz
3. Mesaj kaydetme ve okuma fonksiyonları yazıyoruz
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List, Dict
import os

# Veritabanı bağlantı URL'si (docker-compose.yml'den gelir)
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://chatuser:chatpass@localhost:5432/chatdb"
)


def get_connection():
    """
    PostgreSQL'e bağlantı oluşturur.
    RealDictCursor ile sonuçları sözlük olarak alırız.
    """
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn


def init_db():
    """
    Veritabanını ve tabloyu oluşturur.

    Bu fonksiyon uygulama başladığında bir kez çağrılır.
    "IF NOT EXISTS" sayesinde tablo zaten varsa hata vermez.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # SQL komutu: Tablo oluştur
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index ekle (session_id'ye göre hızlı arama için)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)
    """)

    conn.commit()
    cursor.close()
    conn.close()

    print("PostgreSQL veritabanı hazır!")


def save_message(session_id: str, role: str, message: str):
    """
    Yeni bir mesajı veritabanına kaydeder.

    Parametreler:
    - session_id: Sohbet oturumunun ID'si
    - role: "user" veya "assistant"
    - message: Mesaj içeriği
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO conversations (session_id, role, message) VALUES (%s, %s, %s)",
        (session_id, role, message)
    )

    conn.commit()
    cursor.close()
    conn.close()


def get_conversation(session_id: str) -> List[Dict]:
    """
    Bir sohbetin tüm mesajlarını getirir.

    Parametreler:
    - session_id: Hangi sohbetin mesajlarını istiyoruz?

    Dönüş:
    - Mesajların listesi, her biri {"role": "...", "content": "..."} formatında
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT role, message, created_at FROM conversations WHERE session_id = %s ORDER BY created_at",
        (session_id,)
    )

    rows = cursor.fetchall()

    # Sonuçları LangChain'in beklediği formata çeviriyoruz
    messages = []
    for row in rows:
        messages.append({
            "role": row["role"],
            "content": row["message"],
            "created_at": str(row["created_at"]) if row["created_at"] else None
        })

    cursor.close()
    conn.close()
    return messages


def get_all_sessions() -> List[str]:
    """
    Tüm benzersiz session ID'lerini getirir.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT session_id FROM conversations")
    rows = cursor.fetchall()
    sessions = [row["session_id"] for row in rows]

    cursor.close()
    conn.close()
    return sessions


# Bu dosya direkt çalıştırılırsa veritabanını oluştur
if __name__ == "__main__":
    init_db()
    print("PostgreSQL tablosu başarıyla oluşturuldu!")
