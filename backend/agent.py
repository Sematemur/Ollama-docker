"""
AGENT.PY - LangChain ile Chat Agent

LangChain Nedir?
----------------
LangChain, LLM (Large Language Model) uygulamaları geliştirmek için bir framework'tür.
- Farklı LLM'lerle (OpenAI, Ollama, vb.) kolayca çalışmanızı sağlar
- Sohbet geçmişi yönetimi yapar
- Agent'lar oluşturmanızı sağlar (araç kullanan akıllı asistanlar)

Bu dosyada ne yapıyoruz?
------------------------
1. LiteLLM'e bağlanan bir ChatOpenAI nesnesi oluşturuyoruz
2. Sohbet geçmişini yöneten bir memory oluşturuyoruz
3. Basit bir sohbet chain'i kuruyoruz
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict
import os

# ============================================
# LiteLLM Konfigürasyonu
# ============================================
# Tüm gizli bilgiler .env dosyasından okunur

API_KEY = os.getenv("LITELLM_API_KEY", "")
BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:8023/")
MODEL_NAME = os.getenv("LITELLM_MODEL_NAME", "ollama/qwen3:0.6b")


def create_llm():
    """
    LangChain'in ChatOpenAI sınıfını kullanarak LLM bağlantısı oluşturur.

    ChatOpenAI, OpenAI API formatıyla uyumlu herhangi bir servisle çalışır.
    LiteLLM de OpenAI API formatını desteklediği için bu sınıfı kullanabiliriz.

    Parametreler:
    - openai_api_key: API anahtarı (LiteLLM için gerekli)
    - openai_api_base: LiteLLM'in çalıştığı adres
    - model_name: Kullanılacak model
    - temperature: Cevapların yaratıcılık seviyesi (0=deterministik, 1=yaratıcı)
    """
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model_name=MODEL_NAME,
        temperature=0.7,  # Dengeli bir yaratıcılık seviyesi
    )
    return llm


def format_messages_for_langchain(history: List[Dict]) -> List:
    """
    Veritabanından gelen mesaj geçmişini LangChain formatına çevirir.

    Veritabanındaki format:
    {"role": "user", "content": "Merhaba"}

    LangChain formatı:
    HumanMessage(content="Merhaba")  veya  AIMessage(content="...")

    Bu dönüşüm gerekli çünkü LangChain kendi mesaj sınıflarını kullanır.
    """
    messages = []

    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    return messages


def get_chat_response(user_message: str, history: List[Dict] = None) -> str:
    """
    Kullanıcı mesajına cevap üretir.

    Parametreler:
    - user_message: Kullanıcının gönderdiği mesaj
    - history: Önceki mesajların listesi (sohbet bağlamı için)

    Nasıl çalışır:
    1. LLM bağlantısı oluşturulur
    2. Geçmiş mesajlar LangChain formatına çevrilir
    3. Yeni mesaj eklenir
    4. LLM'den cevap alınır

    Not: history parametresi None ise, bu yeni bir sohbet demektir.
    """
    llm = create_llm()

    # Mesaj listesini hazırla
    messages = []

    # Sistem mesajı ekle (agent'ın nasıl davranacağını belirler)
    system_prompt = """Sen yardımcı bir asistansın. Kullanıcının sorularına
    Türkçe olarak, samimi ve anlaşılır bir şekilde cevap ver.
    Kısa ve öz cevaplar vermeye çalış."""

    messages.append(SystemMessage(content=system_prompt))

    # Sohbet geçmişini ekle (varsa)
    if history:
        messages.extend(format_messages_for_langchain(history))

    # Yeni kullanıcı mesajını ekle
    messages.append(HumanMessage(content=user_message))

    # LLM'den cevap al
    response = llm.invoke(messages)

    # AIMessage nesnesinden sadece içeriği al
    return response.content


# Test için
if __name__ == "__main__":
    # Basit bir test
    print("Agent test ediliyor...")
    print(f"API Key: {API_KEY[:10]}..." if API_KEY else "API Key bulunamadı!")
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    response = get_chat_response("Merhaba, nasılsın?")
    print(f"Cevap: {response}")
