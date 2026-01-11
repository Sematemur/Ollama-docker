/**
 * CHAT.JSX - Chat Bileşeni
 *
 * Bu bileşen chat arayüzünün tamamını oluşturur:
 * - Mesaj listesi
 * - Input alanı
 * - Gönder butonu
 *
 * React Hooks kullanıyoruz:
 * - useState: Bileşen durumunu tutmak için (mesajlar, input değeri, vs.)
 * - useEffect: Yan etkileri yönetmek için (session ID oluşturma)
 * - useRef: DOM elementlerine referans tutmak için (otomatik scroll)
 */

import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

// Backend API adresi
const API_URL = 'http://localhost:8000'

function Chat() {
  // ============================================
  // State Tanımlamaları
  // ============================================

  // Mesajların listesi - her mesaj { role: 'user'|'assistant', content: '...' } formatında
  const [messages, setMessages] = useState([])

  // Kullanıcının yazdığı metin
  const [inputValue, setInputValue] = useState('')

  // Cevap beklenirken true olur (loading durumu)
  const [isLoading, setIsLoading] = useState(false)

  // Sohbet oturumu ID'si - her sayfa açılışında yeni bir ID oluşturulur
  const [sessionId, setSessionId] = useState(null)

  // Hata mesajı
  const [error, setError] = useState(null)

  // Mesaj listesinin sonuna scroll yapmak için referans
  const messagesEndRef = useRef(null)

  // ============================================
  // useEffect - Session ID Oluşturma
  // ============================================

  useEffect(() => {
    // Sayfa yüklendiğinde benzersiz bir session ID oluştur
    // crypto.randomUUID() modern tarayıcılarda çalışır
    const newSessionId = crypto.randomUUID()
    setSessionId(newSessionId)
    console.log('Yeni session başlatıldı:', newSessionId)
  }, []) // Boş array: sadece ilk render'da çalışır

  // ============================================
  // useEffect - Otomatik Scroll
  // ============================================

  useEffect(() => {
    // Yeni mesaj geldiğinde en alta scroll yap
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages]) // messages değiştiğinde çalışır

  // ============================================
  // Mesaj Gönderme Fonksiyonu
  // ============================================

  const sendMessage = async () => {
    // Boş mesaj gönderme
    if (!inputValue.trim()) return

    // Kullanıcı mesajını state'e ekle (anında görünsün)
    const userMessage = { role: 'user', content: inputValue }
    setMessages(prev => [...prev, userMessage])

    // Input'u temizle
    setInputValue('')

    // Loading durumunu başlat
    setIsLoading(true)
    setError(null)

    try {
      // Backend'e POST isteği gönder
      const response = await axios.post(`${API_URL}/chat`, {
        message: inputValue,
        session_id: sessionId
      })

      // Cevabı state'e ekle
      const assistantMessage = {
        role: 'assistant',
        content: response.data.response
      }
      setMessages(prev => [...prev, assistantMessage])

    } catch (err) {
      console.error('Hata:', err)
      setError('Mesaj gönderilemedi. Backend çalışıyor mu?')
    } finally {
      setIsLoading(false)
    }
  }

  // ============================================
  // Enter Tuşu ile Gönderme
  // ============================================

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // ============================================
  // Render
  // ============================================

  return (
    <div style={styles.container}>
      {/* Mesaj Listesi */}
      <div style={styles.messageList}>
        {messages.length === 0 ? (
          <div style={styles.emptyState}>
            Merhaba! Size nasıl yardımcı olabilirim?
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={index}
              style={{
                ...styles.message,
                ...(msg.role === 'user' ? styles.userMessage : styles.assistantMessage)
              }}
            >
              <div style={styles.messageRole}>
                {msg.role === 'user' ? 'Sen' : 'Asistan'}
              </div>
              <div style={styles.messageContent}>{msg.content}</div>
            </div>
          ))
        )}

        {/* Loading Göstergesi */}
        {isLoading && (
          <div style={{ ...styles.message, ...styles.assistantMessage }}>
            <div style={styles.messageRole}>Asistan</div>
            <div style={styles.loading}>Düşünüyor...</div>
          </div>
        )}

        {/* Hata Mesajı */}
        {error && (
          <div style={styles.error}>{error}</div>
        )}

        {/* Scroll için referans noktası */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Alanı */}
      <div style={styles.inputContainer}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Mesajınızı yazın..."
          style={styles.input}
          disabled={isLoading}
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || !inputValue.trim()}
          style={{
            ...styles.button,
            ...(isLoading || !inputValue.trim() ? styles.buttonDisabled : {})
          }}
        >
          Gönder
        </button>
      </div>
    </div>
  )
}

// ============================================
// Inline Stiller
// ============================================
// Not: Büyük projelerde CSS modülleri veya styled-components kullanılır.
// Basitlik için inline style kullandık.

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: 'calc(100vh - 120px)',
    backgroundColor: '#fff',
    borderRadius: '12px',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
    overflow: 'hidden',
  },
  messageList: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  emptyState: {
    textAlign: 'center',
    color: '#888',
    marginTop: '50px',
    fontSize: '1.1rem',
  },
  message: {
    padding: '12px 16px',
    borderRadius: '12px',
    maxWidth: '80%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007bff',
    color: '#fff',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#f0f0f0',
    color: '#333',
  },
  messageRole: {
    fontSize: '0.75rem',
    marginBottom: '4px',
    opacity: 0.7,
  },
  messageContent: {
    fontSize: '1rem',
    lineHeight: 1.5,
    whiteSpace: 'pre-wrap',
  },
  loading: {
    fontStyle: 'italic',
    color: '#666',
  },
  error: {
    textAlign: 'center',
    color: '#dc3545',
    padding: '10px',
    backgroundColor: '#ffebee',
    borderRadius: '8px',
  },
  inputContainer: {
    display: 'flex',
    padding: '16px',
    borderTop: '1px solid #eee',
    gap: '10px',
  },
  input: {
    flex: 1,
    padding: '12px 16px',
    border: '1px solid #ddd',
    borderRadius: '24px',
    fontSize: '1rem',
    outline: 'none',
    transition: 'border-color 0.2s',
  },
  button: {
    padding: '12px 24px',
    backgroundColor: '#007bff',
    color: '#fff',
    border: 'none',
    borderRadius: '24px',
    fontSize: '1rem',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed',
  },
}

export default Chat
