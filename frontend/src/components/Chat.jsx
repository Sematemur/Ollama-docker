import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function Chat() {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [error, setError] = useState(null)
const messagesEndRef = useRef(null)

  useEffect(() => {
    setSessionId(crypto.randomUUID())
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    if (!inputValue.trim()) return

    const messageText = inputValue
    setMessages(prev => [...prev, { role: 'user', content: messageText }])
    setInputValue('')
    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        message: messageText,
        session_id: sessionId || undefined,
      })
      setMessages(prev => [...prev, { role: 'assistant', content: response.data.response }])
    } catch (err) {
      const detail = err.response?.data?.detail || err.message
      setError(detail || 'Mesaj gönderilemedi. Backend çalışıyor mu?')
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div style={styles.container}>
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
                ...(msg.role === 'user' ? styles.userMessage : styles.assistantMessage),
              }}
            >
              <div style={styles.messageRole}>
                {msg.role === 'user' ? 'Sen' : 'Asistan'}
              </div>
              <div style={styles.messageContent}>{msg.content}</div>
            </div>
          ))
        )}

        {isLoading && (
          <div style={{ ...styles.message, ...styles.assistantMessage }}>
            <div style={styles.messageRole}>Asistan</div>
            <div style={styles.loading}>Düşünüyor...</div>
          </div>
        )}

        {error && <div style={styles.error}>{error}</div>}

        <div ref={messagesEndRef} />
      </div>

<div style={styles.inputContainer}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Mesajınızı yazın..."
          style={styles.input}
          disabled={isLoading}
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || !inputValue.trim()}
          style={{
            ...styles.button,
            ...(isLoading || !inputValue.trim() ? styles.buttonDisabled : {}),
          }}
        >
          Gönder
        </button>
      </div>
    </div>
  )
}

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
