/**
 * APP.JSX - Ana Uygulama Bileşeni
 *
 * Bu bileşen uygulamanın ana yapısını oluşturur.
 * Chat bileşenini içerir.
 */

import Chat from './components/Chat'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Chat Asistanı</h1>
      </header>
      <main>
        <Chat />
      </main>
    </div>
  )
}

export default App
