/**
 * MAIN.JSX - React Uygulamasının Başlangıç Noktası
 *
 * Bu dosya React uygulamasını DOM'a bağlar.
 * index.html'deki "root" div'ine App bileşenini render eder.
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
