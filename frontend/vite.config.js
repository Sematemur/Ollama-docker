import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite yapılandırması
// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // Frontend'in çalışacağı port
  }
})
