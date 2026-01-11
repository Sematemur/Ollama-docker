# ============================================
# FRONTEND DOCKERFILE
# ============================================
# React (Vite) uygulaması için Docker image

# Node.js 20 Alpine image (küçük boyut)
FROM node:20-alpine

# Çalışma dizinini ayarla
WORKDIR /app

# Package dosyalarını kopyala (cache için)
COPY package*.json ./

# Bağımlılıkları yükle
RUN npm install

# Uygulama kodunu kopyala
COPY . .

# Port 5173'ü aç (Vite default)
EXPOSE 5173

# Vite dev server'ı başlat
# --host: Docker dışından erişim için gerekli
CMD ["npm", "run", "dev", "--", "--host"]
