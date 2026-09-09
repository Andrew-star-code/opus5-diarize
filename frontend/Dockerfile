# Фронтенд собирается здесь же, где живёт Caddy: отдельный контейнер под
# статику не нужен, а Caddy всё равно терминирует TLS и проксирует API.
FROM node:22-alpine AS build

WORKDIR /app

# Слой зависимостей отдельно: правка кода не должна тянуть npm заново.
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY . .
RUN npm run build

# ─────────────────────────────────────────────────────────────
FROM caddy:2-alpine

# Шрифты и скрипты уже внутри сборки: в оффлайн-сети внешние CDN
# недоступны, всё должно приезжать с этого же сервера.
COPY --from=build /app/dist /srv

# Caddyfile монтируется из репозитория (см. docker-compose.yml),
# чтобы правка проксирования не требовала пересборки образа.
