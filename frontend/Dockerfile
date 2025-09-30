# syntax=docker/dockerfile:1.7

ARG NODE_IMAGE=node:20-bullseye-slim

FROM ${NODE_IMAGE} AS deps

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

FROM ${NODE_IMAGE} AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD wget -qO- http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
