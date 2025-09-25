# --- Bygg steg ---
FROM node:20-alpine AS build

WORKDIR /app

# Kopier kun package.json først for bedre cache
COPY package*.json ./

RUN npm ci --silent

# Kopier resten av prosjektet
COPY . .

# Bygg React/Vite frontend
RUN npm run build

# --- Produksjons steg ---
FROM nginx:alpine

# Fjern default nginx-html
RUN rm -rf /usr/share/nginx/html/*

# Kopier bygget app fra build-steg
COPY --from=build /app/dist /usr/share/nginx/html

# Valgfri: Egendefinert nginx-konfig hvis du har en nginx.conf
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
