# Build stage
FROM node:20-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
# Install TailwindCSS and its dependencies
RUN npm install -D tailwindcss postcss autoprefixer
COPY . .
RUN npm run build
# Ensure early-fix is injected into the built index.html so it runs before the app bundle
# Note: early-fix.js is included in the built dist/index.html. No sed injection needed.

# Production stage
FROM nginx:1.23-alpine
WORKDIR /usr/share/nginx/html
RUN rm -rf ./*
COPY --from=build /app/dist .
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
