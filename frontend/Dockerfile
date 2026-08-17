# LegalAId frontend — build with Node, serve the static bundle with nginx.
#
# nginx also reverse-proxies /api to the gateway, so the browser sees a single origin and there
# is no CORS configuration to keep in sync between dev and Docker.

FROM node:22-alpine AS build

WORKDIR /app

COPY package.json ./
RUN npm install --no-audit --no-fund

COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src

# In the container the API is reached through nginx on the same origin, so no absolute base URL.
ENV VITE_API_BASE=""
RUN npm run build


FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# Probe the served index page. Note: this image's BusyBox `nc` does not support `-z`, which is
# why an apparently obvious port check fails here; curl is present and unambiguous.
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost/ -o /dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
