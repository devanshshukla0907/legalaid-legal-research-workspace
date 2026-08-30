import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The gateway authenticates every route except health. In local development the key lives in
 * data/.dev-api-key, written by the backend on first start, so the proxy injects it and the
 * developer never has to paste a key. Production builds send the key from the app itself
 * (VITE_API_KEY or the Settings page), because there is no proxy in front of them.
 */
function devApiKey(): string {
  for (const candidate of [
    process.env.API_KEY,
    process.env.VITE_API_KEY,
  ]) {
    if (candidate?.trim()) return candidate.trim();
  }
  try {
    return readFileSync(resolve(__dirname, "../data/.dev-api-key"), "utf-8").trim();
  } catch {
    // Backend not started yet. Requests will get a 401 with an explanatory message, and a
    // restart of the dev server picks the key up.
    return "";
  }
}

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      // The UI talks to the gateway through a proxy so there is one origin in dev and no CORS
      // surprises. In Docker this is overridden by VITE_API_BASE.
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
        configure: (proxy) => {
          const key = devApiKey();
          if (!key) return;
          proxy.on("proxyReq", (proxyReq) => {
            if (!proxyReq.getHeader("X-API-Key")) {
              proxyReq.setHeader("X-API-Key", key);
            }
          });
        },
      },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
