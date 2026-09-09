import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// В проде статику раздаёт Caddy, он же проксирует /api и /ws.
// Прокси ниже нужен только для `npm run dev`, чтобы фронт на :5173
// ходил в тот же backend без правки адресов в коде.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8001", ws: true },
    },
  },
  build: {
    outDir: "dist",
    // Шрифты и wavesurfer заметно весят; разносим, чтобы обновление кода
    // не сбрасывало их из кеша браузера.
    rollupOptions: {
      output: {
        manualChunks: {
          wavesurfer: ["wavesurfer.js"],
        },
      },
    },
  },
});
