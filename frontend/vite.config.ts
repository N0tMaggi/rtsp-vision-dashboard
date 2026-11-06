import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const backend = process.env.SECURITYCAM_BACKEND ?? "http://127.0.0.1:5000"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/static/app/",
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  build: {
    outDir: "../securitycam/static/app",
    emptyOutDir: true,
    chunkSizeWarningLimit: 700,
  },
  server: {
    proxy: {
      "/api": backend,
      "/video_feed": backend,
    },
  },
})
