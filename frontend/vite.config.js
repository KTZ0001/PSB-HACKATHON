import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Single-screen demo client for the Aegis trust-scoring API.
// Proxy /api to the backend so the browser talks same-origin (no CORS surprises).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
