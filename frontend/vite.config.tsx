import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8001", // FastAPI backend on new port
        ws: true,
        changeOrigin: true,
        secure: false,
        // NO rewrite - keep /api/v1 paths intact
      },
      "/ws": {
        target: "http://127.0.0.1:8001",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
      // Direct backend routes (fallback for any routes not prefixed with /api)
      "^/(signals|layout|health|stats|trades|chart|settings|binance)": {
        target: "http://127.0.0.1:8001",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
