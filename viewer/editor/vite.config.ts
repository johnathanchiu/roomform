import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  define: {
    "process.env.NEXT_PUBLIC_ASSETS_CDN_URL": JSON.stringify(""),
  },
  optimizeDeps: {
    esbuildOptions: {
      define: {
        "process.env.NEXT_PUBLIC_ASSETS_CDN_URL": JSON.stringify(""),
      },
    },
  },
  plugins: [tailwindcss(), react()],
  server: {
    // dev mode reads scenes from the running serve_editor.py backend
    proxy: {
      "/api": "http://127.0.0.1:8792",
      "/artifacts": "http://127.0.0.1:8792",
    },
  },
});
