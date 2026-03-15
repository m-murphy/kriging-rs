import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: ".",
  plugins: [react()],
  server: {
    port: 5173,
    // Allow serving pkg from linked kriging-rs-wasm (file:../npm/...) which lives outside www/
    fs: {
      allow: [path.resolve(__dirname, "..")],
    },
  },
});
