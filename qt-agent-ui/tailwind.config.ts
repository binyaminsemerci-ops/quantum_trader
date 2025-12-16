import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      borderRadius: {
        xl: "0.9rem",
        "2xl": "1.2rem",
      },
      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.06)",
      },
      colors: {
        brand: {
          DEFAULT: "#2563EB",
          50: "#EFF6FF",
          100: "#DBEAFE",
          200: "#BFDBFE",
          300: "#93C5FD",
          400: "#60A5FA",
          500: "#3B82F6",
          600: "#2563EB",
          700: "#1D4ED8",
          800: "#1E40AF",
          900: "#1E3A8A",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
