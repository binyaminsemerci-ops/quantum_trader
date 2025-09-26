/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",  // fanger opp alle React-komponenter
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "#2563eb", // blå
          dark: "#1e40af",
          light: "#3b82f6",
        },
      },
    },
  },
  plugins: [],
};
