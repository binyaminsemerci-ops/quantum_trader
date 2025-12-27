/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        quantum: {
          bg: '#0a0a0f',
          dark: '#111118',
          card: '#1a1a24',
          border: '#2a2a38',
          text: '#e5e7eb',
          muted: '#9ca3af',
          accent: '#22c55e',
          danger: '#ef4444',
        },
      },
    },
  },
  plugins: [],
};
