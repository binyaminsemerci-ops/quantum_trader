/** Generated Tailwind config (restored) */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'media',
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/line-clamp'),
  ],
};/** Canonical Tailwind CJS config */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  darkMode: 'class',
  theme: {
    container: { center: true, padding: '1rem', screens: { '2xl': '1600px' } },
    extend: {
      screens: { mdp: '900px' },
      colors: {
        brand: {
          DEFAULT: '#2563eb',
          dark: '#1e40af',
          light: '#3b82f6',
          accent: '#6366f1',
        },
        surface: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          800: '#1e293b',
          900: '#0f172a'
        }
      },
      fontFamily: {
        sans: [ 'Inter', 'system-ui', 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', 'sans-serif' ],
        mono: [ 'JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace' ]
      },
      boxShadow: {
        elevated: '0 4px 12px -2px rgba(0,0,0,0.08), 0 2px 4px -1px rgba(0,0,0,0.06)',
        glow: '0 0 0 1px rgba(99,102,241,0.4), 0 4px 20px -2px rgba(99,102,241,0.35)'
      },
      animation: { 'pulse-slow': 'pulse 3s linear infinite' },
      spacing: { '4.5': '1.125rem' },
      borderRadius: { xl2: '1.15rem' },
      backdropBlur: { xs: '2px' }
    },
  },
  plugins: [
    require('@tailwindcss/forms')
  ],
};
