/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        obsidian: {
          950: '#050507',
          900: '#0a0a0f',
          800: '#111118',
          700: '#1a1a24',
          600: '#252530',
        },
        accent: {
          cyan: '#00e5ff',
          blue: '#3b82f6',
          violet: '#8b5cf6',
          red: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['SF Pro Display', '-apple-system', 'BlinkMacSystemFont', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['SF Mono', 'JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
