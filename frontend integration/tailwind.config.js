/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        obsidian: {
          base: '#060608',
          surface: '#0c0c10',
          raised: '#141418',
          card: '#1a1a20',
          high: '#222228',
          muted: '#2a2a32',
        },
        crimson: {
          DEFAULT: '#e03e4a',
          dark: '#b91c2c',
          light: '#ff6b6b',
          pale: '#ffb3b1',
          ghost: 'rgba(224, 62, 74, 0.06)',
        },
        neutral: {
          text: '#e8e8ec',
          secondary: '#8e8e96',
          tertiary: '#5a5a64',
          faint: '#3a3a42',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'system-ui', 'sans-serif'],
        serif: ['Newsreader', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
