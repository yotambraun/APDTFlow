/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,ts,jsx,tsx,md,mdx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0a0e1a',       // near-black hero background
        surface: '#0f1424',  // lifted cards / nav
        navy: '#0f1b3d',      // brand navy panels
        navy2: '#15235a',     // lighter navy borders/hover
        cyan: '#22d3ee',      // primary accent
        green: '#34d399',     // secondary accent (wave end, win cells)
        ink: '#e6edf6',       // primary text on dark
        muted: '#8aa0bd',     // secondary text
        line: '#1c2740',      // hairline borders
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      backgroundImage: {
        wave: 'linear-gradient(90deg, #22d3ee 0%, #34d399 100%)',
        'hero-glow':
          'radial-gradient(60% 50% at 50% 30%, rgba(34,211,238,.18), rgba(52,211,153,.10) 40%, transparent 70%)',
      },
      boxShadow: {
        glow: '0 0 40px -8px rgba(34,211,238,.35)',
        'glow-green': '0 0 40px -8px rgba(52,211,153,.30)',
      },
      maxWidth: {
        content: '72rem',
      },
    },
  },
  plugins: [],
};
