/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/features/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // US-1401.1 AC: Foundry Dark theme palette
        'forge-navy': '#1a1a2e',
        'forge-crimson': '#e94560',
        'forge-blue': '#16213e',
        'forge-accent': '#4ecca3',
        // Extended Foundry Dark colors
        'forge-slate': '#0f0f1a',
        'forge-charcoal': '#252538',
        'forge-steel': '#3d3d5c',
        'forge-ember': '#ff6b6b',
        'forge-gold': '#ffd93d',
        'forge-teal': '#6bcb77',
      },
      // US-1401.1 AC: Responsive grid for 1080p and 4K
      screens: {
        '3xl': '1920px',
        '4k': '2560px',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
    },
  },
  plugins: [],
  darkMode: 'class',
}
