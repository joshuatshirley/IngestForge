/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./jest.setup.js'], // Reuse existing setup
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
