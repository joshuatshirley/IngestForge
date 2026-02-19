/**
 * Jest Configuration for IngestForge Portal
 * US-1401.1, US-1402.1: Test infrastructure for React components
 */
const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: './',
});

/** @type {import('jest').Config} */
const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  testEnvironment: 'jest-environment-jsdom',
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  testMatch: ['**/__tests__/**/*.test.{ts,tsx}', '**/*.test.{ts,tsx}'],
  collectCoverageFrom: [
    'src/components/**/*.{ts,tsx}',
    '!src/components/**/*.d.ts',
  ],
  // Transform D3 and related ESM modules
  transformIgnorePatterns: [
    '/node_modules/(?!(d3|d3-array|d3-axis|d3-brush|d3-chord|d3-color|d3-contour|d3-delaunay|d3-dispatch|d3-drag|d3-dsv|d3-ease|d3-fetch|d3-force|d3-format|d3-geo|d3-hierarchy|d3-interpolate|d3-path|d3-polygon|d3-quadtree|d3-random|d3-scale|d3-scale-chromatic|d3-selection|d3-shape|d3-time|d3-time-format|d3-timer|d3-transition|d3-zoom|internmap|delaunator|robust-predicates)/)',
  ],
};

module.exports = createJestConfig(customJestConfig);
