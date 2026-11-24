import { defineConfig } from 'vitest/config';

// `base: './'` makes the built site work from any sub-path, e.g. https://<user>.github.io/flipgpu/
export default defineConfig({
  base: './',
  build: { target: 'es2022', outDir: 'dist', sourcemap: true },
  // CPU reference simulations take a few seconds each; CI runners can be slow
  test: { include: ['tests/**/*.test.ts'], environment: 'node', testTimeout: 120000 },
});
