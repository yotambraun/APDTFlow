// @ts-check
import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

// Project page at https://yotambraun.github.io/APDTFlow/
// `base` MUST be exactly '/APDTFlow' (CamelCase, no trailing slash).
// All internal links/assets go through the url() helper in src/data/site.ts.
export default defineConfig({
  site: 'https://yotambraun.github.io',
  base: '/APDTFlow',
  output: 'static',
  trailingSlash: 'ignore',
  integrations: [tailwind()],
});
