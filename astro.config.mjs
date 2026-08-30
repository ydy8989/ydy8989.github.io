import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkIal } from './src/plugins/remark-ial.mjs';

export default defineConfig({
  site: 'https://ydy8989.github.io',
  // Jekyll 시절 URL(`/YYYY-MM-DD-slug/`)이 전부 뒤 슬래시를 달고 있다.
  // Disqus 식별자·검색 색인·외부 링크가 여기 걸려 있으므로 반드시 유지한다.
  trailingSlash: 'always',
  integrations: [mdx(), sitemap()],
  vite: { plugins: [tailwindcss()] },
  markdown: {
    remarkPlugins: [remarkMath, remarkIal],
    rehypePlugins: [rehypeKatex],
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } },
  },
});
