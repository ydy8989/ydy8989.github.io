import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// 콘텐츠 "계약". 여기 정의되지 않은 필드를 쓰거나
// 필수 필드를 빠뜨리면 빌드가 실패한다.
const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updated: z.coerce.date().optional(),
    // Jekyll permalink(`/YYYY-MM-DD-slug/`)을 그대로 재현하기 위한 슬러그.
    slug: z.string(),
    categories: z.array(z.string()).default([]),
    tags: z.array(z.string()).default([]),
    // Disqus 스레드 식별자 = 원래 글의 절대 URL. 댓글 보존의 핵심.
    disqusId: z.string(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { blog };
