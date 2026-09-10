import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string().optional(),
    tags: z.array(z.string()).default([]),
    // research = technical notes/results, personal = essays/reflections
    category: z.enum(['research', 'personal']).default('research'),
    // if the canonical version lives elsewhere (LessWrong, arXiv, etc.)
    external: z.string().url().optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { blog };
