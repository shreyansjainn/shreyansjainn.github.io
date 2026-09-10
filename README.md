# shreyansjainn.github.io

Personal / research site for **Shreyans Jain** — [shreyansjainn.github.io](https://shreyansjainn.github.io).

Built with [Astro](https://astro.build). Content is Markdown + a few TypeScript data files; there's no Ruby, and deploys are automated via GitHub Actions.

## Quick start

```bash
npm install     # one-time (Node version pinned in .nvmrc)
npm run dev      # local preview at http://localhost:4321
npm run build    # production build -> dist/
```

## Editing the site

See **[MAINTENANCE.md](MAINTENANCE.md)** for how to add a post, add a publication, update the CV, change socials, and how deployment works.

Quick map:

| To change… | Edit |
| --- | --- |
| A blog post | `src/content/blog/*.md` |
| Publications | `src/data/publications.ts` |
| CV | `src/data/cv.ts` (+ the linked PDF) |
| Research directions / software | `src/data/research.ts` |
| Repeat reads | `src/data/reads.ts` |
| Name, tagline, nav, socials | `src/data/site.ts` |
| Home-page copy | `src/pages/index.astro` |
| Colors, fonts, light/dark | `src/styles/global.css` |

## Deploy

Push to `master` → the [`Deploy site`](.github/workflows/deploy.yml) GitHub Action builds and publishes to GitHub Pages. One-time: set **Settings → Pages → Source → "GitHub Actions"**.
