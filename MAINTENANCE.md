# Maintaining this site

This site is built with [Astro](https://astro.build). Content lives in **Markdown** and a few **TypeScript data files** — there's **no Ruby**, and updates are just editing text files. This guide covers everything you'll do day to day.

---

## 1. One-time setup

You need [Node.js](https://nodejs.org) (version pinned in [`.nvmrc`](.nvmrc) — currently 22).

```bash
# from the repo root
npm install
```

That's it. No Bundler, no gems, no ImageMagick.

> If you use `nvm`, run `nvm use` to switch to the pinned Node version.

---

## 2. Local preview

```bash
npm run dev
```

Open the URL it prints (usually <http://localhost:4321>). The page hot-reloads as you edit — save a file and the browser updates instantly.

To check the exact thing that will be deployed:

```bash
npm run build     # outputs to dist/
npm run preview   # serves the built site locally
```

---

## 3. Common changes

Everything below is a plain text edit. Save, check locally, commit, push.

### Add a blog post

Create a file in [`src/content/blog/`](src/content/blog/), e.g. `my-new-post.md`. The **filename becomes the URL** (`/blog/my-new-post`). Start it with front matter:

```markdown
---
title: "My New Post"
date: 2026-02-01
category: research        # "research" or "personal"
description: "One-line summary shown in listings and previews."
tags: ["mech-interp"]     # optional
# external: "https://..." # optional: if it's cross-posted somewhere canonical
# draft: true             # optional: hide it until you're ready
---

Your content here. Standard Markdown.
```

- **Images:** drop them in `public/assets/img/…` and reference them as `/assets/img/your-image.png`.
- **Math:** wrap inline math in `$…$` and display math in `$$…$$` (KaTeX renders it).
- **Code blocks:** fenced triple-backticks get syntax highlighting automatically.
- A post with `draft: true` is hidden from the site (and from the build) until you remove that line.

### Add / edit a publication

Edit [`src/data/publications.ts`](src/data/publications.ts). Add a new object at the **top** of the array (newest first):

```ts
{
  title: 'Paper Title',
  authors: [
    { name: 'Shreyans Jain', me: true },   // me: true bolds your name
    { name: 'Co Author', eq: true },        // eq: true adds a * (equal contribution)
  ],
  venue: 'EMNLP 2026 (Findings)',
  year: 2026,
  status: 'Under review',                   // optional
  selected: true,                           // optional: flag for highlighting
  links: [
    { label: 'arXiv', href: 'https://arxiv.org/abs/...' },
    { label: 'code', href: 'https://github.com/...' },
  ],
},
```

### Update the CV

Edit [`src/data/cv.ts`](src/data/cv.ts) — it has `summary`, `researchExperience`, `industryExperience`, `education`, and `selectedWriting`. The **"Full CV (PDF)" button** points to the `cvPdf` link in [`src/data/site.ts`](src/data/site.ts) (currently a Google Drive share link) — update that when you upload a new CV.

### Update research directions / software

Edit [`src/data/research.ts`](src/data/research.ts) (`directions` and `software` arrays).

### Change your name, tagline, nav, or social links

Edit [`src/data/site.ts`](src/data/site.ts) — one file holds identity, navigation, and all social links.

### Edit the home page copy (bio, section intros)

Edit [`src/pages/index.astro`](src/pages/index.astro). The prose lives near the top; styling is in the `<style>` block at the bottom.

---

## 4. How deployment works

Deployment is fully automated by GitHub Actions — you never build or upload anything by hand.

```
git push  ──►  GitHub Action ([.github/workflows/deploy.yml])
                 1. checks out the repo
                 2. installs Node (from .nvmrc) + npm ci
                 3. npm run build  ──►  dist/
                 4. uploads dist/ and publishes to GitHub Pages
```

**Any push to `master`** triggers it. Watch progress under the repo's **Actions** tab; a green check means it's live at <https://shreyansjainn.github.io> (usually within ~1–2 minutes). You can also trigger it manually from the Actions tab ("Run workflow").

### One-time GitHub Pages switch (do this once, after merging)

The old site deployed via Jekyll to a `gh-pages` branch. The new workflow uses the modern **GitHub Actions** Pages source. In the repo:

**Settings → Pages → Build and deployment → Source → select "GitHub Actions".**

After that, every push just works.

---

## 5. Typical workflow, end to end

```bash
nvm use                       # match the pinned Node version (optional)
npm run dev                   # preview while you edit
# ...make your edits...
git add -A
git commit -m "add post: my-new-post"
git push                      # GitHub Action builds + deploys automatically
```

---

## 6. Project structure

```
src/
  content/blog/      your posts (Markdown)      ← add posts here
  content.config.ts  blog front-matter schema (validates your posts)
  data/              site.ts, publications.ts, cv.ts, research.ts  ← structured content
  components/        reusable UI (Nav, Footer, PublicationList, Icon)
  layouts/           Base.astro (head, nav, footer, theme), post layout
  pages/             index.astro (home), blog/ (list + post pages)
  styles/global.css  design tokens + base styles (colors, fonts, light/dark)
public/assets/       images and the CV PDF (served at /assets/…)
astro.config.mjs     build config (math, sitemap)
.nvmrc               pinned Node version
```

---

## 7. Troubleshooting

- **Build fails after editing a post:** you probably have a front-matter typo. The error names the file and field. Every post must have `title` and a valid `date` (`YYYY-MM-DD`).
- **A post isn't showing:** it likely has `draft: true`, or a `date` in the future.
- **An image is broken:** confirm the file is under `public/assets/…` and the path in Markdown starts with `/assets/…` (leading slash, no `public`).
- **Math looks wrong:** check `$…$` / `$$…$$` are balanced. KaTeX warns (but doesn't fail) on unsupported Unicode — prefer `x'` over `x′`, `\in` over `∈`, etc.
- **Deploy didn't run:** make sure Pages Source is set to "GitHub Actions" (section 4) and you pushed to `master`.
- **Wrong Node version locally:** `nvm use`, or install the version in `.nvmrc`.
```
