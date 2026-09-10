# Website Revamp Plan — Astro rebuild

**Branch:** `revamp/astro-site`
**Goal:** A clean, research-oriented personal site that is easy to maintain — no Ruby, few dependencies, content in plain Markdown.

---

## Why we're moving off Jekyll/al-folio

The current site is the **al-folio** theme. Your maintenance pain is structural, not accidental:

- ~25 Ruby gems / ~20 Jekyll plugins (`jekyll-scholar`, `jekyll-imagemagick` needs an ImageMagick binary, `jekyll-jupyter-notebook`, a Twitter plugin, minifier, paginate, archives…).
- Every one is a version that can break against a new Ruby / bundler — which is the exact symptom you described.
- A lot of the theme is machinery you don't use.

**Astro** fixes this at the root:
- No Ruby. Node/npm only, ~5 direct dependencies.
- Content stays in **Markdown/MDX** — your posts and `.bib` move over almost unchanged.
- Fast static output, first-class GitHub Pages deploy via GitHub Actions.
- Typed "content collections" keep the site from silently breaking when a field is missing.

## What's real vs. placeholder today (audit)

**Keep (real content):**
- `_pages/about.md` — strong interpretability-focused bio.
- `_bibliography/papers.bib` — 3 real papers.
- 7 real posts in `_posts/` (research notes + personal essays).
- Profile pic (`assets/img/profile_pic.jpg`), CV PDF (`assets/pdf/CV-Shreyans Jain.pdf`).
- Socials: GitHub `shreyansjainn`, LinkedIn `shreyans-jain-4b063667`, Google Scholar `kPbV2RYAAAAJ`, X `py_parrot`, email.

**Drop (template placeholder):**
- `_data/cv.yml` — still **Einstein** (University of Zurich, Nobel Prize 1921).
- `_news/` — lorem-ipsum ("Jean shorts raw denim Vice normcore…").
- `about_einstein.md`, `dropdown.md`, `favblog.md`, `teaching.md`, `repositories.md`, placeholder `_projects/`.
- Docker/devcontainer, lighthouse, all-contributors, tweet-cache, purgecss — al-folio scaffolding.

## Site structure (new)

**One long-scroll home page + a real blog.** Sections live on the home page and only get promoted to a separate page when the content is substantial enough to need its own URL. Nav becomes anchor links that scroll the home page, plus **Blog**.

**Home page sections (scroll down):**

| Section | Source | Why inline |
|---------|--------|-----------|
| **About / Hero** | `src/content/home.md` | Bio + research focus + profile pic + socials. |
| **Publications** | `src/content/papers.bib` | Only 3 papers — reads better as an inline list than a near-empty page. Scales via `.bib`. |
| **Research** | `src/content/home.md` (or `research/*.md`) | Ongoing threads: Behaviour Compositions, Multi-Lingual Interp, `visualizing-training` (ICLR 2025 poster), `mech-interp`. Short cards. |
| **Writing** | latest 3 from `src/content/blog/` | Teaser list → links into the full Blog. |
| **Now + Contact** | `src/content/home.md` | "What I'm focused on now" + email/socials. Too small to be its own page. |
| **CV** | inline highlights + PDF button | Real history (replaces Einstein) with a button to `CV-Shreyans Jain.pdf`. |

**Separate pages (content warrants its own URLs):**

| Page | Source | Notes |
|------|--------|-------|
| **Blog index** | `src/content/blog/` | Your 7 posts; optional tag filter (`mech-interp`, `paper-notes`, personal). |
| **Blog post** | `src/content/blog/*.md` | One page per post — some are long-form, need real URLs for sharing. |

If any inline section later grows (e.g. a full CV, or Research becomes a rich portfolio), promoting it to its own page is a small, mechanical change — the content is already isolated Markdown.

## Design direction (my default — easy to change)

- Clean, minimal, text-first academic look. Generous whitespace, one accent color.
- Readable serif or high-quality sans for body; system-font stack (no webfont download to break).
- Light/dark toggle. Responsive, phone-first.
- No heavy JS; content and typography do the work.

## Maintenance model (the payoff)

- **Add a post:** drop a `.md` file in `src/content/blog/`. Done.
- **Add a paper:** paste a BibTeX entry into `papers.bib`. Done.
- **Update research/now:** edit one Markdown file.
- **Deploy:** push to `master` → GitHub Action builds and publishes. No local Ruby needed.
- A short `README` documents exactly these four actions.

## Build steps

1. **Scaffold** Astro in the branch (`package.json`, `astro.config`, `tsconfig`), content-collection schemas for blog/research, `.nvmrc` pinning Node.
2. **Layout & design system** — base layout, nav, footer, theme tokens, dark mode, typography.
3. **Home/About** — port bio, selected-work + latest-posts sections.
4. **Blog** — migrate all 7 posts (front-matter normalized), index + tag filter, per-post pages, images.
5. **Publications** — parse `papers.bib` → publications list.
6. **Research** + **Now** pages from your real project material.
7. **CV** — rebuild from your actual history (I'll draft from the site content; you fill gaps) + PDF link.
8. **Migrate assets** — profile pic, CV PDF, post images.
9. **GitHub Actions deploy** workflow; verify build; local preview.
10. **Cleanup** — remove al-folio scaffolding once the new site builds cleanly.
11. **README** with the maintenance model above.

## Source of truth: CV PDF (provided)

The CV PDF is now the canonical source for the CV section **and** a fuller publications list:

**Publications (6, reverse-chron):**
1. *Gotta Catch Them All: The Modes of Sycophancy* — Jain, Yost, Abdullah. Under review.
2. *Measure What Matters: Psychometric Evaluation of AI with Situational Judgment Tests* — Yost\*, Jain\*, Raval, Corser, Roush, Xu, Hammack, Shwartz-Ziv, Abdullah. EMNLP 2026 (Findings). arXiv:2510.22170
3. *Beyond Linear Steering: Unified Multi-Attribute Control for Language Models* — Oozeer, Marks, Jain, Barez, Abdullah. EMNLP 2025 (Findings).
4. *How to Visualize Training Dynamics in Neural Networks* — Hu, Jain, Chaulagain, Saphra. ICLR 2025 Blogpost Track.
5. *Sycophancy as Compositions of Atomic Psychometric Traits* — Jain, Yost, Abdullah. BlackboxNLP 2025 (Extended Abstract). arXiv:2508.19316
6. *Towards Discovering Linguistic Indicators for Misalignment in Language Models* — Jain, Raval. BlackboxNLP 2025 (Extended Abstract). Zenodo:16988484

**CV sections:** Summary · Research Experience (LASR Labs, Thoughtworks ×2, Martian, Independent) · Software (K-Steering, visualizing-training) · Selected Writing · Prior Industry (Physarum, GEP, BookMyShow, Hotstar, Fractal) · Education (B.Tech Civil, MNIT Jaipur 2011–2015; AISF BlueDot Nov 2024).

## Open items for you (non-blocking)

- arXiv/PDF links for the papers that don't have one yet (Gotta Catch Them All; Beyond Linear Steering) — I'll leave placeholders/venue links until you have them.
- React to the design once the first pass is up.

---

*Nothing here deletes your old site until the new one builds — al-folio files stay until step 10, and it's all on the `revamp/astro-site` branch.*
