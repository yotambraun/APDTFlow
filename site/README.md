# APDTFlow showcase site

The marketing/demo website for APDTFlow, built with [Astro](https://astro.build)
and Tailwind CSS and deployed to GitHub Pages at
**https://yotambraun.github.io/APDTFlow/**.

This directory is completely separate from the Python package: it is never
included in the wheel/sdist (packaging uses `find_packages()`), and `ci.yml`
ignores `site/**`, so changes here never touch PyPI or run the Python test suite.

## Develop

```bash
cd site
npm install
npm run dev        # http://localhost:4321/APDTFlow/
```

The dev server honors the `/APDTFlow/` base path so local previews match
production and base-path bugs surface immediately.

## Build & preview

```bash
npm run build      # outputs to site/dist
npm run preview    # serves the production build at /APDTFlow/
```

## Deploy

Pushing changes under `site/**` to `main` triggers
`.github/workflows/site.yml`, which builds and deploys to GitHub Pages.

**One-time setup:** in the repo, Settings → Pages → Build and deployment →
Source = **GitHub Actions**.

## Editing content

Almost everything is driven by `src/data/site.ts` — links, stats fallbacks,
the evidence/benchmark tables, the capability matrix, and the `url()` base-path
helper. Showcase images live in `public/images/` (copied from the repo's
`assets/images/` and `examples/`). All numbers are sourced from the project
README and `experiments/` — keep them truthful, including the rows where
APDTFlow loses.
