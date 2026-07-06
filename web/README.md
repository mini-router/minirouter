# MiniRouter Competition

Competition site for the MiniRouter optimization challenge.

## Local development

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```

## GitHub Pages

The app is configured for GitHub Pages with a relative Vite base and a hash-based router, so it can be hosted without extra rewrites.

This site is published from the separate Pages repository at `mini-router/mini-router.github.io`.
From this `web/` folder, build the app and push the generated `dist/` output to that repo's
`gh-pages` branch:

```bash
npm run build
npx gh-pages -d dist -r https://github.com/mini-router/mini-router.github.io.git -b gh-pages
```

The build output includes a `.nojekyll` marker so GitHub Pages serves the generated assets directly.

You do not need a GitHub Actions workflow for manual deploys. Use a workflow only if you want the
site to auto-deploy on push from this repo.
