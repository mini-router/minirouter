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

The repository now publishes this site with the workflow in `.github/workflows/deploy-web.yml`.
GitHub Pages must be enabled for the repository and set to use **GitHub Actions** as the source.
The workflow builds `web/` and deploys the generated `dist/` output directly to Pages.

```bash
npm run build
```

The build output includes a `.nojekyll` marker so GitHub Pages serves the generated assets directly.

You can still build locally with `npm run build` for manual checks.
