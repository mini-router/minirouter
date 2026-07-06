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

The app is configured for GitHub Pages with a relative Vite base and a hash-based router, so it can be hosted from the repository Pages URL without extra rewrites.

Publishing is handled by the `gh-pages` branch. Run `npm run deploy` to build the site and push the static output to that branch.

The build output includes a `.nojekyll` marker so GitHub Pages serves the generated assets directly.
