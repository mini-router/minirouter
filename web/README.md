# MiniRouter Competition

Competition site for the MiniRouter optimization challenge.

The frontend reads its backend URL from `web/.env.production` at build time.
Update `VITE_API_BASE_URL` there if the validator host changes.

## Local development

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```

## Backend connection

The frontend talks to the validator API through `VITE_API_BASE_URL`.

- production: `web/.env.production`
- local development: `web/.env.development`

The shipped default points at `https://minirouter.work.gd`, which is the public
validator host. The GitHub Pages workflow picks up the production env file during
build, so changing that file and pushing to `main` updates the deployed frontend.

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
