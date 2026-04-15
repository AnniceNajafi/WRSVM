# WRSVM Project Page

This folder is the source for the WRSVM GitHub Pages site.

## Files

- `index.html`: landing page (single-page site with tabs)
- `style.css`: styling
- `.nojekyll`: tells GitHub Pages to skip Jekyll processing and serve the files as-is

## Enabling GitHub Pages

1. Push the repo to GitHub.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**:
   - Source: **Deploy from a branch**
   - Branch: `main` (or `master`)
   - Folder: `/docs`
4. Click **Save**. The site will be live in 1 to 2 minutes at:

   ```
   https://<your-username>.github.io/<repo-name>/
   ```

   For this repo, that's `https://annicenajafi.github.io/wrsvm/` (assuming the repo is named `wrsvm`).

## Local preview

Open `index.html` directly in a browser, or serve the folder:

```bash
cd docs
python -m http.server 8000
# then visit http://localhost:8000
```

## Editing

The site is intentionally framework-free: vanilla HTML, CSS, and a few lines of JS for the tab switcher. Edit `index.html` to change content, `style.css` to restyle. No build step required.

## Custom domain (optional)

To use e.g. `wrsvm.org`:

1. Add a `CNAME` file in this folder containing the bare domain (one line, no protocol).
2. In your DNS, add a `CNAME` record pointing to `<username>.github.io`.
3. In **Settings → Pages**, set the custom domain and enable **Enforce HTTPS**.
