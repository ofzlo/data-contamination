# Style Before Memory · demo page

Static demo site for [ofzlo/data-contamination](https://github.com/ofzlo/data-contamination) (branch `main`).
This branch holds only the page; code, splits and the full pipeline live on `main`.

- `index.html` – the page (no build step; open via any static server, e.g. `python -m http.server`)
- `data/demo.json` – sample captions, dose summary, G1 gate excerpt, short-prompt contrasts (split A)
- `img/` – the 9 NoCaps photos shown (Open Images, CC BY 2.0, downscaled to 640px)
- `tools/build_demo_data.py` – regenerates `data/` and `img/` from a `main` checkout with `outputs/` and `data/`
- `tools/page.html` + `tools/wrap_page.py` – page source and the wrapper that writes `index.html`

```bash
python tools/build_demo_data.py --repo /path/to/data-contamination --out .
python tools/wrap_page.py
```

To publish: Settings → Pages → Deploy from branch `demo`, folder `/ (root)`.
