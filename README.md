# APEX — Play Store Intelligence Dashboard

Glassmorphism dark-mode analytics dashboard, built from your Python/Plotly ETL
notebook. Pure HTML/CSS/JS — no build step, no npm install required.

```
apex-dashboard/
├── index.html        # page structure
├── css/
│   └── style.css      # design tokens, glass cards, aurora background, layout
├── js/
│   ├── data.js         # RAW_ROWS (your dataset) + ETL cleaning (mirrors
│   │                    load_and_clean_dataset() from the notebook)
│   └── charts.js       # all 6 Plotly charts + KPI strip + nav scroll-spy
└── README.md
```

## Run it in VS Code

1. Open the `apex-dashboard` folder in VS Code (`File → Open Folder…`).
2. Install the **Live Server** extension (by Ritwick Dey) if you don't have it —
   search "Live Server" in the Extensions panel (`Ctrl/Cmd+Shift+X`).
3. Right-click `index.html` → **"Open with Live Server"**.
4. It opens at `http://127.0.0.1:5500` and hot-reloads on save.

(You *can* just double-click `index.html` to open it directly in a browser too —
everything is loaded from CDNs (Google Fonts, Plotly.js) so it'll render fine,
but Live Server gives you auto-refresh while you edit.)

## Where things live

- **Edit the data** → `js/data.js`, the `RAW_ROWS` array. Add rows in the same
  shape (`App, Category, Rating, Reviews, Size, Installs, Sentiment_Subjectivity`)
  and every chart recalculates automatically — nothing else to touch.
- **Swap in a real CSV/XLSX later** → replace `RAW_ROWS` with a `fetch()` call
  (e.g. using [PapaParse](https://www.papaparse.com/) for CSV) that populates
  the same array shape before `df_app` is built.
- **Edit chart logic** → `js/charts.js`. Each chart is labeled `CHART 1` through
  `CHART 6` and mirrors the corresponding block in your Python notebook
  (`fig1` → Quality Benchmark, `fig2` → Global Reach, etc.).
- **Edit colors/spacing/fonts** → `css/style.css`, the `:root` block at the top
  has every design token (purple/cyan/pink accents, glass opacity, radius, fonts).

## Known limits with only 11 rows

- Charts that take "top 5 / top 10 categories" will just show however many
  distinct categories exist (9 here) — nothing breaks, they just look sparse.
- Chart 06 (Monetization) has no Paid apps in this dataset, so it falls back to
  the same dummy estimate your notebook uses: `Installs_Paid = Installs_Free × 0.05`,
  `Revenue_Paid = Installs_Paid × 2.99`.
- Chart 03/04's "monthly trend" isn't from a real date column (your notebook
  doesn't have one either) — it's a fixed growth-factor projection off the
  current install totals, same as the original code.
