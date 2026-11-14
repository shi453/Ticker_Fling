# 📈 Breakout Scanner Pro — v4

Everything in [v3](README_BreakoutScanner_v3.md) — scoring, retest engine, verdict, playbook, backtest, positions — **plus a 🗂 Tracker**: a year-long pipeline of ideas and trades (Watching → Bought → Closed) with dated notes, stored in a private GitHub repo so it survives Streamlit Cloud restarts and is editable from your phone.

Runs from `Breakout_checker_v4.py`. **v1, v2 and v3 are frozen and untouched** — all four can be deployed side by side from the same repo.

> ⚠️ **Not investment advice.** Decision-support and educational tool. Data is delayed (~15 min) and unofficial. Verify levels in your broker; the author is not a SEBI-registered advisor.

---

## What's new in v4

### 🗂 Tracker — the whole pipeline in one store

Enter a stock, the date you started watching it, the price then, and a target. Everything else is **derived live** and never stored, so it can't go stale:

| You type | The app computes every time you open it |
|---|---|
| ticker, start date, tracking price | current price, % since tracking, days tracked |
| category: 🚀 breakout / 📈 swing / 🏛 long term | % to target, best and worst price since start |
| status: 👀 Watching / 💰 Bought / 🏁 Closed / 🚫 Dropped | **target-hit date** |
| target, stop | P&L (open uses the live price, closed uses your sell price) |
| buy date, quantity, buy price; sell date, sell price | corporate-action warning |
| dated notes | hit-rate summary across everything you track |

**The target-hit date is derived, not typed.** The app replays immutable history from your start date and finds the first bar that traded at or through your target — the same stateless as-of technique the retest engine uses. It is always **strictly after** the start date (a same-day spike doesn't count), blank if never reached, and a manual override is still available if you want one.

Notes are a **dated log**, not one overwritten box. After a year, `12 Aug: entered per playbook` / `18 Sep: thesis weakening` is the record that teaches you something.

**Delete** is permanent and behind a confirm. To stop following something while keeping its history, use **Drop idea** or **Close position** instead.

### 📔 Positions is now a view over the same store

Positions are simply the records whose status is **Bought** — there is no second list and no hardcoded `PORTFOLIO`. Mark an idea bought in the Tracker and it appears on the Positions page with its stop, target and notes already attached. Closing it there writes straight back to the same record, which becomes your trade journal.

The management rules are now **category-aware**: the dead-money timeout and the book-partial-at-+1R rule are *swing* rules, so a 🏛 long-term holding is no longer told to bail after 20 flat bars. The stop is still absolute for every category.

Two new one-tap actions on each position card: **🏁 Close at today's price** and **⬆️ Move stop to <suggested>** (which also writes a dated note). Trailing the stop never changes your R statistics — R stays measured against the **initial** stop.

### A one-time import from v3

v3's hardcoded list is kept in the file as `V3_POSITIONS`, purely as an import source. The Positions page offers to import any entry not already in your store — **import only what you still hold**, since those fills date from v3. Afterwards, use v4 only; tracking the same trades in both versions would recreate the drift the merge removes.

---

## Storage — how a year of data survives Streamlit Cloud

Streamlit Cloud has **no persistent disk**: anything the app writes is wiped on restart or redeploy. v4 therefore persists one JSON document through the **GitHub Contents API**. Every save is a commit, so you also get history and undo for free.

Three backends, picked automatically, with the live one always shown at the top of the page and in the sidebar:

| Badge | When | Notes |
|---|---|---|
| ☁️ **Saved to GitHub** | a token is in `st.secrets` | what your phone uses |
| 💾 **Saved to tracker.json** | desktop, no token | beside the script; git-ignored |
| 🔸 **Session only** | nothing configured | warns loudly; use Backup to download |

**Concurrent edits are guarded.** GitHub refuses a write whose version is stale, so if another device or tab saved first you get a clear *CONFLICT — reload, then redo this change* instead of a silent overwrite. Saves happen on form submit, never per keystroke.

**If GitHub is unreachable or the token expires**, the page keeps the data it already loaded, goes read-only, and explains why. Your data is safe in the repo.

**💾 Backup & restore** (on every Tracker/Positions screen) downloads the whole document as JSON and restores from one. That works on every backend and needs no setup — do it monthly.

### Setting it up for mobile (about 10 minutes, one time)

1. **Create a private data repo**, e.g. `you/trading-data`.
   ⚠️ Use a repo **other than the one this app deploys from** — writing to the deployed repo would restart the app on every save.
2. **Create a fine-grained personal access token**: GitHub → Settings → Developer settings → Fine-grained tokens. Scope it to **only that repo**, permission **Contents: read & write**. Note the expiry date in your calendar — fine-grained tokens expire.
3. **Deploy** on [share.streamlit.io](https://share.streamlit.io), main file `Breakout_checker_v4.py`, app set to private (your email only).
4. **Paste into the app's Secrets** (⋮ → Settings → Secrets):
   ```toml
   [github]
   token  = "github_pat_..."
   repo   = "you/trading-data"
   path   = "tracker.json"
   branch = "main"
   ```
5. Open the URL on your phone → Chrome ⋮ → **Add to Home screen**. The sidebar panel should show ☁️ *Saved to GitHub*.

The file is created on first save; you don't need to add it to the repo yourself.

---

## Year-long accuracy notes

- **Prices for the Tracker are unadjusted on purpose.** Your tracking and target prices are what you saw on the day, so target-hit detection compares like with like. (The *analysis* engine still uses adjusted prices, which is correct for its own job.)
- **Corporate actions are flagged.** If a split or bonus happens inside a tracking window, the record shows a ⚠️ warning, because the stored prices and today's quotes are no longer comparable — edit them to the adjusted values when that happens.
- **One batched download** covers every tracked ticker (cached 15 minutes), so a growing list doesn't hammer Yahoo. A ticker with no data shows `—` and never blocks the page.
- The full analysis (verdict, retest, backtest) still runs **on demand** per stock, from the 🔍 *Analyse this stock* button — not for the whole list.

## Limitations

- **One buy and one sell per record.** Booking a third at +1R is best recorded as a dated note plus a quantity edit; proper multi-lot support is a later step.
- **One store means one dependency** — if the token expires, both Tracker and Positions go read-only until you fix it. That's the trade-off for having no database, and the Backup button is the escape hatch.
- Everything else inherits v3's caveats: delayed data, small backtest samples, no earnings calendar for many smallcaps.

---

*Files: `Breakout_checker_v4.py` + this README. v1–v3 and their READMEs are preserved unchanged; `requirements.txt` (now including `requests`) and `watchlist.json` are shared.*
