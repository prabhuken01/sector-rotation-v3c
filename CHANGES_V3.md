# Stock Rotation v3 – Changes & Fixes

## Version: 3.0 (Feb 2026)

This version is saved at:
`E:\Personal\Trading_Champion\Projects\Sector-rotation-v2\Sector-rotation-v3-claude`

---

## Bugs Fixed

### 1. `company_symbols.py` – Fallback Sector Assignments (v2.3.1 spec)

**Problem:** The static fallback had `IBREALEST.NS` (IB Realtime, a real estate company)
incorrectly placed in the **Energy** sector.

**Fix:**
- Removed `IBREALEST.NS` from Energy
- Added `ADANIGREEN.NS` (Adani Green Energy) to Energy — a proper energy company
- `MANAPPURAM.NS` was already correctly under `Fin Services` ✅
- `BEL.NS` (Bharat Electronics) was already correctly under `Defence` ✅
- Energy now contains only energy/oil & gas names per spec:
  Reliance, NTPC, Power Grid, ONGC, GAIL, IOC, Petronet, Adani Green

### 2. `company_symbols.py` – SectorCompany.xlsx Path

**Problem:** The local fallback only looked for `Sector-Company.xlsx` (old name)
but the v3 folder ships with `SectorCompany.xlsx`.

**Fix:**
- Added `SectorCompany.xlsx` as the primary local path to try
- `Sector-Company.xlsx` kept as secondary fallback
- Both paths are checked in order: configured E-drive path → SectorCompany.xlsx → Sector-Company.xlsx

### 3. `streamlit_app.py` – `end_date` Type Bug in Confluence Loop

**Problem:** In `display_stock_screener_tab` → Part 3 Confluence, `analysis_date`
is a `datetime.date` object, but `fetch_sector_data()` expects a `datetime.datetime`.
This caused data fetches to fail silently on some date-specific queries, resulting
in confluence analysis returning no results.

**Fix:**
```python
from datetime import datetime as _dt
_analysis_end = _dt.combine(analysis_date, _dt.min.time()) \
    if hasattr(analysis_date, 'year') and not isinstance(analysis_date, _dt) \
    else analysis_date
data_1d = fetch_sector_data(symbol, end_date=_analysis_end, interval='1d')
data_entry_raw = fetch_sector_data(symbol, end_date=_analysis_end, interval='1h')
```

### 4. `config.py` – Excel Path Updated for v3

**Change:** Updated `SECTOR_COMPANY_EXCEL_PATH` to point to the v3 folder:
```
E:\Personal\Trading_Champion\Projects\Sector-rotation-v2\Sector-rotation-v3-claude\SectorCompany.xlsx
```

---

## Core Logic Preserved (No Changes)

The following core logic is **unchanged** from v2.3.5 — it was already correct:

| Component | Status |
|---|---|
| Sector rotation: Top 4 (bullish) + Bottom 6 (bearish) | ✅ Correct |
| Per-date top4/bottom6 in Historical Rankings | ✅ Correct (lines 2438–2476) |
| Bullish confluence hard gates (RSI up both TFs, MA Bullish, Uptrend HH/HL, Near HL) | ✅ Correct |
| Bearish confluence scoring (RSI falling = +pts, Near LH = ideal short) | ✅ Correct |
| Top 8 Bullish / Top 8 Bearish display with gate filter (score > -5) | ✅ Correct |
| Confluence fallback when screener has no rows (use SECTOR_COMPANIES universe) | ✅ Correct |
| Trending mode: Z(RSI) + Z(CMF) 50/50 | ✅ Correct |
| Historical mode: rank-based ADX_Z + RS_Rating + RSI + DI_Spread | ✅ Correct |
| Stock Screener MA+RSI+VWAP scoring | ✅ Correct |
| Pivot structure detection (detect_swing_structure) | ✅ Correct |
| Price position Near HL / Near LH (3% threshold) | ✅ Correct |

---

## File Structure

```
stock-rotation-v3/
├── streamlit_app.py         ← Main app (v3.0 — end_date bug fixed)
├── confluence_fixed.py      ← Confluence scoring (unchanged, correct)
├── company_symbols.py       ← Fixed: Energy sector, SectorCompany.xlsx path
├── config.py                ← Updated: Excel path for v3 folder
├── data_fetcher.py          ← Unchanged
├── indicators.py            ← Unchanged
├── analysis.py              ← Unchanged
├── company_analysis.py      ← Unchanged
├── SectorCompany.xlsx       ← Primary sector-company data source
├── sector_company.csv       ← CSV backup
├── requirements.txt         ← Unchanged
└── CHANGES_V3.md            ← This file
```

---

## How to Run

```bash
cd "E:\Personal\Trading_Champion\Projects\Sector-rotation-v2\Sector-rotation-v3-claude"
streamlit run streamlit_app.py
```

---

## Backtest Flow (preserved logic)

1. **Sector rotation** → Momentum Ranking (Trending or Historical mode)
2. **Top 4 sectors** → bullish universe | **Bottom 6 sectors** → bearish universe
3. **Stock Screener (MA+RSI+VWAP)** → scores all stocks in universe
4. **Confluence gates** → hard filter: RSI up both TFs + MA Bullish + Uptrend HH/HL + Near HL
5. **Rank by confluence score** → Top 8 Bullish / Top 8 Bearish
6. **Historical Rankings (30-day)** → per-date top4/bottom6 + next-day returns
7. **Pick Top 1-2** each day from the ranked list

The 30-day historical table with **1D/2D/3D/1W % returns** is the key backtesting
output — use it to validate which confluence stocks actually moved in the expected direction.
