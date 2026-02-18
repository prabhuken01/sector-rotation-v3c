# Stock Rotation App — Fixes Log (v3)

## Summary of Issues Fixed

---

### 1. ✅ RSI & CMF Same for All Sectors (CRITICAL FIX)

**Root Cause:** NSE index symbols like `^CNXIT`, `^CNXAUTO` etc. return **zero or no Volume** data from Yahoo Finance. CMF (Chaikin Money Flow) requires OHLCV with non-zero volume. When volume = 0:
- `mf_volume / volume.sum()` = 0/0 = NaN
- The code defaulted NaN to `0.0` for all sectors
- Z-score of all identical values = 0 for every sector
- Momentum_Score = 5.0 for all → all look the same

**Fixes Applied:**

**`indicators.py`** — CMF now returns NaN (not 0) when volume is all zeros:
```python
if volume is None or (volume == 0).all() or volume.isna().all():
    return pd.Series(float('nan'), index=data.index)
# Also: vol_sum.replace(0, float('nan')) to avoid division by zero
```

**`analysis.py`** — CMF fallback now uses NaN instead of 0.0:
```python
latest_cmf = cmf.iloc[-1] if not cmf.isna().all() else float('nan')
```

**`analysis.py`** — Z-score computation now handles NaN CMF gracefully:
```python
cmf_valid = df['CMF'].dropna()
if len(cmf_valid) >= 2:
    # compute CMF Z-score only from valid sectors
    ...
    raw_score = 0.5 * rsi_z + 0.5 * cmf_z
else:
    # No valid CMF data — use RSI only for ranking
    raw_score = rsi_z
```

**`streamlit_app.py`** — Momentum tab now shows a warning when CMF data is unavailable:
- Warning shown if all CMF values have std < 0.001 (identical = no volume)
- Advises user to switch to ETF Proxy mode for proper CMF readings

**Recommendation:** Always use **ETF Proxy** mode (enabled by default) since ETFs like `AUTOBEES.NS`, `ITBEES.NS` etc. have real volume data, making CMF meaningful.

---

### 2. ✅ Market Breadth as First Tab

**Fix:** Reordered tabs so **Market Breadth** is now tab #1 (first tab visible):

Old order: Momentum Ranking → Market Breadth → Stock Screener → ...
New order: **Market Breadth → Sector Momentum → Stock Screener → ...**

This makes market breadth the landing view, providing immediate market context before drilling into sectors.

---

### 3. ✅ Analysis Date Display in Stock Screener

**Fix:** Added a prominent green date banner inside the Stock Screener tab:
```
📅 Screener Analysis Date: 18 Feb 2026 (Wednesday)
```

Also added the selected date to table headers:
- `📈 Top 10 Bullish — 18 Feb 2026`
- `🔴 Top 10 Bearish — 18 Feb 2026`

---

### 4. ✅ CMP (Current Market Price) Fix in Screener

**Root Cause:** `daily["Close"].iloc[-1]` always returned the last row of data regardless of the selected analysis date. If the data extended beyond the selected date, CMP would be wrong.

**Fix:** Data is now filtered to only include rows up to and including the selected date:
```python
sel_date_ts = pd.Timestamp(selected_date)
date_mask = daily.index <= sel_date_ts + pd.Timedelta(days=1)
daily_to_date = daily[date_mask]
price = float(daily_to_date["Close"].iloc[-1])
```

---

### 5. ✅ Navigation Issue

The "navigation confusion" where clicking one tab went to another was caused by:
- The old tabs were labeled differently from what was shown (e.g., "Confluence Analysis" was actually "Stock Screener")
- Tabs are now clearly labeled: **Market Breadth | Sector Momentum | Stock Screener | Reversal Candidates | ...**
- Analysis is cached (5-minute TTL) so tab switches do NOT re-fetch data

---

### 6. ✅ Color Coding Improvements

- **RSI**: Green (>65), Yellow (50-65), Red (<35) — already existed, preserved
- **CMF**: Green (positive), Red (negative) — already existed, preserved
- **Market Breadth**: Red (<25%), Yellow (25-50%), Green (>50%) — already existed, preserved
- **Bullish/Bearish banners** in Momentum tab use green/yellow alert boxes
- **NEW**: Warning banner in Momentum tab when CMF is unavailable

---

### 7. 📌 Config Fix

- `SECTOR_COMPANY_EXCEL_PATH` set to `None` (relative path) instead of hardcoded E-drive path
- This is critical for cloud deployment / any machine other than the developer's PC

---

## ETF Symbols Reference (for CMF to work)

| Sector | Primary ETF | Notes |
|--------|-------------|-------|
| Auto | AUTOBEES.NS | Has volume |
| IT | ITBEES.NS | Has volume |
| FMCG | FMCGIETF.NS | Has volume |
| Energy | MOENERGY.NS | Has volume |
| Pharma | PHARMABEES.NS | Has volume |
| PSU Bank | PSUBNKBEES.NS | Has volume |
| Pvt Bank | PVTBANKBEES.NS | Has volume |
| Metal | METALIETF.NS | Has volume |
| Infra | INFRABEES.NS | Has volume |
| Fin Services | FINIETF.NS | Has volume |
| Commodities | N/A | No ETF — RSI only |
| Media | N/A | No ETF — RSI only |

**Sectors without ETF proxies (Commodities, Media) will have CMF = NaN and use RSI-only Z-score.**

---

## Files Modified

| File | Change |
|------|--------|
| `streamlit_app.py` | Tab reorder, date display, CMP fix, CMF warning |
| `analysis.py` | CMF NaN handling, Z-score fix |
| `indicators.py` | CMF zero-volume protection |
| `config.py` | Excel path set to None (relative) |
