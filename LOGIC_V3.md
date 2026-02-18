# Stock Rotation v3 — Core Logic

## Goal
Every trading day, identify the **top 1–2 bullish** and **top 1–2 bearish** stocks from ~135 NSE stocks, using a 3-stage funnel:

```
Stage 1: SECTOR ROTATION  →  Which sectors are strong/weak?
Stage 2: STOCK SCREENING   →  Within selected sectors, score every stock
Stage 3: CONFLUENCE RANKING →  Multi-factor ranking to pick final top 1–2
```

---

## Stage 1 — Sector Momentum Ranking

**Input:** Daily OHLCV for 15 sector indices + Nifty 50 benchmark.

**Method:** For each sector, compute:
- RSI(14) on daily close
- CMF(20) on daily OHLCV
- Cross-sectional Z-scores: `Z(RSI)` and `Z(CMF)`
- Momentum Score = `0.5 × Z(RSI) + 0.5 × Z(CMF)`, scaled to 1–10

**Output (sorted by Momentum Score, descending):**
- **Top 4 sectors** → Bullish universe (strong momentum, buy candidates)
- **Bottom 6 sectors** → Bearish universe (weak momentum, short candidates)

> This runs **per date** for historical rankings, so each date has its own Top 4/Bottom 6.

---

## Stage 2 — Stock Screening (within selected sectors)

**Input:** Daily OHLCV for all stocks in Top 4 sectors (bullish) and Bottom 6 sectors (bearish).

**Screener Score** (0–7 pts, simple and fast):
| Factor | +1 pt if... |
|--------|-------------|
| RSI 1W direction | RSI(14) on weekly is rising (vs 1 bar ago) |
| RSI 1D direction | RSI(14) on daily is rising |
| Price > 8 SMA | Close > SMA(8) on daily |
| Price > 20 SMA | Close > SMA(20) on daily |
| Price > 50 SMA | Close > SMA(50) on daily |
| Volume confirm | Recent 5-day avg volume > 20-day avg volume × 1.2 |
| CMF positive | CMF(20) > 0 |

- **Bullish candidates**: Top-scoring stocks from Top 4 sectors (sorted descending)
- **Bearish candidates**: Bottom-scoring stocks from Bottom 6 sectors (sorted ascending)

> This is a fast first pass. Only the top ~20 bullish and bottom ~20 bearish proceed to Stage 3.

---

## Stage 3 — Confluence Scoring (final ranking)

### 3A. Analysis per stock

For each shortlisted stock, compute on the **entry timeframe** (4H default, resampled from 1H) and **confirmation timeframe** (1H):

1. **Swing structure** (detect_swing_structure on entry TF):
   - Classify as Uptrend (HH/HL), Downtrend (LL/LH), or Sideways
   - Record last pivot low (HL candidate) and last pivot high (LH candidate)

2. **Price position** relative to pivots:
   - "Near HL" = within 3% of last pivot low → ideal bullish entry
   - "Near LH" = within 3% of last pivot high → ideal bearish entry
   - "Middle" = neither

3. **MA alignment** (Price vs 20 SMA vs 50 SMA):
   - Bullish: Price > SMA20 > SMA50
   - Bearish: Price < SMA20 < SMA50
   - Mixed: otherwise

4. **RSI direction** (rising = RSI > RSI_prev + 0.5; falling = RSI < RSI_prev − 0.5)

5. **MA crossover** (20 SMA vs 50 SMA within 1.5% → crossover forming)

6. **RSI divergence** (price vs RSI on last 10 bars)

7. **Volume status** (recent 5-bar avg vs 20-bar avg)

### 3B. Bullish Confluence Scoring (graduated, NO hard gate rejection)

**Key change from v2:** Instead of all-or-nothing gates that reject 95% of stocks, 
v3 uses a **graduated penalty system**. Opposing conditions lose points but don't 
automatically get −5. This ensures we always get rankings even in mixed markets.

| # | Factor | Bullish +Pts | Penalty −Pts | Weight |
|---|--------|-------------|-------------|--------|
| 1 | Trend (entry TF) | HH/HL: **+4** | LL/LH: **−3**, Sideways: **+0.5** | Core |
| 2 | Trend (conf TF) | HH/HL: **+3** | LL/LH: **−2**, Sideways: **+0** | Core |
| 3 | MA Align (entry) | Bullish: **+3** | Bearish: **−2**, Mixed: **+0** | Core |
| 4 | MA Align (conf) | Bullish: **+2** | Bearish: **−1**, Mixed: **+0** | Core |
| 5 | Price Position | Near HL: **+3** | Near LH: **−1**, Middle: **+1** | Core |
| 6 | RSI (entry) | Rising 40–70: **+2**, Rising other: **+1** | Falling: **−1**, OB>70: **−1** | Signal |
| 7 | RSI (conf) | Rising 40–70: **+1.5**, Rising: **+0.5** | Falling: **−0.5**, OB>70: **−0.5** | Signal |
| 8 | MA Crossover | Bullish X: **+1.5** | Bearish X: **−1** | Supporting |
| 9 | RSI Divergence | Bullish div: **+1.5** | Bearish div: **−1** | Supporting |
| 10 | Volume | High: **+1** | — | Supporting |

**Max possible: ~22 pts.  Score ranges:**
- ≥ 12: Excellent — high-probability setup
- 9–12: Good/strong
- 5–9: Moderate — some confirmation needed
- < 5: Weak/avoid
- Negative: Opposing setup (bearish stock in bullish ranking = low rank, not rejected)

### 3C. Bearish Confluence Scoring (mirror of bullish)

| # | Factor | Bearish +Pts | Penalty −Pts |
|---|--------|-------------|-------------|
| 1 | Trend (entry) | LL/LH: **+4** | HH/HL: **−3**, Sideways: **+0.5** |
| 2 | Trend (conf) | LL/LH: **+3** | HH/HL: **−2**, Sideways: **+0** |
| 3 | MA Align (entry) | Bearish: **+3** | Bullish: **−2** |
| 4 | MA Align (conf) | Bearish: **+2** | Bullish: **−1** |
| 5 | Price Position | Near LH: **+3** | Near HL: **−2**, Middle: **+1** |
| 6 | RSI (entry) | Falling 30–60: **+2** | Rising: **−1**, OS<30: **−1** |
| 7 | RSI (conf) | Falling 30–60: **+1.5** | Rising: **−0.5** |
| 8 | MA Crossover | Bearish X: **+1.5** | Bullish X: **−1** |
| 9 | RSI Divergence | Bearish div: **+1.5** | Bullish div: **−1** |
| 10 | Volume | High at LH: **+1.5**, High: **+0.5** | — |

### 3D. Final Selection

- **Top 2 Bullish**: Highest bullish confluence scores from Top 4 sector stocks
- **Top 2 Bearish**: Highest bearish confluence scores from Bottom 6 sector stocks
- Also show Top 8 for review

---

## Historical Rankings (per-date backtesting)

For each of the last 15 trading days:

1. **Sector ranking** for that date (using data up to that date only)
2. **Stock screening** for that date (daily data only — NO hourly fetches to avoid slowness)
3. **Confluence scoring** for that date (using daily TF as entry, weekly as confirmation — avoids hourly data dependency)
4. **Forward returns**: Next 1D, 2D, 3D, 1W % for each selected stock

> **Critical fix:** v2 tried fetching hourly data for 135 stocks × 30 dates in a loop = timeout.  
> v3 uses **daily-only** confluence for historical backtesting (faster, still meaningful).  
> Live/current-day analysis can optionally use 4H+1H for finer entry timing.

---

## File Structure (v3)

```
stock-rotation-v3/
├── streamlit_app.py      # Main UI (~800 lines max, tabs + display only)
├── config.py             # Constants, sector symbols, weights
├── data_fetcher.py       # Yahoo Finance data fetching + caching
├── indicators.py         # RSI, ADX, CMF, Z-score, Mansfield RS
├── sector_momentum.py    # Stage 1: Sector ranking
├── stock_screener.py     # Stage 2: Stock screening score
├── confluence.py         # Stage 3: Confluence analysis + scoring
├── historical.py         # Historical rankings (per-date backtesting)
├── company_symbols.py    # Sector-company mapping from Excel/CSV
├── requirements.txt
└── sector_company.csv    # Sector-company data
```
