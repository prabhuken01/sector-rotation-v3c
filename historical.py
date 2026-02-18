"""
historical.py — Stock Rotation v3
Historical rankings: per-date backtesting over last 15 trading days.

KEY FIX from v2: Uses daily data ONLY for confluence (no hourly fetches per date).
This makes it 10–20× faster and avoids timeout/empty data issues.
"""

import pandas as pd
import numpy as np
from sector_momentum import rank_sectors_at_date, get_top_bottom_sectors
from stock_screener import compute_screener_score_at_date
from confluence import analyze_stock, score_bullish, score_bearish, grade_label
from config import HISTORICAL_LOOKBACK_DAYS


def compute_historical_rankings(
    sector_data: dict[str, pd.DataFrame],
    company_data: dict[str, pd.DataFrame],
    symbol_sector: dict[str, str],
    symbol_name: dict[str, str],
    benchmark_data: pd.DataFrame,
    lookback: int = HISTORICAL_LOOKBACK_DAYS,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Build a historical rankings table for the last `lookback` trading days.
    
    For each date:
      1. Rank sectors (point-in-time)
      2. Screen stocks (daily data, point-in-time)
      3. Confluence scoring (daily entry TF, weekly conf TF — NO hourly)
      4. Record top 2 bullish, top 2 bearish + forward returns
    
    Returns DataFrame with one row per date.
    """
    if benchmark_data is None or len(benchmark_data) < lookback + 10:
        return pd.DataFrame()

    dates = benchmark_data.index[-lookback:].tolist()
    rows = []

    for i, date_t in enumerate(dates):
        if progress_callback:
            progress_callback(i, len(dates), date_t)

        date_str = date_t.strftime('%Y-%m-%d')
        row = {'Date': date_str}

        # ── Stage 1: Sector momentum for this date ──
        df_sectors = rank_sectors_at_date(sector_data, date_t)
        if df_sectors.empty:
            rows.append(row)
            continue

        top_sectors, bot_sectors = get_top_bottom_sectors(df_sectors)
        top2_sectors = df_sectors['Sector'].tolist()[:2]
        row['Mom #1 Sector'] = top2_sectors[0] if len(top2_sectors) >= 1 else ''
        row['Mom #2 Sector'] = top2_sectors[1] if len(top2_sectors) >= 2 else ''
        row['Top 4 Sectors'] = ', '.join(top_sectors)
        row['Bot 6 Sectors'] = ', '.join(bot_sectors)

        # ── Stage 2: Screen stocks at this date ──
        screener_scores = []
        for sym, data in company_data.items():
            sector = symbol_sector.get(sym)
            if sector is None:
                continue
            sc = compute_screener_score_at_date(data, date_t)
            if sc is not None:
                screener_scores.append({
                    'symbol': sym, 'sector': sector,
                    'name': symbol_name.get(sym, sym), 'score': sc,
                })

        # Bullish candidates from top sectors, bearish from bottom sectors
        bull_pool = [r for r in screener_scores if r['sector'] in top_sectors]
        bull_pool.sort(key=lambda x: x['score'], reverse=True)
        bear_pool = [r for r in screener_scores if r['sector'] in bot_sectors]
        bear_pool.sort(key=lambda x: x['score'])

        # Take top 15 for confluence
        bull_shortlist = bull_pool[:15]
        bear_shortlist = bear_pool[:15]

        # ── Stage 3: Confluence scoring (DAILY only — fast) ──
        bull_results = []
        for rec in bull_shortlist:
            sym = rec['symbol']
            data = company_data.get(sym)
            if data is None:
                continue
            try:
                idx = data.index.get_indexer([date_t], method='ffill')[0]
                if idx < 59:
                    continue
                subset = data.iloc[:idx + 1]
                analysis = analyze_stock(subset, entry_tf='daily')
                if analysis is None:
                    continue
                bscore, reasons = score_bullish(analysis)
                bull_results.append({
                    'symbol': sym, 'sector': rec['sector'],
                    'name': rec['name'], 'conf_score': bscore,
                    'price': analysis['current_price'],
                    'grade': grade_label(bscore),
                    'data': data, 'idx': idx,
                })
            except Exception:
                continue

        bear_results = []
        for rec in bear_shortlist:
            sym = rec['symbol']
            data = company_data.get(sym)
            if data is None:
                continue
            try:
                idx = data.index.get_indexer([date_t], method='ffill')[0]
                if idx < 59:
                    continue
                subset = data.iloc[:idx + 1]
                analysis = analyze_stock(subset, entry_tf='daily')
                if analysis is None:
                    continue
                bscore, reasons = score_bearish(analysis)
                bear_results.append({
                    'symbol': sym, 'sector': rec['sector'],
                    'name': rec['name'], 'conf_score': bscore,
                    'price': analysis['current_price'],
                    'grade': grade_label(bscore),
                    'data': data, 'idx': idx,
                })
            except Exception:
                continue

        # Sort and pick top 2
        bull_results.sort(key=lambda x: x['conf_score'], reverse=True)
        bear_results.sort(key=lambda x: x['conf_score'], reverse=True)

        # Forward returns helper
        def _fwd(rec, days_fwd):
            d, idx = rec['data'], rec['idx']
            if idx + days_fwd < len(d):
                c0 = float(d['Close'].iloc[idx])
                c1 = float(d['Close'].iloc[idx + days_fwd])
                return round((c1 / c0 - 1) * 100, 1) if c0 > 0 else None
            return None

        for rank_i in range(2):
            prefix = f'Bull #{rank_i + 1}'
            if rank_i < len(bull_results):
                r = bull_results[rank_i]
                row[f'{prefix} Stock'] = r['name']
                row[f'{prefix} Sector'] = r['sector']
                row[f'{prefix} CMP'] = r['price']
                row[f'{prefix} Score'] = r['conf_score']
                row[f'{prefix} Grade'] = r['grade']
                row[f'{prefix} 1D %'] = _fwd(r, 1)
                row[f'{prefix} 2D %'] = _fwd(r, 2)
                row[f'{prefix} 3D %'] = _fwd(r, 3)
                row[f'{prefix} 1W %'] = _fwd(r, 5)
            else:
                row[f'{prefix} Stock'] = ''
                row[f'{prefix} Score'] = None

        for rank_i in range(2):
            prefix = f'Bear #{rank_i + 1}'
            if rank_i < len(bear_results):
                r = bear_results[rank_i]
                row[f'{prefix} Stock'] = r['name']
                row[f'{prefix} Sector'] = r['sector']
                row[f'{prefix} CMP'] = r['price']
                row[f'{prefix} Score'] = r['conf_score']
                row[f'{prefix} Grade'] = r['grade']
                row[f'{prefix} 1D %'] = _fwd(r, 1)
                row[f'{prefix} 2D %'] = _fwd(r, 2)
                row[f'{prefix} 3D %'] = _fwd(r, 3)
                row[f'{prefix} 1W %'] = _fwd(r, 5)
            else:
                row[f'{prefix} Stock'] = ''
                row[f'{prefix} Score'] = None

        rows.append(row)

    return pd.DataFrame(rows)
