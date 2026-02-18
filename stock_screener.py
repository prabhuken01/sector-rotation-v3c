"""
stock_screener.py — Stock Rotation v3, Stage 2
Fast first-pass scoring for all stocks in selected sectors.
Score 0–7 based on simple daily-data checks.
"""

import pandas as pd
import numpy as np
from indicators import calculate_rsi, calculate_cmf, sma


def compute_screener_score(daily_data: pd.DataFrame) -> int | None:
    """
    Fast screener score (0–7) for a single stock using daily data.
    
    +1 pt each for:
      1. RSI(14) weekly rising (vs 1 bar ago on weekly resample)
      2. RSI(14) daily rising
      3. Close > SMA(8)
      4. Close > SMA(20)
      5. Close > SMA(50)
      6. Volume confirm (recent 5-day vol > 20-day avg × 1.2)
      7. CMF(20) > 0
    """
    if daily_data is None or len(daily_data) < 50:
        return None

    try:
        score = 0
        close = daily_data['Close']
        price = float(close.iloc[-1])

        # 1. Weekly RSI rising
        try:
            weekly = daily_data.resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            if len(weekly) >= 16:
                rsi_w = calculate_rsi(weekly)
                if len(rsi_w.dropna()) >= 2 and float(rsi_w.iloc[-1]) > float(rsi_w.iloc[-2]) + 0.5:
                    score += 1
        except Exception:
            pass

        # 2. Daily RSI rising
        rsi_d = calculate_rsi(daily_data)
        if len(rsi_d.dropna()) >= 2 and float(rsi_d.iloc[-1]) > float(rsi_d.iloc[-2]) + 0.5:
            score += 1

        # 3–5. Price vs SMAs
        sma8 = sma(close, 8)
        sma20 = sma(close, 20)
        sma50 = sma(close, 50)
        if not pd.isna(sma8.iloc[-1]) and price > float(sma8.iloc[-1]):
            score += 1
        if not pd.isna(sma20.iloc[-1]) and price > float(sma20.iloc[-1]):
            score += 1
        if not pd.isna(sma50.iloc[-1]) and price > float(sma50.iloc[-1]):
            score += 1

        # 6. Volume confirm
        if 'Volume' in daily_data.columns and len(daily_data) >= 20:
            vol5 = float(daily_data['Volume'].tail(5).mean())
            vol20 = float(daily_data['Volume'].tail(20).mean())
            if vol20 > 0 and vol5 > vol20 * 1.2:
                score += 1

        # 7. CMF > 0
        cmf = calculate_cmf(daily_data)
        if not cmf.isna().all() and float(cmf.iloc[-1]) > 0:
            score += 1

        return score

    except Exception:
        return None


def compute_screener_score_at_date(daily_data: pd.DataFrame, date: pd.Timestamp) -> int | None:
    """Same as compute_screener_score but sliced to a specific date (point-in-time)."""
    if daily_data is None or len(daily_data) < 50:
        return None
    try:
        idx = daily_data.index.get_indexer([date], method='ffill')[0]
        if idx < 49:
            return None
        subset = daily_data.iloc[:idx + 1]
        return compute_screener_score(subset)
    except Exception:
        return None


def screen_stocks(
    company_data: dict[str, pd.DataFrame],
    symbol_sector: dict[str, str],
    bullish_sectors: list[str],
    bearish_sectors: list[str],
    top_n: int = 20,
    bot_n: int = 20,
) -> tuple[list[dict], list[dict]]:
    """
    Screen all stocks and return top-N bullish and bottom-N bearish candidates.
    
    Returns:
        (bullish_list, bearish_list) — each is a list of dicts:
        {'symbol': str, 'sector': str, 'score': int}
    """
    all_scores = []
    for sym, data in company_data.items():
        sector = symbol_sector.get(sym)
        if sector is None:
            continue
        score = compute_screener_score(data)
        if score is not None:
            all_scores.append({'symbol': sym, 'sector': sector, 'score': score})

    # Bullish: highest scores from bullish sectors
    bull_pool = [r for r in all_scores if r['sector'] in bullish_sectors]
    bull_pool.sort(key=lambda x: x['score'], reverse=True)
    bullish = bull_pool[:top_n]

    # Bearish: lowest scores from bearish sectors
    bear_pool = [r for r in all_scores if r['sector'] in bearish_sectors]
    bear_pool.sort(key=lambda x: x['score'])
    bearish = bear_pool[:bot_n]

    return bullish, bearish
