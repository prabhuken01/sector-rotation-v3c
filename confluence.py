"""
confluence.py — Stock Rotation v3, Stage 3
Multi-factor confluence scoring for final stock ranking.

KEY CHANGE from v2:
  v2 used hard gates (all 4 conditions or score = -5) → 95% of stocks rejected.
  v3 uses GRADUATED penalties — opposing conditions lose points but stocks are
  always ranked. This ensures we always get top 1–2 picks even in mixed markets.
"""

import pandas as pd
import numpy as np
from indicators import calculate_rsi, sma


# ─── Pivot detection (swing high/low) ───

def _pivot_highs_lows(data: pd.DataFrame, left: int = 3, right: int = 3):
    """Detect pivot highs and lows (Pine-Script style)."""
    highs = data['High'].values
    lows = data['Low'].values
    n = len(data)
    ph, pl = [], []
    for i in range(left, n - right):
        h_win = highs[i - left: i + right + 1]
        if highs[i] == h_win.max() and (h_win == highs[i]).sum() == 1:
            ph.append((i, float(highs[i])))
        l_win = lows[i - left: i + right + 1]
        if lows[i] == l_win.min() and (l_win == lows[i]).sum() == 1:
            pl.append((i, float(lows[i])))
    return ph, pl


def detect_swing_structure(data: pd.DataFrame, left: int = 3, right: int = 3, min_pivots: int = 3) -> dict:
    """
    Classify swing structure:
      'Uptrend (HH/HL)'   — majority HH + HL
      'Downtrend (LL/LH)' — majority LL + LH
      'Sideways'           — neither
    """
    empty = {'trend': 'Sideways', 'last_hl_price': None, 'last_lh_price': None}
    needed = left + right + min_pivots * 2
    if data is None or len(data) < needed:
        return empty

    ph, pl = _pivot_highs_lows(data, left, right)
    if len(ph) < 2 or len(pl) < 2:
        return {**empty, 'last_hl_price': pl[-1][1] if pl else None,
                'last_lh_price': ph[-1][1] if ph else None}

    hh = sum(1 for i in range(1, len(ph)) if ph[i][1] > ph[i-1][1])
    lh = sum(1 for i in range(1, len(ph)) if ph[i][1] < ph[i-1][1])
    hl = sum(1 for i in range(1, len(pl)) if pl[i][1] > pl[i-1][1])
    ll = sum(1 for i in range(1, len(pl)) if pl[i][1] < pl[i-1][1])

    ph_pairs = len(ph) - 1
    pl_pairs = len(pl) - 1

    is_up = (ph_pairs > 0 and pl_pairs > 0 and
             hh / ph_pairs >= 0.55 and hl / pl_pairs >= 0.55)
    is_down = (ph_pairs > 0 and pl_pairs > 0 and
               lh / ph_pairs >= 0.55 and ll / pl_pairs >= 0.55)

    trend = ('Uptrend (HH/HL)' if is_up
             else 'Downtrend (LL/LH)' if is_down
             else 'Sideways')

    return {
        'trend': trend,
        'last_hl_price': pl[-1][1] if pl else None,
        'last_lh_price': ph[-1][1] if ph else None,
    }


# ─── Helper functions ───

def _ma_alignment(price: float, dma20: float, dma50: float) -> str:
    if pd.isna(dma20) or pd.isna(dma50):
        return 'N/A'
    if price > dma20 > dma50:
        return 'Bullish'
    if price < dma20 < dma50:
        return 'Bearish'
    return 'Mixed'


def _ma_crossover(dma20: float, dma50: float) -> str:
    if pd.isna(dma20) or pd.isna(dma50):
        return 'None'
    diff_pct = abs((dma20 - dma50) / max(dma50, 1e-6) * 100)
    if diff_pct < 1.5:
        return 'Bullish Crossover' if dma20 > dma50 else 'Bearish Crossover'
    return 'None'


def _price_position(price: float, swing: dict, threshold_pct: float = 3.0) -> str:
    hl = swing.get('last_hl_price')
    lh = swing.get('last_lh_price')
    near_hl = hl is not None and abs(price - hl) / max(hl, 1e-6) * 100 <= threshold_pct
    near_lh = lh is not None and abs(price - lh) / max(lh, 1e-6) * 100 <= threshold_pct
    if near_hl and near_lh:
        return 'Near HL' if abs(price - hl) <= abs(price - lh) else 'Near LH'
    if near_hl:
        return 'Near HL'
    if near_lh:
        return 'Near LH'
    return 'Middle'


def _detect_divergence(data: pd.DataFrame, rsi_series: pd.Series) -> str:
    if len(data) < 10:
        return 'None'
    try:
        rl = data['Low'].tail(10).values
        rh = data['High'].tail(10).values
        rrs = rsi_series.tail(10).values
        if len(rrs) < 5 or np.isnan(rrs).all():
            return 'None'
        if rl[-1] < rl[-3] and rrs[-1] > rrs[-3]:
            return 'Bullish'
        if rh[-1] > rh[-3] and rrs[-1] < rrs[-3]:
            return 'Bearish'
    except Exception:
        pass
    return 'None'


def _volume_status(data: pd.DataFrame, lookback: int = 5) -> str:
    if 'Volume' not in data.columns or len(data) < lookback * 4:
        return 'N/A'
    try:
        rv = data['Volume'].tail(lookback).mean()
        avg = data['Volume'].tail(lookback * 4).mean()
        return 'High' if avg > 0 and rv > avg * 1.2 else 'Normal'
    except Exception:
        return 'N/A'


# ─── Core analysis function ───

def analyze_stock(
    data_entry: pd.DataFrame,
    data_conf: pd.DataFrame | None = None,
    entry_tf: str = 'daily',
) -> dict | None:
    """
    Build per-stock analysis dict. Works with:
    - entry_tf='daily': entry = daily data, conf = weekly (resampled from daily)
    - entry_tf='4h': entry = 4H (resampled from 1H), conf = 1H
    - entry_tf='2h': entry = 2H (resampled from 1H), conf = 1H
    
    For historical backtesting, we use daily/weekly (no hourly dependency).
    For live analysis, 4H+1H or 2H+1H gives finer entry timing.
    """
    try:
        # Prepare entry data
        if entry_tf in ('4h', '2h'):
            freq = '4h' if entry_tf == '4h' else '2h'
            entry = data_entry.resample(freq).agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            conf = data_conf if data_conf is not None else data_entry
        elif entry_tf == 'daily':
            entry = data_entry
            # Conf = weekly resampled from daily
            conf = data_entry.resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
        else:
            entry = data_entry
            conf = data_conf if data_conf is not None else data_entry

        if entry is None or len(entry) < 50:
            return None
        if conf is None or len(conf) < 15:
            return None

        # Entry TF calculations
        entry = entry.copy()
        entry['DMA_20'] = sma(entry['Close'], 20)
        entry['DMA_50'] = sma(entry['Close'], 50)
        rsi_e = calculate_rsi(entry)

        price = float(entry['Close'].iloc[-1])
        dma20_e = float(entry['DMA_20'].iloc[-1])
        dma50_e = float(entry['DMA_50'].iloc[-1])
        rsi_e_val = float(rsi_e.iloc[-1]) if not rsi_e.isna().all() else 50.0
        rsi_e_prev = float(rsi_e.iloc[-2]) if len(rsi_e.dropna()) >= 2 else rsi_e_val

        # Conf TF calculations
        conf = conf.copy()
        conf['DMA_20'] = sma(conf['Close'], 20)
        conf['DMA_50'] = sma(conf['Close'], 50)
        rsi_c = calculate_rsi(conf)

        rsi_c_val = float(rsi_c.iloc[-1]) if not rsi_c.isna().all() else 50.0
        rsi_c_prev = float(rsi_c.iloc[-2]) if len(rsi_c.dropna()) >= 2 else rsi_c_val
        dma20_c = float(conf['DMA_20'].iloc[-1])
        dma50_c = float(conf['DMA_50'].iloc[-1])
        price_c = float(conf['Close'].iloc[-1])

        # Swing structure (entry TF)
        window = min(80, len(entry))
        swing = detect_swing_structure(entry.tail(window), left=3, right=3)

        # Swing structure (conf TF)
        window_c = min(40, len(conf))
        swing_c = detect_swing_structure(conf.tail(window_c), left=2, right=2)

        return {
            'current_price': round(price, 2),
            'trend_entry': swing['trend'],
            'trend_conf': swing_c['trend'],
            'ma_alignment_entry': _ma_alignment(price, dma20_e, dma50_e),
            'ma_alignment_conf': _ma_alignment(price_c, dma20_c, dma50_c),
            'ma_crossover_entry': _ma_crossover(dma20_e, dma50_e),
            'rsi_entry': round(rsi_e_val, 1),
            'rsi_entry_prev': round(rsi_e_prev, 1),
            'rsi_conf': round(rsi_c_val, 1),
            'rsi_conf_prev': round(rsi_c_prev, 1),
            'divergence': _detect_divergence(entry, rsi_e),
            'price_position': _price_position(price, swing),
            'last_hl_price': swing.get('last_hl_price'),
            'last_lh_price': swing.get('last_lh_price'),
            'volume_status': _volume_status(entry),
        }

    except Exception:
        return None


# ─── Bullish confluence scoring (GRADUATED — no hard gates) ───

def score_bullish(analysis: dict) -> tuple[float, list[str]]:
    """
    Graduated bullish confluence scoring.
    Returns (score, [reason_strings]).
    
    Max ~22 pts. Opposing conditions get penalties but never flat -5 rejection.
    """
    score = 0.0
    reasons = []

    # 1. Trend (entry TF) — Core
    t = analysis['trend_entry']
    if t == 'Uptrend (HH/HL)':
        score += 4; reasons.append('+4  Uptrend (HH/HL) entry TF')
    elif t == 'Downtrend (LL/LH)':
        score -= 3; reasons.append('−3  Downtrend entry TF — penalised')
    else:
        score += 0.5; reasons.append('+0.5 Sideways entry TF')

    # 2. Trend (conf TF) — Core
    tc = analysis['trend_conf']
    if tc == 'Uptrend (HH/HL)':
        score += 3; reasons.append('+3  Uptrend conf TF')
    elif tc == 'Downtrend (LL/LH)':
        score -= 2; reasons.append('−2  Downtrend conf TF — penalised')

    # 3. MA alignment (entry) — Core
    ma = analysis['ma_alignment_entry']
    if ma == 'Bullish':
        score += 3; reasons.append('+3  MA Bullish entry TF')
    elif ma == 'Bearish':
        score -= 2; reasons.append('−2  MA Bearish entry TF')

    # 4. MA alignment (conf) — Core
    mac = analysis['ma_alignment_conf']
    if mac == 'Bullish':
        score += 2; reasons.append('+2  MA Bullish conf TF')
    elif mac == 'Bearish':
        score -= 1; reasons.append('−1  MA Bearish conf TF')

    # 5. Price position — Core
    pos = analysis.get('price_position', 'Middle')
    hl = analysis.get('last_hl_price')
    lh = analysis.get('last_lh_price')
    if pos == 'Near HL':
        score += 3
        lbl = f'{hl:.2f}' if hl else '?'
        reasons.append(f'+3  Price near HL ({lbl}) — ideal BUY zone')
    elif pos == 'Near LH':
        score -= 1
        lbl = f'{lh:.2f}' if lh else '?'
        reasons.append(f'−1  Price near LH ({lbl}) — at resistance')
    else:
        score += 1; reasons.append('+1  Price in middle range')

    # 6. RSI (entry TF) — Signal
    rsi_e = analysis['rsi_entry']
    rsi_ep = analysis['rsi_entry_prev']
    rising = rsi_e > rsi_ep + 0.5
    falling = rsi_e < rsi_ep - 0.5
    if rising and 40 <= rsi_e <= 70:
        score += 2; reasons.append(f'+2  RSI rising in 40–70 ({rsi_e})')
    elif rising:
        score += 1; reasons.append(f'+1  RSI rising ({rsi_e})')
    elif falling:
        score -= 1; reasons.append(f'−1  RSI falling ({rsi_e})')
    if rsi_e > 70:
        score -= 1; reasons.append(f'−1  RSI overbought ({rsi_e})')
    elif rsi_e < 30 and rising:
        score += 0.5; reasons.append(f'+0.5 RSI oversold turning up ({rsi_e})')

    # 7. RSI (conf TF) — Signal
    rsi_c = analysis['rsi_conf']
    rsi_cp = analysis['rsi_conf_prev']
    if rsi_c > rsi_cp + 0.5 and 40 <= rsi_c <= 70:
        score += 1.5; reasons.append(f'+1.5 Conf RSI rising 40–70 ({rsi_c})')
    elif rsi_c > rsi_cp + 0.5:
        score += 0.5; reasons.append(f'+0.5 Conf RSI rising ({rsi_c})')
    elif rsi_c < rsi_cp - 0.5:
        score -= 0.5; reasons.append(f'−0.5 Conf RSI falling ({rsi_c})')
    if rsi_c > 70:
        score -= 0.5; reasons.append(f'−0.5 Conf RSI overbought ({rsi_c})')

    # 8. MA crossover — Supporting
    xo = analysis['ma_crossover_entry']
    if xo == 'Bullish Crossover':
        score += 1.5; reasons.append('+1.5 Bullish MA crossover')
    elif xo == 'Bearish Crossover':
        score -= 1; reasons.append('−1  Bearish MA crossover')

    # 9. RSI divergence — Supporting
    div = analysis['divergence']
    if div == 'Bullish':
        score += 1.5; reasons.append('+1.5 Bullish RSI divergence')
    elif div == 'Bearish':
        score -= 1; reasons.append('−1  Bearish RSI divergence')

    # 10. Volume — Supporting
    vol = analysis.get('volume_status', 'N/A')
    if vol == 'High':
        score += 1; reasons.append('+1  High volume')

    return round(score, 2), reasons


# ─── Bearish confluence scoring (GRADUATED) ───

def score_bearish(analysis: dict) -> tuple[float, list[str]]:
    """Bearish confluence scoring. Mirror of bullish."""
    score = 0.0
    reasons = []

    # 1. Trend (entry)
    t = analysis['trend_entry']
    if t == 'Downtrend (LL/LH)':
        score += 4; reasons.append('+4  Downtrend (LL/LH) entry TF')
    elif t == 'Uptrend (HH/HL)':
        score -= 3; reasons.append('−3  Uptrend entry TF — penalised')
    else:
        score += 0.5; reasons.append('+0.5 Sideways entry TF')

    # 2. Trend (conf)
    tc = analysis['trend_conf']
    if tc == 'Downtrend (LL/LH)':
        score += 3; reasons.append('+3  Downtrend conf TF')
    elif tc == 'Uptrend (HH/HL)':
        score -= 2; reasons.append('−2  Uptrend conf TF — penalised')

    # 3. MA (entry)
    ma = analysis['ma_alignment_entry']
    if ma == 'Bearish':
        score += 3; reasons.append('+3  MA Bearish entry TF')
    elif ma == 'Bullish':
        score -= 2; reasons.append('−2  MA Bullish entry TF')

    # 4. MA (conf)
    mac = analysis['ma_alignment_conf']
    if mac == 'Bearish':
        score += 2; reasons.append('+2  MA Bearish conf TF')
    elif mac == 'Bullish':
        score -= 1; reasons.append('−1  MA Bullish conf TF')

    # 5. Price position
    pos = analysis.get('price_position', 'Middle')
    hl = analysis.get('last_hl_price')
    lh = analysis.get('last_lh_price')
    if pos == 'Near LH':
        score += 3
        lbl = f'{lh:.2f}' if lh else '?'
        reasons.append(f'+3  Price near LH ({lbl}) — ideal SHORT zone')
    elif pos == 'Near HL':
        score -= 2
        lbl = f'{hl:.2f}' if hl else '?'
        reasons.append(f'−2  Price near HL ({lbl}) — at support')
    else:
        score += 1; reasons.append('+1  Price in middle range')

    # 6. RSI (entry)
    rsi_e = analysis['rsi_entry']
    rsi_ep = analysis['rsi_entry_prev']
    falling = rsi_e < rsi_ep - 0.5
    rising = rsi_e > rsi_ep + 0.5
    if falling and 30 <= rsi_e <= 60:
        score += 2; reasons.append(f'+2  RSI falling 30–60 ({rsi_e})')
    elif falling:
        score += 1; reasons.append(f'+1  RSI falling ({rsi_e})')
    elif rising:
        score -= 1; reasons.append(f'−1  RSI rising ({rsi_e}) — wrong for bearish')
    if rsi_e < 30:
        score -= 1; reasons.append(f'−1  RSI oversold ({rsi_e}) — late short')

    # 7. RSI (conf)
    rsi_c = analysis['rsi_conf']
    rsi_cp = analysis['rsi_conf_prev']
    if rsi_c < rsi_cp - 0.5 and 30 <= rsi_c <= 60:
        score += 1.5; reasons.append(f'+1.5 Conf RSI falling 30–60 ({rsi_c})')
    elif rsi_c < rsi_cp - 0.5:
        score += 0.5; reasons.append(f'+0.5 Conf RSI falling ({rsi_c})')
    elif rsi_c > rsi_cp + 0.5:
        score -= 0.5; reasons.append(f'−0.5 Conf RSI rising ({rsi_c})')
    if rsi_c < 30:
        score -= 0.5; reasons.append(f'−0.5 Conf RSI oversold ({rsi_c})')

    # 8. MA crossover
    xo = analysis['ma_crossover_entry']
    if xo == 'Bearish Crossover':
        score += 1.5; reasons.append('+1.5 Bearish MA crossover')
    elif xo == 'Bullish Crossover':
        score -= 1; reasons.append('−1  Bullish MA crossover')

    # 9. RSI divergence
    div = analysis['divergence']
    if div == 'Bearish':
        score += 1.5; reasons.append('+1.5 Bearish RSI divergence')
    elif div == 'Bullish':
        score -= 1; reasons.append('−1  Bullish RSI divergence')

    # 10. Volume
    vol = analysis.get('volume_status', 'N/A')
    if vol == 'High' and pos == 'Near LH':
        score += 1.5; reasons.append('+1.5 High vol at LH (distribution)')
    elif vol == 'High':
        score += 0.5; reasons.append('+0.5 High volume')

    return round(score, 2), reasons


# ─── Grade description ───

def grade_label(score: float) -> str:
    if score >= 12:
        return '🟢 EXCELLENT'
    if score >= 9:
        return '🟢 GOOD'
    if score >= 5:
        return '🟡 MODERATE'
    return '🔴 WEAK'
