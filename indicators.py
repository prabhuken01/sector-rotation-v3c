"""
indicators.py — Stock Rotation v3
Technical indicators: RSI, ADX, CMF, Z-score.
"""

import pandas as pd
import numpy as np
from config import RSI_PERIOD, ADX_PERIOD, CMF_PERIOD


def calculate_rsi(data: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    """RSI using Wilder's smoothing (matches TradingView)."""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def calculate_cmf(data: pd.DataFrame, period: int = CMF_PERIOD) -> pd.Series:
    """Chaikin Money Flow."""
    h, l, c, v = data['High'], data['Low'], data['Close'], data['Volume']
    mfm = ((c - l) - (h - c)) / (h - l)
    mfm = mfm.fillna(0.0)
    mfv = mfm * v
    return mfv.rolling(period).sum() / v.rolling(period).sum()


def calculate_adx(data: pd.DataFrame, period: int = ADX_PERIOD):
    """ADX with +DI, -DI, DI_Spread using Wilder's smoothing."""
    h, l, c = data['High'], data['Low'], data['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up = h - h.shift()
    dn = l.shift() - l
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=data.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=data.index)

    def wilder(s, p):
        r = s.copy() * np.nan
        r.iloc[p - 1] = s.iloc[:p].mean()
        for i in range(p, len(s)):
            r.iloc[i] = (r.iloc[i - 1] * (p - 1) + s.iloc[i]) / p
        return r

    atr = wilder(tr, period)
    pdm = wilder(plus_dm, period)
    mdm = wilder(minus_dm, period)
    pdi = 100 * pdm / atr
    mdi = 100 * mdm / atr
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi)).fillna(0.0)
    adx = wilder(dx, period)
    return adx, pdi, mdi, pdi - mdi


def calculate_z_score(series: pd.Series) -> float:
    """Z-score of the latest value in a series."""
    if series.isna().all() or len(series) < 2:
        return 0.0
    m, s = series.mean(), series.std()
    if s == 0 or pd.isna(s):
        return 0.0
    return float((series.iloc[-1] - m) / s)


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()
