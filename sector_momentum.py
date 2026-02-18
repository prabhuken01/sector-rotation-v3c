"""
sector_momentum.py — Stock Rotation v3, Stage 1
Rank sectors by momentum (Z-score of RSI + CMF).
Identify top 4 bullish and bottom 6 bearish sectors.
"""

import pandas as pd
import numpy as np
from indicators import calculate_rsi, calculate_cmf
from config import TOP_N_BULLISH_SECTORS, BOTTOM_N_BEARISH_SECTORS


def rank_sectors(
    sector_data: dict[str, pd.DataFrame],
    benchmark_name: str = "Nifty 50",
) -> pd.DataFrame:
    """
    Rank all sectors by momentum score.
    
    Method: Cross-sectional Z-scores of RSI and CMF.
    Score = 0.5 × Z(RSI) + 0.5 × Z(CMF), scaled to 1–10.
    
    Returns DataFrame with columns:
        Sector, RSI, CMF, Z_RSI, Z_CMF, Momentum_Score
    sorted descending by Momentum_Score.
    """
    rows = []
    for name, data in sector_data.items():
        if name == benchmark_name:
            continue
        if data is None or len(data) < 20:
            continue
        try:
            rsi = calculate_rsi(data)
            cmf = calculate_cmf(data)
            rsi_val = float(rsi.iloc[-1]) if not rsi.isna().all() else 50.0
            cmf_val = float(cmf.iloc[-1]) if not cmf.isna().all() else 0.0
            rows.append({'Sector': name, 'RSI': rsi_val, 'CMF': cmf_val})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=['Sector', 'RSI', 'CMF', 'Z_RSI', 'Z_CMF', 'Momentum_Score'])

    df = pd.DataFrame(rows)

    # Cross-sectional Z-scores
    if len(df) > 1:
        for col in ['RSI', 'CMF']:
            m, s = df[col].mean(), df[col].std()
            df[f'Z_{col}'] = ((df[col] - m) / s if s > 0 else 0.0)
    else:
        df['Z_RSI'] = 0.0
        df['Z_CMF'] = 0.0

    # Composite score: 50/50 Z(RSI) + Z(CMF), scaled to 1–10
    raw = 0.5 * df['Z_RSI'] + 0.5 * df['Z_CMF']
    rmin, rmax = raw.min(), raw.max()
    if rmax > rmin:
        df['Momentum_Score'] = 1 + (raw - rmin) / (rmax - rmin) * 9
    else:
        df['Momentum_Score'] = 5.0

    df = df.sort_values('Momentum_Score', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    return df


def rank_sectors_at_date(
    sector_data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    benchmark_name: str = "Nifty 50",
) -> pd.DataFrame:
    """
    Point-in-time sector ranking: uses data up to (and including) `date`.
    Same method as rank_sectors but sliced to date.
    """
    sliced = {}
    for name, data in sector_data.items():
        if data is None or len(data) < 20:
            continue
        try:
            idx = data.index.get_indexer([date], method='ffill')[0]
            if idx < 13:
                continue
            sliced[name] = data.iloc[:idx + 1]
        except Exception:
            continue
    return rank_sectors(sliced, benchmark_name=benchmark_name)


def get_top_bottom_sectors(
    df_ranked: pd.DataFrame,
    top_n: int = TOP_N_BULLISH_SECTORS,
    bottom_n: int = BOTTOM_N_BEARISH_SECTORS,
) -> tuple[list[str], list[str]]:
    """
    From a ranked sector DataFrame, return:
    - top_n sectors (bullish, highest momentum)
    - bottom_n sectors (bearish, lowest momentum)
    """
    if df_ranked.empty:
        return [], []
    sectors = df_ranked['Sector'].tolist()
    top = sectors[:top_n]
    bottom = sectors[-bottom_n:] if len(sectors) >= bottom_n else sectors
    return top, bottom
