"""
data_fetcher.py — Stock Rotation v3
Yahoo Finance data fetching with simple caching.
"""

import os
import hashlib
import time
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


CACHE_DIR = "data_cache"
CACHE_TTL_SECONDS = 300  # 5 minutes


def _cache_path(symbol: str, interval: str, period: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(f"{symbol}_{interval}_{period}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def fetch_symbol(symbol: str, interval: str = "1d", period: str = "1y") -> pd.DataFrame | None:
    """Fetch OHLCV data for a single symbol with file-based caching."""
    cp = _cache_path(symbol, interval, period)
    if os.path.exists(cp):
        age = time.time() - os.path.getmtime(cp)
        if age < CACHE_TTL_SECONDS:
            try:
                return pd.read_pickle(cp)
            except Exception:
                pass
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is not None and len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_pickle(cp)
            return df
    except Exception:
        pass
    return None


def fetch_many(symbols: list[str], interval: str = "1d", period: str = "1y",
               max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """Fetch multiple symbols in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_symbol, s, interval, period): s for s in symbols}
        for f in as_completed(futures):
            sym = futures[f]
            try:
                data = f.result()
                if data is not None and len(data) > 0:
                    results[sym] = data
            except Exception:
                pass
    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sector_data(sectors: dict, interval: str = "1d", period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch all sector index data. Returns {sector_name: DataFrame}."""
    sym_to_name = {v: k for k, v in sectors.items()}
    raw = fetch_many(list(sectors.values()), interval=interval, period=period)
    return {sym_to_name[sym]: df for sym, df in raw.items() if sym in sym_to_name}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_company_data(symbols: list[str], interval: str = "1d", period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch company-level data for all symbols."""
    return fetch_many(symbols, interval=interval, period=period)
