"""
config.py — Stock Rotation v3
Constants, sector index symbols, default weights.
"""

APP_VERSION = "3.0.0"

# ── Sector index symbols (Yahoo Finance tickers for NSE sector indices) ──
SECTORS = {
    'Nifty 50':     '^NSEI',
    'Auto':         '^CNXAUTO',
    'Commodities':  '^CNXCOMMODITIES',
    'Defence':      '^NSEDEFENCE.NS',
    'Energy':       '^CNXENERGY',
    'FMCG':         '^CNXFMCG',
    'IT':           '^CNXIT',
    'Infra':        '^CNXINFRA',
    'Media':        '^CNXMEDIA',
    'Metal':        '^CNXMETAL',
    'Fin Services': '^NIFTYFINSERV',
    'Pharma':       '^CNXPHARMA',
    'PSU Bank':     '^NIFTYPSUBANK',
    'Pvt Bank':     '^CNXPVTBANK',
    'Realty':       '^CNXREALTY',
    'Oil & Gas':    '^CNXOILGAS',
}

# ── Technical indicator periods ──
RSI_PERIOD = 14
ADX_PERIOD = 14
CMF_PERIOD = 20

# ── Sector momentum weights (Trending mode: Z-score based) ──
DEFAULT_MOMENTUM_WEIGHTS = {
    'RSI': 50.0,
    'CMF': 50.0,
}

# ── Confluence ──
TOP_N_BULLISH_SECTORS = 4
BOTTOM_N_BEARISH_SECTORS = 6

# Screener pass-through counts (how many go from Stage 2 → Stage 3)
SCREENER_TOP_N = 20
SCREENER_BOT_N = 20

# ── Historical rankings ──
HISTORICAL_LOOKBACK_DAYS = 15  # last 15 trading days

# ── Formatting ──
DECIMAL_PLACES = {
    'RSI': 1, 'ADX': 1, 'ADX_Z': 1, 'DI_Spread': 1,
    'CMF': 2, 'RS_Rating': 1, 'Momentum_Score': 1,
}
