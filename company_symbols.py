"""
company_symbols.py — Stock Rotation v3
Load sector → company mapping from CSV or Excel.
"""

import os
import pandas as pd


def load_sector_companies(
    csv_path: str = "sector_company.csv",
    xlsx_path: str = "SectorCompany.xlsx",
) -> dict[str, dict[str, dict]]:
    """
    Load sector-company mapping.
    Returns: {sector_name: {symbol: {'name': str, 'weight': float}}}
    
    Tries CSV first (faster), then Excel sheet 'Main', then sheet 'Sheet1'.
    """
    df = None

    # Try CSV
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None

    # Try Excel
    if df is None and os.path.isfile(xlsx_path):
        for sheet in ['Main', 'Sheet1', 0]:
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                break
            except Exception:
                continue

    if df is None or df.empty:
        raise FileNotFoundError(
            f"Cannot load sector-company data from {csv_path} or {xlsx_path}"
        )

    # Normalize columns
    df.columns = df.columns.str.strip()
    required = {'Sector', 'Symbol'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV/Excel must have columns: {required}. Found: {list(df.columns)}")

    # Build mapping
    mapping: dict[str, dict[str, dict]] = {}
    for _, row in df.iterrows():
        sector = str(row['Sector']).strip()
        symbol = str(row['Symbol']).strip()
        name = str(row.get('Company Name', symbol)).strip()
        weight = float(row['Weight (%)']) if pd.notna(row.get('Weight (%)')) else 0.0

        if not sector or not symbol or symbol == 'nan':
            continue

        if sector not in mapping:
            mapping[sector] = {}
        mapping[sector][symbol] = {'name': name, 'weight': weight}

    return mapping


def get_all_symbols(sector_companies: dict) -> list[str]:
    """Get flat list of all symbols across all sectors."""
    symbols = []
    for sector, companies in sector_companies.items():
        symbols.extend(companies.keys())
    return list(set(symbols))


def get_symbol_sector_map(sector_companies: dict) -> dict[str, str]:
    """Returns {symbol: sector_name}."""
    m = {}
    for sector, companies in sector_companies.items():
        for sym in companies:
            m[sym] = sector
    return m


def get_symbol_name_map(sector_companies: dict) -> dict[str, str]:
    """Returns {symbol: company_name}."""
    m = {}
    for sector, companies in sector_companies.items():
        for sym, info in companies.items():
            m[sym] = info.get('name', sym)
    return m
