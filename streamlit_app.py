"""
streamlit_app.py — Stock Rotation v3
Main Streamlit UI. Clean, modular, with 4 tabs:
  1. Sector Momentum
  2. Stock Screener
  3. Confluence Analysis
  4. Historical Rankings
"""

import streamlit as st
import pandas as pd
import numpy as np

from config import APP_VERSION, SECTORS, TOP_N_BULLISH_SECTORS, BOTTOM_N_BEARISH_SECTORS
from company_symbols import load_sector_companies, get_all_symbols, get_symbol_sector_map, get_symbol_name_map
from data_fetcher import fetch_sector_data, fetch_company_data, fetch_many
from sector_momentum import rank_sectors, get_top_bottom_sectors
from stock_screener import screen_stocks
from confluence import analyze_stock, score_bullish, score_bearish, grade_label
from historical import compute_historical_rankings

# ── Page config ──
st.set_page_config(page_title="Stock Rotation v3", page_icon="📊", layout="wide")

# ── Load sector-company mapping ──
@st.cache_data(show_spinner=False)
def _load_companies():
    return load_sector_companies()

try:
    SECTOR_COMPANIES = _load_companies()
except Exception as e:
    st.error(f"❌ Cannot load sector-company data: {e}")
    st.stop()

ALL_SYMBOLS = get_all_symbols(SECTOR_COMPANIES)
SYMBOL_SECTOR = get_symbol_sector_map(SECTOR_COMPANIES)
SYMBOL_NAME = get_symbol_name_map(SECTOR_COMPANIES)

# ── Header ──
st.title("📊 NSE Sector Rotation & Stock Ranking")
st.caption(f"v{APP_VERSION} | {len(ALL_SYMBOLS)} stocks across {len(SECTOR_COMPANIES)} sectors")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    data_interval = st.selectbox("Sector data interval", ['1d', '1wk'], index=0)
    data_period = st.selectbox("Data period", ['6mo', '1y', '2y'], index=1)
    st.markdown("---")
    st.markdown(f"**Sectors:** {len(SECTORS) - 1} (excl. Nifty 50)")
    st.markdown(f"**Stocks:** {len(ALL_SYMBOLS)}")
    st.markdown(f"**Bullish sectors:** Top {TOP_N_BULLISH_SECTORS}")
    st.markdown(f"**Bearish sectors:** Bottom {BOTTOM_N_BEARISH_SECTORS}")

# ── Fetch sector data ──
with st.spinner("📡 Fetching sector data..."):
    sector_data = fetch_sector_data(SECTORS, interval=data_interval, period=data_period)

benchmark_data = sector_data.get('Nifty 50')
if benchmark_data is None or len(benchmark_data) < 20:
    st.error("❌ Could not fetch Nifty 50 benchmark data. Check your internet connection.")
    st.stop()

# ── Stage 1: Sector Momentum (always computed) ──
df_sectors = rank_sectors(sector_data)
top_sectors, bot_sectors = get_top_bottom_sectors(df_sectors)

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sector Momentum",
    "🔍 Stock Screener",
    "🏆 Confluence Analysis",
    "📅 Historical Rankings",
    "📋 Data Sources",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1: Sector Momentum
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📈 Sector Momentum Ranking")
    st.caption(
        "Sectors ranked by momentum score = 50% Z(RSI) + 50% Z(CMF). "
        "Top 4 → bullish stock universe. Bottom 6 → bearish stock universe."
    )

    if df_sectors.empty:
        st.warning("⚠️ No sector data available.")
    else:
        # Highlight top 4 and bottom 6
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**✅ Bullish (top {TOP_N_BULLISH_SECTORS}):** {', '.join(top_sectors)}")
        with col2:
            st.warning(f"**⚠️ Bearish (bottom {BOTTOM_N_BEARISH_SECTORS}):** {', '.join(bot_sectors)}")

        # Show table
        display_df = df_sectors[['Rank', 'Sector', 'RSI', 'CMF', 'Momentum_Score']].copy()
        display_df['RSI'] = display_df['RSI'].round(1)
        display_df['CMF'] = display_df['CMF'].round(3)
        display_df['Momentum_Score'] = display_df['Momentum_Score'].round(1)

        def _color_score(val):
            if val >= 7:
                return 'background-color: #1a472a; color: #4ade80'
            elif val <= 3:
                return 'background-color: #4a1a1a; color: #f87171'
            return ''

        styled = display_df.style.applymap(_color_score, subset=['Momentum_Score'])
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2: Stock Screener
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 Stock Screener (Stage 2)")
    st.caption(
        "Fast first-pass scoring (0–7) on daily data. "
        "Bullish = top scores from top 4 sectors. Bearish = lowest scores from bottom 6."
    )

    # Fetch company data
    with st.spinner(f"📡 Fetching data for {len(ALL_SYMBOLS)} stocks..."):
        company_data = fetch_company_data(ALL_SYMBOLS, interval='1d', period=data_period)

    st.info(f"Fetched data for **{len(company_data)}** / {len(ALL_SYMBOLS)} stocks")

    if company_data:
        bullish_screen, bearish_screen = screen_stocks(
            company_data, SYMBOL_SECTOR, top_sectors, bot_sectors
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### 🟢 Top Bullish Candidates ({len(bullish_screen)})")
            if bullish_screen:
                bull_df = pd.DataFrame(bullish_screen)
                bull_df['name'] = bull_df['symbol'].map(SYMBOL_NAME)
                bull_df = bull_df[['name', 'symbol', 'sector', 'score']]
                bull_df.columns = ['Company', 'Symbol', 'Sector', 'Score']
                st.dataframe(bull_df, use_container_width=True, hide_index=True)
            else:
                st.info("No bullish candidates found.")

        with col2:
            st.markdown(f"#### 🔴 Top Bearish Candidates ({len(bearish_screen)})")
            if bearish_screen:
                bear_df = pd.DataFrame(bearish_screen)
                bear_df['name'] = bear_df['symbol'].map(SYMBOL_NAME)
                bear_df = bear_df[['name', 'symbol', 'sector', 'score']]
                bear_df.columns = ['Company', 'Symbol', 'Sector', 'Score']
                st.dataframe(bear_df, use_container_width=True, hide_index=True)
            else:
                st.info("No bearish candidates found.")
    else:
        st.warning("⚠️ No company data fetched. Check connection.")


# ════════════════════════════════════════════════════════════════════
# TAB 3: Confluence Analysis
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🏆 Confluence Analysis (Stage 3)")

    # Scoring table
    with st.expander("📖 How Confluence Scoring works (10 factors, max ~22 pts)", expanded=False):
        st.markdown("""
| # | Factor | Bullish +Pts | Penalty −Pts | Description |
|---|--------|-------------|-------------|-------------|
| 1 | Trend (entry TF) | HH/HL: **+4** | LL/LH: **−3** | Swing high/low structure |
| 2 | Trend (conf TF) | HH/HL: **+3** | LL/LH: **−2** | Confirmation TF trend |
| 3 | MA Align (entry) | Bullish: **+3** | Bearish: **−2** | Price > 20 > 50 SMA |
| 4 | MA Align (conf) | Bullish: **+2** | Bearish: **−1** | Same on conf TF |
| 5 | Price Position | Near HL: **+3** | Near LH: **−1**, Mid: **+1** | vs pivot structure |
| 6 | RSI (entry) | Rising 40–70: **+2** | Falling: **−1**, OB: **−1** | RSI direction |
| 7 | RSI (conf) | Rising 40–70: **+1.5** | Falling: **−0.5** | Conf TF RSI |
| 8 | MA Crossover | Bullish X: **+1.5** | Bearish X: **−1** | 20/50 SMA crossover |
| 9 | RSI Divergence | Bullish: **+1.5** | Bearish: **−1** | Price vs RSI divergence |
| 10 | Volume | High: **+1** | — | Recent vol > 1.2× avg |

**Scores:** ≥12 = 🟢 Excellent | 9–12 = 🟢 Good | 5–9 = 🟡 Moderate | <5 = 🔴 Weak

**v3 change:** NO hard gate rejection. Opposing conditions get penalties but stocks 
are always ranked — ensures we always get top picks even in mixed markets.
""")

    # TF selector
    conf_tf = st.radio(
        "Entry timeframe for confluence:",
        ["Daily + Weekly (recommended for backtesting)", "4H + 1H (finer entry, needs 1H data)"],
        horizontal=True,
        index=0,
    )
    use_daily_tf = "Daily" in conf_tf

    st.success(f"**✅ Bullish sectors (top {TOP_N_BULLISH_SECTORS}):** {', '.join(top_sectors)}")
    st.warning(f"**⚠️ Bearish sectors (bottom {BOTTOM_N_BEARISH_SECTORS}):** {', '.join(bot_sectors)}")

    if st.button("🚀 Run Confluence Analysis", type="primary"):
        # Get shortlists from screener
        if not company_data:
            st.warning("⚠️ No company data. Go to Stock Screener tab first.")
        else:
            # Build shortlists
            bullish_screen, bearish_screen = screen_stocks(
                company_data, SYMBOL_SECTOR, top_sectors, bot_sectors
            )
            bull_syms = [r['symbol'] for r in bullish_screen[:20]]
            bear_syms = [r['symbol'] for r in bearish_screen[:20]]

            # Fetch hourly data if needed
            hourly_data = {}
            if not use_daily_tf:
                with st.spinner("📡 Fetching 1H data for shortlisted stocks..."):
                    all_conf_syms = list(set(bull_syms + bear_syms))
                    hourly_data = fetch_many(all_conf_syms, interval='1h', period='1mo')

            progress = st.progress(0)
            status = st.empty()

            # Bullish confluence
            bull_results = []
            all_syms = bull_syms + bear_syms
            total = len(all_syms)

            for idx_i, sym in enumerate(bull_syms):
                progress.progress((idx_i + 1) / total)
                status.text(f"Analyzing {SYMBOL_NAME.get(sym, sym)} ({idx_i + 1}/{total})...")

                data = company_data.get(sym)
                if data is None:
                    continue

                if use_daily_tf:
                    analysis = analyze_stock(data, entry_tf='daily')
                else:
                    h_data = hourly_data.get(sym)
                    if h_data is None or len(h_data) < 50:
                        analysis = analyze_stock(data, entry_tf='daily')
                    else:
                        analysis = analyze_stock(h_data, data_conf=data, entry_tf='4h')

                if analysis is None:
                    continue

                bscore, reasons = score_bullish(analysis)
                bull_results.append({
                    'Company': SYMBOL_NAME.get(sym, sym),
                    'Symbol': sym,
                    'Sector': SYMBOL_SECTOR.get(sym, ''),
                    'CMP': analysis['current_price'],
                    'Score': bscore,
                    'Grade': grade_label(bscore),
                    'Trend': analysis['trend_entry'],
                    'MA': analysis['ma_alignment_entry'],
                    'RSI': analysis['rsi_entry'],
                    'Position': analysis['price_position'],
                    'Reasons': ' | '.join(reasons),
                })

            # Bearish confluence
            bear_results = []
            for idx_i, sym in enumerate(bear_syms):
                progress.progress((len(bull_syms) + idx_i + 1) / total)
                status.text(f"Analyzing {SYMBOL_NAME.get(sym, sym)} ({len(bull_syms) + idx_i + 1}/{total})...")

                data = company_data.get(sym)
                if data is None:
                    continue

                if use_daily_tf:
                    analysis = analyze_stock(data, entry_tf='daily')
                else:
                    h_data = hourly_data.get(sym)
                    if h_data is None or len(h_data) < 50:
                        analysis = analyze_stock(data, entry_tf='daily')
                    else:
                        analysis = analyze_stock(h_data, data_conf=data, entry_tf='4h')

                if analysis is None:
                    continue

                bscore, reasons = score_bearish(analysis)
                bear_results.append({
                    'Company': SYMBOL_NAME.get(sym, sym),
                    'Symbol': sym,
                    'Sector': SYMBOL_SECTOR.get(sym, ''),
                    'CMP': analysis['current_price'],
                    'Score': bscore,
                    'Grade': grade_label(bscore),
                    'Trend': analysis['trend_entry'],
                    'MA': analysis['ma_alignment_entry'],
                    'RSI': analysis['rsi_entry'],
                    'Position': analysis['price_position'],
                    'Reasons': ' | '.join(reasons),
                })

            progress.empty()
            status.empty()

            # Display results
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🟢 Top Bullish — Confluence Ranked")
                if bull_results:
                    df_bull = pd.DataFrame(bull_results)
                    df_bull = df_bull.sort_values('Score', ascending=False).reset_index(drop=True)
                    df_bull.index = range(1, len(df_bull) + 1)
                    df_bull.index.name = '#'

                    # Top 8
                    st.dataframe(
                        df_bull[['Company', 'Sector', 'CMP', 'Score', 'Grade', 'Trend', 'MA', 'RSI', 'Position']].head(8),
                        use_container_width=True,
                    )

                    # Top 2 highlight
                    if len(df_bull) >= 1:
                        top1 = df_bull.iloc[0]
                        st.success(f"**🥇 #{1}: {top1['Company']}** ({top1['Sector']}) — Score: {top1['Score']} {top1['Grade']}")
                    if len(df_bull) >= 2:
                        top2 = df_bull.iloc[1]
                        st.success(f"**🥈 #{2}: {top2['Company']}** ({top2['Sector']}) — Score: {top2['Score']} {top2['Grade']}")

                    with st.expander("📋 Full scoring details"):
                        for _, r in df_bull.head(8).iterrows():
                            st.markdown(f"**{r['Company']}** ({r['Symbol']}) — Score: {r['Score']}")
                            st.caption(r['Reasons'])
                else:
                    st.info("No bullish confluence results.")

            with col2:
                st.markdown("#### 🔴 Top Bearish — Confluence Ranked")
                if bear_results:
                    df_bear = pd.DataFrame(bear_results)
                    df_bear = df_bear.sort_values('Score', ascending=False).reset_index(drop=True)
                    df_bear.index = range(1, len(df_bear) + 1)
                    df_bear.index.name = '#'

                    st.dataframe(
                        df_bear[['Company', 'Sector', 'CMP', 'Score', 'Grade', 'Trend', 'MA', 'RSI', 'Position']].head(8),
                        use_container_width=True,
                    )

                    if len(df_bear) >= 1:
                        top1 = df_bear.iloc[0]
                        st.error(f"**🥇 #{1}: {top1['Company']}** ({top1['Sector']}) — Score: {top1['Score']} {top1['Grade']}")
                    if len(df_bear) >= 2:
                        top2 = df_bear.iloc[1]
                        st.error(f"**🥈 #{2}: {top2['Company']}** ({top2['Sector']}) — Score: {top2['Score']} {top2['Grade']}")

                    with st.expander("📋 Full scoring details"):
                        for _, r in df_bear.head(8).iterrows():
                            st.markdown(f"**{r['Company']}** ({r['Symbol']}) — Score: {r['Score']}")
                            st.caption(r['Reasons'])
                else:
                    st.info("No bearish confluence results.")


# ════════════════════════════════════════════════════════════════════
# TAB 4: Historical Rankings
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📅 Historical Rankings (Last 15 Trading Days)")
    st.caption(
        "Per-date backtesting: sector rotation → stock screening → confluence scoring → forward returns. "
        "Uses **daily** entry TF (no hourly dependency) for speed and reliability."
    )

    if st.button("🔄 Compute Historical Rankings", type="primary"):
        if not company_data:
            st.warning("⚠️ No company data. Go to Stock Screener tab first to fetch data.")
        else:
            progress = st.progress(0)
            status = st.empty()

            def _progress(i, total, dt):
                progress.progress((i + 1) / total)
                status.text(f"Processing {dt.strftime('%Y-%m-%d')} ({i + 1}/{total})...")

            df_hist = compute_historical_rankings(
                sector_data=sector_data,
                company_data=company_data,
                symbol_sector=SYMBOL_SECTOR,
                symbol_name=SYMBOL_NAME,
                benchmark_data=benchmark_data,
                progress_callback=_progress,
            )

            progress.empty()
            status.empty()

            if df_hist.empty:
                st.warning("⚠️ No historical data could be computed.")
            else:
                st.success(f"✅ Computed rankings for {len(df_hist)} dates")

                # Split into Bullish and Bearish tables
                st.markdown("#### 🟢 Bullish Confluence — Historical")
                bull_cols = ['Date', 'Mom #1 Sector', 'Mom #2 Sector',
                             'Bull #1 Stock', 'Bull #1 Sector', 'Bull #1 CMP', 'Bull #1 Score', 'Bull #1 Grade',
                             'Bull #1 1D %', 'Bull #1 2D %', 'Bull #1 3D %', 'Bull #1 1W %',
                             'Bull #2 Stock', 'Bull #2 Sector', 'Bull #2 CMP', 'Bull #2 Score',
                             'Bull #2 1D %', 'Bull #2 2D %', 'Bull #2 3D %', 'Bull #2 1W %']
                avail_cols = [c for c in bull_cols if c in df_hist.columns]
                if avail_cols:
                    st.dataframe(df_hist[avail_cols], use_container_width=True, hide_index=True)

                st.markdown("#### 🔴 Bearish Confluence — Historical")
                bear_cols = ['Date',
                             'Bear #1 Stock', 'Bear #1 Sector', 'Bear #1 CMP', 'Bear #1 Score', 'Bear #1 Grade',
                             'Bear #1 1D %', 'Bear #1 2D %', 'Bear #1 3D %', 'Bear #1 1W %',
                             'Bear #2 Stock', 'Bear #2 Sector', 'Bear #2 CMP', 'Bear #2 Score',
                             'Bear #2 1D %', 'Bear #2 2D %', 'Bear #2 3D %', 'Bear #2 1W %']
                avail_cols = [c for c in bear_cols if c in df_hist.columns]
                if avail_cols:
                    st.dataframe(df_hist[avail_cols], use_container_width=True, hide_index=True)

                # Summary stats
                st.markdown("#### 📊 Backtest Summary")
                with st.expander("Forward return statistics", expanded=True):
                    for prefix in ['Bull #1', 'Bull #2', 'Bear #1', 'Bear #2']:
                        col_1d = f'{prefix} 1D %'
                        col_1w = f'{prefix} 1W %'
                        if col_1d in df_hist.columns:
                            vals_1d = df_hist[col_1d].dropna()
                            vals_1w = df_hist[col_1w].dropna() if col_1w in df_hist.columns else pd.Series()
                            if len(vals_1d) > 0:
                                direction = "Bullish" if "Bull" in prefix else "Bearish"
                                if direction == "Bullish":
                                    win_1d = (vals_1d > 0).sum()
                                    win_1w = (vals_1w > 0).sum() if len(vals_1w) > 0 else 0
                                else:
                                    win_1d = (vals_1d < 0).sum()
                                    win_1w = (vals_1w < 0).sum() if len(vals_1w) > 0 else 0

                                wr_1d = win_1d / len(vals_1d) * 100 if len(vals_1d) > 0 else 0
                                wr_1w = win_1w / len(vals_1w) * 100 if len(vals_1w) > 0 else 0
                                avg_1d = vals_1d.mean()
                                avg_1w = vals_1w.mean() if len(vals_1w) > 0 else 0

                                st.markdown(
                                    f"**{prefix}:** "
                                    f"1D win rate {wr_1d:.0f}% (avg {avg_1d:+.1f}%) | "
                                    f"1W win rate {wr_1w:.0f}% (avg {avg_1w:+.1f}%) | "
                                    f"n={len(vals_1d)}"
                                )

                # CSV download
                csv_data = df_hist.to_csv(index=False)
                st.download_button(
                    "📥 Download Historical Rankings CSV",
                    csv_data,
                    "historical_rankings_v3.csv",
                    "text/csv",
                )


# ════════════════════════════════════════════════════════════════════
# TAB 5: Data Sources
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📋 Data Sources & Sector-Company Mapping")

    st.markdown(f"**Total sectors:** {len(SECTOR_COMPANIES)}")
    st.markdown(f"**Total unique stocks:** {len(ALL_SYMBOLS)}")

    for sector in sorted(SECTOR_COMPANIES.keys()):
        companies = SECTOR_COMPANIES[sector]
        with st.expander(f"{sector} ({len(companies)} stocks)"):
            rows = []
            for sym, info in companies.items():
                rows.append({
                    'Symbol': sym,
                    'Company': info.get('name', sym),
                    'Weight %': info.get('weight', 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Sector Indices:**")
    idx_rows = [{'Sector': k, 'Symbol': v} for k, v in SECTORS.items()]
    st.dataframe(pd.DataFrame(idx_rows), use_container_width=True, hide_index=True)
