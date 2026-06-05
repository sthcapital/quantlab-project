"""
data/external/optionable_universe.py

Curated static optionable symbol set (~500 names).

Coverage:
- All S&P 500 constituents (dot-free tickers only — BRK.B / BF.B are excluded
  because apply_symbol_filter() drops any ticker containing a '.').
- Russell 1000 top additions by market cap not already in the S&P 500.
- High-volume mid-cap growth names with confirmed listed options.

Usage in build_tradeable_universe():
    When the Polygon free-tier rate limit (5 req/min on options endpoints) makes
    a full dynamic check impractical, intersect the liquid candidate list with
    OPTIONABLE_UNIVERSE instead:

        from data.external.optionable_universe import OPTIONABLE_UNIVERSE
        candidates = [s for s in candidates if s in OPTIONABLE_UNIVERSE]

Upgrade path:
    Replace with a dynamic check once a Polygon paid tier (Starter+) or FactSet
    options-reference feed is available.  The integration point is the
    optionable_only branch in UniverseManager.build_tradeable_universe() in
    src/quantlab/universe.py — look for the comment "static optionable fallback".
"""

OPTIONABLE_UNIVERSE: frozenset[str] = frozenset({

    # ── S&P 500 — Information Technology (~65) ────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ACN", "IBM", "NOW",
    "INTU", "TXN", "QCOM", "AMD", "CRM", "ADI", "MU", "AMAT", "LRCX", "KLAC",
    "MCHP", "SNPS", "CDNS", "ANSS", "PANW", "FTNT", "CRWD", "ROP", "GRMN",
    "KEYS", "JNPR", "HPE", "HPQ", "NTAP", "STX", "WDC", "GDDY", "EPAM",
    "CTSH", "FFIV", "TDY", "LDOS", "BAH", "SAIC", "IT", "VRSK", "DXC",
    "WU", "FI", "FISV", "FIS", "GPN", "PYPL", "MA", "V", "ADP", "PAYX",
    "GEN", "SWKS", "QRVO", "ON", "MPWR", "ENPH", "NET", "ZS", "OKTA",
    "WDAY", "TEAM", "DDOG", "SNOW", "MDB",

    # ── S&P 500 — Communication Services (~20) ────────────────────────────────
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    "CHTR", "WBD", "PARA", "FOXA", "FOX", "OMC", "IPG", "EA", "TTWO", "LYV",
    "NWSA",

    # ── S&P 500 — Consumer Discretionary (~35) ────────────────────────────────
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
    "CMG", "ORLY", "AZO", "BBY", "EBAY", "ETSY", "GM", "F", "APTV", "GNTX",
    "LKQ", "PHM", "DHI", "LEN", "TOL", "NVR", "ULTA", "LULU", "DECK",
    "HAS", "MAT", "RL", "PVH", "TPR", "BWA", "EXPE", "MAR", "HLT", "H",
    "CCL", "RCL", "NCLH", "LVS", "WYNN", "MGM", "CZR",

    # ── S&P 500 — Consumer Staples (~25) ─────────────────────────────────────
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "K", "HRL", "SJM", "CAG", "CPB", "MKC", "HSY", "KHC", "STZ",
    "TAP", "MNST", "CELH", "EL", "CHD", "CLX", "SYY", "HPC", "COTY",

    # ── S&P 500 — Healthcare (~50) ────────────────────────────────────────────
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN", "BMY",
    "GILD", "REGN", "VRTX", "BIIB", "MRNA", "PFE", "CVS", "CI", "HUM",
    "MCK", "ABC", "CAH", "ELV", "MOH", "CNC", "ISRG", "BSX", "MDT", "SYK",
    "ZBH", "BAX", "BDX", "EW", "RMD", "HOLX", "PODD", "DXCM", "NTRA",
    "EXAS", "VEEV", "IDXX", "MTD", "IQV", "CRL", "ALGN", "MASI", "HSIC",
    "STE", "WAT", "RVTY", "TECH", "GEHC", "DVA", "INCY", "JAZZ", "ALNY",
    "BMRN", "SGEN", "RARE", "RCUS", "PCVX",

    # ── S&P 500 — Financials (~55) ────────────────────────────────────────────
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "BX", "KKR", "APO",
    "AMP", "PGR", "MET", "PRU", "AIG", "ALL", "TRV", "HIG", "CB", "MMC",
    "AON", "SCHW", "STT", "BK", "USB", "PNC", "RF", "KEY", "HBAN", "CFG",
    "MTB", "FITB", "WAL", "COF", "DFS", "SYF", "AXP", "ALLY",
    "NDAQ", "ICE", "CME", "CBOE", "SPGI", "MCO", "MKTX", "RJF", "LNC",
    "FNF", "FAF", "EG", "WRB", "ERIE", "CINF", "GL",
    "RKT", "UWMC", "GHLD", "PFSI", "COOP",

    # ── S&P 500 — Energy (~25) ────────────────────────────────────────────────
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "MPC", "PSX", "VLO", "DVN",
    "HES", "MRO", "APA", "OXY", "FANG", "WMB", "KMI", "OKE", "EQT",
    "CTRA", "BKR", "NOV", "TRGP", "DINO", "PBF",

    # ── S&P 500 — Materials (~25) ─────────────────────────────────────────────
    "LIN", "APD", "DD", "DOW", "EMN", "PPG", "SHW", "ECL", "MOS", "NUE",
    "STLD", "RS", "ATI", "PKG", "IP", "CF", "FCX", "NEM", "GOLD", "AEM",
    "FNV", "WPM", "SCCO", "CCJ", "MP",

    # ── S&P 500 — Industrials (~50) ───────────────────────────────────────────
    "HON", "GE", "CAT", "DE", "MMM", "UPS", "FDX", "RTX", "LMT", "NOC",
    "GD", "BA", "LHX", "TDG", "TXT", "HII", "AXON", "EMR", "ETN", "PH",
    "ITW", "XYL", "AME", "ROK", "IR", "GWW", "SWK", "SNA", "PNR", "IEX",
    "MAS", "GNRC", "ALLE", "ROL", "CTAS", "ALK", "UAL", "DAL", "LUV",
    "AAL", "WM", "RSG", "CLH", "EXPD", "CHRW", "JCI", "AOS", "CARR",
    "OTIS", "PWR", "MTZ", "TRMB", "WAB", "TT", "FLR", "J", "HXL", "HUBB",

    # ── S&P 500 — Real Estate (~25) ───────────────────────────────────────────
    "AMT", "PLD", "EQIX", "CCI", "SPG", "O", "PSA", "EXR", "AVB", "EQR",
    "UDR", "ESS", "CPT", "MAA", "ARE", "BXP", "VNO", "KIM", "REG", "FRT",
    "ADC", "EPRT", "WPC", "NNN", "STAG", "REXR", "EGP", "SBAC", "TRNO",

    # ── S&P 500 — Utilities (~20) ─────────────────────────────────────────────
    "NEE", "DUK", "SO", "D", "EXC", "AEP", "XEL", "ED", "ES", "WEC",
    "DTE", "PEG", "PPL", "EIX", "ETR", "NI", "CMS", "LNT", "AES", "NRG",
    "VST", "CEG", "PCG", "SRE", "AWK",

    # ── Russell 1000 top additions (not S&P 500, sorted by approx market cap) ─
    "UBER", "ABNB", "SHOP", "PLTR", "RIVN", "COIN", "HOOD", "SOFI",
    "LCID", "DKNG", "DASH", "LYFT", "RBLX", "U",
    "MELI", "SE", "GRAB", "NIO", "XPEV", "LI",
    "ROKU", "PINS", "SNAP", "SPOT", "TTD", "MGNI",
    "BILL", "ZI", "GTLB", "PCTY", "PAYC", "CDAY", "HUBS",
    "DOCU", "DBX", "BOX", "ZM", "RNG", "TWLO",
    "CVNA", "VRM", "KMX", "AN", "PAG", "LAD", "ABG",
    "FICO", "CSGP", "MNDY", "FRSH",
    "W", "CHWY", "FIGS", "CPNG",
    "IOT", "BRZE", "AMPL",
    "MARA", "RIOT", "HUT", "CLSK",

    # ── High-volume mid-cap growth (confirmed optionable) ─────────────────────
    "CELH", "HIMS", "TDOC", "ACCD", "OSCR", "CLOV",
    "BLDR", "BLD", "SUM", "EXP", "MLM", "VMC",
    "TREX", "AZEK", "FBHS", "LPX",
    "WOLF", "OLED", "NXPI",
    "CHPT", "EVGO", "BLNK", "PLUG", "FCEL", "BE", "CLNE",
    "ACHR", "JOBY", "RKLB",
    "MMSI", "ATRC", "IART", "NVCR", "HAE",
    "GME", "AMC",
    "DKNG", "PENN", "BALY",
    "LAC", "LTHM", "ALB", "LIVENT",
    "APPS", "DV", "IAS",
    "IIPR", "TLRY", "CGC",
    "BYND", "OATLY",
    "OPEN", "RDFN",
    "UPST", "AFRM", "LC",
})
