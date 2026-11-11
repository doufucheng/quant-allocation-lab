# Quant Allocation Lab

A production-tested toolbox of **16 quantitative allocation, style-rotation, and timing models** for China markets — risk-parity asset allocation, seven style-rotation engines, and eight market-timing frameworks, all with long-history backtests and institutional-grade documentation of design decisions.

Built and maintained over 8+ years of live buy-side use. Each model is self-contained, documented inline (methodology → signal rules → data lineage → known pitfalls), and runs on standard pandas/numpy.

> Backtest results are historical simulations and do not guarantee future returns.

## Repository Map

| Module | Models | What it does |
|---|---|---|
| `allocation/` | `allocation_rp` | Risk-parity multi-asset allocation (equity / bond / commodity / gold ETF pool) |
| `style_rotation/` | 8 models | Style & factor rotation engines: rate-driven, momentum, valuation-spread, Pring cycle, crowding, pair-switch, alpha-composite |
| `timing/` | 6 models | Market timing: annual-MA dividend rule, dual momentum, gold valuation, index valuation signal, LPPLS bubble detection, bond-futures timing |

## Model Highlights

### 1. US-Rate-Driven Style Rotation (`style_rotation/rotation_rate.py`)
The 10Y US Treasury yield as the global pricing anchor for A-share style rotation. An 18-month rolling Z-score of the yield switches between three regimes:

- **Z > +0.7** (rates falling → risk appetite rising): overweight growth + quality (50/50)
- **Z < −0.7** (rates rising → risk appetite falling): overweight dividend + value (50/50)
- Neutral band: equal-weight 25/25/25/25

Backtested 2012–2026 against a broad-market total-return benchmark; annual excess ≈ **3.2%**. The inline docs walk through the exact indicator construction, Excel-lineage cross-checks, and why the Z-window is 18 months.

### 2. Risk-Parity Allocation Engine (`allocation/allocation_rp.py`)
Classic equal-risk-contribution allocation across a multi-asset ETF pool (CN dividend, 30Y/10Y government bond indices, commodities, gold), 2003–history weight construction, target-vol targeting optional.

### 3. Pring Six-Phase Cycle Rotation (`style_rotation/rotation_pring.py)
Maps Martin Pring's six-phase liquidity–economy–inflation cycle onto CN market indicators with a 126-day rolling classifier; rotates equity/bond/commodity legs by phase. Exits at phase six (stagnation).

### 4. LPPLS Bubble Detection (`timing/lppls_model.py`)
Fitted Log-Periodic Power Law Singularity model for bubble/crash regime detection on broad indices.

### 5. Annual-MA Dividend Timing (`timing/annual_ma.py`)
Statistical study + tradable rule: buying a dividend low-vol index **below its 250-day MA** delivered +16.8% average 1-year returns with 94.9% positive-hit rate (514 samples, 2015–2026) vs. +10.4% / 86.1% unconditionally — with full distribution tables, sample counts, and caveats in the docstring.

## Design Principles

1. **Methodology-first docs.** Every model file opens with the full reasoning chain: hypothesis → indicator construction → signal rules → backtest → known failure modes. Reviewers can audit the logic without running code.
2. **Data lineage transparency.** Which index code feeds which signal, which total-return series is used, and where the raw data comes from — all documented per model.
3. **No look-ahead, no survivorship shortcuts.** Rolling windows computed as-of; index histories use total-return versions where dividends matter.
4. **Excel cross-validation.** Several models were first prototyped in spreadsheets; the Python ports cross-check against the Excel formulas line by line (noted inline).

## Tech Stack

Python · pandas · numpy · matplotlib (report figures) · openpyxl (Excel outputs for non-technical stakeholders)

## Related

- [ai-finance-workflows](https://github.com/doufucheng/ai-finance-workflows) — AI/agent engineering work (multi-source aggregators, scoring engines, LLM automation)

## License

MIT
