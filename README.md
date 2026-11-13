<div align="center">

# Quant Allocation Lab

**A two-layer quantitative investment system for China markets**

Layer 1 · Smart Beta style rotation → Layer 2 · Tactical asset allocation overlay

8+ years of live buy-side development · PKU M.Eng. CS · Python/pandas

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Models](https://img.shields.io/badge/Models-16-orange) ![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## The Two-Layer Framework

```
┌─────────────────────────────────────────────────┐
│  Layer 2 · Asset Allocation Overlay             │
│  Momentum-gated tactical weights across         │
│  equity / bonds / commodities / gold ETFs       │
│  → beats fixed all-weather weights by +17%/yr   │
├─────────────────────────────────────────────────┤
│  Layer 1 · Smart Beta Style Rotation            │
│  Dividend / Growth / Quality / Value legs,      │
│  rotated by rate, momentum, cycle, crowding     │
│  → beats 4-style equal weight over 20 years     │
├─────────────────────────────────────────────────┤
│  Data · Wind total-return indices, no look-ahead│
└─────────────────────────────────────────────────┘
```

## Layer 1 — Smart Beta Style Rotation

### 12-Month Momentum Rotation — 20.5 years (2006–2026)

![Smart Beta Layer](results/smartbeta_layer_nav.png)

**Rotation CAGR 20.5% vs 18.3% equal-weight** (+2.2%/yr excess, 20.5 years). Each panel is self-contained: top = absolute NAV, bottom = cumulative excess vs benchmark. Term-structure analysis (in-model docs) shows 12M is the sweet spot: 2–3M mean-reverts, 6–12M trends persist. Win rate 54.6%, monthly turnover only 7.9%.

### Supporting rotation engines

- **US 10Y rate Z-score** (3-regime switch, 2014–2026): ![Rate](results/rate_rotation_nav.png)
- Annual-MA conditional entry (94.9% win rate below MA): ![AnnualMA](results/annual_ma_table.png)

## Layer 2 — Tactical Asset Allocation Overlay

### Momentum Overlay vs Fixed All-Weather Weights — 11.8 years (2014–2025)

![Allocation Layer](results/allocation_layer_nav.png)

The fixed-weight benchmark is the classic all-weather book (30Y CGB 30% / gold 15% / 10Y CGB 10% / equity styles 25% / commodities 25%). The momentum overlay re-weights monthly: **tactical CAGR 22.7% vs fixed 5.4% — +17.3%/yr excess**. Fee honesty documented in-model: 20.6% gross → 16.5–18.9% net.

### Asset-level dual momentum (single-leg view)

![Dual Momentum](results/dual_momentum_nav.png)

> **Disclaimer:** Historical backtests on total-return indices; not a guarantee of future returns. Models document their own failure modes (regime breaks, fee drag) inline.

---

## Model Map

| Layer | Module | Models |
|---|---|---|
| 2 | `allocation/` | Risk-parity engine, all-weather fixed weights, momentum overlay |
| 1 | `style_rotation/` | Momentum (12M), rate Z-score, Pring 6-phase, valuation spread, crowding, pair-switch, alpha-composite, combined |
| 1 | `timing/` | Dual momentum, annual-MA rule, gold valuation, LPPLS bubbles, bond-futures timing, index valuation signal |

Every model opens with: **hypothesis → indicator → signal rules → backtest → failure modes** — auditable without running code.

## Engineering Principles

1. **No look-ahead.** Rolling windows as-of; total-return indices where dividends matter.
2. **Honest attribution.** Failure modes and fee drag documented, not hidden.
3. **Excel-lineage cross-checks** for models prototyped in spreadsheets first.
4. **Institutional data hygiene.** Explicit index-code lineage per signal.

## Author

**Fucheng Dou** — fund manager (index & multi-asset), 9 years buy-side. PKU M.Eng. CS + B.Sc. Physics.
