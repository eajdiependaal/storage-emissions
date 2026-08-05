# Model Guide

This document explains what the model does, how the three files relate to
each other, and how to run it.

## What the model does

The model answers a single question: **does a grid-scale battery reduce or
increase CO₂ emissions when it operates in a day-ahead electricity market,
and how does the answer change depending on how it is operated?**

It does this in three steps:

1. **Reconstruct the emission signal.** For each hour of the year, it finds
   the power plant that set the electricity price. The CO₂ intensity of that
   plant is the *marginal emission intensity* (MEI). When the battery charges,
   it effectively causes that plant to generate more. When it discharges, it
   causes it to generate less. MEI is the conversion factor between MW and
   kg CO₂.

2. **Model the storage system.** The battery is described by physical
   parameters - power rating, energy capacity, charge/discharge
   efficiency, SOC bounds - which constrain what actions are physically
   possible at each hour, plus two operational parameters - a per-MWh
   degradation cost and a per-MWh embodied emission cost - which enter
   the dispatch objective in Layer 3 but do not affect what's physically
   feasible.

3. **Choose when to charge and discharge.** Five strategies are compared:
   a profit-maximising dispatch, an emission-minimising dispatch, two
   lexicographic dispatches (Lexico-E: minimise emissions and break ties
   by profit; Lexico-P: maximise profit and break ties by emissions - the
   same idea, opposite priority order), and a simple greedy heuristic that
   serves as a benchmark for the optimal DP. Both lexicographic orders are
   included rather than assuming one is redundant: whether they coincide
   depends on how often the *primary* signal ties exactly, which is a
   property of the specific price/MEI data and storage technology, not a
   guarantee of the algorithm.

## File structure

```
model/
├── market.py       Layer 1 - Merit order and MEI curve
├── storage.py      Layer 2 - Battery physical & operational parameters
├── algorithms.py   Layer 3 - Dispatch algorithms
├── run.py          Wires the three layers; edit this to change parameters
└── GUIDE.md        This file
```

Each layer is independent. You can read `market.py` without touching
`storage.py` or `algorithms.py`. The only file you need to edit to run a
different scenario is `run.py`.

## Layer 1 - market.py

**Input:** a CSV of power plants, daily commodity prices, hourly day-ahead
market prices.

**What it does:**

For each plant, it computes the Short-Run Marginal Cost using the formula
from Section 3.2 of the paper:

```
SRMC_i = fuel_price / η  +  CO2_price × emission_intensity  +  var_OM
```

where `emission_intensity = thermal_CO2_factor × (1 / η)`.

It then finds, for each hour, the most expensive plant that was dispatched
(the plant with the highest SRMC still at or below the market price). That
plant's CO₂ intensity is MEI(t).

**Output:** a time series of MEI values, one per hour.

**Key design choices documented in the code:**
- Biomass gets a zero emission factor (EU ETS Article 38(2)).
- Blast furnace gas uses its own thermal CO₂ factor (247.4 kg/GJ).
- When the price falls below all plants (negative prices), the cheapest
  plant is used as a fallback.

## Layer 2 - storage.py

**Input:** seven parameters.

**What it does:**

Defines a `StorageUnit` dataclass holding the unit's physical and
operational parameters, plus the SOC transition equations as methods that
follow mechanically from them - no decision logic lives here. The
equations are from Section 3.3:

```
Charging:    SOC_{t+1} = SOC_t + P_c × η_c          (Δt = 1 h)
Discharging: SOC_{t+1} = SOC_t − P_d / η_d
```

The class also calculates the maximum feasible power at any given SOC,
so that Layer 3 algorithms never violate the physical constraints.

Two of the seven parameters are per-MWh-discharged cost proxies for a
specific piece of hardware - one economic, one environmental:
`cycle_cost_eur_mwh` (degradation, EUR/MWh) and
`embodied_emission_kg_mwh` (embodied manufacturing GWP, kg CO2eq/MWh,
e.g. from an Environmental Product Declaration). Both are fixed
attributes of the unit; which one applies to a given dispatch run is
chosen at the call site (see Layer 3 below), because that depends on
whether the signal being optimised is price or MEI, not on the unit
itself.

**How to create a unit:**

```python
from storage import StorageUnit

unit = StorageUnit.from_roundtrip_efficiency(
    p_max_mw                 = 1.0,   # MW
    e_cap_mwh                = 1.0,   # MWh
    roundtrip_efficiency     = 0.85,  # → η_c = η_d = √0.85 ≈ 0.922
    cycle_cost_eur_mwh       = 30.0,  # EUR/MWh degradation proxy
    embodied_emission_kg_mwh = 20.3,  # kg CO2eq/MWh embodied-GWP proxy
)
```

## Layer 3 - algorithms.py

**Input:** a signal array (price or MEI), a `StorageUnit`, initial SOC,
and a `discharge_cost` matching the signal's units.

**What it does:**

Two functions, both returning `(actions, charge_mw, discharge_mw)`:

### `greedy(signal, unit, soc0, discharge_cost=0.0)`

Evaluates every possible (buy hour, sell hour) pair in the horizon.
Charge/discharge power (c_mw, d_mw) depend only on soc0, so they are fixed
for the whole search; selects the pair maximising the actual reward
`d_mw * (signal[sell] - discharge_cost) - c_mw * signal[buy]` - not the
bare price spread, since c_mw and d_mw generally differ. Executes at most
one charge-discharge cycle.

This is O(n²) in the horizon length. For a 36-hour horizon it evaluates
630 pairs. Simple, transparent, and sub-optimal.

### `dp(signal, unit, soc0, discharge_cost=0.0, secondary_signal=None)`

Solves the Bellman equation backwards over the full horizon, then recovers
the optimal policy forwards. This is the main algorithm in the paper.

```
V_t(s) = max over actions [ reward_t(s, a) + V_{t+1}(s') ]
```

Rewards:
```
Discharge: r = power × (signal − discharge_cost)
Charge:    r = −power × signal
Idle:      r = 0
```

`signal` and `discharge_cost` must share the same units - pass price
[EUR/MWh] with `unit.cycle_cost_eur_mwh` for the profit objective, or MEI
[kg CO2/kWh_e] with `unit.embodied_emission_kg_mwh / 1000` for the
emission objective. The DP itself doesn't know or care which objective it
is optimising - that choice lives entirely in what the caller passes in.

The SOC axis is discretised into 200 grid points. Future values between
grid points are recovered by linear interpolation (`np.interp`).

When `secondary_signal` is provided (the lexicographic strategy), the
primary objective decides the action. The secondary is used only when two
actions produce values within 1×10⁻⁶ of each other.

## run.py - the driver

This is the only file a user needs to interact with. It has a CONFIG
section at the top:

```python
START_DATE = "2022-01-01"
END_DATE   = "2022-12-31"

UNIT = StorageUnit.from_roundtrip_efficiency(
    p_max_mw                 = 1.016,   # MW
    e_cap_mwh                = 2.032,   # MWh
    roundtrip_efficiency     = 0.913,
    cycle_cost_eur_mwh       = 55.75,   # EUR/MWh degradation proxy
    embodied_emission_kg_mwh = 20.3,    # kg CO2eq/MWh embodied-GWP proxy
)
```

Change these values and run, from the `storageemissions/` directory:

```
python model/run.py
```

This runs the full pipeline end to end for `START_DATE`–`END_DATE`: loads
data, builds the MEI curve, runs all five dispatch strategies, writes the
per-strategy CSVs and MEI curve to `data/processed/model/`, and saves the
figures.

The `rolling_dispatch()` function in `run.py` handles the re-planning
structure: at 12:00 each day, the algorithm plans the next 36 hours.
The plan executes hour-by-hour. At the next 12:00, it re-plans with
updated prices. This mirrors the EPEX SPOT day-ahead gate closure.

## Input files

| File | Format | Key columns |
|---|---|---|
| `meritorder_NL.csv` | CSV | name, fuel, efficiency, var_om_eur_mwh |
| `commodity_prices_BZNL.csv` | CSV | Datum, CO2, Gas, Kolen |
| `BZ_NL.csv` | CSV | Date (interval format), Price |

The commodity prices file uses Dutch column names (Kolen = coal).
`run.py` renames them to English on load.

## Output files

All outputs go to `data/processed/model/`:

| File | Contents |
|---|---|
| `dispatch_profit_dp.csv` | Hourly dispatch under profit maximisation |
| `dispatch_emission_dp.csv` | Hourly dispatch under emission minimisation |
| `dispatch_lexico_emission_dp.csv` | Hourly dispatch, Lexico-E (emission primary, profit secondary) |
| `dispatch_lexico_profit_dp.csv` | Hourly dispatch, Lexico-P (profit primary, emission secondary) |
| `dispatch_profit_greedy.csv` | Hourly dispatch for the greedy benchmark |
| `mei_curve.csv` | MEI signal, marginal plant, and DAM price per hour |
| `summary.csv` | KPI comparison table |

Each dispatch CSV has columns: `action`, `charge_mw`, `discharge_mw`,
`soc_mwh`, `mei_kg_per_kwh`, `charge_emissions_kg`, `discharge_avoided_kg`,
`net_emissions_kg`, `net_emissions_tco2_cum`.

## How to adapt to a different bidding zone

1. Replace `meritorder_NL.csv` with a CSV for your zone. Keep the same
   column names. The `fuel` column is normalised automatically
   ("Gas CCGT", "gas", "CCGT" all map to the canonical key "gas").

2. Replace `commodity_prices_BZNL.csv` with prices for your zone. The
   CO2 column is the EU ETS price (common across zones). Replace Gas and
   Coal with zone-specific market prices if available.

3. Replace `BZ_NL.csv` with the EPEX or equivalent day-ahead prices for
   your zone.

4. Update `FUEL_PRICE_MAP` in `run.py` if your price file uses different
   column names.

No changes to `market.py`, `storage.py`, or `algorithms.py` are needed.
