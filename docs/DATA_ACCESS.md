# Data access — commodity price series

`data/raw/BZ_NL/commodity_prices_BZNL.csv` ships in this repository with
three columns removed because the underlying data is licensed and cannot
be redistributed. This document lists what was removed, where to obtain
it, and how to rebuild the file.

## What ships, what doesn't

The shipped file has two columns: `Datum` (date) and `blast furnace gas`
(present for schema compatibility; it is zero for the full period — see
`docs/DATA_PROVENANCE.md`). Three columns are missing and must be added
before running the model:

| Column | Series | Source | License reason |
|---|---|---|---|
| `CO2` | EU ETS EUA auction clearing price, EUR/tCO2 | EEX (European Energy Exchange), primary auction reports | EEX's terms of use prohibit systematic republication of a substantial amount of data without permission; 8 years of daily data does not qualify as an incidental/insubstantial excerpt |
| `Gas` | Dutch TTF natural gas front-month futures, EUR/MWh | Investing.com | Investing.com's terms prohibit redistribution of its historical data exports |
| `Kolen` | API2 CIF ARA coal futures, EUR/MWh | OilpriceAPI (Argus-McCloskey API2 assessment) | Argus/McCloskey data is commercially licensed |

Once you have rebuilt the CSV, run `python support/validate_commodity_data.py`
to check row counts, date coverage, and per-year mean/min/max against the
paper's original dataset before running the model.

## Step-by-step reconstruction

### 1. `CO2` — EUA auction price

1. Go to the EEX primary auction report page:
   https://www.eex.com/en/market-data/environmentals/eu-ets-auctions
2. Download the "Emission Spot Primary Market Auction Report" for each
   year 2018-2025 (EEX publishes one report per year covering all
   auctions that year; the report includes the auction clearing price
   per auction date).
3. For each calendar day without an auction (auctions do not run every
   day), forward-fill from the most recent auction date — this matches
   how the paper's dataset was built (daily series, auction-day
   resolution).
4. Build a `Datum, CO2` series (date, EUR/tCO2), 2018-01-01 to
   2025-12-31.

### 2. `Gas` — TTF front-month futures

1. Obtain "Dutch TTF Natural Gas Futures" daily historical data,
   2018-01-01 to 2025-12-31, in EUR/MWh. This series is commonly
   available from Investing.com (subscription/ToS-gated for bulk
   export) or from a data vendor with equivalent TTF front-month
   futures coverage (e.g. ICE, Refinitiv, Bloomberg — any source
   reporting the same instrument will do; the paper does not depend on
   the vendor, only on the series).
2. Build a `Datum, Gas` series (date, EUR/MWh).

### 3. `Kolen` — API2 CIF ARA coal futures

1. Obtain "Coal (API2) CIF ARA (Argus-McCloskey)" daily historical
   futures data, 2018-01-01 to 2025-12-31, in EUR/MWh (thermal
   equivalent). The paper's own series was obtained via
   [OilpriceAPI](https://oilpriceapi.com/) (its coal endpoint reports the
   Argus-McCloskey API2 assessment); Argus Media directly, or any other
   data vendor carrying the same API2 assessment, is an equally valid
   source — the paper does not depend on the vendor, only on the series.
2. Build a `Datum, Kolen` series (date, EUR/MWh).

### 4. Assemble and validate

Merge the three series with the shipped `Datum, blast furnace gas`
columns into one CSV with the exact header:

```
Datum,CO2,Gas,Kolen,blast furnace gas
```

(column order does not matter to the model; the loader reads by name).
Save as `data/raw/BZ_NL/commodity_prices_BZNL.csv`, replacing the shipped
placeholder. Then run:

```bash
python support/validate_commodity_data.py
```

This checks row count, date coverage, and per-year mean/min/max for each
of `CO2`, `Gas`, `Kolen` against the values from the original dataset
used in the paper. A pass means your reconstruction is numerically
equivalent for modeling purposes even if the exact vendor/method differs
from the original compilation.

## Day-ahead prices (`BZ_NL.csv`) — no action needed

`BZ_NL.csv` (EPEX day-ahead market prices, Netherlands bidding zone)
ships as-is. It is ENTSO-E Transparency Platform data, published under
CC-BY 4.0 for the list of series marked open for reuse, which day-ahead
prices are. Attribution: "Contains data from the ENTSO-E Transparency
Platform (transparency.entsoe.eu), CC-BY 4.0."
