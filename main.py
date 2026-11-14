#!/usr/bin/env python3
"""btc-cent-forecast-01 entry (Python 3.13, standard library only).

Forecasts Bitcoin's US-dollar price 240 minutes after each checkpoint.
Reads every <id>.in case file in the input directory (one market,
4,320 one-minute closes over 72 hours, last close = checkpoint) and
writes <id>.out with one forecast per case.

Model (selected by local validation, not tuned on the practice score):
    forecast = last + 0.1 * (last - EMA60)
where last is the checkpoint close and EMA60 is the exponential moving
average of span 60 over all 4,320 closes (alpha = 2/61, seeded with the
first close, iterated chronologically, float64). The 10% tilt is the
smallest momentum correction that stayed non-negative on the largest
validation sample (163 walk-forward cases); larger tilts won the
practice window but lost out-of-sample, and medians, mean reversion,
drift anchors and OLS extrapolation all lost everywhere. The tilt is
deliberately tiny: the honest expectation is a coin flip against the
persistence baseline, with a small measured downside.

Runs offline, deterministic (no randomness, no clock, no network), and
far inside the 8-second cap: one linear pass over the prices per case.
"""

import json
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# Model parameters (exact, validated): EMA span 60, tilt 0.1.
# Do not re-tune these against the practice score: the selection
# discipline that produced them forbids post-hoc changes without a
# fresh, untouched confirmation set.
EMA_SPAN = 60
TILT = 0.1


def price5(x: float) -> float:
    """x kept on the grid's 5 decimals and inside the scorer's (0, 10M)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def forecast(prices: list) -> float:
    """One forecast from a case's closes, oldest first."""
    if not prices:
        raise ValueError("market has no prices")
    alpha = 2.0 / (EMA_SPAN + 1)
    ema = float(prices[0])
    for p in prices[1:]:
        ema = alpha * float(p) + (1.0 - alpha) * ema
    last = float(prices[-1])
    if last != last or last in (float("inf"), float("-inf")):
        raise ValueError("last price is not a finite number")
    return price5(last + TILT * (last - ema))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text(encoding="utf-8"))
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        prices = markets[0].get("prices")
        if not isinstance(prices, list) or not prices:
            raise ValueError(f"{path.name} holds no prices")
        guess = forecast(prices)
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        (OUTPUT_DIR / f"{case_id}.out").write_text(
            json.dumps(out) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
