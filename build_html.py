#!/usr/bin/env python3
"""Pack the collected listings into analysis.html (template.html + embedded data)."""
import glob
import json
from pathlib import Path

import derive

ROOT = Path(__file__).resolve().parent
MODELS = ["RAV4", "CR-V", "Outback", "Forester", "RX"]
PRICE_FLOOR = 5000   # below this the listing is a lease/rental payment, not a sale price
MAKE_OF = {"RAV4": "Toyota", "CR-V": "Honda", "Outback": "Subaru", "Forester": "Subaru", "RX": "Lexus"}


def norm_fuel(f):
    if not f:
        return "Other"
    if "Hybrid" in f:
        return "Plug-in / Hybrid" if "Plug" in f else "Hybrid"
    if f == "Electric":
        return "Electric"
    if f == "Gasoline":
        return "Gasoline"
    return "Other"


def main():
    src = sorted(glob.glob(str(ROOT / "data" / "autotrader_montreal_*.json")))[-1]
    raw = json.load(open(src, encoding="utf-8"))
    listings = raw["listings"]

    descs = derive.load_descriptions()
    dt_src = {"spec": 0, "stated": 0, "none": 0}

    mpath = ROOT / "data" / "price_model.json"
    pmodel = json.load(open(mpath, encoding="utf-8")) if mpath.exists() else None
    deal = pmodel["deal"] if pmodel else {}

    cities, sellers, fuels, trims = [], [], [], []
    drivetrains, colours = list(derive.DRIVETRAINS), list(derive.COLOURS)
    idx = {"city": {}, "seller": {}, "fuel": {}, "trim": {}}

    def intern(kind, store, val):
        val = val or "—"
        if val not in idx[kind]:
            idx[kind][val] = len(store)
            store.append(val)
        return idx[kind][val]

    rows, dropped_price, dropped_year = [], 0, 0
    for r in listings:
        mg = r.get("model_group")
        if mg not in MODELS:
            continue
        price = r.get("price_cad")
        if not price or price < PRICE_FLOOR:   # $1 = "price on request"; sub-floor = lease/rental payment
            dropped_price += 1
            continue
        year = r.get("model_year")
        if not year:
            dropped_year += 1
            continue
        trim = (r.get("trim") or "").strip()[:70]
        dt, dsrc = derive.drivetrain(r, descs.get(r.get("id"), ""))
        dt_src[dsrc] += 1
        rows.append([
            MODELS.index(mg),
            year,
            price,
            r.get("mileage_km"),
            1 if r.get("offer_type") == "N" else 0,
            intern("fuel", fuels, norm_fuel(r.get("fuel"))),
            1 if r.get("seller_type") == "PrivateSeller" else 0,
            intern("city", cities, (r.get("city") or "").title()),
            intern("seller", sellers, r.get("seller_name")),
            intern("trim", trims, trim),
            r.get("url") or "",
            *(deal.get(r.get("id")) or [None, None]),   # fitted fair value, residual %
            drivetrains.index(dt),
            colours.index(derive.colour(r)),
            r.get("id"),
        ])

    payload = {
        "models": MODELS,
        "makes": [MAKE_OF[m] for m in MODELS],
        "fuels": fuels,
        "cities": cities,
        "sellers": sellers,
        "trims": trims,
        "drivetrains": drivetrains,
        "colours": colours,
        "rows": rows,
        "model": pmodel,
        "meta": {
            "collected_at": raw["collected_at"],
            "location": raw["location"],
            "radius_km": raw["radius_km"],
            "total_collected": len(listings),
            "excluded_price_on_request": dropped_price,
            "price_floor": PRICE_FLOOR,
            "excluded_no_year": dropped_year,
            "drivetrain_stated": dt_src["stated"],
            "drivetrain_spec": dt_src["spec"],
            "drivetrain_unknown": dt_src["none"],
        },
    }

    tpl = (ROOT / "template.html").read_text(encoding="utf-8")
    out = tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if pmodel is None:
        raise SystemExit("run model_prices.py first - data/price_model.json is missing")
    dest = ROOT / "analysis.html"
    dest.write_text(out, encoding="utf-8")
    print(f"{len(rows)} rows embedded  (dropped: {dropped_price} price-on-request, {dropped_year} no-year)")
    print(f"wrote {dest}  ({dest.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
