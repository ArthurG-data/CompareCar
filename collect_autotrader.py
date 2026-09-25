#!/usr/bin/env python3
"""One-off collector: autotrader.ca listings for a few models around Montréal.

Fetches search-result pages, parses the embedded __NEXT_DATA__ JSON, flattens each
listing to a simple record and writes data/autotrader_montreal_<date>.json.
Raw per-page listings are kept in data/raw/ so flattening can be redone offline.
"""
import json
import re
import statistics
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

MODELS = [("toyota", "rav4"), ("honda", "cr-v"), ("subaru", "outback"), ("subaru", "forester"),
          ("lexus", "rx-350")]
RADIUS_KM = 100
LAT, LON = "45.50884", "-73.58781"  # Montréal
DELAY_S = 2.0
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0 Safari/537.36")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)


def build_url(make, model, page):
    return (f"https://www.autotrader.ca/cars/{make}/{model}/reg_qc/cit_montreal/"
            f"?cy=CA&damaged_listing=exclude&desc=0&sort=standard&atype=C"
            f"&zip=Montr%C3%A9al&lat={LAT}&lon={LON}&zipr={RADIUS_KM}"
            f"&ustate=N%2CU&size=20&page={page}")


def fetch_page(session, make, model, page):
    url = build_url(make, model, page)
    last_err = None
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                m = NEXT_DATA_RE.search(r.text)
                if not m:
                    raise RuntimeError("no __NEXT_DATA__ in page")
                pp = json.loads(m.group(1))["props"]["pageProps"]
                q = pp.get("pageQuery", {})
                if q.get("zip") != "Montréal, QC" or q.get("zipr") != str(RADIUS_KM) or q.get("offer") != "N,U":
                    sys.exit(f"ABORT: location/offer filter lost on {url}: {q}")
                return pp
            last_err = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
            last_err = repr(e)
        time.sleep(DELAY_S * (attempt + 2))
    sys.exit(f"ABORT: failed {url} after 3 attempts: {last_err}")


def parse_km(s):
    if not s:
        return None
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None


def flatten(l):
    v = l.get("vehicle") or {}
    p = l.get("price") or {}
    loc = l.get("location") or {}
    s = l.get("seller") or {}
    images = l.get("images") or []
    return {
        "id": l.get("id"),
        "cross_reference_id": l.get("crossReferenceId"),
        "url": l.get("url"),
        "make": v.get("make"),
        "model": v.get("model"),
        "model_group": v.get("modelGroup"),
        "variant": v.get("variant"),
        "trim": v.get("modelVersionInput"),
        "model_year": v.get("modelYear"),
        "offer_type": v.get("offerType"),
        "is_new": l.get("isOfferNew"),
        "price_cad": p.get("priceRaw"),
        "price_evaluation": p.get("priceEvaluation"),
        "is_conditional_price": p.get("isConditionalPrice"),
        "mileage_km": parse_km(v.get("mileageInKm")),
        "transmission": v.get("transmission"),
        "fuel": v.get("fuel"),
        "is_damaged": v.get("isCurrentlyDamaged"),
        "availability": l.get("availability"),
        "city": loc.get("city"),
        "province": loc.get("provinceCode"),
        "postal_code": loc.get("zip"),
        "street": loc.get("street"),
        "seller_type": s.get("type"),
        "seller_name": s.get("companyName"),
        "seller_id": s.get("id"),
        "search_result_type": l.get("searchResultType"),
        "ad_tier": l.get("adTier"),
        "cover_image": images[0] if images else None,
    }


def collect_model(session, make, model):
    key = f"{make} {model}"
    listings = {}
    pp = fetch_page(session, make, model, 1)
    reported = pp.get("numberOfResults")
    pages = pp.get("numberOfPages") or 1
    page = 1
    while True:
        batch = pp.get("listings") or []
        (RAW / f"{make}-{model}-p{page}.json").write_text(json.dumps(batch, ensure_ascii=False))
        for l in batch:
            if l.get("id"):
                listings.setdefault(l["id"], l)
        print(f"  {key}: page {page}/{pages} ({len(batch)} listings, {len(listings)} unique)", flush=True)
        if not batch or page >= pages:
            break
        page += 1
        time.sleep(DELAY_S)
        pp = fetch_page(session, make, model, page)
    return key, reported, list(listings.values())


def summarize(records):
    by = {}
    for r in records:
        by.setdefault(f"{r['make']} {r['model']}", []).append(r)
    print("\nmodel               cond   n     min      median   max")
    for key in sorted(by):
        for cond, label in (("N", "new"), ("U", "used")):
            prices = [r["price_cad"] for r in by[key] if r["offer_type"] == cond and r["price_cad"]]
            if prices:
                print(f"{key:<20}{label:<6}{len(prices):<5}{min(prices):>8,}{int(statistics.median(prices)):>10,}{max(prices):>9,}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "en-CA,en;q=0.9"})
    counts, records = {}, []
    for make, model in MODELS:
        print(f"Collecting {make} {model} …", flush=True)
        key, reported, raw = collect_model(session, make, model)
        flat = [flatten(l) for l in raw]
        counts[key] = {"reported": reported, "collected": len(flat)}
        records.extend(flat)
        time.sleep(DELAY_S)
    out = {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "autotrader.ca",
        "location": "Montréal, QC",
        "radius_km": RADIUS_KM,
        "offer_types": ["N", "U"],
        "damaged_excluded": True,
        "counts": counts,
        "listings": records,
    }
    out_path = DATA / f"autotrader_montreal_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nWrote {len(records)} listings to {out_path}")
    print("counts:", json.dumps(counts))
    summarize(records)


if __name__ == "__main__":
    main()
