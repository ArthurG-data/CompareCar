#!/usr/bin/env python3
"""Fields AutoTrader does not expose as structured data, recovered from what it does.

Drivetrain and colour are not fields in the search feed. Drivetrain is written into
the trim string and the dealer's description; colour is written into the listing URL
slug. Both derivations are shared by build_html.py and model_prices.py so the page
and the model always agree on what a listing is.
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# drivetrain
# --------------------------------------------------------------------------- #
_AWD = re.compile(r'\b(AWD|4WD|4X4|4RM|ALL[- ]?WHEEL|TRACTION\s+INT[EÉ]GRALE|INT[EÉ]GRALE|TI)\b', re.I)
_FWD = re.compile(r'\b(FWD|2WD|4X2|2RM|FRONT[- ]?WHEEL|TRACTION\s+AVANT)\b', re.I)
_TAG = re.compile(r'<[^>]+>')

# Subaru sells Outback and Forester in Canada with symmetrical AWD on every trim,
# and Toyota/Honda make AWD standard on their electrified compact SUVs here.
_ALWAYS_AWD_MODELS = {"Outback", "Forester"}


def load_descriptions():
    """id -> plain-text dealer description, from the saved raw search pages."""
    out = {}
    for f in glob.glob(str(ROOT / "data" / "raw" / "*.json")):
        try:
            for r in json.load(open(f, encoding="utf-8")):
                if r.get("id") and r.get("description"):
                    out[r["id"]] = _TAG.sub(" ", r["description"])
        except (ValueError, OSError):
            continue
    return out


def drivetrain(listing, description=""):
    """Returns (value, source) where value is 'AWD' | 'FWD' | 'Unknown'."""
    if listing.get("model_group") in _ALWAYS_AWD_MODELS:
        return "AWD", "spec"
    text = (listing.get("trim") or "") + " | " + (description or "")
    awd, fwd = bool(_AWD.search(text)), bool(_FWD.search(text))
    if fwd and not awd:
        return "FWD", "stated"
    if awd and not fwd:
        return "AWD", "stated"
    fuel = listing.get("fuel") or ""
    if "Hybrid" in fuel or fuel == "Electric":
        return "AWD", "spec"
    return "Unknown", "none"


# --------------------------------------------------------------------------- #
# colour — the token before "-cat_" in the listing URL
# --------------------------------------------------------------------------- #
_SLUG = re.compile(r'-([a-z]+)-cat_')
_MAIN = {"black": "Black", "white": "White", "grey": "Grey", "silver": "Silver",
         "blue": "Blue", "red": "Red", "green": "Green"}
_RARE = {"beige", "brown", "orange", "gold", "violet", "bronze", "yellow", "purple"}

COLOURS = ["Black", "White", "Grey", "Silver", "Blue", "Red", "Green", "Other", "Unknown"]
DRIVETRAINS = ["AWD", "FWD", "Unknown"]


def colour(listing):
    m = _SLUG.search(listing.get("url") or "")
    if not m:
        return "Unknown"
    tok = m.group(1)
    if tok in _MAIN:
        return _MAIN[tok]
    if tok in _RARE:
        return "Other"
    return "Unknown"          # the slot held a fuel or trim word: no colour published
