#!/usr/bin/env python3
"""Fit a joint age + mileage price model and write data/price_model.json.

Why this exists: model-year and odometer are ~0.78 correlated in this market, so a
median-by-year curve silently blends "this car is old" with "this car was driven
hard". This fits both together (restricted cubic splines on each, on log price) so
the two effects are separated, then reuses the fitted surface to measure the hybrid
premium, the seller/region spreads and the cost-per-remaining-year sweet spot as
*adjusted* effects rather than raw medians.
"""
import glob
import json
from pathlib import Path

import numpy as np

import derive

ROOT = Path(__file__).resolve().parent
MODELS = ["RAV4", "CR-V", "Outback", "Forester", "RX"]
BASE_YEAR = 2026            # newest model year present; age = BASE_YEAR - model_year
KM_OFFSET = 5000.0          # keeps log() finite and tames the near-zero end
PRICE_FLOOR = 5000          # below this the listing is a lease/rental payment, not a sale price
RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- #
# restricted cubic spline basis (Harrell): k knots -> k-1 columns
# --------------------------------------------------------------------------- #
def rcs(x, knots):
    x = np.asarray(x, float)
    t = np.asarray(knots, float)
    k = len(t)
    denom = (t[-1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        def cube(u):
            return np.maximum(u, 0.0) ** 3
        term = (cube(x - t[j])
                - cube(x - t[-2]) * (t[-1] - t[j]) / (t[-1] - t[-2])
                + cube(x - t[-1]) * (t[-2] - t[j]) / (t[-1] - t[-2]))
        cols.append(term / denom)
    return np.column_stack(cols)


def knots_for(x, n=4):
    qs = {3: [.10, .50, .90], 4: [.05, .35, .65, .95], 5: [.05, .275, .50, .725, .95]}[n]
    return list(np.quantile(x, qs))


# --------------------------------------------------------------------------- #
# geography from postal FSA
# --------------------------------------------------------------------------- #
REGIONS = ["Montréal Island", "Laval", "North Shore", "South Shore",
           "Vaudreuil–Valleyfield", "Montérégie East", "Ottawa (ON)", "Other"]


def region_of(postal, province):
    p = (postal or "").replace(" ", "").upper()
    if province == "ON":
        return "Ottawa (ON)"
    if len(p) < 2:
        return "Other"
    a, b = p[0], p[1]
    if a == "H":
        return "Laval" if b == "7" else "Montréal Island"
    if a == "J":
        if b in "35":
            return "South Shore" if b == "3" else "North Shore"
        if b == "4":
            return "South Shore"
        if b == "7":
            return "North Shore"
        if b == "6":
            return "Vaudreuil–Valleyfield"
        if b == "2":
            return "Montérégie East"
    if a == "K":
        return "Ottawa (ON)"
    return "Other"


def fuel_class(f):
    f = f or ""
    if "Plug" in f:
        return "plugin"
    if "Hybrid" in f:
        return "hybrid"
    if f == "Electric":
        return "electric"
    return "gas"


# --------------------------------------------------------------------------- #
# OLS with HC1 robust standard errors
# --------------------------------------------------------------------------- #
def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    # HC1 sandwich
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = XtX_inv @ S @ XtX_inv * (n / max(n - p, 1))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, se, 1 - ss_res / ss_tot, resid


def cv_rmse_dollars(build, rows, y_log, price, folds=5):
    """K-fold CV, scored in dollars after a smearing back-transform."""
    idx = RNG.permutation(len(rows))
    parts = np.array_split(idx, folds)
    errs, ape = [], []
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[i] for i in range(folds) if i != f])
        Xtr, Xte = build(tr), build(te)
        beta, *_ = np.linalg.lstsq(Xtr, y_log[tr], rcond=None)
        smear = float(np.mean(np.exp(y_log[tr] - Xtr @ beta)))
        pred = np.exp(Xte @ beta) * smear
        errs.append((pred - price[te]) ** 2)
        ape.append(np.abs(pred - price[te]) / price[te])
    return float(np.sqrt(np.concatenate(errs).mean())), float(np.concatenate(ape).mean() * 100)


def main():
    src = sorted(glob.glob(str(ROOT / "data" / "autotrader_montreal_*.json")))[-1]
    raw = json.load(open(src, encoding="utf-8"))

    rows = [r for r in raw["listings"]
            if r.get("model_group") in MODELS
            and r.get("price_cad") and r["price_cad"] >= PRICE_FLOOR
            and r.get("model_year") and r.get("mileage_km") is not None]
    descs = derive.load_descriptions()
    for r in rows:
        r["_dt"] = derive.drivetrain(r, descs.get(r.get("id"), ""))[0]
        r["_col"] = derive.colour(r)
    used = [r for r in rows if r.get("offer_type") == "U"]
    new = [r for r in rows if r.get("offer_type") == "N"]

    age = np.array([BASE_YEAR - r["model_year"] for r in used], float)
    km = np.array([r["mileage_km"] for r in used], float)
    price = np.array([r["price_cad"] for r in used], float)
    y = np.log(price)
    logkm = np.log(km + KM_OFFSET)
    mdl = np.array([MODELS.index(r["model_group"]) for r in used])
    fuel = np.array([fuel_class(r.get("fuel")) for r in used])
    priv = np.array([1.0 if r.get("seller_type") == "PrivateSeller" else 0.0 for r in used])
    reg = np.array([region_of(r.get("postal_code"), r.get("province")) for r in used])
    dtv = np.array([r["_dt"] for r in used])
    col = np.array([r["_col"] for r in used])

    ak = knots_for(age, 4)
    kk = knots_for(logkm, 4)
    A = rcs(age, ak)
    K = rcs(logkm, kk)
    one = np.ones((len(used), 1))
    MD = np.column_stack([(mdl == i).astype(float) for i in range(1, len(MODELS))])  # RAV4 = reference
    FUEL_CATS = [f for f in ["hybrid", "plugin", "electric"] if (fuel == f).sum() >= 10]
    FU = np.column_stack([(fuel == f).astype(float) for f in FUEL_CATS])
    RG = np.column_stack([(reg == g).astype(float) for g in REGIONS[1:]])          # Montréal Island = reference
    DT = np.column_stack([(dtv == g).astype(float) for g in derive.DRIVETRAINS[1:]])   # AWD = reference
    COLS = [c for c in derive.COLOURS[1:] if (col == c).sum() >= 15]                   # Black = reference
    CO = np.column_stack([(col == c).astype(float) for c in COLS])
    PV = priv[:, None]
    MD_AGE = MD * age[:, None]                                                     # per-model depreciation slope
    # AutoTrader files RAV4 Primes (plug-in hybrids) under "Electric", so the
    # retention term is defined over ALL electrified powertrains, explicitly.
    ELECTRIFIED = np.isin(fuel, ["hybrid", "plugin", "electric"]).astype(float)
    HY_AGE = (ELECTRIFIED * age)[:, None]                                          # does electrification hold value?

    specs = {
        "S1 age only (linear)":            lambda i: np.column_stack([one[i], age[i, None].T if False else age[i][:, None]]),
        "S2 spline(age)":                  lambda i: np.column_stack([one[i], A[i]]),
        "S3 spline(age) + spline(km)":     lambda i: np.column_stack([one[i], A[i], K[i]]),
        "S4 + model/fuel/seller/region":   lambda i: np.column_stack([one[i], A[i], K[i], MD[i], FU[i], PV[i], RG[i]]),
        "S5 + drivetrain + colour":        lambda i: np.column_stack([one[i], A[i], K[i], MD[i], FU[i], PV[i], RG[i], DT[i], CO[i]]),
        "S6 + model×age, electrified×age": lambda i: np.column_stack([one[i], A[i], K[i], MD[i], FU[i], PV[i], RG[i], DT[i], CO[i], MD_AGE[i], HY_AGE[i]]),
    }

    print("=" * 74)
    print(f"USED LISTINGS WITH PRICE, YEAR AND ODOMETER: n = {len(used)}   (new: {len(new)})")
    print(f"corr(age, km) = {np.corrcoef(age, km)[0,1]:.3f}   "
          f"corr(age, log km) = {np.corrcoef(age, logkm)[0,1]:.3f}")
    print("=" * 74)

    # baseline: predict the median price of the same model+year (what the page did before)
    med = {}
    for m in range(len(MODELS)):
        for a in set(age[mdl == m]):
            v = price[(mdl == m) & (age == a)]
            if len(v) >= 3:
                med[(m, a)] = float(np.median(v))
    gmed = float(np.median(price))
    pred0 = np.array([med.get((m, a), gmed) for m, a in zip(mdl, age)])
    rmse0 = float(np.sqrt(((pred0 - price) ** 2).mean()))
    mape0 = float((np.abs(pred0 - price) / price).mean() * 100)
    print(f"\n{'specification':<34}{'CV RMSE':>12}{'CV MAPE':>10}{'R² (log)':>11}")
    print(f"{'S0 median by model+year (in-sample)':<34}{'$'+format(round(rmse0),','):>12}{mape0:>9.1f}%{'—':>11}")

    fits = {}
    for name, build in specs.items():
        X = build(np.arange(len(used)))
        beta, se, r2, resid = ols(X, y)
        rmse, mape = cv_rmse_dollars(build, used, y, price)
        fits[name] = (X, beta, se, r2, resid, rmse, mape)
        print(f"{name:<34}{'$'+format(round(rmse),','):>12}{mape:>9.1f}%{r2:>11.3f}")

    NAME = "S6 + model×age, electrified×age"
    X, beta, se, r2, resid, rmse, mape = fits[NAME]
    smear = float(np.mean(np.exp(resid)))
    sigma = float(np.std(resid, ddof=X.shape[1]))

    # ---- column bookkeeping (must mirror the S5 stack above) ----
    c = 0
    def take(n):
        nonlocal c
        s = c; c += n
        return list(range(s, s + n))
    i_int = take(1); i_age = take(A.shape[1]); i_km = take(K.shape[1])
    i_md = take(len(MODELS) - 1); i_fu = take(len(FUEL_CATS)); i_pv = take(1); i_rg = take(len(REGIONS) - 1)
    i_dt = take(len(derive.DRIVETRAINS) - 1); i_co = take(len(COLS))
    i_mdage = take(len(MODELS) - 1); i_hyage = take(1)
    assert c == X.shape[1], (c, X.shape[1])

    pct = lambda b: (np.exp(b) - 1) * 100

    print("\n" + "-" * 74)
    print("SEPARATING AGE FROM MILEAGE  (holding the other at its median)")
    print("-" * 74)
    med_age, med_km = float(np.median(age)), float(np.median(km))
    print(f"typical used listing: {med_age:.0f} years old, {med_km:,.0f} km, "
          f"{np.median(km/np.maximum(age,1)):,.0f} km/year")

    def surface(a_val, k_val, m=0, fu="gas", pv=0.0, rg="Montréal Island",
                dt="AWD", co="Black"):
        v = np.zeros(X.shape[1]); v[i_int[0]] = 1.0
        v[i_age] = rcs(np.array([a_val]), ak)[0]
        v[i_km] = rcs(np.array([np.log(k_val + KM_OFFSET)]), kk)[0]
        if m > 0: v[i_md[m-1]] = 1.0; v[i_mdage[m-1]] = a_val
        if fu in FUEL_CATS:
            v[i_fu[FUEL_CATS.index(fu)]] = 1.0
        if fu in ("hybrid", "plugin", "electric"): v[i_hyage[0]] = a_val
        v[i_pv[0]] = pv
        if rg != "Montréal Island": v[i_rg[REGIONS[1:].index(rg)]] = 1.0
        if dt in derive.DRIVETRAINS[1:]: v[i_dt[derive.DRIVETRAINS[1:].index(dt)]] = 1.0
        if co in COLS: v[i_co[COLS.index(co)]] = 1.0
        return float(np.exp(v @ beta) * smear), v

    print("\n  age effect at a CONSTANT 60,000 km (RAV4, gas, dealer, Montréal):")
    for a in [1, 3, 5, 7, 10]:
        p, _ = surface(a, 60000)
        print(f"    {a:>2} yr old -> {p:>9,.0f}")
    print("\n  mileage effect at a CONSTANT 5 years old:")
    for k in [20000, 60000, 100000, 150000, 200000]:
        p, _ = surface(5, k)
        print(f"    {k:>7,} km -> {p:>9,.0f}")

    # marginal effects at the median point
    p_ref, _ = surface(med_age, med_km)
    p_age1, _ = surface(med_age + 1, med_km)
    p_km10, _ = surface(med_age, med_km + 10000)
    p_both, _ = surface(med_age + 1, med_km + float(np.median(km/np.maximum(age,1))))
    print(f"\n  at the median point ({med_age:.0f} yr / {med_km:,.0f} km), holding the other fixed:")
    print(f"    +1 year, same odometer      : {p_age1 - p_ref:>+9,.0f}  ({(p_age1/p_ref-1)*100:+.1f}%)")
    print(f"    +10,000 km, same model year : {p_km10 - p_ref:>+9,.0f}  ({(p_km10/p_ref-1)*100:+.1f}%)")
    print(f"    +1 year AND its typical km  : {p_both - p_ref:>+9,.0f}  ({(p_both/p_ref-1)*100:+.1f}%)")
    print("    (the third line is what a median-by-year curve actually measures)")

    print("\n" + "-" * 74)
    print("ADJUSTED EFFECTS  (same age, same odometer, same model)")
    print("-" * 74)
    LAB = {"hybrid": "Hybrid", "plugin": "Plug-in hybrid",
           "electric": "Electric / plug-in (AutoTrader label)"}
    lab_fu = [LAB[f] for f in FUEL_CATS]
    for j, nm in zip(i_fu, lab_fu):
        print(f"  {nm:<28}{pct(beta[j]):>+7.1f}%   (± {pct(beta[j]+1.96*se[j])-pct(beta[j]):.1f} pp)")
    print(f"  {'Electrified × year of age':<28}{pct(beta[i_hyage[0]]):>+7.1f}%/yr "
          f"  (± {pct(beta[i_hyage[0]]+1.96*se[i_hyage[0]])-pct(beta[i_hyage[0]]):.1f} pp)")
    print(f"  {'Private seller':<28}{pct(beta[i_pv[0]]):>+7.1f}%   (± {pct(beta[i_pv[0]]+1.96*se[i_pv[0]])-pct(beta[i_pv[0]]):.1f} pp)")
    print("\n  region (vs Montréal Island):")
    for j, nm in zip(i_rg, REGIONS[1:]):
        n_r = int((reg == nm).sum())
        print(f"    {nm:<26}{pct(beta[j]):>+7.1f}%   n={n_r}")
    print("\n  drivetrain (vs AWD):")
    for j, nm in zip(i_dt, derive.DRIVETRAINS[1:]):
        print(f"    {nm:<26}{pct(beta[j]):>+7.1f}%   n={int((dtv == nm).sum())}")
    print("\n  colour (vs Black):")
    for j, nm in zip(i_co, COLS):
        print(f"    {nm:<26}{pct(beta[j]):>+7.1f}%   n={int((col == nm).sum())}")

    print("\n  model (vs RAV4), level and depreciation slope:")
    for j, ja, nm in zip(i_md, i_mdage, MODELS[1:]):
        print(f"    {nm:<26}{pct(beta[j]):>+7.1f}%   {pct(beta[ja]):>+6.2f}%/yr extra")

    # ---------------- sweet spot ---------------- #
    LIFE_KM, LIFE_YR = 320000.0, 22.0
    ANN_KM = float(np.median(km / np.maximum(age, 1)))
    print("\n" + "-" * 74)
    print(f"COST PER REMAINING YEAR  (life {LIFE_KM:,.0f} km / {LIFE_YR:.0f} yr, "
          f"{ANN_KM:,.0f} km/yr, upkeep $400 + $130/yr of age)")
    print("-" * 74)
    # Purchase price alone falls monotonically with age, so "price per remaining year"
    # has no interior optimum - it always says buy the oldest car on the lot. Running
    # costs are what create a real trade-off, so they are part of the metric and the
    # reader sets them: maint(age) = base + rate x age, charged for every year owned.
    MAINT_BASE, MAINT_RATE = 400.0, 130.0

    def tco(price, age, km, ann_km, base, rate, life_km=LIFE_KM, life_yr=LIFE_YR):
        rem = min(life_yr - age, (life_km - km) / ann_km)
        if rem < 2:
            return None, None
        n = rem
        upkeep = n * base + rate * (age * n + n * (n - 1) / 2)
        return (price + upkeep) / n, rem

    sweet = {}
    for m, nm in enumerate(MODELS):
        best, grid = None, []
        # only score cells the market actually offers: at least MIN_SUP real listings
        # of this model nearby. Without this the optimum drifts into empty corners of
        # the surface (a 2026 model year showing 40,000 km) where the fit extrapolates.
        MIN_SUP = 3
        m_age, m_km = age[mdl == m], km[mdl == m]
        for a in range(0, 16):
            for k in range(0, 260000, 10000):
                sup = int(((np.abs(m_age - a) <= 1.5) & (np.abs(m_km - k) <= 25000)).sum())
                if sup < MIN_SUP:
                    continue
                p, _ = surface(a, k, m=m)
                cpy, rem = tco(p, a, k, ANN_KM, MAINT_BASE, MAINT_RATE)
                if cpy is None:
                    continue
                cell = {"age": a, "km": k, "price": round(p), "rem": round(rem, 2),
                        "cpy": round(cpy), "n": sup}
                grid.append(cell)
                if best is None or cpy < best["cpy"]:
                    best = cell
        sweet[nm] = {"best": best, "grid": grid}
        print(f"  {nm:<10} cheapest: {best['age']} yr / {best['km']:,} km -> "
              f"${best['price']:,} + upkeep over {best['rem']:.1f} yr = ${best['cpy']:,}/yr")

    # ---------------- per-listing residuals ---------------- #
    pred_used = np.exp(X @ beta) * smear
    resid_pct = (price - pred_used) / pred_used * 100
    deal = {}
    for r, pv_, rp in zip(used, pred_used, resid_pct):
        deal[r["id"]] = [round(float(pv_)), round(float(rp), 1)]
    print(f"\nper-listing fair value written for {len(deal)} used listings "
          f"(median |residual| {np.median(np.abs(resid_pct)):.1f}%)")

    # ---------------- partial-effect curves for the page ---------------- #
    age_grid = [round(a, 2) for a in np.linspace(0, 15, 46)]
    km_grid = [int(v) for v in np.linspace(0, 250000, 51)]
    partial = {"age": {}, "km": {}, "retention": {}}
    new_med = {nm: float(np.median([r["price_cad"] for r in new if r["model_group"] == nm]))
               for nm in MODELS}
    for m, nm in enumerate(MODELS):
        partial["age"][nm] = [round(surface(a, med_km, m=m)[0]) for a in age_grid]
        partial["km"][nm] = [round(surface(med_age, k, m=m)[0]) for k in km_grid]
        partial["retention"][nm] = [round(surface(a, a * ANN_KM, m=m)[0] / new_med[nm] * 100, 1)
                                    for a in age_grid]

    out = {
        "n_used": len(used), "n_new": len(new), "price_floor": PRICE_FLOOR,
        "corr_age_km": round(float(np.corrcoef(age, km)[0, 1]), 3),
        "base_year": BASE_YEAR, "km_offset": KM_OFFSET,
        "age_knots": ak, "km_knots": kk,
        "beta": [float(b) for b in beta], "se": [float(s) for s in se],
        "smear": smear, "sigma": sigma, "r2": r2, "cv_rmse": rmse, "cv_mape": mape,
        "baseline": {"rmse": rmse0, "mape": mape0},
        "specs": [{"name": n, "rmse": f[5], "mape": f[6], "r2": f[3]} for n, f in fits.items()],
        "idx": {"int": i_int, "age": i_age, "km": i_km, "md": i_md, "fu": i_fu,
                "pv": i_pv, "rg": i_rg, "dt": i_dt, "co": i_co,
                "mdage": i_mdage, "hyage": i_hyage},
        "regions": REGIONS, "fuel_cats": FUEL_CATS, "models": MODELS,
        "drivetrains": derive.DRIVETRAINS, "colours": COLS, "colour_levels": ["Black"] + COLS,
        "drivetrain_n": {g: int((dtv == g).sum()) for g in derive.DRIVETRAINS},
        "colour_n": {c: int((col == c).sum()) for c in ["Black"] + COLS},
        "region_n": {g: int((reg == g).sum()) for g in REGIONS},
        "median_age": med_age, "median_km": med_km, "annual_km": ANN_KM,
        "new_median": new_med,
        "effects": {
            "fuel": [{"name": n, "pct": pct(beta[j]), "lo": pct(beta[j] - 1.96 * se[j]),
                      "hi": pct(beta[j] + 1.96 * se[j]), "n": int((fuel == f).sum())}
                     for j, n, f in zip(i_fu, lab_fu, ["hybrid", "plugin", "electric"])],
            "electrified_age": {"pct": pct(beta[i_hyage[0]]), "lo": pct(beta[i_hyage[0]] - 1.96 * se[i_hyage[0]]),
                           "hi": pct(beta[i_hyage[0]] + 1.96 * se[i_hyage[0]])},
            "private": {"pct": pct(beta[i_pv[0]]), "lo": pct(beta[i_pv[0]] - 1.96 * se[i_pv[0]]),
                        "hi": pct(beta[i_pv[0]] + 1.96 * se[i_pv[0]]), "n": int(priv.sum())},
            "drivetrain": [{"name": g, "pct": pct(beta[j]), "lo": pct(beta[j] - 1.96 * se[j]),
                            "hi": pct(beta[j] + 1.96 * se[j]), "n": int((dtv == g).sum())}
                           for j, g in zip(i_dt, derive.DRIVETRAINS[1:])],
            "colour": [{"name": c, "pct": pct(beta[j]), "lo": pct(beta[j] - 1.96 * se[j]),
                        "hi": pct(beta[j] + 1.96 * se[j]), "n": int((col == c).sum())}
                       for j, c in zip(i_co, COLS)],
            "region": [{"name": g, "pct": pct(beta[j]), "lo": pct(beta[j] - 1.96 * se[j]),
                        "hi": pct(beta[j] + 1.96 * se[j]), "n": int((reg == g).sum())}
                       for j, g in zip(i_rg, REGIONS[1:])],
            "model": [{"name": nm, "pct": pct(beta[j]), "slope": pct(beta[ja])}
                      for j, ja, nm in zip(i_md, i_mdage, MODELS[1:])],
            "marginal": {"per_year": p_age1 - p_ref, "per_10k_km": p_km10 - p_ref,
                         "confounded": p_both - p_ref, "ref_price": p_ref},
        },
        "partial": {"age_grid": age_grid, "km_grid": km_grid, **partial},
        "sweet": sweet,
        "sweet_assumptions": {"life_km": LIFE_KM, "life_yr": LIFE_YR, "annual_km": ANN_KM,
                              "maint_base": MAINT_BASE, "maint_rate": MAINT_RATE},
        "deal": deal,
    }
    dest = ROOT / "data" / "price_model.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {dest} ({dest.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
