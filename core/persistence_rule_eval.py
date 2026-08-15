"""Test-set metrics for confirmation recipes A-D, on the persistence_evaluation protocol.

Chip is predicted positive iff it intersects any patch in the layer. Pooled over
val + test2 + test3, matching model_evaluation.ipynb section 3.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "/home/zu/Genome/mining-detector/core")
from persistence import Period, PersistenceConfig, compute_onsets
from postprocess import centroid_key

ROOT = Path("/home/zu/Genome/mining-detector")
DATA = ROOT / "data/training_patches2026-05-04T09:47"
M = ROOT / "data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble"
STEM = "Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40"
SPLITS = ["val", "test2", "test3"]
YEARS = list(range(2018, 2026))
QUARTERS = ["Q125", "Q225", "Q325", "Q425", "Q126", "Q226"]


def load_chips():
    geoms, labels = [], []
    for split in SPLITS:
        for lab in (0, 1):
            for fp in sorted(DATA.glob(f"*/{split}/{lab}/*.tif")):
                with rasterio.open(fp) as src:
                    geoms.append(box(*src.bounds))
                labels.append(lab)
    return gpd.GeoDataFrame({"y": labels}, geometry=geoms, crs="EPSG:4326")


def load(period, t_main, t_iso):
    tag = f"t{t_main:g}_d5_3km_t-iso{t_iso:g}"
    p = M / f"postprocessed_{tag}" / f"{STEM}_{period.date_span}_{tag}.geojson"
    g = gpd.read_file(p, columns=["confidence"])
    kx, ky = centroid_key(g)
    return list(zip(kx, ky)), list(g.geometry)


def main():
    chips = load_chips()
    print(f"{len(chips):,} chips, {int(chips.y.sum()):,} positive")

    geom_of, det43, det55 = {}, {}, {}
    for p in [Period(y) for y in YEARS] + [Period.parse(t) for t in QUARTERS]:
        keys, geoms = load(p, 0.43, 0.75)
        det43[p] = set(keys)
        for k, g in zip(keys, geoms):
            geom_of.setdefault(k, g)
        if p.is_annual:
            k55, g55 = load(p, 0.55, 0.8)
            det55[p] = set(k55)
            for k, g in zip(k55, g55):
                geom_of.setdefault(k, g)

    annual = [Period(y) for y in YEARS]
    rules = {
        "A  window2 annual": PersistenceConfig(window=2),
        "B  window2 +quarters": PersistenceConfig(window=2, use_quarterly_witnesses=True),
        "C  window3 annual (classic)": PersistenceConfig(window=3),
        "D  window3 +quarters": PersistenceConfig(window=3, use_quarterly_witnesses=True),
        "D+ window3 +quarters, early": PersistenceConfig(
            window=3, use_quarterly_witnesses=True, early_confirm=True),
    }

    layers = {}
    # references
    layers["or_t055_2025  (current production)"] = set().union(*det55.values())
    layers["or_t043_2025  (loose, no persistence)"] = set().union(
        *[det43[p] for p in annual])

    for name, cfg in rules.items():
        detected = det43 if cfg.use_quarterly_witnesses else {p: det43[p] for p in annual}
        onsets = compute_onsets(detected, cfg)
        confirmed = set(onsets)
        # provisional t0.55 for annual periods whose window has not closed
        prov = set()
        for p in annual:
            if not cfg.window_is_closed(p, annual):
                prov |= det55[p]
        layers[f"{name}"] = confirmed | (prov - confirmed)
        layers[f"{name}  [confirmed only]"] = confirmed

    rows = []
    for name, keys in layers.items():
        gdf = gpd.GeoDataFrame(geometry=[geom_of[k] for k in keys if k in geom_of],
                               crs="EPSG:4326")
        hits = gpd.sjoin(chips[["geometry"]], gdf[["geometry"]],
                         how="inner", predicate="intersects")
        pred = chips.index.isin(hits.index.unique()).astype(int)
        tn, fp, fn, tp = confusion_matrix(chips["y"], pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(layer=name, patches=len(keys), TP=tp, FP=fp, FN=fn, TN=tn,
                         precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4)))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.to_csv(Path(__file__).parent / "rule_metrics.csv", index=False)


if __name__ == "__main__":
    main()
