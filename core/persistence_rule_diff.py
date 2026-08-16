"""Is D+ actually like the current product, or does it just score the same?"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

sys.path.insert(0, "/home/zu/Genome/mining-detector/core")
from persistence import Period, PersistenceConfig, compute_onsets
from postprocess import centroid_key

ROOT = Path("/home/zu/Genome/mining-detector")
DATA = ROOT / "data/training_patches2026-05-04T09:47"
M = ROOT / "data/outputs/48px_v4.10b-18d-20g-21a-22bc-ensemble"
STEM = "Amazon_ACA_48px_v4.10b-18d-20g-21a-22bc-ensemble_0.40"
YEARS = list(range(2018, 2026))
QUARTERS = ["Q125", "Q225", "Q325", "Q425", "Q126", "Q226"]


def load(period, t_main, t_iso):
    tag = f"t{t_main:g}_d5_3km_t-iso{t_iso:g}"
    g = gpd.read_file(M / f"postprocessed_{tag}" / f"{STEM}_{period.date_span}_{tag}.geojson",
                      columns=["confidence"])
    kx, ky = centroid_key(g)
    return list(zip(kx, ky)), list(g.geometry), g["confidence"].to_numpy()


chips_g, chips_y = [], []
for split in ("val", "test2", "test3"):
    for lab in (0, 1):
        for fp in sorted(DATA.glob(f"*/{split}/{lab}/*.tif")):
            with rasterio.open(fp) as src:
                chips_g.append(box(*src.bounds))
            chips_y.append(lab)
chips = gpd.GeoDataFrame({"y": chips_y}, geometry=chips_g, crs="EPSG:4326")

geom_of, conf_of, det43, det55 = {}, {}, {}, {}
for p in [Period(y) for y in YEARS] + [Period.parse(t) for t in QUARTERS]:
    k, g, c = load(p, 0.43, 0.75)
    det43[p] = set(k)
    for kk, gg, cc in zip(k, g, c):
        geom_of.setdefault(kk, gg)
        conf_of.setdefault(kk, cc)
    if p.is_annual:
        k5, g5, c5 = load(p, 0.55, 0.8)
        det55[p] = set(k5)
        for kk, gg, cc in zip(k5, g5, c5):
            geom_of.setdefault(kk, gg); conf_of.setdefault(kk, cc)

annual = [Period(y) for y in YEARS]
cfgC = PersistenceConfig(window=3)
cfgD = PersistenceConfig(window=3, use_quarterly_witnesses=True, early_confirm=True)

prov = set().union(*[det55[p] for p in annual if not cfgC.window_is_closed(p, annual)])
C = set(compute_onsets({p: det43[p] for p in annual}, cfgC)) | prov
Dp = set(compute_onsets(det43, cfgD)) | prov
t055 = set().union(*det55.values())

print(f"{'layer':<10}{'patches':>10}")
for n, s in (("C", C), ("D+", Dp), ("or_t055", t055)):
    print(f"{n:<10}{len(s):>10,}")
print()
print(f"D+ vs or_t055: shared {len(Dp & t055):,}  "
      f"D+ only {len(Dp - t055):,}  or_t055 only {len(t055 - Dp):,}  "
      f"Jaccard {len(Dp & t055)/len(Dp | t055):.3f}")
print(f"D+ vs C:       shared {len(Dp & C):,}  D+ only {len(Dp - C):,}")
extra = Dp - C
if extra:
    print(f"  median confidence of D+'s additions over C: "
          f"{np.median([conf_of[k] for k in extra if k in conf_of]):.3f}")


def hit_chips(keys):
    gdf = gpd.GeoDataFrame(geometry=[geom_of[k] for k in keys if k in geom_of],
                           crs="EPSG:4326")
    h = gpd.sjoin(chips[["geometry"]], gdf[["geometry"]], how="inner",
                  predicate="intersects")
    return set(h.index.unique())


hC, hD, h55 = hit_chips(C), hit_chips(Dp), hit_chips(t055)
neg = set(chips.index[chips.y == 0])
fpC, fpD, fp55 = hC & neg, hD & neg, h55 & neg
print()
print(f"false-positive chips: C {len(fpC)}  D+ {len(fpD)}  or_t055 {len(fp55)}")
print(f"  D+ and or_t055 share {len(fpD & fp55)} of them; "
      f"D+ only {len(fpD - fp55)}, or_t055 only {len(fp55 - fpD)}")
print(f"  D+ vs C: shared {len(fpD & fpC)}, D+ adds {len(fpD - fpC)}")
