#!/usr/bin/env python3
"""
Ground area of EPSG:4326 raster pixels.

Our mask rasters are on a geographic grid, so a pixel is not a fixed area: its
width shrinks with cos(latitude). Across the Amazon basin (roughly +10 to -20
degrees) pixel area varies by about 6%, so counting pixels and multiplying by a
single constant biases any jurisdiction total, and biases it differently for
northern and southern jurisdictions.

Area depends only on the row (latitude band), never on the column, so the whole
correction is a one-dimensional lookup: one value per raster row, computed once.

Uses the exact WGS84 ellipsoidal formula for a cell bounded by two parallels and
two meridians, not a spherical approximation:

    A = dlon * (a^2 (1-e^2) / 2) * [ sin(phi)/(1 - e^2 sin^2(phi))
                                     + (1/2e) ln((1 + e sin(phi))/(1 - e sin(phi))) ]

evaluated between the cell's lower and upper latitude.

Typical use, for a jurisdiction total::

    from raster_area import zonal_area_ha
    areas = zonal_area_ha("mining_mask_2023.tif", jurisdictions_gdf, value=1)

Note that pixel counts are NOT comparable across grid regimes: rasters built
before the fixed-grid change have per-period resolutions differing by ~0.03%,
and the fixed grid uses 0.00009 deg. Always derive area from the raster's own
transform -- which is what everything here does -- rather than caching a
pixel-area constant.

Relation to the existing pipeline
---------------------------------
The website pipeline takes a different route to the same quantity: it
vectorizes the mask (``convert_rasters_to_vector.py``) and then measures
polygon area in a UTM projection
(``preprocess_mining_areas.calculate_area_using_utm``). That is a reasonable
approach -- UTM is conformal, so area distortion within a zone is only about
+/-0.2% -- and it is the right tool when areas must be attributed to
intersected polygons.

This module is the raster-side counterpart: it measures area without
vectorizing, so it is useful as an independent check on the vector path, and
for straight per-jurisdiction totals where no intersection geometry is needed.
Agreement between the two is a good regression test; they share no code and no
assumptions.
"""
from __future__ import annotations

import numpy as np

# WGS84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = 2 * _F - _F * _F
_E = np.sqrt(_E2)

M2_PER_HA = 10_000.0


def _authalic_term(lat_rad: np.ndarray) -> np.ndarray:
    """Inner term of the ellipsoidal zone-area integral."""
    s = np.sin(lat_rad)
    return s / (1.0 - _E2 * s * s) + (1.0 / (2.0 * _E)) * np.log(
        (1.0 + _E * s) / (1.0 - _E * s))


def cell_area_m2(lat_south: np.ndarray, lat_north: np.ndarray,
                 dlon_deg: float) -> np.ndarray:
    """Area (m^2) of cells spanning ``dlon_deg`` between the given parallels."""
    lat_south = np.asarray(lat_south, dtype=np.float64)
    lat_north = np.asarray(lat_north, dtype=np.float64)
    dlon_rad = np.deg2rad(dlon_deg)
    coefficient = _A * _A * (1.0 - _E2) / 2.0
    return dlon_rad * coefficient * (
        _authalic_term(np.deg2rad(lat_north)) - _authalic_term(np.deg2rad(lat_south)))


def pixel_area_m2_by_row(transform, height: int) -> np.ndarray:
    """Per-row pixel area (m^2) for a north-up EPSG:4326 raster.

    ``transform`` is an affine from rasterio; ``transform.e`` is negative for a
    north-up raster. Returns one value per row, top to bottom.
    """
    rows = np.arange(height, dtype=np.float64)
    lat_top = transform.f + rows * transform.e
    lat_bottom = lat_top + transform.e
    south, north = np.minimum(lat_top, lat_bottom), np.maximum(lat_top, lat_bottom)
    return cell_area_m2(south, north, abs(transform.a))


def area_ha(path, value=1, window=None) -> float:
    """Total ground area (ha) of pixels equal to ``value``.

    Reads row-blocks, so memory stays flat on basin-scale rasters.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        window = window or Window(0, 0, ds.width, ds.height)
        row_area = pixel_area_m2_by_row(ds.window_transform(window),
                                        int(window.height))
        total = 0.0
        for start in range(0, int(window.height), 1024):
            n = min(1024, int(window.height) - start)
            block = ds.read(1, window=Window(window.col_off,
                                             window.row_off + start,
                                             window.width, n))
            counts = (block == value).sum(axis=1)
            total += float(np.dot(counts, row_area[start:start + n]))
    return total / M2_PER_HA


def zonal_area_ha(path, zones, value=1, zone_field=None) -> dict:
    """Ground area (ha) of ``value`` pixels within each zone polygon.

    ``zones`` is a GeoDataFrame; results are keyed by ``zone_field`` if given,
    otherwise by positional index. Zones are rasterized onto the raster's own
    grid, so no resampling of the mask occurs.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window, from_bounds

    with rasterio.open(path) as ds:
        zones = zones.to_crs(ds.crs)
        keys = (list(zones[zone_field]) if zone_field
                else list(range(len(zones))))
        out = {}
        for key, geom in zip(keys, zones.geometry):
            if geom is None or geom.is_empty:
                out[key] = 0.0
                continue
            minx, miny, maxx, maxy = geom.bounds
            win = from_bounds(minx, miny, maxx, maxy,
                              ds.transform).round_offsets().round_lengths()
            win = win.intersection(Window(0, 0, ds.width, ds.height))
            if win.width <= 0 or win.height <= 0:
                out[key] = 0.0
                continue
            transform = ds.window_transform(win)
            mask = rasterize([(geom, 1)], out_shape=(int(win.height), int(win.width)),
                             transform=transform, fill=0, dtype="uint8").astype(bool)
            block = ds.read(1, window=win)
            row_area = pixel_area_m2_by_row(transform, int(win.height))
            counts = ((block == value) & mask).sum(axis=1)
            out[key] = float(np.dot(counts, row_area)) / M2_PER_HA
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ground area of mask pixels.")
    ap.add_argument("raster")
    ap.add_argument("--value", type=int, default=1)
    ap.add_argument("--zones", default=None,
                    help="Optional polygon file for per-zone totals.")
    ap.add_argument("--zone-field", default=None)
    args = ap.parse_args()

    if args.zones:
        import geopandas as gpd
        totals = zonal_area_ha(args.raster, gpd.read_file(args.zones),
                               value=args.value, zone_field=args.zone_field)
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"{k}\t{v:.2f}")
        print(f"TOTAL\t{sum(totals.values()):.2f} ha")
    else:
        print(f"{area_ha(args.raster, value=args.value):.2f} ha")
