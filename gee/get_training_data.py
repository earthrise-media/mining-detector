"""
Georeferenced Training Data Pipeline
-----------------------------------

- Align Tile semantics with descarteslabs.geo.DLTile
- Save georeferenced GeoTIFFs per patch

Assumptions
- Patches are square and north-up (no rotation/skew)
- Resolution defaults to 10 m/pixel (S2)

"""
from dataclasses import dataclass, asdict
from datetime import datetime
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import geopandas as gpd
import imageio
import math
import numpy as np
import random
import shapely
from shapely.geometry import Point, box
from affine import Affine
import rasterio
from tqdm import tqdm

# External deps expected to exist in the environment
from descarteslabs.geo import DLTile
import gee
import utils


# -----------------------------
# Tile (DLTile-aligned wrapper)
# -----------------------------
@dataclass
class Tile:
    """DLTile-like tile defined by a center point and fixed pixel grid.

    Parameters
    ----------
    lat, lon : float
        Center coordinates in EPSG:4326.
    tilesize : int
        Width/height of the tile in pixels.
    resolution : float, default 10.0
        Pixel size in meters.

    Notes
    -----
    - CRS is derived from DLTile.from_latlon to match the source UTM/MGRS zone.
    - Bounds are snapped to the resolution grid and derived from center & tilesize.
    - Affine transform is north-up: translation(ulx, uly) * scale(res, -res)
    """

    lat: float
    lon: float
    tilesize: int
    resolution: float = 10.0

    # Computed attributes (filled post-init)
    crs: Optional[object] = None  # rasterio/pyproj CRS object
    x_center: Optional[float] = None
    y_center: Optional[float] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
    transform: Optional[Affine] = None
    geometry: Optional[shapely.geometry.base.BaseGeometry] = None  # WGS84 polygon

    def __post_init__(self) -> None:
        # 1) Snap CRS to DLTile's notion of the correct projected CRS (UTM/MGRS)
        dltile = DLTile.from_latlon(
            self.lat,
            self.lon,
            resolution=float(self.resolution),
            tilesize=int(self.tilesize),
            pad=0,
        )
        self.crs = dltile.crs  # pyproj.CRS (rasterio works with this)

        # 2) Project center to tile CRS
        point = Point(self.lon, self.lat)
        gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326").to_crs(self.crs)
        self.x_center = float(gdf.geometry.x.iloc[0])
        self.y_center = float(gdf.geometry.y.iloc[0])

        # 3) Compute square bounds from center, snapped to pixel grid
        half_m = (self.tilesize * self.resolution) / 2.0
        raw_minx = self.x_center - half_m
        raw_maxx = self.x_center + half_m
        raw_miny = self.y_center - half_m
        raw_maxy = self.y_center + half_m

        # Snap UL (minx for x, maxy for y) to resolution grid to avoid subpixel drift
        res = float(self.resolution)
        ulx = np.floor(raw_minx / res) * res
        uly = np.ceil(raw_maxy / res) * res

        # Construct transform and recompute exact bounds from transform + size
        self.transform = Affine.translation(ulx, uly) * Affine.scale(res, -res)
        width = height = int(self.tilesize)
        minx = ulx
        maxx = ulx + width * res
        maxy = uly
        miny = uly - height * res
        self.bounds = (minx, miny, maxx, maxy)

        # 4) Store WGS84 geometry for convenience/selection
        bbox_proj = box(minx, miny, maxx, maxy)
        self.geometry = (
            gpd.GeoSeries([bbox_proj], crs=self.crs).to_crs("EPSG:4326").iloc[0]
        )

    # Compatibility helpers
    @property
    def x(self) -> float:
        return self.x_center if self.x_center is not None else float("nan")

    @property
    def y(self) -> float:
        return self.y_center if self.y_center is not None else float("nan")

    def tile_id(self) -> str:
        """Stable, filename-safe identifier (UL pixel origin + metadata).

        Format: res{res}_px{ts}_ulx{ulx}_uly{uly}
        """
        assert self.transform is not None
        ulx, uly = self.transform * (0, 0)
        return f"res{int(self.resolution)}_px{int(self.tilesize)}_ulx{ulx:.2f}_uly{uly:.2f}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("geometry", None)
        # Keep transform as tuple for JSON friendliness
        if self.transform is not None:
            d["transform"] = tuple(self.transform)  # type: ignore
        return d


# ----------------------
# Training data pipeline
# ----------------------
class TrainingData:
    def __init__(
        self,
        patch_size: int,
        collection: str = "S2L2A",
        clear_threshold: float = 0.75,
        outdir: str | Path = "../data/training_patches",
    ) -> None:
        self.patch_size = int(patch_size)
        self.clear_threshold = float(clear_threshold)
        self.collection = collection
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def create_tiles(self, sampling_pts: gpd.GeoDataFrame) -> List[Tile]:
        if sampling_pts.crs is None:
            sampling_pts = sampling_pts.set_crs(epsg=4326)
        else:
            sampling_pts = sampling_pts.to_crs(epsg=4326)

        lats = sampling_pts.geometry.y.values
        lons = sampling_pts.geometry.x.values

        tiles: List[Tile] = []
        for lat, lon in zip(lats, lons):
            tile = Tile(float(lat), float(lon), tilesize=self.patch_size, resolution=10.0)
            tiles.append(tile)
        print(f"{len(tiles)} tiles to download")
        return tiles

    def _build_profile(self, height: int, width: int, count: int, dtype: str = "uint16") -> dict:
        return {
            "driver": "GTiff",
            "height": int(height),
            "width": int(width),
            "count": int(count),
            "dtype": dtype,
            "compress": "deflate",
        }

    def _to_rgb(self, pixels: np.ndarray) -> np.ndarray:
        """Convert HxWxB array to uint8 RGB for preview.
        Prefers S2 B4/B3/B2; falls back to first 3 bands if needed.
        """
        # Scale for S2 reflectance-like values
        scale = 3000.0
        # Sentinel-2 RGB indices if available
        if hasattr(self, "bandIds") and self.bandIds is not None:
            try:
                idx = [self.bandIds.index(b) for b in ["B4", "B3", "B2"]]
                rgb = np.clip(pixels[..., idx] / scale, 0, 1)
                return (rgb * 255).astype(np.uint8)
            except ValueError:
                pass
        # Fallback: take first 3 bands
        b = pixels.shape[-1]
        if b == 1:
            rgb = np.repeat(np.clip(pixels[..., [0]] / scale, 0, 1), 3, axis=-1)
        else:
            take = min(3, b)
            pad = 3 - take
            rgb = np.clip(pixels[..., :take] / scale, 0, 1)
            if pad:
                rgb = np.concatenate([rgb, np.zeros_like(rgb[..., :pad])],
                                     axis=-1)
            return (rgb * 255).astype(np.uint8)

    def get_patches(
        self,
        gdf: gpd.GeoDataFrame,
        max_tiles_per_png: int = 1000,
        tilesize: Optional[int] = None,
    ) -> List[Tile]:
        """Extract patches from a GeoDataFrame and write georeferenced GeoTIFFs.

        Parameters
        ----------
        gdf : GeoDataFrame
            Columns: ['source_file', 'label', 'start_date', 'end_date', 'split', 
                'geometry']; geometry CRS must be EPSG:4326 (lat/lon).
        max_tiles_per_png : int
            Randomly sample down to this count for the thumbnail PNG.
        tilesize : Optional[int]
            Override the constructor patch_size if provided.

        Returns
        -------
        tiles_written : List[Tile]
            The Tile objects written.
        """
        assert {'source_file','label','start_date','end_date','split','geometry'}.issubset(gdf.columns), (
            "gdf must have columns ['source_file','label','start_date','end_date','split','geometry']"
        )
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        tiles_written: List[Tile] = []

        for source_file, group in tqdm(gdf.groupby('source_file')):
            print(f"Retrieving data for {source_file}")
            rgb_tiles: List[np.ndarray] = []
            start_dates = group['start_date'].unique()
            end_dates = group['end_date'].unique()
            if len(start_dates) != 1 or len(end_dates) != 1:
                raise ValueError(
                    f"Inconsistent dates in group {source_file}: "
                    f"start_dates={start_dates}, end_dates={end_dates}"
                )
            start_date = str(start_dates[0])
            end_date = str(end_dates[0])
            
            ts = int(tilesize or self.patch_size)
            tiles = [
                Tile(float(row.geometry.y), float(row.geometry.x), ts)
                for _, row in group.iterrows()
            ]

            extractor = gee.GEE_Data_Extractor(
                start_date,
                end_date,
                clear_threshold=self.clear_threshold,
                collection=self.collection,
            )
            self.bandIds = extractor.bandIds

            data, tile_metadata = extractor.get_tile_data_concurrent(tiles)
            for (pixels, tile), (_, row) in zip(zip(data, tile_metadata),
                                                group.iterrows()):
                pixels = np.array(utils.pad_patch(pixels, ts))

                # Write GeoTIFF
                chw = np.moveaxis(pixels.astype('uint16'), -1, 0)  # (B,H,W)
                height, width, bands = pixels.shape
                profile = self._build_profile(height, width, bands,
                                              dtype='uint16')
                meta = profile | {'crs': tile.crs, 'transform': tile.transform}

                tif_name = (f"{self.collection}{tile.tile_id()}" +
                            f"_{row.start_date}_{row.end_date}.tif")
                out_dir = self.outdir / str(row.split) / str(row.label)
                out_dir.mkdir(parents=True, exist_ok=True)
                tif_path = out_dir / tif_name
                with rasterio.open(tif_path, 'w', **meta) as dst:
                    dst.write(chw)

                tiles_written.append(tile)

                # Collect RGB for thumbnail grid
                rgb = self._to_rgb(pixels)
                if rgb.shape[:2] != (ts, ts):
                    rgb = np.array(utils.pad_patch(rgb, ts))
                rgb_tiles.append(rgb)

            # Downsample thumbnails if too many
            if len(rgb_tiles) > max_tiles_per_png:
                rgb_tiles = random.sample(rgb_tiles, max_tiles_per_png)

            if len(rgb_tiles) == 0:
                continue

            # Make a squarish grid
            n = len(rgb_tiles)
            n_cols = int(math.ceil(math.sqrt(n)))
            n_rows = int(math.ceil(n / n_cols))
            H = n_rows * ts
            W = n_cols * ts
            grid = np.zeros((H, W, 3), dtype=np.uint8)

            for i, thumb in enumerate(rgb_tiles):
                r = i // n_cols
                c = i % n_cols
                grid[r*ts:(r+1)*ts, c*ts:(c+1)*ts, :] = thumb

            # Save PNG at the root output_dir (mixing train/val)
            png_name = f"{Path(source_file).stem}_{start_date}_{end_date}.png"
            png_path = self.outdir / png_name
            imageio.imwrite(png_path.as_posix(), grid)

        return tiles_written


def tile_geodataframe(tiles: Iterable[Tile], to_crs: str | int = 4326) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of tile footprints for inspection/QA."""
    geoms = [t.geometry for t in tiles]
    gdf = gpd.GeoDataFrame(
        {
            "tile_id": [t.tile_id() for t in tiles],
            "tilesize": [t.tilesize for t in tiles],
            "resolution": [t.resolution for t in tiles],
        },
        geometry=geoms,
        crs="EPSG:4326",
    )
    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    return gdf
