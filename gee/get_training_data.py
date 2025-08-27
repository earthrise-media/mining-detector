"""
Georeferenced Training Data Pipeline
-----------------------------------

- Save georeferenced GeoTIFFs per patch

Assumptions
- Patches are square and north-up (no rotation/skew)
- Resolution defaults to 10 m/pixel (S2)

"""
import os
from pathlib import Path
from typing import Iterable, List

from descarteslabs.geo import DLTile
import geopandas as gpd
import imageio
import math
import numpy as np
import random
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

import gee
import utils

class TrainingData:
    def __init__(
        self,
        patch_size: int,
        resolution: float = 10.0,
        pad: float = 0.0,
        collection: str = "S2L1C",
        clear_threshold: float = 0.75,
        outdir: str | Path = "../data/training_patches",
    ) -> None:
        self.patch_size = int(patch_size)
        self.resolution = float(resolution)
        self.pad = float(pad)
        self.clear_threshold = float(clear_threshold)
        self.collection = collection
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        
    def create_tiles(self, gdf: gpd.GeoDataFrame) -> list[DLTile]:
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        tiles: list[DLTile] = [
            DLTile.from_latlon(
                float(row.geometry.y),
                float(row.geometry.x),
                resolution=self.resolution,
                tilesize=self.patch_size,
                pad=self.pad,
            )
            for _, row in gdf.iterrows()
            ]
        print(f"{len(tiles)} tiles to created")
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


    def get_patches(self, gdf: gpd.GeoDataFrame, max_tiles_per_png: int = 1000) -> List[DLTile]:
        """Extract patches from a GeoDataFrame and write georeferenced GeoTIFFs.

        Parameters
        ----------
        gdf : GeoDataFrame
            Columns: ['source_file', 'label', 'start_date', 'end_date', 'split', 
                'geometry']; geometry CRS must be EPSG:4326 (lat/lon).
        max_tiles_per_png : int
            Randomly sample down to this count for the thumbnail PNG.

        Returns
        -------
        tiles_written : List[DLTile]
            The DLTile objects written.
        """
        assert {'source_file','label','start_date','end_date','split','geometry'}.issubset(gdf.columns), (
            "gdf must have columns ['source_file','label','start_date','end_date','split','geometry']"
        )
        tiles_written: List[DLTile] = []

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
            ts = self.patch_size
            
            tiles = self.create_tiles(group)

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
                minx, miny, maxx, maxy = tile.bounds
                transform = from_origin(
                    minx,  
                    maxy,  
                    tile.resolution,  # pixel width
                    -tile.resolution   # pixel height
                    )
                meta = profile | {'crs': tile.crs, 'transform': transform}

                tif_name = (f"{self.collection}_clear{self.clear_threshold}" +
                            f"_{tile.key}_{row.start_date}_{row.end_date}.tif")
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

        png_fname = (f"{Path(source_file).stem}_{self.collection}" +
                     f"_clear{self.clear_threshold}" +
                     f"_{start_date}_{end_date}.png")
        png_path = self.outdir / png_fname
        write_thumbnail_grid(rgb_tiles, png_path, ts,
                             max_tiles=max_tiles_per_png)

        return tiles_written


def tile_geodataframe(tiles: Iterable[DLTile], to_crs: str | int = 4326) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of tile footprints for inspection/QA."""
    geoms = [t.geometry for t in tiles]
    gdf = gpd.GeoDataFrame(
        {
            "tile_id": [t.key for t in tiles],
            "tilesize": [t.tilesize for t in tiles],
            "resolution": [t.resolution for t in tiles],
        },
        geometry=geoms,
        crs="EPSG:4326",
    )
    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    return gdf

def write_thumbnail_grid(thumbs: list[np.ndarray], out_path: Path, ts: int,
                         max_tiles: int = 1000) -> None:
    """Downsample, arrange, and save a grid of RGB thumbnails."""
    if len(thumbs) == 0:
        return

    if len(thumbs) > max_tiles:
        thumbs = random.sample(thumbs, max_tiles)

    n = len(thumbs)
    n_cols = int(math.ceil(math.sqrt(n)))
    n_rows = int(math.ceil(n / n_cols))
    H = n_rows * ts
    W = n_cols * ts
    grid = np.zeros((H, W, 3), dtype=np.uint8)

    for i, thumb in enumerate(thumbs):
        r = i // n_cols
        c = i % n_cols
        grid[r*ts:(r+1)*ts, c*ts:(c+1)*ts, :] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path.as_posix(), grid)
