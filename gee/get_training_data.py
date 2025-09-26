"""
Georeferenced Training Data Pipeline
-----------------------------------

- Helper routines to save georeferenced GeoTIFFs centered on input lat/lon points

Assumptions
- Image tiles are square and north-up (no rotation/skew)
- Resolution defaults to 10 m/pixel (S2)

"""
import os
from pathlib import Path
from typing import Iterable, List

import geopandas as gpd
import imageio
import math
import numpy as np
import random
from tqdm import tqdm

from gee import DataConfig, GEE_Data_Extractor
from tile_utils import CenteredTile, ensure_tile_shape

class TrainingData:
    """Manages extraction of training data tiles from GEE and write to disk."""

    def __init__(
        self,
        config: DataConfig,
        resolution: float = 10.0,
        outdir: Path = Path("training_data"),
    ):
        self.config = config
        self.resolution = float(resolution)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def make_rgb_preview(
        self, pixels: np.ndarray, scale: float = 3000.0) -> np.ndarray:
        """Convert multispectral tile to RGB preview (uint8).

        Parameters
        ----------
        pixels : np.ndarray
            HxWxB array of reflectance-like values.
        scale : float, default=3000.0
            Normalization scale factor (default chosen for Sentinel-2 L1C).
        """
        # Sentinel-2 preferred RGB bands if available
        try:
            idx = [self.config.bands.index(b) for b in ["B4", "B3", "B2"]]
            rgb = np.clip(pixels[..., idx] / scale, 0, 1)
            return (rgb * 255).astype(np.uint8)
        except ValueError:
            pass

        # Fallback: use first 3 bands (or repeat if single-band)
        b = pixels.shape[-1]
        if b == 1:
            rgb = np.repeat(np.clip(pixels[..., [0]] / scale, 0, 1),
                            3, axis=-1)
        else:
            take = min(3, b)
            pad = 3 - take
            rgb = np.clip(pixels[..., :take] / scale, 0, 1)
            if pad:
                rgb = np.concatenate([rgb, np.zeros_like(rgb[..., :pad])],
                                     axis=-1)

        return (rgb * 255).astype(np.uint8)

    def fetch_tiles_for_points(
        self, gdf: gpd.GeoDataFrame, max_tiles_per_png: int = 1000
        ) -> List[CenteredTile]:
        """Extract image data for a GeoDataFrame of lat/lon Points.

        Returns
        -------
        tiles_written : List[CenteredTile]

        Additional Outputs
        ------------------
        Written PNGs of RGB thumbnails, collected by 'source_file'
        """
        assert {'source_file','label','start_date','end_date','split','geometry'}.issubset(gdf.columns), (
            "gdf must have columns ['source_file','label','start_date','end_date','split','geometry']"
        )
        
        tiles_written = []
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

            extractor = GEE_Data_Extractor(
                start_date,
                end_date,
                config=self.config
            )
            
            tiles = []
            for idx, row in group.iterrows():
                tile = CenteredTile(
                    lat=row.geometry.y, lon=row.geometry.x,
                    tilesize=self.config.tilesize,
                    resolution=self.resolution)
                tiles.append(tile)

            data = extractor.get_tile_data_concurrent(tiles)
            
            for (pixels, tile), (_, row) in zip(zip(data, tiles),
                                                group.iterrows()):
                outdir = self.outdir / str(row.split) / str(row.label)
                extractor.save_tile(pixels, tile, outdir)
                tiles_written.append(tile)

                rgb = self.make_rgb_preview(pixels)
                rgb_tiles.append(rgb)

            png_fname = (f"{Path(source_file).stem}_{self.config.collection}"
                         f"_clear{self.config.clear_threshold}" +
                         f"_{start_date}_{end_date}.png")
            png_path = self.outdir / png_fname
            write_thumbnail_grid(rgb_tiles, png_path, self.config.tilesize,
                                 max_tiles=max_tiles_per_png)

        return tiles_written

def write_thumbnail_grid(thumbs: list[np.ndarray], out_path: Path,
                         tilesize: int, max_tiles: int = 1000) -> None:
    """Downsample, arrange, and save a grid of RGB thumbnails."""
    if len(thumbs) == 0:
        return

    if len(thumbs) > max_tiles:
        thumbs = random.sample(thumbs, max_tiles)

    n = len(thumbs)
    n_cols = int(math.ceil(math.sqrt(n)))
    n_rows = int(math.ceil(n / n_cols))
    ts = tilesize
    H = n_rows * ts
    W = n_cols * ts
    grid = np.zeros((H, W, 3), dtype=np.uint8)

    for i, thumb in enumerate(thumbs):
        r = i // n_cols
        c = i % n_cols
        grid[r*ts:(r+1)*ts, c*ts:(c+1)*ts, :] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path.as_posix(), grid)
