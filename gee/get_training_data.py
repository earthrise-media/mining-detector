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

import geopandas as gpd
import imageio
import math
import numpy as np
import random
import rasterio
from tqdm import tqdm

import gee
from tile_utils import CenteredTile, pad_patch

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
        
    def _points_to_tiles(self, gdf: gpd.GeoDataFrame) -> list[CenteredTile]:
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        tiles: list[CenteredTile] = [
            CenteredTile(
                float(row.geometry.y),
                float(row.geometry.x),
                resolution=self.resolution,
                tilesize=self.patch_size,
                pad=self.pad,
            )
            for _, row in gdf.iterrows()
            ]
        print(f"{len(tiles)} tiles created")
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

    def _write_tile_geotiff(
        self,
        pixels: np.ndarray,
        tile: CenteredTile,
        split: str,
        label: str,
        start_date: str,
        end_date: str,
    ) -> Path:
        """
        Writes a single tile's pixels to GeoTIFF and returns its path.

        Parameters
        ----------
        pixels : np.ndarray
            HxWxB array of pixel values.
        tile : CenteredTile
            Tile object with CRS and geotransform.
        split : str
            Dataset split (e.g. 'train', 'val', 'test').
        label : str
            Label/class name.
        start_date : str
            Acquisition start date.
        end_date : str
            Acquisition end date.

        Returns
        -------
        tif_path : Path
            Path to written GeoTIFF.
        """
        chw = np.moveaxis(pixels.astype("uint16"), -1, 0)  # (B,H,W)
        height, width, bands = pixels.shape
        profile = self._build_profile(height, width, bands, dtype="uint16")
        profile |= {"crs": tile.crs, "transform": tile.geotrans}

        tif_name = (
            f"{self.collection}_clear{self.clear_threshold}"
            f"_{tile.key}_{start_date}_{end_date}.tif"
        )
        out_dir = self.outdir / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        tif_path = out_dir / tif_name

        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(chw)

        return tif_path

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
        if hasattr(self, "bandIds") and self.bandIds is not None:
            try:
                idx = [self.bandIds.index(b) for b in ["B4", "B3", "B2"]]
                rgb = np.clip(pixels[..., idx] / scale, 0, 1)
                return (rgb * 255).astype(np.uint8)
            except ValueError:
                pass

        # Fallback: use first 3 bands (or repeat if single-band)
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
        max_tiles_per_png: int = 1000
    ) -> List[CenteredTile]:
        """Extract image patches for a GeoDataFrame of lat/lon Points.

        Parameters
        ----------
        gdf : GeoDataFrame
            Columns: ['source_file', 'label', 'start_date', 'end_date', 'split', 
                'geometry']; geometry CRS must be EPSG:4326 (lat/lon).
        max_tiles_per_png : int
            Randomly sample down to this count for the thumbnail PNG.

        Returns
        -------
        tiles_written : List[CenteredTile]
            The CenteredTile objects written.

        Additional Outputs
        ------------------
        Written PNGs of RGB thumbnails, collected by 'source_file'
        """
        assert {'source_file','label','start_date','end_date','split','geometry'}.issubset(gdf.columns), (
            "gdf must have columns ['source_file','label','start_date','end_date','split','geometry']"
        )
        tiles_written: List[CenteredTile] = []

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
            
            tiles = self._points_to_tiles(group)

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
                pixels = pad_patch(pixels, self.patch_size)
                tif_path = self._write_tile_geotiff(
                    pixels, tile,
                    split=str(row.split),
                    label=str(row.label),
                    start_date=str(row.start_date),
                    end_date=str(row.end_date),
                )
                tiles_written.append(tile)

                rgb = self.make_rgb_preview(pixels)
                rgb_tiles.append(rgb)

            png_fname = (f"{Path(source_file).stem}_{self.collection}" +
                         f"_clear{self.clear_threshold}" +
                         f"_{start_date}_{end_date}.png")
            png_path = self.outdir / png_fname
            write_thumbnail_grid(rgb_tiles, png_path, ts,
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
