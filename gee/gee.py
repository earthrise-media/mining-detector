from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, ClassVar, Dict, Any

import ee
from google.api_core import retry
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from concurrent.futures import ThreadPoolExecutor, as_completed

from descarteslabs.geo import DLTile
from tile_utils import CenteredTile, ensure_tile_shape

TileType = Union[DLTile, CenteredTile]

EE_PROJECT = os.environ.get("EE_PROJECT", "earthindex")
_ee_initialized = False


def _ensure_earth_engine_initialized() -> None:
    """Lazily initialize the Earth Engine client (first GEE_Data_Extractor use).

    If ``GOOGLE_APPLICATION_CREDENTIALS`` is set to a path of a service account JSON
    file, uses :class:`google.oauth2.service_account.Credentials` with the Earth Engine
    scope. Otherwise uses the default client (e.g. user ``earthengine authenticate``).
    """
    global _ee_initialized
    if _ee_initialized:
        return
    key_path = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    init_kw: Dict[str, Any] = {
        "opt_url": "https://earthengine-highvolume.googleapis.com",
        "project": EE_PROJECT,
    }
    if key_path:
        if not os.path.isfile(key_path):
            raise FileNotFoundError(
                f"GOOGLE_APPLICATION_CREDENTIALS={key_path!r} is not a readable file"
            )
        from google.oauth2 import service_account

        ee_scopes = ["https://www.googleapis.com/auth/earthengine"]
        credentials = service_account.Credentials.from_service_account_file(
            key_path, scopes=ee_scopes
        )
        init_kw["credentials"] = credentials
    ee.Initialize(**init_kw)
    _ee_initialized = True


PathLike = Union[str, Path]

@dataclass
class DataConfig:
    tilesize: int = 576
    pad: int = 0
    collection: str = "S2L1C"
    bands: Optional[List[str]] = None
    clear_threshold: float = 0.6
    max_workers: int = 8
    image_cache_dir: Optional[PathLike] = None #If given, will write/read images

    _BAND_IDS: ClassVar[Dict[str, List[str]]] = {
        "S1": ["VV", "VH"],
        "S2L1C": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                   "B8A", "B8", "B9", "B10", "B11", "B12"],
        "S2L1C-12band": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                          "B8A", "B8", "B9", "B11", "B12"],
        "S2L2A": ["B1", "B2", "B3", "B4", "B5", "B6", "B7",
                  "B8A", "B8", "B9", "B11", "B12"],
        "AlphaEarth": [f"A{x:02d}" for x in range(64)],
    }

    _NDVI_BANDS: ClassVar[Dict[str, str]] = {
        "red": "B4",
        "nir": "B8A"
    }

    def __post_init__(self):
        # Automatically resolve bands if not provided
        if self.bands is None:
            if self.collection not in self._BAND_IDS:
                raise ValueError(f"No band mapping defined for collection "
                                 f"'{self.collection}'")
            self.bands = self._BAND_IDS[self.collection]

        if self.image_cache_dir:
            self.image_cache_dir = Path(self.image_cache_dir)
            self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def available_collections(cls) -> List[str]:
        """Return the list of supported collection IDs."""
        return list(cls._BAND_IDS.keys())
    

class GEE_Data_Extractor:
    def __init__(self, start_date: str, end_date: str, config: DataConfig):
        _ensure_earth_engine_initialized()
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.composite = self._build_composite(start_date, end_date)

    def _build_composite(self, start_date: str, end_date: str):
        collection = self.config.collection
        clear_threshold = self.config.clear_threshold
      
        if collection in ['S2L1C', 'S2L1C-12band']:
            s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            csPlus = ee.ImageCollection(
                "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            QA_BAND = "cs_cdf"
            composite = (
                s2.filterDate(start_date, end_date)
                .linkCollection(csPlus, [QA_BAND])
                .map(lambda img:
                    img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
                .median())

        elif collection == 'S2L2A':
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            csPlus = ee.ImageCollection(
                "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            QA_BAND = "cs_cdf"
            composite = (
                s2.filterDate(start_date, end_date)
                .linkCollection(csPlus, [QA_BAND])
                .map(lambda img:
                    img.updateMask(img.select(QA_BAND).gte(clear_threshold)))
                .median())

        elif collection == 'S1':
            s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
            composite = (
                s1.filterDate(start_date, end_date)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VH"))
                .mosaic())

        elif collection == 'AlphaEarth':
            emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            composite = emb.filterDate(start_date, end_date).mosaic()

        else:
            raise ValueError(f'Collection {collection} not recognized.')
        return composite

    @retry.Retry(timeout=240)
    def get_tile_data(self, tile: TileType) -> np.ndarray:
        """Download or load cached image data for a tile.

        Inputs:
        - tile: a TileType object
        Outputs:
        - pixels: a numpy array with shape (H, W, bands)
        """
        image_cache_dir = getattr(self.config, "image_cache_dir", None)
        collection = self.config.collection
        start, end = self.start_date, self.end_date
        tif_name = f"{collection}_{tile.key}_{start}_{end}.tif"

        if image_cache_dir:
            tif_path = image_cache_dir / tif_name
            if tif_path.exists():
                try:
                    return self.load_tile(tif_path)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load cached tile {tile.key}: {e}")

        tile_geom = ee.Geometry.Rectangle(tile.geometry.bounds)
        out_size = tile.tilesize + 2 * tile.pad
        composite_tile = self.composite.clipToBoundsAndScale(
            geometry=tile_geom,
            width=out_size,
            height=out_size
        )

        pixels = ee.data.computePixels(
            {
                "bandIds": self.config.bands,
                "expression": composite_tile,
                "fileFormat": "NUMPY_NDARRAY",
            }
        )

        if isinstance(pixels, np.ndarray):
            if pixels.dtype.fields is not None:
                pixels = np.stack([pixels[band] for band in self.config.bands],
                                 axis=-1)
        else:
            pixels = np.array(pixels)

        pixels = ensure_tile_shape(pixels, out_size)
        pixels = pixels.astype(np.float32, copy=False)

        if image_cache_dir:
            try:
                self.save_tile(pixels, tile, image_cache_dir)
            except Exception as e:
                self.logger.warning(f"Failed to cache tile {tile.key}: {e}")

        return pixels

    def get_tile_data_concurrent(
        self, tiles: List[TileType]) -> List[Optional[np.ndarray]]:
        """
        Download all tile data concurrently.

        Per-tile failures do not abort the whole batch: failed indices are
        collected and retried in a second concurrent pass (helps transient EE
        errors such as user memory limit). Output order matches ``tiles``.
        Entries that still fail after retry are ``None``; an error is logged.
        """
        log = logging.getLogger(__name__)
        if not tiles:
            return []

        n = len(tiles)
        results: List[Optional[np.ndarray]] = [None] * n

        def run_indices(indices: List[int]) -> List[int]:
            failed_local: List[int] = []
            if not indices:
                return failed_local
            with ThreadPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                future_to_i = {
                    executor.submit(self.get_tile_data, tiles[i]): i
                    for i in indices
                }
                for fut in as_completed(future_to_i):
                    i = future_to_i[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as exc:
                        log.warning(
                            "get_tile_data failed for tile %s: %s",
                            tiles[i].key,
                            exc,
                        )
                        failed_local.append(i)
            return failed_local

        failed = run_indices(list(range(n)))
        if failed:
            log.info(
                "Retrying %d failed tile(s) after initial concurrent batch",
                len(failed),
            )
            failed = run_indices(failed)
        if failed:
            sample = [tiles[i].key for i in failed[:5]]
            log.error(
                "get_tile_data still failed for %d tile(s) after retry "
                "(entries will be None). Sample keys: %r",
                len(failed),
                sample,
            )

        return list(results)

    @staticmethod
    def affine_from_tile(tile, width: int, height: int):
        """Construct an EPSG:4326 affine transform from tile metadata."""
        return from_bounds(*tile.geometry.bounds, width, height)

    def save_tile(
        self, pixels: np.ndarray, tile: TileType, outdir: Path,
        product_type: Literal["image", "mask", "logits"] = "image") -> Path:
        """
        Write a tile GeoTIFF in EPSG:4326.

        product_type:
            - "image"  -> satellite imagery
            - "mask"   -> uint8 segmentation mask 
            - "logits" -> float32 model logits 
        """
        if product_type == "mask":
            dtype = "uint8"
            nodata = 2
            suffix = "-msk"
        elif product_type == "logits":
            dtype = "float32"
            nodata = np.nan
            suffix = "-logits"
        else:  
            if self.config.collection[:2] == "S2":
                dtype = "uint16"
            else:
                dtype = "float32"
            nodata = None
            suffix = ""

        pixels = np.moveaxis(pixels.astype(dtype, copy=False), -1, 0)
        bands, height, width = pixels.shape

        transform = self.affine_from_tile(tile, width, height)
        crs = "EPSG:4326"

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
            "tiled": True,
        }

        if nodata is not None:
            profile["nodata"] = nodata

        base = (
            f"{self.config.collection}_{tile.key}_"
            f"{self.start_date}_{self.end_date}"
        )

        tif_name = f"{base}{suffix}.tif"
        outpath = outdir / tif_name

        with rasterio.open(outpath, "w", **profile) as dst:
            dst.write(pixels)

        return outpath


    def load_tile(self, path: Path) -> np.ndarray:
        """Load a tile’s pixels from GeoTIFF."""
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (B,H,W)
            arr = np.moveaxis(arr, 0, -1)  # back to (H,W,B)
        return arr


from inference_engine import (
    InferenceConfig,
    InferenceEngine,
    MaskConfig,
    SAM2_Masker,
    resolve_default_embed_model_path,
    split_parent_pixels_to_embed_windows,
)
