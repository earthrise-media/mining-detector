"""Tile inference: foundation-model embeddings, Keras probe, optional SAM2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import queue
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
import warnings

from affine import Affine
from descarteslabs.geo import DLTile
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage as ndi
import tensorflow as tf

try:
    from .tf_darwin import apply_darwin_tf_compat
except ImportError:
    from tf_darwin import apply_darwin_tf_compat

apply_darwin_tf_compat()

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from shapely.geometry import box

from tile_utils import CenteredTile, cut_chips, create_tiles, ensure_tile_shape

from dense_embedding_cache import (
    DenseCachePaths,
    build_patch_cell_geometries,
    cls_column_names,
    load_dense_embedding_parquets,
    make_dense_cache_paths,
    merge_cls_patch_for_probe,
    save_dense_embedding_parquets,
)

if TYPE_CHECKING:
    from gee import DataConfig, GEE_Data_Extractor

TileType = Union[DLTile, CenteredTile]
PathLike = Union[str, Path]

REPO_ROOT = Path(__file__).resolve().parent.parent

# Frozen full-model pickle for ``embedding_strategy='cls_only'``
# (:meth:`InferenceEngine._load_embed_model_frozen`).
SSL4EO_PATH = str(
    (REPO_ROOT / "models/SSL4EO/pretrained/dino_vit_small_patch16_224.pt").resolve()
)

# ViT-S/16 state dict for ``embedding_strategy='cls_patch'``
# (:meth:`InferenceEngine._load_embed_model` / :meth:`InferenceEngine.embed_dense`).
SSL4EO_CLS_PATCH_WEIGHTS_PATH = str(
    (
        REPO_ROOT
        / "models/SSL4EO/pretrained/ssl4eo_vit_small_patch16_weights.pth"
    ).resolve()
)

SAM2_PATH = str((REPO_ROOT / "models/sam2").resolve())
DEFAULT_MASK_DIR = str((REPO_ROOT / "data/outputs/sam2").resolve())
DEFAULT_INFERENCE_OUTPUT_BASE = str((REPO_ROOT / "data/outputs").resolve())

DEFAULT_SAM2_HYDRA_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def resolve_default_embed_model_path(
    embed_model_name: Optional[str],
    embedding_strategy: Literal["cls_only", "cls_patch"],
) -> str:
    """Default foundation-model weights for known (name, strategy) pairs.

    ``embed_model_path=None`` on :class:`InferenceConfig` is resolved via this
    helper (not called for ``embedding_strategy=='none'``). Other names require
    an explicit path.
    """
    if embed_model_name != "ssl4eo_vit_s16":
        raise ValueError(
            "InferenceConfig.embed_model_path must be set when "
            f"embed_model_name is not 'ssl4eo_vit_s16' (got {embed_model_name!r}, "
            f"embedding_strategy={embedding_strategy!r})"
        )
    if embedding_strategy == "cls_only":
        return SSL4EO_PATH
    return SSL4EO_CLS_PATCH_WEIGHTS_PATH


def _cls_only_parquet_embedding_matrix(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Rows of ``*_embeddings.parquet`` as float32; ``dim`` = count of ``cls{n}`` columns."""
    dim = sum(1 for c in gdf.columns if c.startswith("cls") and c[3:].isdigit())
    return gdf[cls_column_names(dim)].to_numpy(dtype=np.float32)


@dataclass
class InferenceConfig:
    # Path to Keras classifier (.h5). Optional for embedding-only runs; required
    # for ``bulk_predict``, ``predict_on_tile*`` classification, and
    # ``embedding_strategy='none'`` pixel inference.
    model_path: Optional[PathLike] = None
    pred_threshold: float = 0.5
    embed_model_name: Optional[str] = "ssl4eo_vit_s16"
    #: If ``None``, resolved from :func:`resolve_default_embed_model_path` for
    #: ``ssl4eo_vit_s16``; must be set explicitly for other foundation models.
    embed_model_path: Optional[str] = None
    # The next 3 parameters are required if using an embedding model
    embed_model_chip_size: Optional[int] = 224
    embedding_batch_size: Optional[int] = 32
    geo_chip_size: Optional[int] = 224
    #: ``cls_only``: frozen FM + :meth:`InferenceEngine.embed` + legacy
    #: ``*_embeddings.parquet``. ``cls_patch``: ViT + :meth:`InferenceEngine.embed_dense`
    #: + ``*_embed_dense_{cls,patch}.parquet`` pair. ``none``: Keras classifier only on
    #: pixel chips (no FM, no embedding cache reads). No cross-format cache fallback.
    embedding_strategy: Literal["cls_only", "cls_patch", "none"] = "cls_patch"
    embeddings_cache_dir: Optional[PathLike] = None
    run_sam2: bool = False
    # Base directory for prediction GeoJSONs; subfolder per model version at runtime.
    inference_output_base: PathLike = DEFAULT_INFERENCE_OUTPUT_BASE
    stride_ratio: int = 1  # stride is computed as chip_size // stride_ratio.
    tries: int = 2
    max_concurrent_tiles: int = 500
    #: Post-probe spatial pooling in ViT patch grid (``cls_patch`` only). ``1`` is a
    #: no-op and matches legacy outputs (no ``pooled_confidence`` column).
    post_probe_pool_size: int = 1
    #: ``mean``, ``max``, or ``median`` over the ``post_probe_pool_size`` neighborhood.
    post_probe_pool_method: Literal["mean", "max", "median"] = "mean"

    def __post_init__(self):
        if self.model_path is not None:
            self.model_path = Path(self.model_path)

        if self.embedding_strategy == "none":
            self.embed_model_path = ""
        elif self.embed_model_path is None:
            self.embed_model_path = resolve_default_embed_model_path(
                self.embed_model_name, self.embedding_strategy
            )
        else:
            self.embed_model_path = str(self.embed_model_path)

        self._validate_embedding_config()

        if self.embeddings_cache_dir:
            self.embeddings_cache_dir = Path(self.embeddings_cache_dir)
            self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)

        self.inference_output_base = Path(self.inference_output_base)

        if self.post_probe_pool_size < 1:
            raise ValueError(
                f"post_probe_pool_size must be >= 1, got {self.post_probe_pool_size}"
            )
        _m = str(self.post_probe_pool_method).lower()
        if _m not in {"mean", "max", "median"}:
            raise ValueError(
                "post_probe_pool_method must be 'mean', 'max', or 'median'; "
                f"got {self.post_probe_pool_method!r}"
            )
        self.post_probe_pool_method = _m  # type: ignore[assignment]

    def _validate_embedding_config(self):
        if self.embedding_strategy == "none":
            return
        if not self.embed_model_name or not self.embed_model_path:
            raise ValueError(
                "embed_model_name and embed_model_path (or a default for "
                "'ssl4eo_vit_s16') are required when "
                f"embedding_strategy={self.embedding_strategy!r}"
            )

        required = {
            "embed_model_chip_size": self.embed_model_chip_size,
            "embedding_batch_size": self.embedding_batch_size,
            "geo_chip_size": self.geo_chip_size,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                "Embedding model enabled but missing required parameters: "
                + ", ".join(missing)
            )

        if self.embedding_strategy == "cls_patch":
            if self.geo_chip_size != self.embed_model_chip_size:
                raise ValueError(
                    "For embedding_strategy='cls_patch', set geo_chip_size == "
                    "embed_model_chip_size (per-window FM input, no resize). "
                    f"Got geo_chip_size={self.geo_chip_size}, "
                    f"embed_model_chip_size={self.embed_model_chip_size}."
                )

@dataclass
class MaskConfig:
    prior_sigma: float = 12.0   # spatial prior sigma (pixels)
    smoothing_sigma: float = 2.5  # gaussian smoothing after upsampling (pixels)

    sam2_repo_path: PathLike = SAM2_PATH
    sam2_checkpoint: Optional[PathLike] = None
    finetuned_weights: Optional[PathLike] = None
    # Hydra config name for ``build_sam2`` (e.g. configs/sam2.1/...), not a path.
    sam2_model_cfg: Optional[PathLike] = None
    mask_dir: PathLike = DEFAULT_MASK_DIR

    def __post_init__(self):
        sam2_repo = Path(self.sam2_repo_path)
        self.sam2_repo_path = str(sam2_repo)

        # Keep these as strings: SAM2 internals expect str paths.
        self.sam2_checkpoint = str(
            Path(self.sam2_checkpoint) if self.sam2_checkpoint
            else sam2_repo / "sam2.1_hiera_small.pt"
        )
        self.finetuned_weights = str(
            Path(self.finetuned_weights) if self.finetuned_weights
            else sam2_repo / "SAM_model_96_px_final.pth"
        )
        self.sam2_model_cfg = self._resolve_sam2_hydra_config(
            self.sam2_model_cfg, sam2_repo
        )

        self.mask_dir = Path(self.mask_dir)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_sam2_hydra_config(
        sam2_model_cfg: Optional[PathLike], sam2_repo: Path
    ) -> str:
        """Return Hydra ``config_name`` for ``sam2.build_sam.build_sam2``."""
        if not sam2_model_cfg:
            return DEFAULT_SAM2_HYDRA_CONFIG
        raw = str(sam2_model_cfg).strip()
        path = Path(raw).expanduser()
        configs_root = (sam2_repo / "sam2" / "configs").resolve()
        if path.is_file():
            try:
                rel = path.resolve().relative_to(configs_root)
                return f"configs/{rel.as_posix()}"
            except ValueError:
                pass
        # Already a Hydra package-relative name (or user override to experiment).
        return raw

def split_parent_pixels_to_embed_windows(
    pixels: np.ndarray,
    tile: TileType,
    window_px: int,
) -> Tuple[np.ndarray, gpd.GeoDataFrame]:
    """
    Split a parent H×W raster into a ``n_h×n_w`` grid of ``window_px`` squares
    and build matching EPSG:4326 window polygons (row-major: wr then wc).

    Pixel row 0 is the northern edge (matches :func:`tile_utils.cut_chips`).
    """
    if pixels.ndim != 3:
        raise ValueError(f"pixels must be H×W×C, got shape {pixels.shape}")
    h, w, _ = pixels.shape
    if h % window_px != 0 or w % window_px != 0:
        raise ValueError(
            f"pixels {h}×{w} are not divisible by window_px={window_px}"
        )
    n_h = h // window_px
    n_w = w // window_px
    west, south, east, north = tile.geometry.bounds
    chips: List[np.ndarray] = []
    polys: List = []
    dx_win = (east - west) / n_w
    dy_win = (north - south) / n_h
    for wr in range(n_h):
        for wc in range(n_w):
            r0, r1 = wr * window_px, (wr + 1) * window_px
            c0, c1 = wc * window_px, (wc + 1) * window_px
            chips.append(pixels[r0:r1, c0:c1, :])
            minx_w = west + wc * dx_win
            maxx_w = west + (wc + 1) * dx_win
            maxy_w = north - wr * dy_win
            miny_w = north - (wr + 1) * dy_win
            polys.append(box(minx_w, miny_w, maxx_w, maxy_w))
    stacked = np.stack(chips, axis=0)
    gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
    return stacked, gdf


def apply_patch_grid_pooling(
    probs: np.ndarray,
    meta: pd.DataFrame,
    pool_size: int,
    method: str,
) -> np.ndarray:
    """
    For each ViT patch cell, pool probe probabilities over a ``pool_size×pool_size``
    neighborhood in **patch index space** (per 224 window), using partial windows
    at edges. ``meta`` row order must match ``probs`` (as from
    :func:`dense_embedding_cache.merge_cls_patch_for_probe`).
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if len(probs) != len(meta):
        raise ValueError(
            f"probs length {len(probs)} != meta rows {len(meta)}"
        )
    if pool_size == 1:
        return probs.astype(np.float32, copy=True)

    method_l = method.lower()
    if method_l not in {"mean", "max", "median"}:
        raise ValueError(
            f"pool method must be mean, max, or median; got {method!r}"
        )

    n_win = int(meta["quadrant"].max()) + 1
    h = int(meta["patch_row"].max()) + 1
    w = int(meta["patch_col"].max()) + 1
    expected = n_win * h * w
    if len(probs) != expected:
        raise ValueError(
            f"prob length {len(probs)} != n_win*h*w ({n_win}*{h}*{w})"
        )

    grid = np.zeros((n_win, h, w), dtype=np.float64)
    for i in range(len(probs)):
        q = int(meta["quadrant"].iloc[i])
        r = int(meta["patch_row"].iloc[i])
        c = int(meta["patch_col"].iloc[i])
        grid[q, r, c] = probs[i]

    out = np.empty(len(probs), dtype=np.float64)
    for i in range(len(probs)):
        q = int(meta["quadrant"].iloc[i])
        r = int(meta["patch_row"].iloc[i])
        c = int(meta["patch_col"].iloc[i])
        r0 = max(0, r - (pool_size - 1) // 2)
        r1 = min(h, r + pool_size // 2 + 1)
        c0 = max(0, c - (pool_size - 1) // 2)
        c1 = min(w, c + pool_size // 2 + 1)
        block = grid[q, r0:r1, c0:c1]
        if method_l == "mean":
            out[i] = float(block.mean())
        elif method_l == "max":
            out[i] = float(block.max())
        else:
            out[i] = float(np.median(block))

    return out.astype(np.float32)


class InferenceEngine:
    """Handles tile inference, retries, and aggregation."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        data_config: DataConfig,
        config: InferenceConfig,
        mask_config: Optional[MaskConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        from gee import GEE_Data_Extractor

        self.data_extractor = GEE_Data_Extractor(
            start_date, end_date, data_config)
        self.config = config
        self.logger = logger or logging.getLogger()

        self.model = None
        if config.model_path:
            self.model = tf.keras.models.load_model(
                str(config.model_path), compile=False)

        # Add a lock to serialize model access (in-process)
        self._tf_model_lock = threading.Lock()
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if config.embed_model_path:
            if config.embedding_strategy == "cls_patch":
                self.embed_model = self._load_embed_model()
            elif config.embedding_strategy == "cls_only":
                self.embed_model = self._load_embed_model_frozen()
            else:
                self.embed_model = None
        else:
            self.embed_model = None

        if self.config.run_sam2:
            if not mask_config:
                mask_config = MaskConfig()
            self.masker = SAM2_Masker(self.data_extractor, mask_config)
        else:
            self.masker = None

    def _ensure_keras_model(self) -> None:
        """Raise if the Keras classifier was not loaded (``model_path`` unset)."""
        if self.model is None:
            raise ValueError(
                "InferenceConfig.model_path must be set to a Keras .h5 file for "
                "this operation (classification / bulk_predict). "
                "Embedding-only use (embed / embed_dense) does not require it."
            )

    def predictions_geojson_path(self, region_name: str) -> Path:
        """Path for bulk-inference positive-chip GeoJSON under ``inference_output_base``.

        ``region_name`` is a label for the output file only (e.g. ``Path(geojson).stem``);
        it is not used to load geometry.

        Uses ``config.model_path`` stem (first two ``_``-separated tokens as model
        version), ``config.pred_threshold``, ``config.inference_output_base``, and
        the data extractor's date range.
        """
        if not self.config.model_path:
            raise ValueError(
                "InferenceConfig.model_path must be set to build output path"
            )
        mp = Path(self.config.model_path)
        model_version = "_".join(mp.stem.split("_")[:2])
        period = f"{self.data_extractor.start_date}_{self.data_extractor.end_date}"
        outdir = self.config.inference_output_base / model_version
        outdir.mkdir(parents=True, exist_ok=True)
        pred = self.config.pred_threshold
        return outdir / (
            f"{region_name}_{model_version}_{pred:.2f}_{period}.geojson"
        )

    def _load_embed_model_frozen(self):
        if self.config.embed_model_name == "ssl4eo_vit_s16":
            embed_model = torch.load(self.config.embed_model_path,
                                     weights_only=False)
        else:
            raise ValueError(
                f"Unknown embedding model: {self.config.embed_model_name}")

        embed_model = embed_model.to(self.device)
        embed_model.eval()
        return embed_model

    def _load_embed_model(self):
        if self.config.embed_model_name == "ssl4eo_vit_s16":
            import sys
            sys.path.insert(0, str(REPO_ROOT / 'models/SSL4EO-S12/src/benchmark/transfer_classification/'))
            from models.dino.vision_transformer import vit_small
            
            embed_model = vit_small(
                patch_size=16,
                num_classes=0, 
                in_chans=13
            )
            state_dict = torch.load(self.config.embed_model_path,
                                    map_location=self.device)
            embed_model.load_state_dict(state_dict)
            
        else:
            raise ValueError(
                f"Unknown embedding model: {self.config.embed_model_name}")

        embed_model = embed_model.to(self.device)
        embed_model.eval()
        return embed_model

    def _make_embedding_cache_path(self, tile: TileType) -> Optional[Path]:
        """Return Path to an embedding cache file; None if disabled."""
        emb_cache_dir = getattr(self.config, "embeddings_cache_dir", None)
        if emb_cache_dir is None:
            return None

        collection = self.data_extractor.config.collection
        start = self.data_extractor.start_date
        end = self.data_extractor.end_date
        emb_name = f"{collection}_{tile.key}_{start}_{end}_embeddings.parquet"
        return emb_cache_dir / emb_name

    def _make_dense_embedding_cache_paths(
        self, tile: TileType
    ) -> Optional[DenseCachePaths]:
        """Paths for the cls / patch Parquet pair; None if caching disabled."""
        emb_cache_dir = getattr(self.config, "embeddings_cache_dir", None)
        if emb_cache_dir is None:
            return None
        emb_cache_dir = Path(emb_cache_dir)
        collection = self.data_extractor.config.collection
        start = self.data_extractor.start_date
        end = self.data_extractor.end_date
        return make_dense_cache_paths(
            emb_cache_dir, collection, tile.key, start, end
        )

    def embed(
        self,
        chips: np.ndarray,
        chip_geoms: gpd.GeoDataFrame,
        tile: Optional[TileType] = None) -> np.ndarray:
        """Embed chips via a foundation model, with optional caching."""
        if self.config.embedding_strategy == "none":
            raise ValueError(
                "embed() is not used when embedding_strategy='none'"
            )
        if self.config.embedding_strategy != "cls_only":
            raise ValueError(
                "embed() requires embedding_strategy='cls_only'; "
                "use embed_dense() when embedding_strategy is 'cls_patch'"
            )

        emb_path = (
            self._make_embedding_cache_path(tile) if
            tile is not None else None
        )

        if emb_path is not None and emb_path.exists():
            try:
                gdf = gpd.read_parquet(emb_path)
                return _cls_only_parquet_embedding_matrix(gdf)
            except Exception as e:
                self.logger.warning(
                    f"Failed to load cached embeddings for {tile.key}: {e}")

        model_chip_size = self.config.embed_model_chip_size
        geo_chip_size = self.config.geo_chip_size

        tensor = torch.from_numpy(chips).permute(0, 3, 1, 2)  # NHWC → NCHW

        embeddings_list = []
        batch_size = self.config.embedding_batch_size
        batch_iter = range(0, len(tensor), batch_size)
        # Show batch progress for notebook embedding runs (no tile context)
        if tile is None:
            batch_iter = tqdm(batch_iter, desc="Embedding batches")
            
        with torch.no_grad():
            for i in batch_iter:
                batch = tensor[i:i+batch_size].to(
                    self.device, dtype=torch.float32)
                if geo_chip_size != model_chip_size:
                    batch = F.interpolate(
                        batch,
                        size=(model_chip_size, model_chip_size),
                        mode='bicubic',
                        align_corners=False,
                    )
                out = self.embed_model(batch)
                if isinstance(out, dict):
                    out = out[list(out.keys())[0]]
                embeddings_list.append(out.cpu())

        embeddings = torch.cat(embeddings_list, dim=0).numpy()
                            
        if emb_path is not None:
            try:
                gdf = gpd.GeoDataFrame(
                    embeddings,
                    columns=cls_column_names(embeddings.shape[1]),
                    geometry=chip_geoms["geometry"],
                    crs="EPSG:4326",
                )
                gdf.to_parquet(emb_path, index=False)
            except Exception as e:
                self.logger.warning(
                    f"Failed to save embeddings cache for {tile.key}: {e}")

        return embeddings

    def embed_dense(
        self,
        chips: np.ndarray,  # (N, H, W, C) typically square H=W=embed_model_chip_size
        chip_geoms: gpd.GeoDataFrame,
        tile: Optional[TileType] = None) -> dict:
        """
        Returns a dictionary containing:
        - 'cls': Global embeddings (N, Dim)
        - 'spatial': Patch embeddings in a grid (N, H, W, Dim) with H=W=sqrt(n_patches)
        """
        if self.config.embedding_strategy == "none":
            raise ValueError(
                "embed_dense() is not used when embedding_strategy='none'"
            )
        if self.config.embedding_strategy != "cls_patch":
            raise ValueError(
                "embed_dense() requires embedding_strategy='cls_patch' "
                "(ViT with intermediate layers + dense cache)"
            )

        dense_paths = (
            self._make_dense_embedding_cache_paths(tile)
            if tile is not None
            else None
        )

        if dense_paths is not None:
            try:
                cached = load_dense_embedding_parquets(dense_paths)
                if cached is not None:
                    return {
                        "cls": cached["cls"],
                        "spatial": cached["spatial"],
                    }
            except Exception as e:
                self.logger.warning(
                    f"Failed to load dense embedding cache for {tile.key}: {e}"
                )

        model_chip_size = self.config.embed_model_chip_size
        geo_chip_size = self.config.geo_chip_size
        
        tensor = torch.from_numpy(chips).permute(0, 3, 1, 2).to(torch.float32)
        if geo_chip_size != model_chip_size:
            tensor = F.interpolate(
                tensor, size=(model_chip_size, model_chip_size),
                mode='bicubic', align_corners=False)
            
        cls_list = []
        spatial_list = []
        batch_size = self.config.embedding_batch_size
        batch_iter = range(0, len(tensor), batch_size)

        with torch.no_grad():
            for i in batch_iter:
                batch = tensor[i:i+batch_size].to(self.device)
                out = self.embed_model.get_intermediate_layers(batch, n=1)[0] 

                # Extract CLS
                cls_tokens = out[:, 0, :] # [B, Dim]
            
                # Extract and Reshape Patches
                # Index 1: is the start of the 196 patches
                patches = out[:, 1:, :]
                dim = patches.shape[-1]
                n_patches = patches.shape[1]
                grid_side = int(round(n_patches**0.5))
                if grid_side * grid_side != n_patches:
                    raise ValueError(
                        f"embed_dense: expected square patch grid, got n_patches={n_patches}"
                    )
                spatial_features = patches.reshape(-1, grid_side, grid_side, dim)
                cls_list.append(cls_tokens.cpu())
                spatial_list.append(spatial_features.cpu())

        out = {
            "cls": torch.cat(cls_list, dim=0).numpy(),
            "spatial": torch.cat(spatial_list, dim=0).numpy(),
        }

        if dense_paths is not None and tile is not None:
            try:
                save_dense_embedding_parquets(
                    dense_paths,
                    out["cls"],
                    out["spatial"],
                    chip_geoms,
                    parent_key=tile.key,
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to save dense embedding cache for {tile.key}: {e}"
                )

        return out

    def _resolve_chip_params(self):
        """Return (chip_size, stride) based on model/config."""
        if self.config.geo_chip_size is None:
            self._ensure_keras_model()
            input_shape = self.model.layers[0].input_shape
            if isinstance(input_shape, list):  # ensemble of models
                input_shape = input_shape[0]
            chip_size = input_shape[1]
        else:
            chip_size = self.config.geo_chip_size
        stride = chip_size // self.config.stride_ratio
        return chip_size, stride

    def _model_infer(self, x):
        """Thread-safe wrapper around model.predict for inference."""
        self._ensure_keras_model()
        with self._tf_model_lock:
            return self.model.predict(x, verbose=0)

    @staticmethod
    def _probs_from_predictions(preds: np.ndarray) -> np.ndarray:
        """Positive-class probability per row (matches :meth:`_preds_to_gdf`)."""
        if preds.ndim == 2:
            if preds.shape[1] == 1:
                # Use [:, 0] not squeeze(): for batch size 1, squeeze() becomes 0-d
                # and mean_preds[idx] raises "too many indices for array".
                return np.asarray(preds[:, 0], dtype=np.float64)
            if preds.shape[1] == 2:
                return np.asarray(preds[:, 1], dtype=np.float64)
            return np.asarray(np.mean(preds, axis=1), dtype=np.float64)
        return np.asarray(np.atleast_1d(preds), dtype=np.float64)

    def _preds_to_gdf(
        self,
        preds: np.ndarray,
        chip_geoms: gpd.GeoDataFrame,
        *,
        patch_grid_meta: Optional[pd.DataFrame] = None,
        pooled_probs: Optional[np.ndarray] = None,
    ) -> gpd.GeoDataFrame:
        """Convert model preds -> preds_gdf (optionally with dense grid metadata)."""
        mean_preds = self._probs_from_predictions(preds)

        idx = np.where(mean_preds > self.config.pred_threshold)[0]
        if len(idx) == 0:
            return gpd.GeoDataFrame(
                columns=["geometry", "confidence"], crs="EPSG:4326")

        preds_gdf = gpd.GeoDataFrame(
            geometry=chip_geoms.loc[idx, "geometry"].reset_index(drop=True),
            crs="EPSG:4326",
        )
        preds_gdf["confidence"] = mean_preds[idx]
        if pooled_probs is not None:
            preds_gdf["pooled_confidence"] = pooled_probs[idx]

        if patch_grid_meta is not None:
            meta_take = patch_grid_meta.iloc[idx].reset_index(drop=True)
            for col in (
                "parent_key",
                "quadrant",
                "patch_row",
                "patch_col",
                "window_id",
            ):
                if col in meta_take.columns:
                    preds_gdf[col] = meta_take[col].values

        if preds.ndim == 2 and preds.shape[1] > 2:
            preds_gdf["preds"] = [str(list(v)) for v in preds[idx]]

        preds_gdf = preds_gdf.set_crs("epsg:4326", allow_override=True)
        return preds_gdf

    def _ensure_gdf(self, df):
        """Ensure input is a GeoDataFrame with geometry column and CRS.
        
        Returns: GeoDataFrame
        Raises: ValueError: If df is not a GeoDataFrame and not empty.
        """
        if isinstance(df, gpd.GeoDataFrame):
            return df
        elif df.empty:
            return gpd.GeoDataFrame(df, geometry="geometry", crs='epsg:4326')
        else:
            raise ValueError(f"Expected a gdf, got {type(df)} length {len(df)}")

    def produce_tile_input(self, tile: TileType) -> Dict[str, Any]:
        """
        Producer work for a single tile:
        - ``embedding_strategy == 'cls_patch'``: try dense cls/patch Parquet pair;
          on hit return patch-level ``embeddings``, patch cell ``chip_geoms``, and
          ``patch_grid_meta`` (parent key, window id, patch row/col per row).
        - ``embedding_strategy == 'cls_only'``: try legacy ``*_embeddings.parquet``.
        - ``embedding_strategy == 'none'``: skip embedding caches; fetch pixels only.
        - Otherwise fetch pixels -> ``{'mode':'pixels', ...}``.

        Note: When mode is ``embeddings``, the consumer runs classification only
        from cached vectors (no tile pixels in the queue). Inline SAM2 masking
        (``InferenceConfig.run_sam2``) is therefore skipped for those tiles even
        if there are positive detections. For production runs with SAM2, leave
        ``embeddings_cache_dir`` unset or avoid hitting the cache for tiles that
        need masks; use the standalone SAM2 scripts for a separate masking pass.
        """
        if self.config.embedding_strategy == "cls_patch":
            dense_paths = self._make_dense_embedding_cache_paths(tile)
            if dense_paths is not None:
                try:
                    loaded = load_dense_embedding_parquets(dense_paths)
                    if loaded is not None:
                        features, meta = merge_cls_patch_for_probe(loaded)
                        grid_side = loaded["spatial"].shape[1]
                        patch_geoms = build_patch_cell_geometries(
                            loaded["chip_geoms"], grid_side
                        )
                        return {
                            "mode": "embeddings",
                            "embeddings": features,
                            "chip_geoms": patch_geoms,
                            "patch_grid_meta": meta,
                            "tile": tile,
                        }
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load dense embedding cache for {tile.key}: {e}. "
                        "Will fetch pixels."
                    )
        elif self.config.embedding_strategy == "cls_only":
            emb_path = self._make_embedding_cache_path(tile)
            if emb_path and emb_path.exists():
                try:
                    embeddings_gdf = gpd.read_parquet(emb_path)
                    chip_geoms = embeddings_gdf[["geometry"]].copy()
                    embeddings = _cls_only_parquet_embedding_matrix(embeddings_gdf)
                    return {
                        "mode": "embeddings",
                        "embeddings": embeddings,
                        "chip_geoms": chip_geoms,
                        "tile": tile
                    }
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load embedding for {tile.key}: {e}. "
                        f"Will fetch pixels."
                    )

        pixels = self.data_extractor.get_tile_data(tile)
        return {"mode": "pixels", "pixels": pixels, "tile": tile}

    def predict_on_tile_pixels(
        self,
        pixels: np.ndarray,
        tile: TileType) -> tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """
        Run the per-tile pipeline starting from pixels (I/O already done).
        This is intended to run inside the consumer (serialized for GPU use).
        """
        if self.config.embedding_strategy == "cls_patch":
            return self._predict_on_tile_pixels_dense(pixels, tile)

        chip_size, stride = self._resolve_chip_params()
        tile_width = tile.tilesize + 2 * tile.pad
        if tile_width % stride != 0:
            self.logger.warning(
                f"Padded tile width {tile_width}px is not evenly divisible "
                f"by stride {stride}px (chip_size={chip_size}, "
                f"stride_ratio={self.config.stride_ratio}). "
                f"Inference may miss some pixels."
            )

        normed = np.clip(pixels / 10000.0, 0, 1).astype(np.float32)
        chips, chip_geoms = cut_chips(
            normed, tile.geometry.bounds, chip_size, stride, crs='epsg:4326')

        try: 
            if self.embed_model is not None:
                embeddings = self.embed(chips, chip_geoms, tile)
                preds = self.predict_on_tile_embeddings(
                    embeddings, chip_geoms, tile)
                preds_gdf, failed_tile = preds
                if failed_tile is not None:
                    return gpd.GeoDataFrame(), failed_tile
            else:
                preds = self._model_infer(chips)
                preds_gdf = self._preds_to_gdf(preds, chip_geoms)

            if not preds_gdf.empty and self.masker:
                self.masker.predict(pixels, tile, preds_gdf)

            return preds_gdf, None

        except Exception as e:
            self.logger.error(f"Error predicting for tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile

    def _predict_on_tile_pixels_dense(
        self,
        pixels: np.ndarray,
        tile: TileType,
    ) -> tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """448×448-style parent → ``embed_model_chip_size`` windows → dense probe."""
        if self.embed_model is None:
            self.logger.error(
                f"embedding_strategy='cls_patch' requires embed_model (tile {tile.key})"
            )
            return gpd.GeoDataFrame(), tile
        window_px = self.config.embed_model_chip_size
        if window_px is None:
            self.logger.error("embed_model_chip_size is required for dense inference")
            return gpd.GeoDataFrame(), tile
        try:
            chips, window_geoms = split_parent_pixels_to_embed_windows(
                pixels, tile, int(window_px)
            )
        except ValueError as e:
            self.logger.error(f"Dense window split failed for tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile

        normed = np.clip(chips / 10000.0, 0, 1).astype(np.float32)
        try:
            dense_out = self.embed_dense(normed, window_geoms, tile)
            nwin = dense_out["cls"].shape[0]
            loaded = {
                "cls": dense_out["cls"],
                "spatial": dense_out["spatial"],
                "parent_key": tile.key,
                "window_ids": [f"{tile.key}_q{i}" for i in range(nwin)],
            }
            features, meta = merge_cls_patch_for_probe(loaded)
            grid_side = dense_out["spatial"].shape[1]
            patch_geoms = build_patch_cell_geometries(window_geoms, grid_side)
            preds_gdf, failed_tile = self.predict_on_tile_embeddings(
                features, patch_geoms, tile, patch_grid_meta=meta
            )
            if failed_tile is not None:
                return gpd.GeoDataFrame(), failed_tile
            if not preds_gdf.empty and self.masker:
                self.masker.predict(pixels, tile, preds_gdf)
            return preds_gdf, None
        except Exception as e:
            self.logger.error(f"Error in dense predict for tile {tile.key}: {e}")
            return gpd.GeoDataFrame(), tile

    def predict_on_tile_embeddings(
        self,
        embeddings: np.ndarray,
        chip_geoms: gpd.GeoDataFrame,
        tile: TileType,
        patch_grid_meta: Optional[pd.DataFrame] = None,
    ) -> tuple[gpd.GeoDataFrame, Optional[TileType]]:
        """Run the TF classifier on already-available embeddings."""
        try:
            preds = self._model_infer(embeddings)
            pooled_probs: Optional[np.ndarray] = None
            if (
                patch_grid_meta is not None
                and self.config.embedding_strategy == "cls_patch"
                and self.config.post_probe_pool_size > 1
            ):
                mean_flat = self._probs_from_predictions(preds)
                pooled_probs = apply_patch_grid_pooling(
                    mean_flat,
                    patch_grid_meta.reset_index(drop=True),
                    self.config.post_probe_pool_size,
                    self.config.post_probe_pool_method,
                )
            preds_gdf = self._preds_to_gdf(
                preds,
                chip_geoms,
                patch_grid_meta=patch_grid_meta,
                pooled_probs=pooled_probs,
            )
            return preds_gdf, None
        except Exception as e:
            tile_key = getattr(tile, "key", "unknown")
            self.logger.error(f"Error predicting on embeddings for tile "
                              f"{tile_key}: {e}")
            return gpd.GeoDataFrame(), tile

    def _consumer(
        self, q: "queue.Queue[Dict[str, Any]]", sentinel: object,
        nonlocal_predictions: List[gpd.GeoDataFrame],
        fails: List[TileType], consumer_done: threading.Event,
        lock: threading.Lock):
        """Consume items from the queue, serially running inference."""
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break

                mode = item.get("mode")
                if mode == "embeddings":
                    embeddings = item["embeddings"]
                    chip_geoms = item["chip_geoms"]
                    tile_local = item["tile"]
                    patch_grid_meta = item.get("patch_grid_meta")
                    preds = self.predict_on_tile_embeddings(
                        embeddings,
                        chip_geoms,
                        tile_local,
                        patch_grid_meta=patch_grid_meta,
                    )
                elif mode == "pixels":
                    pixels = item["pixels"]
                    tile_local = item["tile"]
                    preds = self.predict_on_tile_pixels(pixels, tile_local)
                else:
                    self.logger.error(f"Unknown queue mode: {mode}")
                    preds = (gpd.GeoDataFrame(), item.get("tile"))
                preds_gdf, failed_tile = preds
                
                with lock:
                    if failed_tile is not None:
                        fails.append(failed_tile)
                    else:
                        if not preds_gdf.empty:
                            nonlocal_predictions.append(preds_gdf)
                        
                q.task_done()

        except Exception as e:
            self.logger.error(f"Consumer error: {e}")
        finally:
            consumer_done.set()
        
    def bulk_predict(
        self,
        tiles: List[TileType],
        region_name: str,
    ) -> gpd.GeoDataFrame:
        """
        Producer-consumer bulk inference, with retry logic:
         - producers load cache per ``embedding_strategy`` (cls_patch pair vs legacy
           parquet), or fetch pixels
         - consumer serializes GPU work: embedding model (if required) and
           TF classifier

        If ``embeddings_cache_dir`` is set and a tile is served from cache
        (``mode == "embeddings"``), inline SAM2 masking does not run for that
        tile (see ``produce_tile_input``). Dense cache rows are **patch-cell**
        geometries, not stride chips.

        ``region_name`` labels the output GeoJSON only (e.g. ``region_path.stem``).

        Writes cumulative predictions to the GeoJSON path from
        :meth:`predictions_geojson_path` (after each batch merge).

        """
        self._ensure_keras_model()
        outpath = self.predictions_geojson_path(region_name)

        predictions = gpd.GeoDataFrame({
            "geometry": gpd.GeoSeries(dtype="geometry"),
            "confidence": gpd.pd.Series(dtype="float"),
            },
            crs="epsg:4326"
        )
        
        retry_tiles = tiles.copy()
        tries_remaining = self.config.tries
        max_concurrent_tiles = self.config.max_concurrent_tiles
        max_workers = self.data_extractor.config.max_workers

        while tries_remaining and retry_tiles:
            self.logger.info(f"{tries_remaining} tries remaining.")
            fails: List[TileType] = []

            for i in tqdm(range(0, len(retry_tiles), max_concurrent_tiles)):
                batch_tiles = retry_tiles[i : i + max_concurrent_tiles]
                q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=64)
                sentinel = object()
                consumer_done = threading.Event()

                collected_preds: List[gpd.GeoDataFrame] = []
                lock = threading.Lock()
                
                consumer_thread = threading.Thread(
                    target=self._consumer,
                    args=(q, sentinel, collected_preds, fails,
                          consumer_done, lock),
                    daemon=True
                )
                consumer_thread.start()

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    future_to_tile = {
                        ex.submit(self.produce_tile_input, tile):
                        tile for tile in batch_tiles
                    }
                    for fut in as_completed(future_to_tile):
                        tile = future_to_tile[fut]
                        try:
                            item = fut.result()
                            q.put(item)
                        except Exception as e:
                            self.logger.error(
                                f"Producer failed for tile {tile.key}: {e}")
                            fails.append(tile)

                q.put(sentinel)
                consumer_done.wait()

                if collected_preds:
                    batch_gdf = self._ensure_gdf(
                        gpd.pd.concat(collected_preds, ignore_index=True))
                    batch_gdf = batch_gdf.set_crs(
                        'epsg:4326', allow_override=True)
                    predictions = predictions.set_crs(
                        'epsg:4326', allow_override=True)
                    predictions = self._ensure_gdf(
                        gpd.pd.concat([predictions, batch_gdf],
                                      ignore_index=True))
                    self.logger.info(f"Found {len(batch_gdf)} new positives.")
                    print(f"Found {len(batch_gdf)} new positives.", flush=True)
                    
                    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
                    predictions.to_file(outpath, index=False)

            self.logger.info(f"{len(fails)} failed tiles.")
            retry_tiles = fails
            tries_remaining -= 1

        return predictions

    def predict_on_tile(self, tile: TileType) -> gpd.GeoDataFrame:
        """
        Convenience wrapper for debugging: run full inference on a single tile.
        Reuses the same logic as bulk_predict but without producer/consumer.
        """
        emb_cache_dir = getattr(self.config, "embeddings_cache_dir", None)
        if emb_cache_dir is not None:
            if self.config.embedding_strategy == "cls_patch":
                dense_paths = self._make_dense_embedding_cache_paths(tile)
                if dense_paths is not None:
                    loaded = load_dense_embedding_parquets(dense_paths)
                    if loaded is not None:
                        features, meta = merge_cls_patch_for_probe(loaded)
                        grid_side = loaded["spatial"].shape[1]
                        patch_geoms = build_patch_cell_geometries(
                            loaded["chip_geoms"], grid_side
                        )
                        preds_gdf, _ = self.predict_on_tile_embeddings(
                            features, patch_geoms, tile, patch_grid_meta=meta
                        )
                        return preds_gdf
            elif self.config.embedding_strategy == "cls_only":
                emb_path = self._make_embedding_cache_path(tile)
                if emb_path.exists():
                    gdf = gpd.read_parquet(emb_path)
                    chip_geoms = gdf[["geometry"]].copy()
                    embeddings = _cls_only_parquet_embedding_matrix(gdf)
                    preds_gdf, _ = self.predict_on_tile_embeddings(
                        embeddings, chip_geoms, tile)
                    return preds_gdf

        pixels = self.data_extractor.get_tile_data(tile)
        preds_gdf, _ = self.predict_on_tile_pixels(pixels, tile)
        return preds_gdf

class SAM2_Masker:
    """Computes pixelwise segmentations."""
    def __init__(
        self, data_extractor: GEE_Data_Extractor, config: MaskConfig):
        self.data_extractor = data_extractor
        self.config = config

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._sam_lock = threading.Lock()
        
        self.predictor = self._load_model()

    def _load_model(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_model = build_sam2(
            self.config.sam2_model_cfg,
            self.config.sam2_checkpoint,
            device=self.device)

        state = torch.load(self.config.finetuned_weights, map_location="cpu")
        sam2_model.load_state_dict(state, strict=False)
        sam2_model.eval()
        
        return SAM2ImagePredictor(sam2_model)

    @staticmethod
    def polygon_gdf_to_pixel_bbox(
        gdf: gpd.GeoDataFrame, transform: Affine,
        width: int, height: int) -> Optional[np.ndarray]:
        """Convert georeferenced polygons to a box prompt in pixel coords."""
        if gdf.empty:
            return None

        geom = gdf.geometry.union_all()
        if geom.is_empty:
            return None

        minx, miny, maxx, maxy = geom.bounds
        inv = ~transform

        x0, y0 = inv * (minx, maxy)
        x1, y1 = inv * (maxx, miny)

        x0 = int(np.clip(np.floor(x0), 0, width - 1))
        x1 = int(np.clip(np.ceil(x1), 0, width - 1))
        y0 = int(np.clip(np.floor(y0), 0, height - 1))
        y1 = int(np.clip(np.ceil(y1), 0, height - 1))

        if x1 <= x0 or y1 <= y0:
            return None

        return np.array([x0, y0, x1, y1], dtype=np.int32)

    def get_rgb(self, pixels: np.ndarray) -> np.ndarray:
        """Convert Sentinel-2 reflectance -> uint8 RGB w/ constrast stretch."""
        rgb = pixels[..., [3, 2, 1]].astype(np.float32)
        rgb = np.clip(rgb / 3000.0, 0, 1)
        return (rgb * 255).astype(np.uint8)

    def soft_spatial_prior(
        self, preds_gdf: gpd.GeoDataFrame, transform: Affine,
        width: int, height: int) -> np.ndarray:
        """
        Piecewise spatial logit prior:
        - Inside detection mask: 0
        - Outside: quadratic negative penalty increasing with distance
        """
        raster = rasterio.features.rasterize(
            ((geom, 1) for geom in preds_gdf.geometry
            if geom is not None and not geom.is_empty),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

        dist_outside = ndi.distance_transform_edt(~raster)

        penalty = -(dist_outside / self.config.prior_sigma) ** 2
        prior = np.zeros_like(penalty, dtype=np.float32)
        prior[~raster] = penalty[~raster]

        return prior

    @staticmethod
    def resize_prior_to_logits(
        prior: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample prior field to SAM2 logit resolution."""

        scale_y = target_shape[0] / prior.shape[0]
        scale_x = target_shape[1] / prior.shape[1]

        out = ndi.zoom(prior, (scale_y, scale_x), order=1)
        return out.astype(np.float32)

    def upsample_logits(
        self, logits: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample SAM2 logits to target raster resolution, with optional
            Gaussian smoothing for spatial regularization.
        """
        logits_tensor = torch.from_numpy(logits[None, None, ...]).float()  
        upsampled = F.interpolate(
            logits_tensor,
            size=target_shape,
            mode='bilinear',
            align_corners=False
        )[0,0].numpy()

        sigma = self.config.smoothing_sigma
        if sigma and sigma > 0:
            upsampled = ndi.gaussian_filter(upsampled, sigma=sigma)

        return upsampled

    def predict(self, pixels: np.ndarray, tile: TileType,
                preds_gdf: gpd.GeoDataFrame):
        """Run SAM2 using polygon-derived box prompts.

        pixels: np.ndarray: reflectance values (H, W, B)
        tile: TileType
        preds_gdf: GeoDataFrame in EPSG:4326 (positive chip geometries).
        """
        height, width = pixels.shape[:2]
        transform = self.data_extractor.affine_from_tile(tile, width, height)

        box_prompt = self.polygon_gdf_to_pixel_bbox(
            preds_gdf, transform, width, height)

        if box_prompt is None:
            return {}

        with self._sam_lock:
            self.predictor.set_image(self.get_rgb(pixels))
            labels, scores, prob_logits = self.predictor.predict(
                box=box_prompt,
                multimask_output=True)

        best_logits = prob_logits[np.argmax(scores)]
        
        prior = self.soft_spatial_prior(preds_gdf, transform, width, height)
        prior = self.resize_prior_to_logits(prior, best_logits.shape)
        log_odds = best_logits + prior
        upsampled = self.upsample_logits(log_odds, pixels.shape[:2])
        mask = (upsampled > 0).astype('uint8')

        self.data_extractor.save_tile(
            pixels=mask[..., None],  # single band
            tile=tile,
            outdir=self.config.mask_dir,
            product_type="mask")

        self.data_extractor.save_tile(
            pixels=log_odds[..., None],  
            tile=tile,
            outdir=self.config.mask_dir,
            product_type="logits")

        return {
            "labels": labels,
            "scores": scores,
            "prob_logits": prob_logits,
            "box_prompt": box_prompt,
        }

    def _simplify_for_tiling(
        self, gdf: gpd.GeoDataFrame,
        buffer_width: float = 0.005) -> gpd.GeoDataFrame:
        """Simplify polygons for tile creation.

        Argument buffer_width defines the smoothing scale, in degrees 
            for epsg:4326 geometries.
        """
        gdf = gdf.copy()

        gdf["geometry"] = gdf.geometry.buffer(
            buffer_width, join_style=2).buffer(
            -buffer_width, join_style=2)

        def _drop_holes(geom):
            if geom.is_empty or geom is None:
                return None
            if geom.geom_type == "Polygon":
                return type(geom)(geom.exterior)
            elif geom.geom_type == "MultiPolygon":
                return type(geom)([type(p)(p.exterior) for p in geom.geoms
                                       if not p.is_empty])
            else:
                return geom

        gdf["geometry"] = gdf.geometry.map(_drop_holes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gdf["geometry"] = gdf.buffer(0)

        gdf = gdf[gdf.is_valid & ~gdf.is_empty].reset_index(drop=True)

        return gdf

    def bulk_mask_polygons(
        self, polys_gdf: gpd.GeoDataFrame,
        max_concurrent_tiles=500) -> gpd.GeoDataFrame:
        """Compute segmentation masks for wide-area polygons."""
        
        polys_gdf = polys_gdf.to_crs("EPSG:4326")
        region_gdf = self._simplify_for_tiling(polys_gdf)
        tiles = create_tiles(
            region_gdf.union_all(),
            self.data_extractor.config.tilesize,
            self.data_extractor.config.pad
        )
        print(f'{len(tiles)} tiles created.')

        # --- precompute intersections single-threaded - not thread safe ---
        tiles_w_polys = []
        for tile in tqdm(tiles, desc="Clipping polygons"):
            tile_polys = gpd.clip(polys_gdf, tile.geometry)
            tile_polys = tile_polys[
                tile_polys.is_valid &
                ~tile_polys.geometry.is_empty
            ]
            if not tile_polys.empty:
                tiles_w_polys.append((tile, tile_polys))

        # --- SAM2 masking multi-threaded --- 
        def process_tile(tile: TileType, tile_polys: gpd.GeoDataFrame):
            pixels = self.data_extractor.get_tile_data(tile)
            return self.predict(pixels, tile, tile_polys)

        for i in tqdm(range(0, len(tiles_w_polys), max_concurrent_tiles),
                      desc="Processing tiles"):
            batch = tiles_w_polys[i : i + max_concurrent_tiles]

            with ThreadPoolExecutor(
                max_workers=self.data_extractor.config.max_workers) as ex:
                futures = {
                    ex.submit(process_tile, tile, tp): tile for tile, tp
                        in batch
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Tile failed with error: {e}", flush=True)
    
