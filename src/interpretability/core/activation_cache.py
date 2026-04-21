"""
Activation cache for storing and retrieving model activations.

This module provides efficient in-memory and disk-backed storage for activations
extracted during model forward passes.
"""

from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from collections import OrderedDict
import torch
import numpy as np
import zarr
import json
from datetime import datetime
from loguru import logger

from .types import CacheMetadata


class ActivationCache:
    """
    In-memory cache with LRU eviction and optional disk backing.

    The cache stores activations indexed by unique keys and provides
    methods for efficient storage, retrieval, and persistence.

    Args:
        max_size_mb: Maximum cache size in megabytes (default: 1000 MB = 1 GB)
        disk_path: Optional path for disk-backed storage
        auto_save: Whether to automatically save to disk when evicting
    """

    def __init__(
        self,
        max_size_mb: int = 1000,
        disk_path: Optional[str] = None,
        auto_save: bool = False
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.disk_path = Path(disk_path) if disk_path else None
        self.auto_save = auto_save

        # In-memory storage with LRU ordering
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._metadata: Dict[str, CacheMetadata] = {}
        self._current_size_bytes = 0

        # Disk storage
        self._zarr_store: Optional[zarr.Group] = None
        if self.disk_path:
            self.disk_path.mkdir(parents=True, exist_ok=True)
            self._init_disk_storage()

        logger.info(f"Initialized ActivationCache with max size {max_size_mb} MB")

    def _init_disk_storage(self):
        """Initialize Zarr storage for disk backing."""
        if self.disk_path:
            store_path = self.disk_path / "activations.zarr"
            self._zarr_store = zarr.open_group(str(store_path), mode='a')
            logger.debug(f"Initialized disk storage at {store_path}")

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Calculate size of tensor in bytes."""
        return tensor.element_size() * tensor.nelement()

    def _evict_lru(self, required_space: int):
        """Evict least recently used items to free up space."""
        while self._current_size_bytes + required_space > self.max_size_bytes:
            if not self._cache:
                break

            # Remove oldest item (first in OrderedDict)
            key, tensor = self._cache.popitem(last=False)
            size = self._get_tensor_size(tensor)
            self._current_size_bytes -= size

            # Auto-save to disk if enabled
            if self.auto_save and self._zarr_store is not None:
                self._save_to_disk_single(key, tensor)

            logger.debug(f"Evicted {key} ({size / 1024 / 1024:.2f} MB)")

    def store(
        self,
        key: str,
        activations: torch.Tensor,
        metadata: Optional[CacheMetadata] = None
    ):
        """
        Store activations in cache.

        Args:
            key: Unique identifier for this activation
            activations: Tensor to cache
            metadata: Optional metadata about the cached item
        """
        # Evict if necessary
        tensor_size = self._get_tensor_size(activations)
        if tensor_size > self.max_size_bytes:
            logger.warning(
                f"Tensor size ({tensor_size / 1024 / 1024:.2f} MB) exceeds cache size. "
                "Saving directly to disk."
            )
            if self._zarr_store is not None:
                self._save_to_disk_single(key, activations)
            return

        self._evict_lru(tensor_size)

        # Remove existing entry if present (for LRU update)
        if key in self._cache:
            old_size = self._get_tensor_size(self._cache[key])
            self._current_size_bytes -= old_size
            del self._cache[key]

        # Add to cache
        self._cache[key] = activations.detach().cpu()
        self._current_size_bytes += tensor_size

        if metadata:
            self._metadata[key] = metadata

        # Move to end (most recently used)
        self._cache.move_to_end(key)

        logger.debug(
            f"Stored {key}: {activations.shape} "
            f"({tensor_size / 1024 / 1024:.2f} MB, "
            f"cache {self._current_size_bytes / 1024 / 1024:.2f}/{self.max_size_bytes / 1024 / 1024:.2f} MB)"
        )

    def retrieve(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve activations from cache.

        Args:
            key: Unique identifier for the activation

        Returns:
            Cached tensor if found, None otherwise
        """
        # Check in-memory cache first
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            logger.debug(f"Retrieved {key} from memory cache")
            return self._cache[key]

        # Try loading from disk
        if self._zarr_store is not None and key in self._zarr_store:
            tensor = self._load_from_disk_single(key)
            if tensor is not None:
                # Store in memory cache
                self.store(key, tensor)
                logger.debug(f"Retrieved {key} from disk cache")
                return tensor

        logger.debug(f"Key {key} not found in cache")
        return None

    def contains(self, key: str) -> bool:
        """Check if key exists in cache (memory or disk)."""
        if key in self._cache:
            return True
        if self._zarr_store is not None and key in self._zarr_store:
            return True
        return False

    def get_metadata(self, key: str) -> Optional[CacheMetadata]:
        """Retrieve metadata for a cached item."""
        return self._metadata.get(key)

    def clear(self):
        """Clear all cached data from memory."""
        self._cache.clear()
        self._metadata.clear()
        self._current_size_bytes = 0
        logger.info("Cleared in-memory cache")

    def save_to_disk(self, path: Optional[str] = None):
        """
        Save all in-memory activations to disk.

        Args:
            path: Optional path (uses disk_path if not provided)
        """
        save_path = Path(path) if path else self.disk_path
        if save_path is None:
            raise ValueError("No disk path specified")

        save_path.mkdir(parents=True, exist_ok=True)

        # Open or create Zarr store
        store_path = save_path / "activations.zarr"
        zarr_store = zarr.open_group(str(store_path), mode='a')

        # Save all cached items
        for key, tensor in self._cache.items():
            self._save_to_disk_single(key, tensor, zarr_store)

        # Save metadata
        metadata_path = save_path / "metadata.json"
        metadata_dict = {
            key: {
                "prompt": meta.prompt,
                "model_name": meta.model_name,
                "timestamp": meta.timestamp,
                "shape_info": {k: list(v) for k, v in meta.shape_info.items()}
            }
            for key, meta in self._metadata.items()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info(f"Saved {len(self._cache)} activations to {store_path}")

    def _save_to_disk_single(
        self,
        key: str,
        tensor: torch.Tensor,
        zarr_store: Optional[zarr.Group] = None
    ):
        """Save a single tensor to disk."""
        store = zarr_store or self._zarr_store
        if store is None:
            return

        # Convert to numpy and save
        array = tensor.cpu().numpy()

        # Create or overwrite dataset
        if key in store:
            del store[key]

        store.create_dataset(
            key,
            data=array,
            chunks=True,
            compression='blosc',
            compression_opts={'cname': 'zstd', 'clevel': 3}
        )

    def load_from_disk(self, path: Optional[str] = None):
        """
        Load activations from disk into memory cache.

        Args:
            path: Optional path (uses disk_path if not provided)
        """
        load_path = Path(path) if path else self.disk_path
        if load_path is None:
            raise ValueError("No disk path specified")

        store_path = load_path / "activations.zarr"
        if not store_path.exists():
            logger.warning(f"No cache found at {store_path}")
            return

        # Open Zarr store
        zarr_store = zarr.open_group(str(store_path), mode='r')

        # Load metadata
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                self._metadata = {
                    key: CacheMetadata(
                        prompt=meta["prompt"],
                        model_name=meta["model_name"],
                        timestamp=meta["timestamp"],
                        shape_info={k: tuple(v) for k, v in meta["shape_info"].items()}
                    )
                    for key, meta in metadata_dict.items()
                }

        # Load activations (lazy - only keys)
        self._zarr_store = zarr_store
        logger.info(f"Loaded cache from {store_path} ({len(zarr_store)} items)")

    def _load_from_disk_single(self, key: str) -> Optional[torch.Tensor]:
        """Load a single tensor from disk."""
        if self._zarr_store is None or key not in self._zarr_store:
            return None

        array = self._zarr_store[key][:]
        return torch.from_numpy(array)

    def keys(self) -> List[str]:
        """Get all keys in cache (memory + disk)."""
        memory_keys = list(self._cache.keys())
        disk_keys = list(self._zarr_store.keys()) if self._zarr_store else []
        return list(set(memory_keys + disk_keys))

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_items = len(self._cache)
        disk_items = len(self._zarr_store) if self._zarr_store else 0

        return {
            "memory_items": memory_items,
            "disk_items": disk_items,
            "total_items": len(set(self.keys())),
            "memory_size_mb": self._current_size_bytes / 1024 / 1024,
            "max_size_mb": self.max_size_bytes / 1024 / 1024,
            "memory_utilization": self._current_size_bytes / self.max_size_bytes,
        }

    def __len__(self) -> int:
        """Return number of unique cached items."""
        return len(set(self.keys()))

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.contains(key)

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"ActivationCache("
            f"memory={stats['memory_items']}, "
            f"disk={stats['disk_items']}, "
            f"size={stats['memory_size_mb']:.1f}/{stats['max_size_mb']:.1f} MB)"
        )
