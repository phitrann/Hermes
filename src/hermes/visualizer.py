"""
Visualization helper with cache-busting to prevent chart reuse.
Generates fresh charts for each query and manages cache.
"""
import os
import glob
import logging
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class HermesVisualizer:
    """
    Handles chart generation with cache busting.
    
    Key methods:
    - get_latest_chart(): Returns path to most recent PNG
    - get_latest_chart_as_pil(): Returns fresh PIL Image
    - clear_cache_for_query(): Removes old charts before new generation
    """

    def __init__(self, charts_dir: str, max_cached_charts: int = 5):
        """
        Args:
            charts_dir: Path to store generated charts
            max_cached_charts: Max old charts to keep (older ones deleted)
        """
        self.charts_dir = charts_dir
        self.max_cached_charts = max_cached_charts
        self.latest_chart_path: Optional[str] = None
        Path(charts_dir).mkdir(parents=True, exist_ok=True)

    def get_latest_chart(self) -> Optional[str]:
        """
        Get path to the most recently created chart.
        
        Returns:
            Path to PNG file or None if not found
        """
        try:
            pattern = os.path.join(self.charts_dir, "*.png")
            chart_files = glob.glob(pattern)
            if not chart_files:
                logger.warning("No chart files found in directory")
                return None
            latest = max(chart_files, key=os.path.getmtime)
            self.latest_chart_path = latest
            logger.info(f"Latest chart: {latest}")
            return latest
        except Exception as e:
            logger.error(f"Error retrieving chart: {e}")
            return None

    def get_latest_chart_as_pil(self) -> Optional[Image.Image]:
        """
        Fetch the most recent chart as a fresh PIL Image.
        Always re-opens from disk to avoid in-memory caching.

        Returns:
            PIL Image or None if not found
        """
        chart_path = self.get_latest_chart()
        if not chart_path:
            return None

        try:
            # Always re-open to avoid in-memory reference
            img = Image.open(chart_path)
            img_copy = img.copy()  # Detach from file
            logger.debug(f"Loaded chart as fresh PIL: {chart_path}")
            return img_copy
        except Exception as e:
            logger.error(f"Could not load chart as PIL: {e}")
            return None

    def clear_cache_for_query(self) -> None:
        """
        Delete old charts to prevent file reuse.
        Keeps only the N most recent charts.
        """
        try:
            pattern = os.path.join(self.charts_dir, "*.png")
            chart_files = sorted(
                glob.glob(pattern),
                key=os.path.getmtime,
            )

            # Remove older charts, keep latest N
            if len(chart_files) > self.max_cached_charts:
                for old_chart in chart_files[: -self.max_cached_charts]:
                    try:
                        os.remove(old_chart)
                        logger.debug(f"Deleted old chart: {old_chart}")
                    except Exception as e:
                        logger.warning(f"Could not delete {old_chart}: {e}")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def reset(self) -> None:
        """Clear latest chart reference."""
        self.latest_chart_path = None
