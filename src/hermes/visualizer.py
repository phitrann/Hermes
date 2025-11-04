"""
Visualization helper: read latest chart file produced by PandasAI
"""
import os
import glob
import logging

logger = logging.getLogger(__name__)

class HermesVisualizer:
    def __init__(self, charts_dir: str):
        self.charts_dir = charts_dir

    def get_latest_chart(self):
        try:
            pattern = os.path.join(self.charts_dir, "*.png")
            chart_files = glob.glob(pattern)
            if not chart_files:
                return None
            latest = max(chart_files, key=os.path.getmtime)
            return latest
        except Exception as e:
            logger.error(f"Error retrieving chart: {e}")
            return None