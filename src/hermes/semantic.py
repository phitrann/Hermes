"""
Semantic registration helper using PandasAI dataset registration (pai.create / pai.load)
"""
import logging
import pandas as pd
import pandasai as pai
from .config import SEMANTIC_DATASET_PATH

logger = logging.getLogger(__name__)

def generate_columns_metadata(df: pd.DataFrame):
    cols = []
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            col_type = "integer"
        elif pd.api.types.is_float_dtype(df[col]):
            col_type = "float"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "datetime"
        else:
            col_type = "string"
        desc = _default_description(col)
        cols.append({"name": col, "type": col_type, "description": desc})
    return cols

def _default_description(col_name: str) -> str:
    name = col_name.lower()
    if name in ("id", "shipment_id"):
        return "Unique shipment identifier"
    if name == "route":
        return "Route identifier (e.g., Route A)"
    if name == "warehouse":
        return "Warehouse code"
    if name == "delivery_time":
        return "Delivery duration in days"
    if name == "delay_minutes":
        return "Delay duration in minutes"
    if name == "delay_reason":
        return "Reason for delay"
    if name == "date":
        return "Shipment date"
    if name == "cost":
        return "Operational cost per shipment"
    return f"{col_name} column"

def register_semantic_dataset(df: pd.DataFrame, path: str = SEMANTIC_DATASET_PATH, name: str = "Hermes Shipments", description: str = "Shipments dataset"):
    try:
        # If dataset exists, pai.load will succeed; skip creation
        try:
            pai.load(path)
            logger.info(f"Semantic dataset '{path}' already exists; skipping creation.")
            return
        except Exception:
            pass
        columns_meta = generate_columns_metadata(df)
        pai.create(path=path, name=name, df=df, description=description, columns=columns_meta)
        logger.info(f"Registered semantic dataset at '{path}'.")
    except Exception as e:
        logger.warning(f"Failed to register semantic dataset: {e}")