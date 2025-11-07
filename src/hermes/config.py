"""
Configuration for Hermes AI - paths and PandasAI / LLM config
"""
from pathlib import Path
import os

# Set matplotlib backend to non-interactive to prevent GUI issues
import matplotlib
matplotlib.use('Agg')

from pandasai.helpers.logger import Logger
from pandasai_litellm.litellm import LiteLLM
import pandasai as pai

# Directories
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = os.path.join(ROOT_DIR, "data")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
CACHE_DIR = os.path.join(ROOT_DIR, ".cache")
CHARTS_DIR = os.path.join(ROOT_DIR, "exports/charts")

SHIPMENTS_FILE = os.path.join(DATA_DIR, "shipments.csv")
QUESTIONS_FILE = os.path.join(DATA_DIR, "questions.csv")
SEMANTIC_DATASET_PATH = "hermes/shipments"

for d in (LOGS_DIR, CACHE_DIR, CHARTS_DIR):
    os.makedirs(d, exist_ok=True)

# PandasAI + LLM configuration
pandasai_logger = Logger(save_logs=True, verbose=False)  # Enable verbose logging to capture reasoning

# LiteLLM config - adjust model, base_url, api_key to your environment
llm = LiteLLM(
    model="openai/Qwen/Qwen3-14B-AWQ",
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
    temperature=0.1,
    max_tokens=2000
)

pai.config.set({
    "llm": llm,
    "logger": pandasai_logger,
    "enable_cache": True,  # 🔥 DISABLE to force chart regeneration for each query
    "save_charts": True,
    "save_charts_path": CHARTS_DIR,
    "open_charts": False,
    "verbose": False,  # Enable verbose logging to see reasoning
    "enforce_privacy": True,
    "max_retries": 3
})

# SmartDataFrame configuration
SMART_DF_CONFIG = {
    "llm": llm,
    "enable_cache": False,  # 🔥 Force fresh LLM calls
}