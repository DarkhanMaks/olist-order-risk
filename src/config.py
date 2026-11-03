from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (parent of src/)
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "olist.duckdb"
SQL_DIR = ROOT / "sql"
MODEL_PATH = ROOT / "models" / "model.joblib"
