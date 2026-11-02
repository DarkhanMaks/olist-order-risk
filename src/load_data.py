"""Load every raw Olist CSV into a DuckDB file, one table per CSV."""
import duckdb

from src.config import DB_PATH, RAW_DIR


def table_name(csv_path):
    # olist_orders_dataset.csv -> orders
    return csv_path.stem.removeprefix("olist_").removesuffix("_dataset")


def main():
    con = duckdb.connect(str(DB_PATH))
    for csv in sorted(RAW_DIR.glob("*.csv")):
        name = table_name(csv)
        # read_csv_auto guesses column types (dates become TIMESTAMP)
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS "
            f"SELECT * FROM read_csv_auto('{csv.as_posix()}')"
        )
        rows = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"{name:35s} {rows:>9,} rows")
    con.close()


if __name__ == "__main__":
    main()
