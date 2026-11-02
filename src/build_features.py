"""Build the `features` table in DuckDB from sql/features.sql and print a quick summary."""
import duckdb

from src.config import DB_PATH, SQL_DIR


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute((SQL_DIR / "features.sql").read_text(encoding="utf-8"))

    n, dup, bad = con.execute(
        "SELECT count(*), count(*) - count(DISTINCT order_id), avg(bad_review) FROM features"
    ).fetchone()
    print(f"rows={n:,}  duplicate_order_ids={dup}  bad_review_rate={bad:.3f}")

    df = con.execute("SELECT * FROM features").df()
    print("\nnulls per column:")
    print(df.isna().sum().to_string())
    print("\nnumeric summary:")
    print(df.describe().T[["min", "50%", "max"]].round(2).to_string())
    con.close()


if __name__ == "__main__":
    main()
