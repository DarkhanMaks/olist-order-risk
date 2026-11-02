"""Run every query in sql/sanity_checks.sql and print the results."""
import duckdb

from src.config import DB_PATH, SQL_DIR


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    text = (SQL_DIR / "sanity_checks.sql").read_text(encoding="utf-8")
    for query in text.split(";"):
        query = query.strip()
        if not query:
            continue
        title = query.splitlines()[0].lstrip("- ").strip()  # first "-- comment" line
        print(f"\n### {title}")
        print(con.execute(query).df().to_string(index=False))


if __name__ == "__main__":
    main()
