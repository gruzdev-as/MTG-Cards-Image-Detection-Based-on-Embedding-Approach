from pathlib import Path

SQL_DIR = Path("sql")

def load_sql(filename: str) -> str:
    sql_path = SQL_DIR / filename
    return sql_path.read_text(encoding="utf-8")
