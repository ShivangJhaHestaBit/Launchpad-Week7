import sqlite3

def load_schema(db_path: str) -> dict:
    """
    Extract tables and columns from SQLite database.
    Returns: { table_name: [column1, column2, ...] }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [row[0] for row in cursor.fetchall()]

    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        schema[table] = [row[1] for row in cursor.fetchall()]

    conn.close()
    return schema


def format_schema_for_prompt(schema: dict) -> str:
    """
    Convert schema into LLM-friendly text.
    """
    lines = []
    for table, columns in schema.items():
        cols = ", ".join(columns)
        lines.append(f"Table {table}({cols})")
    return "\n".join(lines)
