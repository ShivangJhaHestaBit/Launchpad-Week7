import sqlite3

from utils.schema_loader import load_schema, format_schema_for_prompt
from generator.sql_generator import SQLGenerator
from generator.llm_client import generate

DB_PATH = "sales.db"

class SQLPipeline:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.schema = load_schema(db_path)
        self.schema_text = format_schema_for_prompt(self.schema)
        self.generator = SQLGenerator()

    def _validate_sql(self, sql: str):
        forbidden = ["insert", "update", "delete", "drop", "alter", "truncate"]
        lowered = sql.lower()

        for word in forbidden:
            if word in lowered:
                raise ValueError(f"Forbidden SQL operation detected: {word}")

        if not lowered.strip().startswith("select"):
            raise ValueError("Only SELECT queries are allowed")

    def _execute_sql(self, sql: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        conn.close()
        return columns, rows

    def _summarize(self, question: str, columns, rows) -> str:
        if not rows:
            return "No results found."

        preview = "\n".join(
            [", ".join(columns)] +
            [", ".join(map(str, r)) for r in rows[:10]]
        )

        prompt = f"""
You are a data analyst.

User question:
{question}

SQL result:
{preview}

Summarize the result clearly in plain English.
"""

        try:
            return generate(prompt)
        except Exception:
            return f"Returned {len(rows)} rows."

    def run(self, question: str):
        sql = self.generator.generate_sql(question, self.schema_text)

        self._validate_sql(sql)
        columns, rows = self._execute_sql(sql)
        summary = self._summarize(question, columns, rows)

        return {
            "question": question,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "summary": summary,
        }
