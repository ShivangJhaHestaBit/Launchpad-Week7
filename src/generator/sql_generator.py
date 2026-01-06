import re
from generator.llm_client import generate

class SQLGenerator:
    def generate_sql(self, question: str, schema_text: str) -> str:
        prompt = f"""
You are an expert SQL generator for SQLite.

Schema:
{schema_text}

User question:
{question}

Rules:
- Generate only ONE SQL query
- Use valid SQLite syntax
- Do NOT include explanations or comments
- Do NOT use DROP, DELETE, UPDATE, INSERT
- Return ONLY the SQL query
"""
        output = generate(prompt)
        match = re.search(
            r"(select\s+.*?;)",
            output,
            re.IGNORECASE | re.DOTALL
        )

        if not match:
            raise ValueError(f"Failed to extract SQL from LLM output:\n{output}")

        sql = match.group(1).strip()
        print(sql)
        return sql
