# SQL-QA Engine (LLM-Powered)

A schema-aware, safe SQL Question-Answering engine that converts natural-language questions into validated, read-only SQLite queries using an LLM (Mistral). The system prevents schema-hallucination and SQL injection, executes only safe `SELECT` statements, and returns concise, human-friendly summaries.


## Architecture
```
User Question
        ↓
Schema Loader
        ↓
Prompt Construction
        ↓
LLM (Mistral)
        ↓
SQL Extraction
        ↓
SQL Validation
        ↓
Safe Execution (SQLite)
        ↓
Result Summarization
        ↓
Final Answer
```

## Pipeline steps

1. Schema extraction (schema_loader.py)
     - Read metadata from SQLite
     - Extract table names, column names, column types
     - Ensures the LLM cannot hallucinate schema elements

2. Prompt construction (sql_generator.py)
     - Inject schema and strict rules into the prompt:
         - Only allowed tables & columns
         - SQLite syntax only
         - `SELECT` queries only
         - No explanations or comments

3. SQL generation (LLM)
     - LLM produces SQL using the schema-aware prompt
     - Output may include additional text (prompt echo or rules)

     Example raw output:
     ```
     ... Rules ... SQL: SELECT artist, SUM(sales) FROM sales GROUP BY artist;
     ```

4. SQL extraction
     - Use a regex to extract only the `SELECT ...;` statement:
     ```
     /SELECT[\s\S]*?;/
     ```
     - This avoids prompt leakage, false validation errors, and reduces injection surface

5. SQL validation (sql_pipeline.py)
     - Enforce:
         - Only `SELECT` statements
         - Block `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`
         - Block unsupported SQLite features (e.g., `FULL OUTER JOIN`)
     - Validation failures → hard stop

6. Safe execution
     - Execute via parameter-safe SQLite connection
     - No string interpolation
     - Read-only access pattern guaranteed

7. Result summarization
     - Return:
         - Number of rows
         - Top N rows (first N)
         - Human-readable summary

     Example output:
     ```
     Returned 3 rows.

     Top results:
     - Artist: Coldplay — Total Sales: 120000
     - Artist: Adele — Total Sales: 98000
     ```

8. Final answer
     - Concise, user-facing explanation and top results
     - No raw SQL or database internals exposed

---