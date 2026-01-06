from pipelines.sql_pipeline import SQLPipeline

def main():
    pipeline = SQLPipeline()

    question = "Show total sales by artist for 2023"

    result = pipeline.run(question)

    print("\nQUESTION")
    print(result["question"])

    print("\nGENERATED SQL")
    print(result["sql"])

    print("\nCOLUMNS")
    print(result["columns"])

    print("\nROWS (first 10)")
    for row in result["rows"][:10]:
        print(row)

    print("\nSUMMARY")
    print(result["summary"])


if __name__ == "__main__":
    main()
