from fastapi import FastAPI, UploadFile, File , Form
from pathlib import Path
import shutil
from memory.memory_store import MemoryStore
from evaluations.rag_eval import RAGEvaluator

from tests.test_retrival import ask_retrival
from tests.test_image_search import image_to_image_final, image_to_text_final, text_to_image_final
from pipelines.sql_pipeline import SQLPipeline

app = FastAPI()

UPLOAD_DIR = Path("tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

memory = MemoryStore()
evaluator = RAGEvaluator()

sql_pipeline = SQLPipeline("sales.db")

@app.post("/ask")
def ask(question: str):
    answer = ask_retrival(question)
    score = evaluator.faithfulness_score(answer["answer"], answer["context"])
    answer["score"] = score
    return answer

@app.post("/ask-image")
def ask_image(
    question: str = Form(None),
    image: UploadFile = File(None),
    top_k: int = 5,
):
    if not question and not image:
        return {"error": "Provide either a question or an image"}

    image_path = None
    if image:
        image_path = UPLOAD_DIR / image.filename
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

    if image_path and question:
        answer =  image_to_text_final(str(image_path), question, top_k)
        score = evaluator.faithfulness_score(answer["answer"], answer["context"])
        answer["score"] = score
        return answer

    if image_path:
        answer =  image_to_image_final(str(image_path), top_k)
        score = evaluator.faithfulness_score(answer["answer"], answer["context"])
        answer["score"] = score
        return answer

    answer =  text_to_image_final(question, top_k)
    score = evaluator.faithfulness_score(answer["answer"], answer["context"])
    answer["score"] = score
    return answer


@app.post("/ask-sql")
def ask_sql(question: str):
    result = sql_pipeline.run(question)
    return result
