import json
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from pathlib import Path

raw_dir = Path("src/data/raw")
output_file = Path("src/data/chunks/chunks.jsonl")

dir_loader = DirectoryLoader(
    raw_dir,
    glob = "**/*.pdf",
    loader_cls = PyMuPDFLoader
)

pdfDocuments = dir_loader.load()

splitter = TokenTextSplitter(
    chunk_size = 650,
    chunk_overlap = 100
)

chunks = splitter.split_documents(pdfDocuments)

with open(output_file, "w", encoding="utf-8") as f:
    for i, doc in enumerate(chunks):
        record = {
            "id": i,
            "text": doc.page_content,
            "metadata": doc.metadata
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")