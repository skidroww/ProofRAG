from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional

from search_test4 import hybrid_search, corpus_info

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    company: Optional[str] = ""


@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.get("/docs_info")
def get_docs_info():
    unique_docs = {}
    for item in corpus_info:
        meta = item.get("metadata", {})
        title = meta.get("title", "제목없음")

        if title not in unique_docs:
            unique_docs[title] = {
                "title": title,
                "company": meta.get("company","기업명없음"),
                "violation": meta.get("violation", "유형없음")
            }
    return list(unique_docs.values())

@app.post("/ask")
def ask(req: QueryRequest):
    search_data = hybrid_search(req.question, req.company)
    top_ids = search_data["top_k_ids"]
    timings = search_data["timings"]

    results = []
    for cid in top_ids:
        matched = next((item for item in corpus_info if item['chunk_id'] == cid), None)
        if matched:
            results.append({
                "chunk_id": cid,
                "text": matched["injected_text"]
            })
    return {
        "question": req.question,
        "results": results,
        "timings": timings
    }