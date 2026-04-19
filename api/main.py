from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_chain

app = FastAPI()
query_fn = get_chain()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    answer = query_fn(q.question)
    return {"answer": answer}