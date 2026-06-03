from fastapi import FastAPI
from pydantic import BaseModel

import google.generativeai as genai

from rag import ask_rag

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(data: ChatRequest):

    answer = ask_rag(data.question)

    return {
        "answer": answer
    }