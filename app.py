"""Веб-интерфейс юридического ассистента (FastAPI)."""
from __future__ import annotations
import re
from pathlib import Path
import httpx
import os

os.environ["LLM_BASE_URL"] = 
os.environ["LLM_API_KEY"] =
os.environ["LLM_MODEL"] =

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import solution as s
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Юридический ассистент")
STATE: dict = {}

@app.on_event("startup")
def startup() -> None:
    s.DOCS.clear()
    # Ищем архив с документами
    kb_path = s.find_file("kb.zip", "knowledge-base.zip", "knowledge_base.zip")
    print(f"Загружаем базу знаний из {kb_path}...")
    chunks = s.load_kb(kb_path)
    s.embed_chunks(chunks)
    STATE["chunks"], STATE["bm"] = chunks, s.BM25(chunks)
    print(f"✅ База загружена, фрагментов: {len(chunks)}")

class Msg(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    messages: list[Msg]

@app.post("/api/chat")
def chat(body: ChatIn):
    msgs = [m for m in body.messages if m.content.strip()]
    if not msgs or msgs[-1].role != "user":
        raise HTTPException(400, "Последним должно быть сообщение пользователя")
    msgs = msgs[-7:]
    if len(msgs) == 1:
        question = msgs[0].content.strip()
    else:
        question = "\n".join(
            ("Пользователь: " if m.role == "user" else "Ассистент: ") + m.content.strip() for m in msgs)
    
    with httpx.Client(timeout=120) as c:
        answer, hits = s.answer_ex(c, STATE["chunks"], STATE["bm"], question)
    
    sources, seen = [], set()
    for h in hits:
        text = h.text.split("\n", 1)[1] if h.text.startswith("[Документ:") else h.text
        key = (h.doc, text[:80])
        if key in seen: continue
        seen.add(key)
        sources.append({
            "doc": h.doc, 
            "text": text.strip(), 
            "whole": h.full,
            "status": re.search(r"статус: ([^\]]+)\]", h.text).group(1) if "статус:" in h.text else ""
        })
    return {"answer": answer, "sources": sources}

@app.get("/api/doc")
def doc(name: str):
    if name not in s.DOCS:
        raise HTTPException(404, "Документ не найден")
    return {"name": name, "text": s.DOCS[name]}

# Подключаем статику (фронтенд)
app.mount("/static", StaticFiles(directory=Path(__file__).parent), name="static")

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")
