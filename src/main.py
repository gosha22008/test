"""FastAPI-сервис агента по внутренней документации.

Эндпоинты:
  GET  /health   — статус для healthcheck и мониторинга
  POST /search   — ИНСТРУМЕНТ агента search_knowledge_base: гибридный ретривал
                   + AutoMerging по Qdrant, без LLM. Его вызывает агентная
                   платформа (config/agent.yaml -> tools.endpoint).

Инференс LLM живёт в Ollama; здесь только ретривал и оркестрация.

Запуск (внутри каталога src): uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retriever import get_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("agent-api")

app = FastAPI(title="Internal Docs RAG Agent", version="1.0.0")

# Ленивый синглтон: ретривер строится при первом /search, а не на старте,
# потому что docstore появляется только ПОСЛЕ индексации (create_index).
# Если бы грузили на старте — сервис падал бы до первой индексации.
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever(similarity_top_k=5)
        log.info("Ретривер инициализирован")
    return _retriever


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Поисковый запрос")
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultItem(BaseModel):
    text: str
    file_name: str | None = None
    page: str | None = None
    score: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Инструмент search_knowledge_base: возвращает релевантные чанки с
    метаданными источника. Пустой список — валидный ответ, по которому агент
    обязан честно сказать «не нашёл информацию» (см. системный промпт)."""
    try:
        retriever = _get_retriever()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        nodes = retriever.retrieve(req.query)[: req.top_k]
    except Exception as exc:  # noqa: BLE001
        log.exception("Ошибка ретривала")
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    results = [
        SearchResultItem(
            text=n.text,
            file_name=n.metadata.get("file_name"),
            page=n.metadata.get("page_label"),
            score=float(n.score) if n.score is not None else None,
        )
        for n in nodes
    ]
    return SearchResponse(results=results)
