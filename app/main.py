from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "documents.jsonl"
STATIC = ROOT / "app" / "static"
CACHE = ROOT / ".cache"
KEY_FILE = ROOT / ".secrets" / "typesafe_api_key.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
CHUNK_SIZE, CHUNK_OVERLAP = 800, 150

PRESETS = [
    {"id": "payment_gateway", "name": "Payment gateway の障害", "description": "通常の肯定条件", "query": "Find postmortems about incidents that affected payment-gateway.", "ground_truth": ["PM-INC-2102", "PM-INC-2113", "PM-INC-2117"]},
    {"id": "billing_not_payment", "name": "Billing、ただし Payment ではない", "description": "同一領域の否定条件", "query": "Find postmortems about incidents that affected billing-engine but did not affect payment-gateway.", "ground_truth": ["PM-INC-2101", "PM-INC-2105", "PM-INC-2110", "PM-INC-2116", "PM-INC-2120"]},
    {"id": "restart_warnings", "name": "再起動禁止の Runbook", "description": "否定形の運用指示", "query": "Find runbooks that explicitly warn against restarting all instances or restarting from scratch.", "ground_truth": ["RB-001", "RB-011"]},
    {"id": "cross_team_responders", "name": "複数チームが対応した障害", "description": "文書内の複合判定", "query": "Find postmortems where responders came from more than one team.", "ground_truth": ["PM-INC-2101", "PM-INC-2102", "PM-INC-2103", "PM-INC-2105", "PM-INC-2109", "PM-INC-2111", "PM-INC-2113", "PM-INC-2114", "PM-INC-2115", "PM-INC-2116", "PM-INC-2117", "PM-INC-2118", "PM-INC-2120"]},
]


def api_key() -> str | None:
    value = os.getenv("TYPESAFE_API_KEY", "").strip()
    if value:
        return value
    if not KEY_FILE.exists():
        return None
    for line in KEY_FILE.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    return None


def chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("invalid chunk configuration")
    return [text[i : i + size] for i in range(0, max(len(text), 1), size - overlap)]


def excerpt(text: str, limit: int = 330) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def metrics(scores: dict[str, float], threshold: float, truth: set[str]) -> dict[str, Any]:
    hits = {doc_id for doc_id, score in scores.items() if score >= threshold}
    tp, fp, fn = hits & truth, hits - truth, truth - hits
    precision = len(tp) / len(hits) if hits else 0.0
    recall = len(tp) / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": sorted(tp), "fp": sorted(fp), "fn": sorted(fn), "precision": precision, "recall": recall, "f1": f1}


class SearchBody(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    preset_id: str | None = None
    jev_concurrency: int = 1

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query is blank")
        return value

    @field_validator("jev_concurrency")
    @classmethod
    def valid_concurrency(cls, value: int) -> int:
        if value not in (1, 4):
            raise ValueError("jev_concurrency must be 1 or 4")
        return value


class SearchEngine:
    def __init__(self) -> None:
        self.documents = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
        self.chunk_rows = [(i, chunk) for i, doc in enumerate(self.documents) for chunk in chunks(doc["text"])]
        self.model: Any = None
        self.embeddings: np.ndarray[Any, Any] | None = None

    def public_doc(self, doc: dict[str, Any]) -> dict[str, str]:
        return {key: doc[key] for key in ("doc_id", "doc_type", "title", "date")}

    def _cache_path(self) -> Path:
        digest = hashlib.sha256(DATASET.read_bytes() + f"{MODEL_NAME}:{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode()).hexdigest()[:20]
        return CACHE / f"embeddings-{digest}.npz"

    def prepare(self) -> None:
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
        if self.embeddings is not None:
            return
        path = self._cache_path()
        if path.exists():
            loaded = np.load(path)["embeddings"]
            if loaded.shape[0] == len(self.chunk_rows):
                self.embeddings = loaded
                return
        CACHE.mkdir(parents=True, exist_ok=True)
        encoded = self.model.encode([row[1] for row in self.chunk_rows], normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = np.asarray(encoded, dtype=np.float32)
        np.savez_compressed(path, embeddings=self.embeddings)

    def cosine(self, query: str) -> list[dict[str, Any]]:
        self.prepare()
        vector = np.asarray(self.model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
        values = self.embeddings @ vector
        best: dict[int, tuple[float, str]] = {}
        for (doc_index, text), value in zip(self.chunk_rows, values):
            if doc_index not in best or float(value) > best[doc_index][0]:
                best[doc_index] = (float(value), text)
        result = [{"document": self.public_doc(doc), "score": round(best[i][0], 6), "excerpt": excerpt(best[i][1])} for i, doc in enumerate(self.documents)]
        return sorted(result, key=lambda row: row["score"], reverse=True)


engine = SearchEngine()
app = FastAPI(title="JEV vs Cosine Search")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/presets")
async def presets() -> dict[str, Any]:
    return {"presets": PRESETS, "defaults": {"cosine_threshold": 0.35, "jev_threshold": 0.5, "jev_concurrency": 1}, "document_count": len(engine.documents), "jev_available": api_key() is not None}


async def jev_call(client: Any, document: dict[str, Any], query: str) -> dict[str, Any]:
    from typesafe_sdk import Noul
    question = Noul(instructions=f"Does this document satisfy every inclusion and exclusion condition in this search request? Search request: {query}", criteria={"true": "The document itself provides sufficient evidence for every condition.", "false": "It is irrelevant, incomplete, or violates any exclusion condition."})
    response = await client.system_one(state={"title": document["title"], "document": document["text"]}, model=JEV_MODEL, questions={"relevant": question})
    answer = response.nouls["relevant"].noul
    usage = getattr(response, "usage", None)
    return {"score": float(answer), "model": str(response.model), "input_tokens": int(getattr(usage, "input_tokens", 0)), "output_tokens": int(getattr(usage, "output_tokens", 0))}


async def stream(body: SearchBody, request: Request) -> AsyncIterator[bytes]:
    started = time.perf_counter()
    preset = next((p for p in PRESETS if p["id"] == body.preset_id and p["query"] == body.query), None)
    truth = set(preset["ground_truth"]) if preset else set()
    yield line({"event": "start", "total": len(engine.documents), "preset_id": preset["id"] if preset else None, "ground_truth": sorted(truth)})
    cosine_started = time.perf_counter()
    cosine_rows = await asyncio.to_thread(engine.cosine, body.query)
    for row in cosine_rows:
        yield line({"event": "cosine_evaluation", **row, "is_relevant": row["document"]["doc_id"] in truth if preset else None})
    cosine_ms = round((time.perf_counter() - cosine_started) * 1000)

    key = api_key()
    if not key:
        yield line({"event": "error", "scope": "jev", "message": "TypeSafe APIキーが未設定です。cosine結果のみ表示します。"})
        yield line({"event": "complete", "cosine_ms": cosine_ms, "jev_ms": None, "total_ms": round((time.perf_counter() - started) * 1000), "jev_errors": len(engine.documents)})
        return

    from typesafe_sdk import AsyncTypeSafeClient
    jev_started = time.perf_counter()
    completed = errors = rejected = tokens_in = tokens_out = 0
    resolved_model: str | None = None
    queue: asyncio.Queue[tuple[dict[str, Any], dict[str, Any] | None, str | None]] = asyncio.Queue()
    semaphore = asyncio.Semaphore(body.jev_concurrency)

    async with AsyncTypeSafeClient(api_key=key) as client:
        async def evaluate(document: dict[str, Any]) -> None:
            async with semaphore:
                try:
                    result = await jev_call(client, document, body.query)
                    await queue.put((document, result, None))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await queue.put((document, None, type(exc).__name__))

        tasks = [asyncio.create_task(evaluate(doc)) for doc in engine.documents]
        try:
            while completed < len(tasks):
                if await request.is_disconnected():
                    break
                document, result, error_name = await queue.get()
                completed += 1
                if result is None:
                    errors += 1
                    yield line({"event": "error", "scope": "document", "doc_id": document["doc_id"], "message": f"JEV判定に失敗しました ({error_name})"})
                else:
                    resolved_model = result["model"]
                    tokens_in += result["input_tokens"]
                    tokens_out += result["output_tokens"]
                    if result["score"] < 0.5:
                        rejected += 1
                    yield line({"event": "jev_evaluation", "document": engine.public_doc(document), "score": round(result["score"], 6), "excerpt": excerpt(document["text"]), "is_relevant": document["doc_id"] in truth if preset else None})
                yield line({"event": "jev_progress", "completed": completed, "total": len(tasks), "rejected_at_default": rejected, "errors": errors})
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    yield line({"event": "complete", "cosine_ms": cosine_ms, "jev_ms": round((time.perf_counter() - jev_started) * 1000), "total_ms": round((time.perf_counter() - started) * 1000), "jev_errors": errors, "input_tokens": tokens_in, "output_tokens": tokens_out, "model": resolved_model})


@app.post("/api/search/stream")
async def search(body: SearchBody, request: Request) -> StreamingResponse:
    return StreamingResponse(stream(body, request), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
