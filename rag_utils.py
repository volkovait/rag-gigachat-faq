"""Chunking, cosine similarity, token estimate."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple


def chunk_text(text: str, max_chars: int = 300) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + max_chars, n)
        if end < n:
            window = text[i:end]
            sp = window.rfind(" ")
            if sp >= max_chars // 2:
                end = i + sp + 1
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end <= i:
            i += 1
        else:
            i = end
    return chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class ScoredChunk(NamedTuple):
    text: str
    embedding: List[float]
    score: float


def top_k_chunks(
    question_embedding: List[float],
    store: List[Dict[str, Any]],
    k: int = 3,
) -> List[ScoredChunk]:
    scored = []
    for item in store:
        text = item.get("text")
        emb = item.get("embedding")
        if not isinstance(text, str) or not isinstance(emb, list):
            continue
        floats = []
        for x in emb:
            if isinstance(x, (int, float)):
                floats.append(float(x))
            else:
                floats = []
                break
        if not floats:
            continue
        s = cosine_similarity(question_embedding, floats)
        scored.append(ScoredChunk(text=text, embedding=floats, score=s))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:k]


def estimate_prompt_tokens_from_messages(messages: List[Dict[str, str]]) -> int:
    joined = "\n".join(m.get("content", "") for m in messages)
    raw = joined.encode("utf-8")
    return max(1, len(raw) // 3)


def build_rag_messages(context_chunks: List[str], question: str) -> List[Dict[str, str]]:
    context_block = "\n\n---\n\n".join(context_chunks)
    system = (
        "Вы официальный помощник по внутренним правилам компании. "
        "Отвечайте строго на основе переданного контекста. "
        "Если в контексте нет ответа, так и скажите: информации в базе знаний недостаточно. "
        "Не выдумывайте факты. Формулировки — деловой русский язык."
    )
    user = (
        f"Контекст (фрагменты из базы знаний):\n{context_block}\n\n"
        f"Вопрос пользователя:\n{question}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass
class PreparedEntry:
    messages: List[Dict[str, str]]
    estimated_prompt_tokens: int
    expires_at: float


PREPARED_TTL_SEC = 300.0


def new_prepared_entry(messages: List[Dict[str, str]]) -> PreparedEntry:
    est = estimate_prompt_tokens_from_messages(messages)
    return PreparedEntry(
        messages=messages,
        estimated_prompt_tokens=est,
        expires_at=time.time() + PREPARED_TTL_SEC,
    )


def purge_expired(prepared: Dict[str, PreparedEntry]) -> None:
    now = time.time()
    dead = [k for k, v in prepared.items() if v.expires_at < now]
    for k in dead:
        del prepared[k]
