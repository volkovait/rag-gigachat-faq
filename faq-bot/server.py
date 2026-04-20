#!/usr/bin/env python3
"""FAQ-bot HTTP server: POST /index, /prepare, /ask; GET /."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from gigachat_rest import GigaChatRestClient, GigaChatSettings
from rag_utils import (
    PreparedEntry,
    build_rag_messages,
    chunk_text,
    new_prepared_entry,
    purge_expired,
    top_k_chunks,
)

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# In-memory store: list of {"text", "embedding"}
CHUNKS = []  # type: list[dict[str, object]]
PREPARED: dict[str, PreparedEntry] = {}

MAX_EMBED_BATCH = 90


def _truthy_env(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def make_client() -> GigaChatRestClient:
    cred = os.environ.get("GIGACHAT_CREDENTIALS", "").strip()
    if not cred:
        raise RuntimeError("Set GIGACHAT_CREDENTIALS in environment or .env")
    settings = GigaChatSettings(
        credentials=cred,
        scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
        api_base=os.environ.get(
            "GIGACHAT_API_BASE", "https://gigachat.devices.sberbank.ru/api"
        ).strip(),
        oauth_url=os.environ.get(
            "GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        ).strip(),
        verify_ssl=_truthy_env("GIGACHAT_VERIFY_SSL", default=False),
        embeddings_model=os.environ.get("GIGACHAT_EMBEDDINGS_MODEL", "Embeddings").strip(),
        chat_model=os.environ.get("GIGACHAT_CHAT_MODEL", "GigaChat").strip(),
    )
    return GigaChatRestClient(settings)


def embed_texts_batched(client: GigaChatRestClient, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    all_emb: list[list[float]] = []
    i = 0
    while i < len(texts):
        batch = texts[i : i + MAX_EMBED_BATCH]
        try:
            all_emb.extend(client.embed_texts(batch))
            i += len(batch)
        except Exception:
            if len(batch) == 1:
                raise
            half = max(1, len(batch) // 2)
            all_emb.extend(embed_texts_batched(client, batch[:half]))
            all_emb.extend(embed_texts_batched(client, batch[half:]))
            i += len(batch)
    return all_emb


def json_response(handler: BaseHTTPRequestHandler, code: int, body: object) -> None:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)


def text_response(
    handler: BaseHTTPRequestHandler, code: int, body: bytes, content_type: str
) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> object:
    length = handler.headers.get("Content-Length")
    if not length:
        return None
    try:
        n = int(length)
    except ValueError:
        return None
    raw = handler.rfile.read(n)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/":
            json_response(self, 404, {"error": "not_found"})
            return
        if not INDEX_HTML.is_file():
            json_response(self, 500, {"error": "index.html missing"})
            return
        body = INDEX_HTML.read_bytes()
        text_response(self, 200, body, "text/html; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        purge_expired(PREPARED)

        if path == "/index":
            self._post_index()
        elif path == "/prepare":
            self._post_prepare()
        elif path == "/ask":
            self._post_ask()
        else:
            json_response(self, 404, {"error": "not_found"})

    def _post_index(self) -> None:
        try:
            client = make_client()
        except RuntimeError as e:
            json_response(self, 500, {"error": str(e)})
            return
        body = read_json_body(self)
        if not isinstance(body, dict):
            json_response(self, 400, {"error": "invalid_json"})
            return
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            json_response(self, 400, {"error": "text required"})
            return
        pieces = chunk_text(text, 300)
        if not pieces:
            json_response(self, 400, {"error": "empty after chunking"})
            return
        try:
            vectors = embed_texts_batched(client, pieces)
        except Exception as e:
            json_response(self, 502, {"error": f"embeddings failed: {e}"})
            return
        global CHUNKS
        CHUNKS = [
            {"text": t, "embedding": v}
            for t, v in zip(pieces, vectors, strict=True)
        ]
        PREPARED.clear()
        json_response(self, 200, {"chunks": len(CHUNKS)})

    def _post_prepare(self) -> None:
        body = read_json_body(self)
        if not isinstance(body, dict):
            json_response(self, 400, {"error": "invalid_json"})
            return
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            json_response(self, 400, {"error": "question required"})
            return
        if not CHUNKS:
            json_response(self, 400, {"error": "index is empty; upload knowledge first"})
            return
        try:
            client = make_client()
        except RuntimeError as e:
            json_response(self, 500, {"error": str(e)})
            return
        try:
            qv = client.embed_texts([question.strip()])[0]
        except Exception as e:
            json_response(self, 502, {"error": f"question embedding failed: {e}"})
            return
        top = top_k_chunks(qv, CHUNKS, k=3)
        if not top:
            json_response(self, 400, {"error": "no valid chunks for retrieval"})
            return
        messages = build_rag_messages([c.text for c in top], question.strip())
        entry = new_prepared_entry(messages)
        pid = str(uuid.uuid4())
        PREPARED[pid] = entry
        json_response(
            self,
            200,
            {
                "prepare_id": pid,
                "estimated_prompt_tokens": entry.estimated_prompt_tokens,
                "chunks_used": len(top),
                "warning": (
                    "Запрос к модели чата тарифицируется по правилам GigaChat. "
                    "Указанное число токенов — приблизительная оценка по длине текста; "
                    "фактический расход может отличаться."
                ),
            },
        )

    def _post_ask(self) -> None:
        body = read_json_body(self)
        if not isinstance(body, dict):
            json_response(self, 400, {"error": "invalid_json"})
            return
        pid = body.get("prepare_id")
        if not isinstance(pid, str) or not pid:
            json_response(self, 400, {"error": "prepare_id required"})
            return
        entry = PREPARED.get(pid)
        if entry is None:
            json_response(
                self,
                400,
                {"error": "unknown or expired prepare_id; run prepare again"},
            )
            return
        if entry.expires_at < time.time():
            del PREPARED[pid]
            json_response(
                self,
                400,
                {"error": "prepare_id expired; run prepare again"},
            )
            return
        try:
            client = make_client()
        except RuntimeError as e:
            json_response(self, 500, {"error": str(e)})
            return
        try:
            answer = client.chat_completions(entry.messages)
        except Exception as e:
            json_response(self, 502, {"error": f"chat failed: {e}"})
            return
        del PREPARED[pid]
        json_response(self, 200, {"answer": answer})


def main() -> None:
    host = os.environ.get("FAQ_BOT_HOST", "127.0.0.1").strip()
    port = int(os.environ.get("FAQ_BOT_PORT", "8765"))
    server = HTTPServer((host, port), Handler)
    print(f"FAQ-bot listening on http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
