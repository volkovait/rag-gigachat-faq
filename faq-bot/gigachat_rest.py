"""GigaChat REST: OAuth token cache, embeddings, chat completions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx


def _authorization_basic(credentials: str) -> str:
    c = credentials.strip()
    if c.lower().startswith("basic "):
        return c
    return f"Basic {c}"


@dataclass
class GigaChatSettings:
    credentials: str
    scope: str = "GIGACHAT_API_PERS"
    api_base: str = "https://gigachat.devices.sberbank.ru/api"
    oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    verify_ssl: bool = True
    embeddings_model: str = "Embeddings"
    chat_model: str = "GigaChat"
    timeout: float = 120.0


class GigaChatRestClient:
    def __init__(self, settings: GigaChatSettings) -> None:
        self._s = settings
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def _client(self) -> httpx.Client:
        return httpx.Client(verify=self._s.verify_ssl, timeout=self._s.timeout)

    def _refresh_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expires_at - 60:
            return self._token

        with self._client() as client:
            r = client.post(
                self._s.oauth_url,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "RqUID": str(uuid.uuid4()),
                    "Authorization": _authorization_basic(self._s.credentials),
                },
                data=urlencode({"scope": self._s.scope}),
            )
            r.raise_for_status()
            data = r.json()
        access = data.get("access_token")
        if not access or not isinstance(access, str):
            raise RuntimeError(f"OAuth response missing access_token: {data!r}")
        expires_at = data.get("expires_at")
        if isinstance(expires_at, (int, float)):
            self._token_expires_at = float(expires_at)
        else:
            self._token_expires_at = now + 25 * 60
        self._token = access
        return access

    def _bearer(self) -> str:
        return self._refresh_token()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        token = self._bearer()
        url = f"{self._s.api_base.rstrip('/')}/v1/embeddings"
        out: list[list[float]] = []
        with self._client() as client:
            r = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={"model": self._s.embeddings_model, "input": texts},
            )
            if r.status_code >= 400:
                raise RuntimeError(
                    f"Embeddings HTTP {r.status_code}: {r.text[:500]}"
                )
            data = r.json()
        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected embeddings response: {json.dumps(data)[:800]}")
        # Sort by index if present
        def sort_key(x: Any) -> int:
            if isinstance(x, dict) and isinstance(x.get("index"), int):
                return x["index"]
            return 0

        items = sorted(items, key=sort_key)
        for item in items:
            if not isinstance(item, dict):
                continue
            emb = item.get("embedding")
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                out.append([float(x) for x in emb])
        if len(out) != len(texts):
            raise RuntimeError(
                f"Embeddings count mismatch: got {len(out)}, expected {len(texts)}"
            )
        return out

    def chat_completions(self, messages: list[dict[str, str]]) -> str:
        token = self._bearer()
        url = f"{self._s.api_base.rstrip('/')}/v1/chat/completions"
        with self._client() as client:
            r = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._s.chat_model,
                    "messages": messages,
                },
            )
            if r.status_code >= 400:
                raise RuntimeError(f"Chat HTTP {r.status_code}: {r.text[:800]}")
            data = r.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Unexpected chat response: {json.dumps(data)[:800]}")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("Invalid choices[0]")
        msg = first.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
        raise RuntimeError(f"No message.content in response: {json.dumps(data)[:800]}")
