"""All calls from the Streamlit UI to the FastAPI backend live here."""
from __future__ import annotations

from typing import Any

import requests


class ApiError(Exception):
    """The backend could not be reached or returned an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail)


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = requests.request(method, f"{self.base_url}{path}", **kwargs)
        except requests.RequestException as error:
            raise ApiError(f"Cannot reach the backend at {self.base_url}.") from error
        if response.status_code >= 400:
            raise ApiError(_detail(response), response.status_code)
        return response.json() if response.content else None

    # ---- documents
    def health(self) -> dict:
        return self._request("GET", "/health", timeout=10)

    def list_documents(self) -> list[dict]:
        return self._request("GET", "/documents", timeout=10)

    def get_document(self, document_id: str) -> dict:
        return self._request("GET", f"/documents/{document_id}", timeout=10)

    def upload(self, filename: str, content: bytes) -> dict:
        return self._request("POST", "/documents/upload", files={"file": (filename, content)})

    def delete(self, document_id: str) -> dict:
        return self._request("DELETE", f"/documents/{document_id}", timeout=30)

    def graph_status(self, document_id: str) -> dict:
        return self._request("GET", f"/documents/{document_id}/graph", timeout=10)

    # ---- question answering
    def query(self, question: str) -> dict:
        return self._request("POST", "/query", json={"question": question})
