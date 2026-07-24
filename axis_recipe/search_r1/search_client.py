from __future__ import annotations

import asyncio
import itertools
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, override

import httpx

from axrl.worker.infer_worker import InferWorker

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    title: str
    text: str
    score: float


@dataclass
class SearchRequestResult:
    documents: list[RetrievedDocument]
    server: str
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None
    status_code: int | None = None


class SearchClient(InferWorker[str, SearchRequestResult]):
    """Async client for retrieval servers with connection pooling and round-robin load balancing.

    Accepts one or more retrieval server URLs.  When multiple URLs are provided,
    requests are distributed across them in round-robin order so that each
    server shares the query load evenly.

    A single :class:`httpx.AsyncClient` is lazily initialised and shared across
    all concurrent ``retrieve()`` calls, bounded by ``httpx.Limits`` to prevent
    file-descriptor exhaustion under high concurrency.

    **Concurrency note:** When all ``max_connections`` slots are busy, additional
    ``retrieve()`` calls are automatically queued by ``httpx`` until a connection
    becomes available — this is transparent to callers.

    **Retry behaviour:** When multiple servers are configured and a request
    fails (network error, timeout, etc.), the client automatically retries on
    the next server in the round-robin cycle up to ``max_retries`` times before
    returning a failure result.  This dramatically reduces the observed failure
    rate in multi-node setups where cross-node HTTP requests are less reliable.

    Args:
        base_urls: One or more retrieval server URLs.  A single string is
            accepted for backward compatibility and converted to a one-element
            list internally.
        topk: Number of documents to retrieve per query.
        request_timeout: HTTP request timeout in seconds.
        max_connections: Upper bound on simultaneous TCP connections.
        max_keepalive_connections: Max idle connections kept alive for reuse
            between rollout batches.
        max_retries: Maximum number of retry attempts on different servers
            when a request fails.  Defaults to 2 (i.e. up to 3 total
            attempts).  Set to 0 to disable retries.
        retry_backoff_seconds: Base delay between retries. The k-th retry waits
            ``k * retry_backoff_seconds`` seconds before the next attempt.
    """

    def __init__(
        self,
        *,
        base_urls: list[str] | str = "http://127.0.0.1:18000",
        topk: int = 3,
        request_timeout: float = 30.0,
        max_connections: int = 128,
        max_keepalive_connections: int = 32,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(base_urls, str):
            base_urls = [base_urls]
        assert len(base_urls) > 0, "At least one retrieval server URL is required."
        assert retry_backoff_seconds >= 0.0, "retry_backoff_seconds must be non-negative."
        self.base_urls = base_urls
        self.topk = topk
        self.request_timeout = request_timeout
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._client: httpx.AsyncClient | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._url_cycle = itertools.cycle(base_urls)
        logger.info(
            "SearchClient initialised with %d server(s), max_retries=%d, retry_backoff_seconds=%.2f: %s",
            len(base_urls),
            max_retries,
            retry_backoff_seconds,
            base_urls,
        )

    # ------------------------------------------------------------------
    # Lazy shared client
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and return the shared ``httpx.AsyncClient``."""
        if self._client is None or self._client.is_closed:
            async with self._lock:
                # Double-check after acquiring lock
                if self._client is None or self._client.is_closed:
                    limits = httpx.Limits(
                        max_connections=self._max_connections,
                        max_keepalive_connections=self._max_keepalive_connections,
                    )
                    self._client = httpx.AsyncClient(
                        timeout=self.request_timeout,
                        limits=limits,
                    )
                    logger.info(
                        "Created shared httpx.AsyncClient (max_conn=%d, keepalive=%d)",
                        self._max_connections,
                        self._max_keepalive_connections,
                    )
        return self._client

    @override
    async def generate(self, req: str) -> SearchRequestResult:
        return await self.retrieve_with_metadata(req)

    async def close(self) -> None:
        """Close the underlying HTTP client and release its connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @override
    def shutdown(self) -> None:
        if self._client is None or self._client.is_closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close())
        else:
            self._close_task = loop.create_task(self.close())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, query: str) -> list[RetrievedDocument]:
        return (await self.retrieve_with_metadata(query)).documents

    async def retrieve_with_metadata(self, query: str) -> SearchRequestResult:
        query_list: list[str] = [query]
        payload: dict[str, Any] = {
            "queries": query_list,
            "return_scores": True,
            "topk": self.topk,
        }

        # Retry in round-robin order across servers. When retries exceed the
        # number of servers, we cycle back through the list.
        max_attempts = self._max_retries + 1
        last_result: SearchRequestResult | None = None

        for attempt in range(max_attempts):
            base_url = next(self._url_cycle)
            if random.random() < 0.001:
                logger.info("Retrieval server ip: %s (attempt %d/%d)", base_url, attempt + 1, max_attempts)

            url = base_url.rstrip("/") + "/retrieve"
            start_time = time.perf_counter()

            try:
                client = await self._get_client()
                response = await client.post(url, json=payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                response.raise_for_status()
                documents = self.parse_data(response.json())
                return SearchRequestResult(
                    documents=documents,
                    server=base_url,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                logger.warning(
                    "Retrieval request failed (attempt %d/%d): server=%s latency_ms=%.2f error_type=%s status_code=%s query=%r error=%s",
                    attempt + 1,
                    max_attempts,
                    base_url,
                    latency_ms,
                    type(exc).__name__,
                    status_code,
                    query,
                    exc,
                )
                last_result = SearchRequestResult(
                    documents=[],
                    server=base_url,
                    latency_ms=latency_ms,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    status_code=status_code,
                )
                if attempt + 1 < max_attempts and self._retry_backoff_seconds > 0.0:
                    retry_index = attempt + 1
                    sleep_seconds = retry_index * self._retry_backoff_seconds
                    logger.info(
                        "Sleeping %.2fs before retry %d/%d for query=%r",
                        sleep_seconds,
                        retry_index,
                        self._max_retries,
                        query,
                    )
                    await asyncio.sleep(sleep_seconds)

        assert last_result is not None
        return last_result

    @staticmethod
    def parse_data(data: dict[str, Any]) -> list[RetrievedDocument]:
        result = data.get("result")
        if not isinstance(result, list) or len(result) != 1:
            raise TypeError(f"Unexpected retrieval response shape: result={result!r}")

        entry = result[0]
        if not isinstance(entry, list):
            raise TypeError(f"Unexpected retrieval entry shape: entry={entry!r}")

        docs: list[RetrievedDocument] = []
        for item in entry:
            document = item["document"]
            score = float(item["score"])
            contents = document["contents"]
            if not isinstance(contents, str):
                raise TypeError(f"Unexpected document contents: {contents!r}")
            lines = contents.split("\n")
            title = lines[0]
            text = "\n".join(lines[1:]).strip()
            docs.append(RetrievedDocument(title=title, text=text, score=score))
        return docs


async def _test_client() -> None:
    from rich.pretty import pprint

    client = SearchClient(topk=3)
    try:
        docs = await client.retrieve(query="Who created the Python programming language?")
        # Note: Guido van Rossum created the Python programming language in 1991.
        pprint(docs)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_test_client())
