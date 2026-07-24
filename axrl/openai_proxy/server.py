from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import ray
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from uvicorn import Server

logger = logging.getLogger(__name__)
_REMOTE_REGISTRY_MAX_CONCURRENCY = 8192


class _OpenAIProxyResponder(Protocol):
    async def respond(self, request_id: str, body: dict[str, Any], *, status_code: int = 200, headers: dict[str, str] | None = None) -> None: ...

    async def fail(self, request_id: str, message: str, *, status_code: int = 500) -> None: ...


@dataclass
class OpenAIProxyResponse:
    body: dict[str, Any]
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class OpenAIPendingRequest:
    session_id: str
    request_id: str
    body: dict[str, Any]
    headers: dict[str, str]
    created_at: float
    _responder: _OpenAIProxyResponder | None = field(default=None, repr=False, compare=False)

    async def respond(self, body: dict[str, Any], *, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        assert self._responder is not None, "OpenAIPendingRequest is detached from its registry."
        await self._responder.respond(self.request_id, body, status_code=status_code, headers=headers)

    async def fail(self, message: str, *, status_code: int = 500) -> None:
        assert self._responder is not None, "OpenAIPendingRequest is detached from its registry."
        await self._responder.fail(self.request_id, message, status_code=status_code)

    def attach_responder(self, responder: _OpenAIProxyResponder) -> OpenAIPendingRequest:
        self._responder = responder
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_responder"] = None
        return state


@dataclass
class _PendingState:
    request: OpenAIPendingRequest
    future: asyncio.Future[OpenAIProxyResponse]


class _OpenAIProxySessionRegistryCore:
    def __init__(self, *, request_timeout_seconds: float = 600.0) -> None:
        self.request_timeout_seconds = request_timeout_seconds
        self._sessions: dict[str, asyncio.Queue[OpenAIPendingRequest]] = {}
        self._requests: dict[str, _PendingState] = {}
        self._closed: set[str] = set()

    async def create_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            raise ValueError(f"OpenAI proxy session already exists: {session_id}")
        self._sessions[session_id] = asyncio.Queue()
        self._closed.discard(session_id)

    async def close_session(self, session_id: str) -> None:
        self._closed.add(session_id)
        queue = self._sessions.pop(session_id, None)
        if queue is not None:
            while not queue.empty():
                pending = queue.get_nowait()
                await self.fail(pending.request_id, "OpenAI proxy session was closed", status_code=499)
        request_ids = [request_id for request_id, state in self._requests.items() if state.request.session_id == session_id]
        for request_id in request_ids:
            await self.fail(request_id, "OpenAI proxy session was closed", status_code=499)

    async def submit_chat_completion(
        self,
        *,
        session_id: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> OpenAIProxyResponse:
        if session_id in self._closed:
            return OpenAIProxyResponse(
                body={"error": {"message": f"OpenAI proxy session is closed: {session_id}", "type": "not_found", "code": 404}},
                status_code=404,
            )
        queue = self._sessions.get(session_id)
        if queue is None:
            return OpenAIProxyResponse(
                body={"error": {"message": f"Unknown OpenAI proxy session: {session_id}", "type": "not_found", "code": 404}},
                status_code=404,
            )
        loop = asyncio.get_running_loop()
        pending = OpenAIPendingRequest(
            session_id=session_id,
            request_id=uuid.uuid4().hex,
            body=body,
            headers=headers,
            created_at=time.time(),
        )
        self._requests[pending.request_id] = _PendingState(request=pending, future=loop.create_future())
        await queue.put(pending)
        try:
            return await asyncio.wait_for(self._requests[pending.request_id].future, timeout=self.request_timeout_seconds)
        except TimeoutError:
            self._requests.pop(pending.request_id, None)
            return OpenAIProxyResponse(
                body={"error": {"message": "Timed out waiting for rollout response", "type": "timeout", "code": 504}},
                status_code=504,
            )

    async def wait_for_request(self, session_id: str, *, timeout_seconds: float | None = None) -> OpenAIPendingRequest:
        queue = self._sessions.get(session_id)
        if queue is None:
            raise KeyError(f"Unknown OpenAI proxy session: {session_id}")
        while True:
            if timeout_seconds is None:
                pending = await queue.get()
            else:
                pending = await asyncio.wait_for(queue.get(), timeout=timeout_seconds)
            if pending.request_id in self._requests:
                return pending

    async def respond(self, request_id: str, body: dict[str, Any], *, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        state = self._requests.pop(request_id, None)
        if state is None or state.future.done():
            return
        state.future.set_result(OpenAIProxyResponse(body=body, status_code=status_code, headers=headers or {}))

    async def fail(self, request_id: str, message: str, *, status_code: int = 500) -> None:
        await self.respond(
            request_id,
            {
                "error": {
                    "message": message,
                    "type": "openai_proxy_error",
                    "code": status_code,
                }
            },
            status_code=status_code,
        )


@ray.remote
class _RemoteOpenAIProxySessionRegistry(_OpenAIProxySessionRegistryCore):
    """Actor-backed OpenAI proxy rendezvous service.

    The actor owns the request futures so HTTP proxy workers and rollout env
    actors on different processes or nodes can share one session registry.
    """


class OpenAIProxySessionRegistry:
    """Shared handoff point between the HTTP server and envs.

    The controller starts one ``OpenAIProxyServer`` for many concurrent rollout
    envs. That means all OpenHands processes send HTTP requests to the same
    FastAPI server. The registry is the object that separates those requests by
    ``session_id`` and hands each request to the matching env.

    The env side registers a session, waits for OpenHands' next model request,
    generates a response, and resolves the pending HTTP request:

        await registry.create_session(session_id)
        pending = await registry.wait_for_request(session_id)

        # pending.body is the raw OpenAI-compatible request from OpenHands.
        # The env converts it to GenerationInput, runs rollout generation, and
        # packs GenerationOutput back into an OpenAI-compatible response.
        await pending.respond(openai_response_json)
        await registry.close_session(session_id)

    The server side calls ``submit_chat_completion(session_id, ...)`` from the
    FastAPI route. That method creates the pending request, places it in the
    matching session queue, and waits for ``await pending.respond(...)``.

    ``await pending.respond(...)`` resolves the exact future that
    ``submit_chat_completion()`` is awaiting, so the original OpenHands HTTP
    request resumes with the packed response JSON.

    This wrapper is always actor-backed. A newly constructed registry creates
    the shared Ray actor, and ``from_remote_actor(...)`` creates lightweight
    wrappers for other processes or nodes.
    """

    def __init__(self, *, request_timeout_seconds: float = 600.0, actor: ActorHandle | None = None, owns_actor: bool | None = None) -> None:
        if actor is None:
            assert ray.is_initialized(), "OpenAIProxySessionRegistry requires ray.init() before construction."
            actor = self.initialize_remote_actor(request_timeout_seconds=request_timeout_seconds)
            owns_actor = True
        self._actor = actor
        self._owns_actor = bool(owns_actor)

    def get_actor_handle(self) -> ActorHandle:
        return self._actor

    @classmethod
    def from_remote_actor(cls, actor: ActorHandle) -> OpenAIProxySessionRegistry:
        return cls(actor=actor, owns_actor=False)

    @classmethod
    def remote(cls, *, request_timeout_seconds: float = 600.0) -> OpenAIProxySessionRegistry:
        return cls(actor=cls.initialize_remote_actor(request_timeout_seconds=request_timeout_seconds), owns_actor=True)

    @staticmethod
    def initialize_remote_actor(*, request_timeout_seconds: float = 600.0) -> ActorHandle:
        return cast(
            "ActorHandle",
            _RemoteOpenAIProxySessionRegistry.options(max_concurrency=_REMOTE_REGISTRY_MAX_CONCURRENCY, num_cpus=1).remote(  # type: ignore[attr-defined]
                request_timeout_seconds=request_timeout_seconds
            ),
        )

    async def create_session(self, session_id: str) -> None:
        await self._actor.create_session.remote(session_id)

    async def close_session(self, session_id: str) -> None:
        await self._actor.close_session.remote(session_id)

    async def submit_chat_completion(
        self,
        *,
        session_id: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> OpenAIProxyResponse:
        return cast(
            "OpenAIProxyResponse",
            await self._actor.submit_chat_completion.remote(session_id=session_id, body=body, headers=headers),
        )

    async def wait_for_request(self, session_id: str, *, timeout_seconds: float | None = None) -> OpenAIPendingRequest:
        pending = cast(
            "OpenAIPendingRequest",
            await self._actor.wait_for_request.remote(session_id, timeout_seconds=timeout_seconds),
        )
        return pending.attach_responder(self)

    async def respond(self, request_id: str, body: dict[str, Any], *, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        await self._actor.respond.remote(request_id, body, status_code=status_code, headers=headers)

    async def fail(self, request_id: str, message: str, *, status_code: int = 500) -> None:
        await self._actor.fail.remote(request_id, message, status_code=status_code)

    def shutdown(self) -> None:
        if not self._owns_actor or not ray.is_initialized():
            return
        with suppress(Exception):
            ray.kill(self._actor, no_restart=True)
        self._owns_actor = False


class OpenAIProxyServer:
    """Session-scoped OpenAI-compatible HTTP proxy for black-box rollouts.

    The server does not call SGLang directly and does not pack model responses.
    It only provides the HTTP boundary that external agents, such as
    OpenHands, see as an OpenAI ``/v1`` endpoint.

    Each rollout receives a unique session URL:

        http://host:port/sessions/{session_id}/v1

    A controller process normally owns one shared ``OpenAIProxyServer`` and one
    shared ``OpenAIProxySessionRegistry``. Individual envs do not create their
    own HTTP servers. Instead, each env calls
    ``await registry.create_session()`` with its unique session id, so the
    registry holds many independent queues:

        session A -> Queue[A] -> Env[A]
        session B -> Queue[B] -> Env[B]

    OpenHands sends chat-completion requests to that scoped URL. The FastAPI
    route extracts ``session_id`` from the path, stores the raw OpenAI-compatible
    request body and headers in an ``OpenAIPendingRequest``, and pushes it into
    only that session's queue. The matching ``OpenHandsEnv`` awaits that queue
    through ``wait_for_request()``, converts the pending request into
    SGLang-ready ``GenerationInput`` with ``OpenAIChatAdapter``, and records the
    converted input in its SGLang I/O trace. After the rollout worker returns
    ``GenerationOutput``, the env asks the same adapter to build the normal
    OpenAI chat-completion response and resolves the pending HTTP future with
    ``await pending.respond(...)``.

    In short: this class owns session routing and request parking; the env owns
    SGLang input capture, generation, response packing, and resuming OpenHands.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        registry: OpenAIProxySessionRegistry | None = None,
        request_timeout_seconds: float = 600.0,
        auth_token: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.registry = registry or OpenAIProxySessionRegistry(request_timeout_seconds=request_timeout_seconds)
        self._owns_registry = registry is None
        self.auth_token = auth_token
        self._server: Server | None = None
        self._server_task: asyncio.Task[None] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def session_base_url(self, session_id: str, *, public_host: str | None = None) -> str:
        host = public_host or self.host
        return f"http://{host}:{self.port}/sessions/{session_id}/v1"

    async def start(self) -> None:
        import uvicorn

        app = self._build_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._server = server
        self._server_task = asyncio.create_task(server.serve())
        await self._wait_until_started()

    async def stop(self) -> None:
        try:
            if self._server is not None:
                self._server.should_exit = True
            if self._server_task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(self._server_task), timeout=5.0)
                except TimeoutError:
                    logger.warning("Timed out waiting for OpenAI proxy server shutdown; cancelling server task.")
                    self._server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._server_task
                except asyncio.CancelledError:
                    if not self._server_task.cancelled():
                        raise
                    logger.debug("OpenAI proxy server task was cancelled during shutdown.")
                except Exception:
                    logger.warning("OpenAI proxy server exited with an exception during shutdown.", exc_info=True)
        finally:
            if self._owns_registry:
                self.registry.shutdown()
            self._server = None
            self._server_task = None

    async def _wait_until_started(self) -> None:
        assert self._server is not None
        for _ in range(500):
            if self._server.started:
                servers = getattr(self._server, "servers", None) or []
                if self.port == 0 and servers:
                    sockets = servers[0].sockets
                    self.port = int(sockets[0].getsockname()[1])
                return
            await asyncio.sleep(0.01)
        raise TimeoutError("Timed out starting OpenAIProxyServer")

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/sessions/{session_id}/v1/models")
        async def models(session_id: str, request: Request) -> JSONResponse:
            unauthorized = self._unauthorized_response(request)
            if unauthorized is not None:
                return unauthorized
            return JSONResponse(
                content={
                    "object": "list",
                    "data": [
                        {
                            "id": f"openai-proxy-session-{session_id}",
                            "object": "model",
                            "owned_by": "axrl",
                        }
                    ],
                }
            )

        @app.post("/sessions/{session_id}/v1/chat/completions")
        async def chat_completions(session_id: str, request: Request) -> JSONResponse:
            unauthorized = self._unauthorized_response(request)
            if unauthorized is not None:
                return unauthorized
            try:
                body = await request.json()
            except ClientDisconnect:
                logger.info("OpenAI proxy client disconnected while reading request body: session=%s.", session_id)
                return JSONResponse(
                    content={
                        "error": {
                            "message": "Client disconnected while sending request body",
                            "type": "client_disconnect",
                            "code": 499,
                        }
                    },
                    status_code=499,
                )
            logger.debug(
                "OpenAI proxy received chat completion: session=%s messages=%s stream=%s",
                session_id,
                len(body.get("messages", [])) if isinstance(body, dict) else None,
                body.get("stream") if isinstance(body, dict) else None,
            )
            response = await self.registry.submit_chat_completion(
                session_id=session_id,
                body=body,
                headers=dict(request.headers),
            )
            return JSONResponse(
                content=response.body,
                status_code=response.status_code,
                headers=response.headers,
            )

        @app.post("/v1/chat/completions")
        async def reject_unscoped_chat() -> JSONResponse:
            return JSONResponse(
                content={
                    "error": {
                        "message": "Use /sessions/{session_id}/v1/chat/completions for openai_proxy rollouts",
                        "type": "session_required",
                        "code": 404,
                    }
                },
                status_code=404,
            )

        return app

    def _unauthorized_response(self, request: Request) -> JSONResponse | None:
        if self.auth_token is None:
            return None
        authorization = request.headers.get("authorization", "")
        expected = f"Bearer {self.auth_token}"
        if secrets.compare_digest(authorization, expected):
            return None
        return JSONResponse(
            content={
                "error": {
                    "message": "Missing or invalid bearer token.",
                    "type": "unauthorized",
                    "code": 401,
                }
            },
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
