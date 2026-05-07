"""
Kriki HTTP API server platform adapter.

Exposes a narrow OpenAI Responses-like endpoint:
- POST /v1/kriki/responses

The adapter intentionally keeps a smaller public surface than
gateway.platforms.api_server while reusing the same agent execution path.
"""

import asyncio
import hmac
import json
import logging
import os
import socket as _socket
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import (
    AIOHTTP_AVAILABLE,
    APIServerAdapter,
    _content_has_visible_payload,
    _multimodal_validation_error,
    _normalize_multimodal_content,
    _openai_error,
    body_limit_middleware,
    cors_middleware,
    security_headers_middleware,
    web,
)
from gateway.platforms.base import SendResult, is_network_accessible

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4088
DEFAULT_AGENT_CACHE_MAX_SIZE = 128
DEFAULT_AGENT_CACHE_IDLE_TTL_SECS = 3600.0
KRIKI_WATCH_COMMAND = "/kriki-watch"


def check_kriki_server_requirements() -> bool:
    """Return True when aiohttp is available."""
    return AIOHTTP_AVAILABLE


class KrikiServerAdapter(APIServerAdapter):
    """Narrow Responses-compatible HTTP adapter for Kriki clients."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config)
        extra = config.extra or {}
        self.platform = Platform.KRIKI_SERVER
        self._host: str = extra.get("host", os.getenv("KRIKI_SERVER_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port", os.getenv("KRIKI_SERVER_PORT", str(DEFAULT_PORT))))
        self._api_key: str = extra.get("key", os.getenv("KRIKI_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("KRIKI_SERVER_CORS_ORIGINS", "")),
        )
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("KRIKI_SERVER_MODEL_NAME", "")),
        )
        self._agent_cache_max_size = int(
            extra.get("agent_cache_max_size", os.getenv("KRIKI_AGENT_CACHE_MAX_SIZE", str(DEFAULT_AGENT_CACHE_MAX_SIZE)))
        )
        self._agent_cache_idle_ttl = float(
            extra.get("agent_cache_idle_ttl", os.getenv("KRIKI_AGENT_CACHE_IDLE_TTL", str(DEFAULT_AGENT_CACHE_IDLE_TTL_SECS)))
        )
        self._agent_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._agent_cache_lock = threading.RLock()

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        if explicit and explicit.strip():
            return explicit.strip()
        return "kriki-agent"

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        if not self._api_key:
            return None

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._api_key):
                return None

        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
    ) -> Any:
        from gateway.run import GatewayRunner, _load_gateway_config, _resolve_gateway_model, _resolve_runtime_agent_kwargs
        from hermes_cli.tools_config import _get_platform_tools
        from run_agent import AIAgent

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        model = _resolve_gateway_model()
        user_config = _load_gateway_config()
        enabled_toolsets = sorted(
            ts for ts in _get_platform_tools(user_config, "kriki_server")
            if ts != "skills"
        )
        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
        fallback_model = GatewayRunner._load_fallback_model()

        return AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            platform="kriki_server",
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
        )

    def _get_cached_agent(
        self,
        session_id: str,
        ephemeral_system_prompt: Optional[str] = None,
    ) -> Any:
        """Return a cached AIAgent for *session_id*, creating one on miss."""
        now = time.time()
        prompt = ephemeral_system_prompt or None

        with self._agent_cache_lock:
            self._prune_agent_cache_locked(now)
            cached = self._agent_cache.get(session_id)
            if cached and cached.get("ephemeral_system_prompt") == prompt:
                cached["last_used"] = now
                self._agent_cache.move_to_end(session_id)
                return cached["agent"]

            agent = self._create_agent(
                ephemeral_system_prompt=prompt,
                session_id=session_id,
            )
            self._agent_cache[session_id] = {
                "agent": agent,
                "ephemeral_system_prompt": prompt,
                "created_at": now,
                "last_used": now,
                "lock": threading.RLock(),
            }
            self._agent_cache.move_to_end(session_id)
            self._prune_agent_cache_locked(now)
            return agent

    def _get_cached_agent_lock(self, session_id: str) -> threading.RLock:
        with self._agent_cache_lock:
            cached = self._agent_cache.get(session_id)
            if cached is None:
                return threading.RLock()
            return cached["lock"]

    def _prune_agent_cache_locked(self, now: Optional[float] = None) -> None:
        if self._agent_cache_idle_ttl > 0:
            current = time.time() if now is None else now
            stale = [
                session_id
                for session_id, cached in self._agent_cache.items()
                if current - float(cached.get("last_used", current)) > self._agent_cache_idle_ttl
            ]
            for session_id in stale:
                self._agent_cache.pop(session_id, None)

        if self._agent_cache_max_size <= 0:
            self._agent_cache.clear()
            return

        while len(self._agent_cache) > self._agent_cache_max_size:
            self._agent_cache.popitem(last=False)

    @staticmethod
    def _strip_kriki_watch_prefix(message: Any) -> Any:
        if not isinstance(message, str):
            return message
        stripped = message.strip()
        if not stripped.startswith(KRIKI_WATCH_COMMAND):
            return message
        remainder = stripped[len(KRIKI_WATCH_COMMAND):].strip()
        return remainder or stripped

    def _build_kriki_watch_message(self, user_message: Any, session_id: str) -> Any:
        if not isinstance(user_message, str):
            return user_message
        try:
            from agent.skill_commands import build_skill_invocation_message

            user_instruction = self._strip_kriki_watch_prefix(user_message)
            msg = build_skill_invocation_message(
                KRIKI_WATCH_COMMAND,
                str(user_instruction),
                task_id=session_id,
                runtime_note=(
                    "Kriki API requests always use the kriki-watch skill. "
                    "Ignore other skills and do not call skill discovery/view tools."
                ),
            )
            return msg or user_message
        except Exception as e:
            logger.warning("Failed to preload %s skill for Kriki request: %s", KRIKI_WATCH_COMMAND, e)
            return user_message

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
    ) -> tuple:
        """Run a Kriki turn using the session-scoped cached AIAgent."""
        loop = asyncio.get_running_loop()
        resolved_session_id = session_id or str(uuid.uuid4())

        def _run():
            agent = self._get_cached_agent(
                resolved_session_id,
                ephemeral_system_prompt=ephemeral_system_prompt,
            )
            if agent_ref is not None:
                agent_ref[0] = agent

            agent_lock = self._get_cached_agent_lock(resolved_session_id)
            with agent_lock:
                previous_callbacks = {
                    "stream_delta_callback": getattr(agent, "stream_delta_callback", None),
                    "tool_progress_callback": getattr(agent, "tool_progress_callback", None),
                    "tool_start_callback": getattr(agent, "tool_start_callback", None),
                    "tool_complete_callback": getattr(agent, "tool_complete_callback", None),
                }
                agent.stream_delta_callback = stream_delta_callback
                agent.tool_progress_callback = tool_progress_callback
                agent.tool_start_callback = tool_start_callback
                agent.tool_complete_callback = tool_complete_callback
                before_input = getattr(agent, "session_prompt_tokens", 0) or 0
                before_output = getattr(agent, "session_completion_tokens", 0) or 0
                before_total = getattr(agent, "session_total_tokens", 0) or 0
                try:
                    result = agent.run_conversation(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        task_id="default",
                    )
                    after_input = getattr(agent, "session_prompt_tokens", 0) or 0
                    after_output = getattr(agent, "session_completion_tokens", 0) or 0
                    after_total = getattr(agent, "session_total_tokens", 0) or 0
                finally:
                    for name, callback in previous_callbacks.items():
                        setattr(agent, name, callback)

            usage = {
                "input_tokens": max(0, after_input - before_input),
                "output_tokens": max(0, after_output - before_output),
                "total_tokens": max(0, after_total - before_total),
            }
            return result, usage

        return await loop.run_in_executor(None, _run)

    @staticmethod
    def _extract_kriki_output_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build Kriki's Responses output without dropping the final answer."""
        items: List[Dict[str, Any]] = []
        for msg in result.get("messages", []):
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                items.append({
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": func.get("arguments", ""),
                    "call_id": tc.get("id", "") or f"call_{uuid.uuid4().hex[:24]}",
                    "name": func.get("name", ""),
                })

        final = result.get("final_response", "") or result.get("error", "")
        if not final and not items:
            final = "(No response generated)"
        if final:
            items.append({
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": final,
                }],
                "role": "assistant",
            })
        return items

    async def _handle_kriki_responses(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        raw_input = body.get("input")
        if raw_input is None:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        input_messages: List[Dict[str, Any]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for idx, item in enumerate(raw_input):
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    try:
                        content = _normalize_multimodal_content(item.get("content", ""))
                    except ValueError as exc:
                        return _multimodal_validation_error(exc, param=f"input[{idx}].content")
                    input_messages.append({"role": role, "content": content})
        else:
            return web.json_response(_openai_error("'input' must be a string or array"), status=400)

        conversation_history: List[Dict[str, Any]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(_openai_error("'conversation_history' must be an array of message objects"), status=400)
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                try:
                    content = _normalize_multimodal_content(entry["content"])
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"conversation_history[{i}].content")
                conversation_history.append({"role": str(entry["role"]), "content": content})

        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        user_message: Any = input_messages[-1].get("content", "") if input_messages else ""
        if not _content_has_visible_payload(user_message):
            return web.json_response(_openai_error("No user message found in input"), status=400)

        session_id = request.headers.get("X-Hermes-Session-Id", "").strip() or str(uuid.uuid4())
        user_message = self._build_kriki_watch_message(user_message, session_id)
        stream = bool(body.get("stream", False))
        if stream:
            import queue as _q

            stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                if delta is not None:
                    stream_q.put(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                return

            def _on_tool_start(tool_call_id, function_name, function_args):
                stream_q.put(("__tool_started__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                stream_q.put(("__tool_completed__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                    "result": function_result,
                }))

            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=body.get("instructions"),
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
            ))

            return await self._write_sse_responses(
                request=request,
                response_id=f"resp_{uuid.uuid4().hex[:28]}",
                model=body.get("model", self._model_name),
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=conversation_history,
                user_message=user_message,
                instructions=body.get("instructions"),
                conversation=None,
                store=False,
                session_id=session_id,
            )

        try:
            result, usage = await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=body.get("instructions"),
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Error running agent for Kriki responses: %s", e, exc_info=True)
            return web.json_response(
                _openai_error(f"Internal server error: {e}", err_type="server_error"),
                status=500,
            )

        created_at = int(time.time())
        response_data = {
            "id": f"resp_{uuid.uuid4().hex[:28]}",
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": body.get("model", self._model_name),
            "output": self._extract_kriki_output_items(result),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        return web.json_response(response_data, headers={"X-Hermes-Session-Id": session_id})

    async def connect(self) -> bool:
        """Start the Kriki aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        try:
            mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware) if mw is not None]
            self._app = web.Application(middlewares=mws)
            self._app["kriki_server_adapter"] = self
            self._app.router.add_post("/v1/kriki/responses", self._handle_kriki_responses)

            if is_network_accessible(self._host) and not self._api_key:
                logger.error(
                    "[%s] Refusing to start: binding to %s requires KRIKI_SERVER_KEY. "
                    "Set KRIKI_SERVER_KEY or use the default 127.0.0.1.",
                    self.name, self._host,
                )
                return False

            if is_network_accessible(self._host) and self._api_key:
                try:
                    from hermes_cli.auth import has_usable_secret
                    if not has_usable_secret(self._api_key, min_length=8):
                        logger.error("[%s] Refusing to start: KRIKI_SERVER_KEY is a placeholder value.", self.name)
                        return False
                except ImportError:
                    pass

            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                    _s.settimeout(1)
                    _s.connect(("127.0.0.1", self._port))
                logger.error(
                    "[%s] Port %d already in use. Set a different port in config.yaml: platforms.kriki_server.port",
                    self.name, self._port,
                )
                return False
            except (ConnectionRefusedError, OSError):
                pass

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            self._mark_connected()
            if not self._api_key:
                logger.warning(
                    "[%s] No API key configured (KRIKI_SERVER_KEY / platforms.kriki_server.key). "
                    "Local requests will be accepted without authentication.",
                    self.name,
                )
            logger.info(
                "[%s] Kriki server listening on http://%s:%d/v1/kriki/responses (model: %s)",
                self.name, self._host, self._port, self._model_name,
            )
            return True
        except Exception as e:
            logger.error("[%s] Failed to start Kriki server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        self._mark_disconnected()
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        with self._agent_cache_lock:
            self._agent_cache.clear()
        self._app = None
        logger.info("[%s] Kriki server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return SendResult(success=False, error="Kriki server uses HTTP request/response, not send()")
