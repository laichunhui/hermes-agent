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

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 4088
DEFAULT_AGENT_CACHE_MAX_SIZE = 128
DEFAULT_AGENT_CACHE_IDLE_TTL_SECS = 3600.0
DEFAULT_AGENT_INACTIVITY_TIMEOUT_SECS = 120.0
KRIKI_WATCH_COMMAND = "/kriki-watch"


def check_kriki_server_requirements() -> bool:
    """Return True when aiohttp is available."""
    return AIOHTTP_AVAILABLE


class KrikiServerAdapter(APIServerAdapter):
    """Narrow Responses-compatible HTTP adapter for Kriki clients."""

    # Cached kriki-watch skill message template: (prefix, suffix) around the
    # user instruction, or (None, None) when the skill doesn't exist.  Built
    # once at init to eliminate per-request disk IO and skill scanning.
    _KRIKI_WATCH_PLACEHOLDER = "___KRIKI_USER_INSTRUCTION___"

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
        self._agent_inactivity_timeout = float(
            extra.get(
                "agent_inactivity_timeout",
                os.getenv("KRIKI_AGENT_INACTIVITY_TIMEOUT", str(DEFAULT_AGENT_INACTIVITY_TIMEOUT_SECS)),
            )
        )
        self._agent_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._agent_cache_lock = threading.RLock()
        # Pre-load the kriki-watch skill template once at startup to eliminate
        # per-request disk IO, file scanning, and string formatting.
        self._kriki_watch_prefix, self._kriki_watch_suffix = self._preload_kriki_watch_template()

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
            _get_platform_tools(user_config, "kriki_server")
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

    def _get_cached_agent(self, session_id: str) -> Any:
        """Return a cached AIAgent for *session_id*, creating one on miss.

        The ``ephemeral_system_prompt`` is intentionally NOT a cache key here.
        It is injected per-turn inside ``_run_agent`` (within the agent lock),
        just like the stream/tool callbacks.  This ensures:

        * The agent is never discarded just because ``language`` changed
          between two requests on the same session.
        * Language (or any other ephemeral directive) is always current for
          the turn that is about to run, not stale from a prior request.
        * Conversation continuity is preserved across language changes.
        """
        now = time.time()

        with self._agent_cache_lock:
            self._prune_agent_cache_locked(now)
            cached = self._agent_cache.get(session_id)
            if cached:
                cached["last_used"] = now
                self._agent_cache.move_to_end(session_id)
                return cached["agent"]

            agent = self._create_agent(session_id=session_id)
            self._agent_cache[session_id] = {
                "agent": agent,
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

    def _preload_kriki_watch_template(self) -> tuple:
        """Pre-load the kriki-watch skill message template at init time.

        Returns (prefix, suffix) where prefix is everything before the user
        instruction placeholder and suffix is everything after.  If the skill
        doesn't exist or fails to load, returns (None, None).

        This eliminates per-request disk IO (file scanning + skill read) and
        string formatting, which were the dominant latency contributors.
        """
        try:
            from agent.skill_commands import build_skill_invocation_message

            msg = build_skill_invocation_message(
                KRIKI_WATCH_COMMAND,
                self._KRIKI_WATCH_PLACEHOLDER,
                task_id=None,
                runtime_note=(
                    "Do not call skill discovery or skill_view tools. "
                    "FALLBACK RULE (overrides any 'output nothing' instruction in the skill): "
                    "When the user's request is unrelated to watch device control or weather "
                    "(e.g. general questions, knowledge, casual chat, stories, math, "
                    "translation), answer it directly using your general knowledge or web "
                    "search — do NOT output 'no response', empty text, or any refusal. "
                    "Reserve silence / 'no response' only for requests that would require an "
                    "unsupported hardware command on the watch itself."
                ),
            )
            if not msg:
                logger.debug(
                    "Kriki-watch skill (%s) not found — messages will be forwarded as-is.",
                    KRIKI_WATCH_COMMAND,
                )
                return None, None

            # Split the formatted message around the placeholder to recover the
            # static prefix and suffix.  The user instruction is injected at
            # request time.
            parts = msg.split(self._KRIKI_WATCH_PLACEHOLDER, 1)
            if len(parts) == 2:
                prefix, suffix = parts[0], parts[1]
            else:
                prefix, suffix = msg, ""
            logger.info(
                "Pre-loaded kriki-watch skill template (%d chars prefix, %d chars suffix).",
                len(prefix), len(suffix),
            )
            return prefix, suffix
        except Exception as e:
            logger.warning(
                "Failed to preload %s skill template: %s — per-request loading will be used as fallback.",
                KRIKI_WATCH_COMMAND, e,
            )
            return None, None

    def _build_kriki_watch_message(
        self,
        user_message: Any,
        session_id: str,
        device_id: str = "",
        language: str = "",
    ) -> Any:
        """Wrap the user message with the kriki-watch skill ONLY when the user
        explicitly types the /kriki-watch slash command.

        Matches the CLI pattern: skills are injected on explicit invocation,
        not applied to every message.  Uses the cached template (built at init
        time) to avoid per-request disk IO.

        *device_id* and *language* are injected as structured context lines
        **inside the user message** so the model sees them at the same attention
        level as the actual instruction.  Putting them in the system prompt
        (ephemeral_system_prompt) would bury them after a long prompt and make
        them easy to ignore, especially when the skill content is in a different
        language.
        """
        if not isinstance(user_message, str):
            return user_message

        # Only inject the skill when the user explicitly invokes /kriki-watch.
        # The user can type "/kriki-watch 打开运动" or just "打开运动" — only
        # the former triggers skill injection.  This matches CLI's slash-command
        # dispatch in cli.py (lines 6415–6426).
        stripped = user_message.strip()

        # Build a compact context prefix that carries both device and language
        # directives in one block, e.g.:
        #   [Target device: watch-abc]
        #   [Response language: zh-CN]
        context_lines = []
        if device_id:
            context_lines.append(f"[Target device: {device_id}]")
        if language:
            context_lines.append(f"[Response language: {language}]")
        context_prefix = "\n".join(context_lines) + "\n" if context_lines else ""

        if not stripped.startswith(KRIKI_WATCH_COMMAND):
            # No explicit slash command — pass the message through, optionally
            # prepending the context block.
            if context_prefix:
                return f"{context_prefix}{stripped}"
            return user_message

        user_instruction = stripped[len(KRIKI_WATCH_COMMAND):].strip()
        prefix, suffix = self._kriki_watch_prefix, self._kriki_watch_suffix

        if prefix is None:
            # Skill wasn't available at init — forward the instruction as-is.
            raw = user_instruction or user_message
            return f"{context_prefix}{raw}" if context_prefix else raw

        return f"{prefix}{context_prefix}{user_instruction}{suffix}"

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
            _t_exec_start = time.monotonic()
            logger.info(
                "[kriki-timing] sid=%s T_exec=0 executor started",
                resolved_session_id,
            )
            # Agent is keyed by session_id only.  ephemeral_system_prompt is
            # injected per-turn (see below) so language changes never cause
            # the agent to be discarded and conversation history is preserved.
            agent = self._get_cached_agent(resolved_session_id)
            logger.info(
                "[kriki-timing] sid=%s T_exec+%.2fs agent ready (cache_hit=%s)",
                resolved_session_id, time.monotonic() - _t_exec_start,
                resolved_session_id in self._agent_cache,
            )
            if agent_ref is not None:
                agent_ref[0] = agent

            agent_lock = self._get_cached_agent_lock(resolved_session_id)
            with agent_lock:
                # Save per-turn overridable attributes and restore them in
                # finally so a cached agent is never left in a dirty state.
                previous_state = {
                    "stream_delta_callback": getattr(agent, "stream_delta_callback", None),
                    "tool_progress_callback": getattr(agent, "tool_progress_callback", None),
                    "tool_start_callback": getattr(agent, "tool_start_callback", None),
                    "tool_complete_callback": getattr(agent, "tool_complete_callback", None),
                    # ephemeral_system_prompt is per-turn: inject the current
                    # request's language/instructions directive for this turn
                    # and restore the previous value (typically None) afterwards.
                    "ephemeral_system_prompt": getattr(agent, "ephemeral_system_prompt", None),
                }
                agent.stream_delta_callback = stream_delta_callback
                agent.tool_progress_callback = tool_progress_callback
                agent.tool_start_callback = tool_start_callback
                agent.tool_complete_callback = tool_complete_callback
                agent.ephemeral_system_prompt = ephemeral_system_prompt or None
                before_input = getattr(agent, "session_prompt_tokens", 0) or 0
                before_output = getattr(agent, "session_completion_tokens", 0) or 0
                before_total = getattr(agent, "session_total_tokens", 0) or 0
                try:
                    _t_llm_start = time.monotonic()
                    logger.info(
                        "[kriki-timing] sid=%s T_exec+%.2fs entering run_conversation",
                        resolved_session_id, _t_llm_start - _t_exec_start,
                    )
                    result = agent.run_conversation(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        task_id="default",
                    )
                    logger.info(
                        "[kriki-timing] sid=%s T_exec+%.2fs run_conversation returned (llm_wall=%.2fs)",
                        resolved_session_id,
                        time.monotonic() - _t_exec_start,
                        time.monotonic() - _t_llm_start,
                    )
                    after_input = getattr(agent, "session_prompt_tokens", 0) or 0
                    after_output = getattr(agent, "session_completion_tokens", 0) or 0
                    after_total = getattr(agent, "session_total_tokens", 0) or 0
                finally:
                    for name, value in previous_state.items():
                        setattr(agent, name, value)

            usage = {
                "input_tokens": max(0, after_input - before_input),
                "output_tokens": max(0, after_output - before_output),
                "total_tokens": max(0, after_total - before_total),
            }
            return result, usage

        return await loop.run_in_executor(None, _run)

    def _persist_response_chain(
        self,
        *,
        response_id: str,
        model: str,
        result: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        original_user_message: Any,
        instructions: Optional[str],
        session_id: str,
        conversation: Optional[str],
        response_envelope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the turn into ``self._response_store`` for chaining via
        ``previous_response_id``.

        Unlike ``api_server._persist_response_snapshot``, this uses
        ``result["messages"]`` as the *single* source of truth (avoiding the
        ``append + extend`` duplication path) and rewrites the current turn's
        user message back to the **unwrapped** content so device_id / language /
        skill-injection metadata never bloats the stored history.
        """
        if not isinstance(result, dict):
            return

        agent_messages = list(result.get("messages") or [])
        if agent_messages:
            full_history: List[Dict[str, Any]] = agent_messages
            # The agent saw the wrapped user message (with device_id /
            # language / skill prefix); rewind the last user turn back to the
            # original content for clean future chaining.
            for i in range(len(full_history) - 1, -1, -1):
                if full_history[i].get("role") == "user":
                    full_history[i] = {"role": "user", "content": original_user_message}
                    break
        else:
            full_history = list(conversation_history)
            full_history.append({"role": "user", "content": original_user_message})
            final = result.get("final_response") or ""
            if final:
                full_history.append({"role": "assistant", "content": final})

        envelope = response_envelope or {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "model": model,
        }
        try:
            self._response_store.put(response_id, {
                "response": envelope,
                "conversation_history": full_history,
                "instructions": instructions,
                "session_id": session_id,
            })
            if conversation:
                self._response_store.set_conversation(conversation, response_id)
        except Exception as exc:
            logger.warning(
                "Failed to persist Kriki response %s for chaining: %s",
                response_id, exc,
            )

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

        # Responses-API style chaining: clients may pass `previous_response_id`
        # (or a `conversation` name shortcut) to resume a prior turn without
        # having to ship the full history themselves.  Explicit
        # `conversation_history` in the body takes precedence.
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = bool(body.get("store", True))

        if conversation and previous_response_id:
            return web.json_response(
                _openai_error("Cannot use both 'conversation' and 'previous_response_id'"),
                status=400,
            )
        if conversation:
            previous_response_id = self._response_store.get_conversation(conversation)

        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                # The prior turn was most likely cancelled before its
                # persistence callback fired, or its record was evicted.
                # Don't 404 — log a warning and proceed with empty history
                # so the client can keep talking without restarting the
                # session.  Callers that need strict chaining can detect
                # the missing context themselves and re-send their full
                # `conversation_history`.
                logger.warning(
                    "[kriki] previous_response_id=%s not found in "
                    "response_store (likely cancelled or evicted); "
                    "continuing with empty history",
                    previous_response_id,
                )
            else:
                conversation_history = list(stored.get("conversation_history", []))
                logger.info(
                    "[kriki] loaded %d messages from previous_response_id=%s",
                    len(conversation_history), previous_response_id,
                )
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        user_message: Any = input_messages[-1].get("content", "") if input_messages else ""
        if not _content_has_visible_payload(user_message):
            return web.json_response(_openai_error("No user message found in input"), status=400)

        # Preserve the original (un-wrapped) user content for history
        # persistence.  device_id / language / skill wrapping is per-turn
        # ephemeral context and must not bloat the stored history.
        original_user_message: Any = user_message

        session_id = request.headers.get("X-Hermes-Session-Id", "").strip() or str(uuid.uuid4())
        _t_req_start = time.monotonic()
        logger.info("[kriki-timing] sid=%s T0 request received", session_id)
        # Extract optional per-request parameters.
        device_id: str = (body.get("device_id") or "").strip()
        language: str = (body.get("language") or "").strip()

        user_message = self._build_kriki_watch_message(
            user_message, session_id, device_id=device_id, language=language
        )

        # Build the ephemeral system prompt.
        #
        # *language* and *device_id* are injected directly into the user
        # message (via _build_kriki_watch_message) to keep them at the same
        # attention level as the actual instruction.
        #
        # *instructions* — when supplied by the caller — is always set as the
        # ephemeral system prompt for this turn, regardless of whether the
        # kriki-watch skill is available.  Callers rely on it as a hard
        # per-turn directive, so it must always reach the LLM.
        _instructions = body.get("instructions")
        ephemeral_system_prompt = (
            str(_instructions) if _instructions is not None else None
        )

        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        stream = bool(body.get("stream", False))
        if stream:
            import queue as _q

            stream_q: _q.Queue = _q.Queue()

            _first_delta_logged = [False]
            # Mutable single-cell list so the executor-thread callbacks and the
            # async watchdog can share a wall-clock timestamp without locks
            # (Python attribute-set on a list element is GIL-atomic).
            _last_activity = [time.monotonic()]

            def _bump_activity():
                _last_activity[0] = time.monotonic()

            def _on_delta(delta):
                if delta is not None:
                    if not _first_delta_logged[0]:
                        _first_delta_logged[0] = True
                        logger.info(
                            "[kriki-timing] sid=%s T+%.2fs first LLM token/delta",
                            session_id, time.monotonic() - _t_req_start,
                        )
                    _bump_activity()
                    stream_q.put(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                _bump_activity()
                return

            def _on_tool_start(tool_call_id, function_name, function_args):
                logger.info(
                    "[kriki-timing] sid=%s T+%.2fs tool_start name=%s",
                    session_id, time.monotonic() - _t_req_start, function_name,
                )
                _bump_activity()
                stream_q.put(("__tool_started__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                _bump_activity()
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
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
            ))

            # Inactivity watchdog: if no streamed delta / tool event arrives
            # for `self._agent_inactivity_timeout` seconds, assume the agent
            # is wedged (hung LLM HTTP call, hung tool, etc.) and force a
            # terminal state.  We call `agent.interrupt()` to nudge the
            # executor thread out of its blocking call, cancel the asyncio
            # future, and push an EOS sentinel so the SSE writer's queue
            # loop wakes up immediately instead of waiting on the next
            # 0.5s poll.  Once `agent_task` resolves (success, exception,
            # or cancellation), api_server._write_sse_responses emits
            # `response.completed` / `response.failed` as appropriate.
            inactivity_timeout = self._agent_inactivity_timeout
            watchdog_task: Optional[asyncio.Task] = None

            if inactivity_timeout > 0:
                async def _inactivity_watchdog() -> None:
                    poll_interval = max(1.0, min(5.0, inactivity_timeout / 4))
                    try:
                        while not agent_task.done():
                            await asyncio.sleep(poll_interval)
                            if agent_task.done():
                                return
                            idle = time.monotonic() - _last_activity[0]
                            if idle < inactivity_timeout:
                                continue
                            logger.warning(
                                "[kriki] sid=%s agent inactive for %.1fs "
                                "(> %.0fs timeout); interrupting and cancelling",
                                session_id, idle, inactivity_timeout,
                            )
                            agent = agent_ref[0]
                            if agent is not None:
                                try:
                                    agent.interrupt(
                                        f"inactivity timeout {int(inactivity_timeout)}s"
                                    )
                                except Exception as exc:
                                    logger.warning(
                                        "[kriki] sid=%s agent.interrupt() failed: %s",
                                        session_id, exc,
                                    )
                            # Wake the SSE drain loop so it doesn't sit on
                            # its 0.5s blocking queue.get for another tick.
                            try:
                                stream_q.put(None)
                            except Exception:
                                pass
                            agent_task.cancel()
                            return
                    except asyncio.CancelledError:
                        return

                watchdog_task = asyncio.ensure_future(_inactivity_watchdog())

                def _stop_watchdog(_t: "asyncio.Future") -> None:
                    if watchdog_task is not None and not watchdog_task.done():
                        watchdog_task.cancel()

                agent_task.add_done_callback(_stop_watchdog)

            # NOTE: store=False on the SSE writer because api_server's
            # _persist_response_snapshot appends user_message AND extends
            # result["messages"] (which already includes the user_message),
            # producing duplicated history that doubles every turn.  We
            # persist cleanly ourselves via a done-callback below.
            if store:
                _kriki_model = body.get("model", self._model_name)

                def _persist_streaming_result(task: "asyncio.Future") -> None:
                    # Persist on every terminal state (success, cancellation,
                    # exception) so the response_id that has already been sent
                    # down the SSE stream stays usable as
                    # `previous_response_id` on the next turn.  Without this,
                    # a cancelled request leaves a dangling id and the
                    # follow-up call 404s.
                    result: Optional[Dict[str, Any]] = None
                    if not task.cancelled():
                        try:
                            if task.exception() is None:
                                result, _usage = task.result()
                        except Exception:
                            result = None

                    if not isinstance(result, dict):
                        # Minimal placeholder — _persist_response_chain
                        # handles empty `messages` by reconstructing history
                        # from conversation_history + original_user_message,
                        # which is exactly what we want for a turn that
                        # never produced an assistant reply.
                        result = {"messages": [], "final_response": ""}

                    self._persist_response_chain(
                        response_id=response_id,
                        model=_kriki_model,
                        result=result,
                        conversation_history=conversation_history,
                        original_user_message=original_user_message,
                        instructions=ephemeral_system_prompt,
                        session_id=session_id,
                        conversation=conversation,
                    )

                agent_task.add_done_callback(_persist_streaming_result)

            return await self._write_sse_responses(
                request=request,
                response_id=response_id,
                model=body.get("model", self._model_name),
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=conversation_history,
                user_message=original_user_message,
                instructions=ephemeral_system_prompt,
                conversation=None,
                store=False,
                session_id=session_id,
            )

        try:
            result, usage = await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=ephemeral_system_prompt,
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
            "id": response_id,
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
        if store:
            self._persist_response_chain(
                response_id=response_id,
                model=response_data["model"],
                result=result,
                conversation_history=conversation_history,
                original_user_message=original_user_message,
                instructions=ephemeral_system_prompt,
                session_id=session_id,
                conversation=conversation,
                response_envelope=response_data,
            )
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
                    _s.connect(("0.0.0.0", self._port))
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
