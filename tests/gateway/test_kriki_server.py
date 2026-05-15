"""Tests for the Kriki Responses-compatible gateway adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import cors_middleware, security_headers_middleware
from gateway.platforms.kriki_server import (
    DEFAULT_PORT,
    KrikiServerAdapter,
    check_kriki_server_requirements,
)


class _FakeRequest:
    def __init__(self, body, headers=None):
        self._body = body
        self.headers = headers or {}

    async def json(self):
        return self._body


def _make_adapter(api_key: str = "") -> KrikiServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return KrikiServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: KrikiServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["kriki_server_adapter"] = adapter
    app.router.add_post("/v1/kriki/responses", adapter._handle_kriki_responses)
    return app


class TestKrikiServerBasics:
    def test_requirements_reflect_aiohttp(self):
        assert check_kriki_server_requirements() is True

    def test_defaults_to_port_4088_and_platform(self):
        adapter = _make_adapter()
        assert adapter.platform == Platform.KRIKI_SERVER
        assert adapter._port == DEFAULT_PORT == 4088
        assert adapter._model_name == "kriki-agent"

    def test_platform_enum_has_kriki_server(self):
        assert Platform.KRIKI_SERVER.value == "kriki_server"

    def test_kriki_server_in_connected_platforms(self):
        config = GatewayConfig()
        config.platforms[Platform.KRIKI_SERVER] = PlatformConfig(enabled=True)
        assert Platform.KRIKI_SERVER in config.get_connected_platforms()

    def test_kriki_server_not_in_connected_when_disabled(self):
        config = GatewayConfig()
        config.platforms[Platform.KRIKI_SERVER] = PlatformConfig(enabled=False)
        assert Platform.KRIKI_SERVER not in config.get_connected_platforms()


class TestKrikiConfigIntegration:
    def test_env_override_enables_kriki_server(self, monkeypatch):
        monkeypatch.setenv("KRIKI_SERVER_ENABLED", "true")
        from gateway.config import load_gateway_config

        config = load_gateway_config()
        assert Platform.KRIKI_SERVER in config.platforms
        assert config.platforms[Platform.KRIKI_SERVER].enabled is True

    def test_env_override_port_host_and_key(self, monkeypatch):
        monkeypatch.setenv("KRIKI_SERVER_KEY", "sk-kriki")
        monkeypatch.setenv("KRIKI_SERVER_PORT", "4999")
        monkeypatch.setenv("KRIKI_SERVER_HOST", "0.0.0.0")
        from gateway.config import load_gateway_config

        config = load_gateway_config()
        extra = config.platforms[Platform.KRIKI_SERVER].extra
        assert extra["key"] == "sk-kriki"
        assert extra["port"] == 4999
        assert extra["host"] == "0.0.0.0"


class TestKrikiResponsesEndpoint:
    @pytest.mark.asyncio
    async def test_missing_input_returns_400(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/kriki/responses", json={"model": "kriki-agent"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_message_output_shape(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        mock_result = {"final_response": "您好，有什么可以帮您的吗？", "messages": []}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
                resp = await cli.post("/v1/kriki/responses", json={"input": "hello"})

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "response"
            assert data["status"] == "completed"
            assert data["output"][0]["id"].startswith("msg_")
            assert data["output"][0]["type"] == "message"
            assert data["output"][0]["status"] == "completed"
            assert data["output"][0]["role"] == "assistant"
            content = data["output"][0]["content"][0]
            assert content == {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": "您好，有什么可以帮您的吗？",
            }

    @pytest.mark.asyncio
    async def test_function_call_output_shape(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        mock_result = {
            "final_response": "",
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_hfJisRds7XluiTL5tv0QrObr",
                    "function": {
                        "name": "operation_exercise_control",
                        "arguments": "{\"operation\":\"open\",\"sport_type\":\"Long jump\"}",
                    },
                }],
            }],
        }
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/kriki/responses", json={"input": "open long jump"})

            assert resp.status == 200
            data = await resp.json()
            assert len(data["output"]) == 1
            item = data["output"][0]
            assert item["id"].startswith("fc_")
            assert item == {
                "id": item["id"],
                "type": "function_call",
                "status": "completed",
                "arguments": "{\"operation\":\"open\",\"sport_type\":\"Long jump\"}",
                "call_id": "call_hfJisRds7XluiTL5tv0QrObr",
                "name": "operation_exercise_control",
            }

    def test_tool_calls_do_not_hide_final_message(self):
        output = KrikiServerAdapter._extract_kriki_output_items({
            "final_response": "最终答复",
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "skill_view",
                        "arguments": "{\"name\":\"kriki-watch\"}",
                    },
                }],
            }],
        })

        assert [item["type"] for item in output] == ["function_call", "message"]
        assert output[0]["name"] == "skill_view"
        assert output[1]["content"][0]["text"] == "最终答复"


class TestKrikiDeviceIdAndLanguage:
    """Tests for the device_id and language request parameters."""

    # ── device_id ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_device_id_injected_into_plain_message(self):
        """device_id is prepended to a plain (non-slash-command) user message."""
        adapter = _make_adapter()
        mock_result = {"final_response": "ok", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await adapter._handle_kriki_responses(_FakeRequest(
                {"input": "打开运动", "device_id": "watch-abc123"},
            ))

        assert resp.status == 200
        user_msg = mock_run.call_args.kwargs["user_message"]
        assert "[Target device: watch-abc123]" in user_msg
        assert "打开运动" in user_msg

    @pytest.mark.asyncio
    async def test_device_id_injected_into_kriki_watch_command(self):
        """device_id appears inside the skill-wrapped message for /kriki-watch."""
        with patch("agent.skill_commands.build_skill_invocation_message",
                   return_value="[skill]\n___KRIKI_USER_INSTRUCTION___\n[end]"):
            adapter = _make_adapter()

        mock_result = {"final_response": "ok", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await adapter._handle_kriki_responses(_FakeRequest(
                {"input": "/kriki-watch 开始跑步", "device_id": "watch-xyz"},
            ))

        assert resp.status == 200
        user_msg = mock_run.call_args.kwargs["user_message"]
        assert "[skill]" in user_msg
        assert "[Target device: watch-xyz]" in user_msg
        assert "开始跑步" in user_msg

    def test_no_device_id_message_passes_through_unchanged(self):
        """When device_id is absent the message is unchanged (plain case)."""
        adapter = _make_adapter()
        result = adapter._build_kriki_watch_message("打开运动", "session-1", device_id="")
        assert result == "打开运动"

    def test_empty_device_id_ignored(self):
        adapter = _make_adapter()
        result = adapter._build_kriki_watch_message("打开运动", "session-1", device_id="  ")
        # Adapter strips whitespace at the handler level; _build_kriki_watch_message
        # receives the already-stripped value.  Empty string → no prefix.
        assert result == "打开运动"

    # ── language ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_language_injected_into_user_message(self):
        """language param is injected as [Response language: X] in the user
        message (not the ephemeral system prompt) so the model sees it at the
        same attention level as the actual instruction."""
        adapter = _make_adapter()
        mock_result = {"final_response": "Hola", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await adapter._handle_kriki_responses(_FakeRequest(
                {"input": "hello", "language": "Spanish"},
            ))

        assert resp.status == 200
        user_msg = mock_run.call_args.kwargs["user_message"]
        assert "[Response language: Spanish]" in user_msg
        # language no longer goes into ephemeral_system_prompt
        ephemeral = mock_run.call_args.kwargs.get("ephemeral_system_prompt")
        assert ephemeral is None

    @pytest.mark.asyncio
    async def test_no_language_keeps_empty_ephemeral_prompt(self):
        """Without language the ephemeral prompt stays None (cache-friendly)."""
        adapter = _make_adapter()
        mock_result = {"final_response": "ok", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            await adapter._handle_kriki_responses(_FakeRequest({"input": "hello"}))

        ephemeral = mock_run.call_args.kwargs.get("ephemeral_system_prompt")
        assert ephemeral is None

    @pytest.mark.asyncio
    async def test_language_in_user_message_instructions_in_ephemeral_when_skill_absent(self):
        """When kriki-watch skill is absent: language goes into user message and
        instructions go into ephemeral_system_prompt (legacy fallback)."""
        with patch("agent.skill_commands.build_skill_invocation_message", return_value=None):
            adapter = _make_adapter()

        mock_result = {"final_response": "ok", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            await adapter._handle_kriki_responses(_FakeRequest({
                "input": "hello",
                "language": "Japanese",
                "instructions": "You are a helpful assistant.",
            }))

        user_msg = mock_run.call_args.kwargs["user_message"]
        ephemeral = mock_run.call_args.kwargs.get("ephemeral_system_prompt", "")
        # language is injected into user message
        assert "[Response language: Japanese]" in user_msg
        # instructions go into ephemeral only when skill is absent
        assert "You are a helpful assistant." in (ephemeral or "")

    @pytest.mark.asyncio
    async def test_language_in_user_message_instructions_ignored_when_skill_present(self):
        """When kriki-watch skill is available: language goes into user message;
        instructions are silently dropped (skill context already provides them)
        and ephemeral_system_prompt stays None."""
        with patch("agent.skill_commands.build_skill_invocation_message",
                   return_value="[skill]\n___KRIKI_USER_INSTRUCTION___\n[end]"):
            adapter = _make_adapter()

        mock_result = {"final_response": "ok", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            await adapter._handle_kriki_responses(_FakeRequest({
                "input": "hello",
                "language": "zh-CN",
                "instructions": "should be ignored",
            }))

        user_msg = mock_run.call_args.kwargs["user_message"]
        ephemeral = mock_run.call_args.kwargs.get("ephemeral_system_prompt")
        # language is in user message, not ephemeral
        assert "[Response language: zh-CN]" in user_msg
        # ephemeral stays None — skill is available and instructions are not merged
        assert ephemeral is None

    # ── device_id + language combined ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_device_id_and_language_together(self):
        """Both device_id and language are injected into the user message."""
        adapter = _make_adapter()
        mock_result = {"final_response": "ok", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await adapter._handle_kriki_responses(_FakeRequest({
                "input": "打开运动",
                "device_id": "watch-001",
                "language": "zh-CN",
            }))

        assert resp.status == 200
        user_msg = mock_run.call_args.kwargs["user_message"]
        assert "[Target device: watch-001]" in user_msg
        assert "[Response language: zh-CN]" in user_msg
        # ephemeral stays None — both directives are in user message
        ephemeral = mock_run.call_args.kwargs.get("ephemeral_system_prompt")
        assert ephemeral is None


class TestKrikiAgentCache:
    @pytest.mark.asyncio
    async def test_reuses_agent_for_same_session_id(self):
        adapter = _make_adapter()
        agent = MagicMock()
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0

        def _run_conversation(**kwargs):
            agent.session_prompt_tokens += 3
            agent.session_completion_tokens += 4
            agent.session_total_tokens += 7
            return {"final_response": "ok", "messages": []}

        agent.run_conversation.side_effect = _run_conversation

        with patch.object(adapter, "_create_agent", return_value=agent) as mock_create:
            first_result, first_usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-1",
            )
            second_result, second_usage = await adapter._run_agent(
                user_message="again",
                conversation_history=[],
                session_id="session-1",
            )

        mock_create.assert_called_once()
        assert first_result["final_response"] == "ok"
        assert second_result["final_response"] == "ok"
        assert first_usage == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
        assert second_usage == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
        assert agent.run_conversation.call_count == 2

    @pytest.mark.asyncio
    async def test_ephemeral_prompt_injected_per_turn_not_per_agent(self):
        """ephemeral_system_prompt (language, instructions) is now per-turn:
        the SAME agent is reused even when the value changes between requests.
        The agent's ephemeral_system_prompt is set before run_conversation and
        restored after — agent creation is NOT triggered by prompt changes."""
        adapter = _make_adapter()
        agent = MagicMock()
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent.ephemeral_system_prompt = None  # initial state

        seen_prompts: list = []

        def _run_conversation(**kwargs):
            # Record the ephemeral prompt at call time so we can assert it
            # was correctly injected per-turn.
            seen_prompts.append(agent.ephemeral_system_prompt)
            agent.session_prompt_tokens += 1
            agent.session_completion_tokens += 1
            agent.session_total_tokens += 2
            return {"final_response": "ok", "messages": []}

        agent.run_conversation.side_effect = _run_conversation

        with patch.object(adapter, "_create_agent", return_value=agent) as mock_create:
            await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                ephemeral_system_prompt="prompt one",
                session_id="session-1",
            )
            await adapter._run_agent(
                user_message="again",
                conversation_history=[],
                ephemeral_system_prompt="prompt two",
                session_id="session-1",
            )
            await adapter._run_agent(
                user_message="final",
                conversation_history=[],
                ephemeral_system_prompt=None,
                session_id="session-1",
            )

        # Agent created exactly once regardless of how many times the prompt changes.
        mock_create.assert_called_once()
        # Each turn saw its own ephemeral prompt.
        assert seen_prompts == ["prompt one", "prompt two", None]
        # After all turns the agent's ephemeral prompt is back to its initial value.
        assert agent.ephemeral_system_prompt is None

    @pytest.mark.asyncio
    async def test_response_includes_session_header_for_reuse(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        mock_result = {"final_response": "ok", "messages": []}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/kriki/responses",
                    json={"input": "hello"},
                    headers={"X-Hermes-Session-Id": "session-1"},
                )

            assert resp.status == 200
            assert resp.headers["X-Hermes-Session-Id"] == "session-1"

    @pytest.mark.asyncio
    async def test_stream_true_uses_responses_sse_writer(self):
        adapter = _make_adapter()
        mock_result = {"final_response": "ok", "messages": []}

        async def _fake_writer(**kwargs):
            result, usage = await kwargs["agent_task"]
            assert result == mock_result
            assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
            assert kwargs["session_id"] == "session-1"
            assert kwargs["store"] is False
            return web.Response(status=200)

        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run, \
             patch.object(adapter, "_write_sse_responses", side_effect=_fake_writer) as mock_sse:
            mock_run.return_value = (mock_result, {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
            resp = await adapter._handle_kriki_responses(_FakeRequest(
                {"input": "hello", "stream": True, "model": "hermes-agent"},
                headers={"X-Hermes-Session-Id": "session-1"},
            ))

        assert resp.status == 200
        mock_sse.assert_awaited_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["stream_delta_callback"] is not None
        assert call_kwargs["tool_start_callback"] is not None
        assert call_kwargs["tool_complete_callback"] is not None

    @pytest.mark.asyncio
    async def test_preloads_kriki_watch_before_agent(self):
        """Verify that the kriki-watch skill template is pre-loaded at init time
        and the cached prefix/suffix is used at request time (no per-request
        build_skill_invocation_message call)."""
        # Preload at init time by mocking skill before adapter creation.
        with patch("agent.skill_commands.build_skill_invocation_message",
                   return_value="[kriki-watch loaded]\n___KRIKI_USER_INSTRUCTION___\n[Runtime note: ...]") as mock_skill_preload:
            adapter = _make_adapter()

        # Verify the skill was loaded exactly once (at init).
        mock_skill_preload.assert_called_once()
        assert adapter._kriki_watch_prefix is not None
        assert adapter._kriki_watch_suffix is not None

        # Now run a request — the cached template should be used without
        # calling build_skill_invocation_message again.
        mock_result = {"final_response": "ok", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run, \
             patch("agent.skill_commands.build_skill_invocation_message") as mock_skill_request:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await adapter._handle_kriki_responses(_FakeRequest(
                {"input": "/kriki-watch 打开运动"},
                headers={"X-Hermes-Session-Id": "session-1"},
            ))

        assert resp.status == 200
        # build_skill_invocation_message must NOT be called during the request.
        mock_skill_request.assert_not_called()
        # The user_message should use the pre-loaded template.
        assert "[kriki-watch loaded]" in mock_run.call_args.kwargs["user_message"]
        assert "打开运动" in mock_run.call_args.kwargs["user_message"]

    def test_runtime_note_contains_fallback_rule(self):
        """The pre-loaded skill template must embed the FALLBACK RULE so the
        model answers general questions instead of returning 'no response'."""
        fake_msg = None
        captured_kwargs = {}

        def _capture(*args, **kwargs):
            nonlocal fake_msg
            captured_kwargs.update(kwargs)
            fake_msg = (
                f"[skill]\n{args[1] if len(args) > 1 else kwargs.get('user_instruction', '')}\n[/skill]"
            )
            return fake_msg

        with patch("agent.skill_commands.build_skill_invocation_message", side_effect=_capture):
            _make_adapter()

        runtime_note = captured_kwargs.get("runtime_note", "")
        rn_lower = runtime_note.lower()
        assert "fallback" in rn_lower, (
            "runtime_note must contain a FALLBACK RULE directive"
        )
        assert "no response" in rn_lower, (
            "runtime_note must mention when NOT to output 'no response'"
        )
        assert "general" in rn_lower or "knowledge" in rn_lower, (
            "runtime_note must explicitly cover general/knowledge questions"
        )
        # Must NOT forbid the agent from answering general questions
        assert "always use the kriki-watch skill" not in rn_lower, (
            "runtime_note must not force-lock the agent to only the kriki-watch skill"
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_raw_message_when_skill_unavailable(self):
        """When kriki-watch skill doesn't exist, messages pass through as-is."""
        with patch("agent.skill_commands.build_skill_invocation_message", return_value=None):
            adapter = _make_adapter()

        assert adapter._kriki_watch_prefix is None
        assert adapter._kriki_watch_suffix is None

        # The message should pass through unmodified.
        mock_result = {"final_response": "ok", "messages": []}
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await adapter._handle_kriki_responses(_FakeRequest(
                {"input": "打开运动"},
                headers={"X-Hermes-Session-Id": "session-1"},
            ))

        assert resp.status == 200
        assert mock_run.call_args.kwargs["user_message"] == "打开运动"

    @pytest.mark.asyncio
    async def test_run_agent_temporarily_sets_stream_callbacks_and_ephemeral_prompt(self):
        """All per-turn overrides (callbacks + ephemeral_system_prompt) are
        set before run_conversation and restored to their previous values in
        the finally block, so the cached agent is never left dirty."""
        adapter = _make_adapter()
        seen_state = {}

        class FakeAgent:
            def __init__(self):
                self.session_prompt_tokens = 0
                self.session_completion_tokens = 0
                self.session_total_tokens = 0
                self.stream_delta_callback = "old-delta"
                self.tool_progress_callback = "old-progress"
                self.tool_start_callback = "old-start"
                self.tool_complete_callback = "old-complete"
                self.ephemeral_system_prompt = "old-ephemeral"

            def run_conversation(self, **kwargs):
                seen_state["stream_delta_callback"] = self.stream_delta_callback
                seen_state["tool_start_callback"] = self.tool_start_callback
                seen_state["ephemeral_system_prompt"] = self.ephemeral_system_prompt
                self.session_prompt_tokens = 1
                self.session_completion_tokens = 2
                self.session_total_tokens = 3
                return {"final_response": "ok", "messages": []}

        agent = FakeAgent()
        delta_cb = object()
        start_cb = object()
        with patch.object(adapter, "_create_agent", return_value=agent):
            result, usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-1",
                stream_delta_callback=delta_cb,
                tool_start_callback=start_cb,
                ephemeral_system_prompt="new-ephemeral",
            )

        assert result["final_response"] == "ok"
        assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        # During run_conversation the per-turn values were active.
        assert seen_state["stream_delta_callback"] is delta_cb
        assert seen_state["tool_start_callback"] is start_cb
        assert seen_state["ephemeral_system_prompt"] == "new-ephemeral"
        # After the call, all values are restored to what the agent had before.
        assert agent.stream_delta_callback == "old-delta"
        assert agent.tool_progress_callback == "old-progress"
        assert agent.tool_start_callback == "old-start"
        assert agent.tool_complete_callback == "old-complete"
        assert agent.ephemeral_system_prompt == "old-ephemeral"


class TestKrikiServerToolset:
    def test_platforms_dict_includes_kriki_server(self):
        from hermes_cli.tools_config import PLATFORMS

        assert "kriki_server" in PLATFORMS
        assert PLATFORMS["kriki_server"]["default_toolset"] == "hermes-kriki-server"

    def test_default_toolset_includes_skill_tools(self):
        from toolsets import resolve_toolset

        tools = resolve_toolset("hermes-kriki-server")
        assert "skill_view" in tools
        assert "skills_list" in tools

    @patch("gateway.platforms.kriki_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_keeps_skills_toolset(self):
        """skills toolset is NOT filtered out — the agent needs skill_view
        and skills_list to interact with skills even when kriki-watch is
        preloaded."""
        adapter = _make_adapter()
        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "test-key",
                "base_url": None,
                "provider": None,
                "api_mode": None,
                "command": None,
                "args": [],
            }
            mock_model.return_value = "test/model"
            mock_config.return_value = {"platform_toolsets": {"kriki_server": ["skills", "memory"]}}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            call_kwargs = mock_agent_cls.call_args.kwargs
            assert set(call_kwargs["enabled_toolsets"]) == {"memory", "skills"}

    @patch("gateway.platforms.kriki_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_uses_kriki_platform_toolsets(self):
        adapter = _make_adapter()
        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {
                "api_key": "test-key",
                "base_url": None,
                "provider": None,
                "api_mode": None,
                "command": None,
                "args": [],
            }
            mock_model.return_value = "test/model"
            mock_config.return_value = {"platform_toolsets": {"kriki_server": ["web", "terminal"]}}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args.kwargs
            assert call_kwargs["platform"] == "kriki_server"
            assert call_kwargs["enabled_toolsets"] == ["terminal", "web"]
