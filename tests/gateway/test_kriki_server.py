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
    async def test_rebuilds_agent_when_instructions_change(self):
        adapter = _make_adapter()
        first_agent = MagicMock()
        second_agent = MagicMock()
        for agent in (first_agent, second_agent):
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            agent.run_conversation.return_value = {"final_response": "ok", "messages": []}

        with patch.object(adapter, "_create_agent", side_effect=[first_agent, second_agent]) as mock_create:
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

        assert mock_create.call_count == 2
        assert mock_create.call_args_list[0].kwargs["ephemeral_system_prompt"] == "prompt one"
        assert mock_create.call_args_list[1].kwargs["ephemeral_system_prompt"] == "prompt two"

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
    async def test_run_agent_temporarily_sets_stream_callbacks(self):
        adapter = _make_adapter()
        seen_callbacks = {}

        class FakeAgent:
            def __init__(self):
                self.session_prompt_tokens = 0
                self.session_completion_tokens = 0
                self.session_total_tokens = 0
                self.stream_delta_callback = "old-delta"
                self.tool_progress_callback = "old-progress"
                self.tool_start_callback = "old-start"
                self.tool_complete_callback = "old-complete"

            def run_conversation(self, **kwargs):
                seen_callbacks["stream_delta_callback"] = self.stream_delta_callback
                seen_callbacks["tool_start_callback"] = self.tool_start_callback
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
            )

        assert result["final_response"] == "ok"
        assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        assert seen_callbacks["stream_delta_callback"] is delta_cb
        assert seen_callbacks["tool_start_callback"] is start_cb
        assert agent.stream_delta_callback == "old-delta"
        assert agent.tool_progress_callback == "old-progress"
        assert agent.tool_start_callback == "old-start"
        assert agent.tool_complete_callback == "old-complete"


class TestKrikiServerToolset:
    def test_platforms_dict_includes_kriki_server(self):
        from hermes_cli.tools_config import PLATFORMS

        assert "kriki_server" in PLATFORMS
        assert PLATFORMS["kriki_server"]["default_toolset"] == "hermes-kriki-server"

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
