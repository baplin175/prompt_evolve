"""Tests for MatchaClient with mocked HTTP calls."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.gateway.base import GatewayResponse
from app.gateway.matcha_client import MatchaClient, _extract_reply_text


# ---------------------------------------------------------------------------
# _extract_reply_text
# ---------------------------------------------------------------------------


class TestExtractReplyText:
    def test_non_dict_returns_empty(self):
        assert _extract_reply_text("not a dict") == ""
        assert _extract_reply_text(42) == ""
        assert _extract_reply_text(None) == ""

    def test_nested_content_text(self):
        data = {
            "output": [
                {
                    "content": [{"text": "Hello world"}],
                }
            ]
        }
        assert _extract_reply_text(data) == "Hello world"

    def test_list_output_joined(self):
        data = {"output": ["line1", "line2", "line3"]}
        assert _extract_reply_text(data) == "line1\nline2\nline3"

    def test_string_output(self):
        data = {"output": "plain string"}
        assert _extract_reply_text(data) == "plain string"

    def test_none_output(self):
        data = {"output": None}
        assert _extract_reply_text(data) == ""

    def test_empty_output(self):
        data = {"output": ""}
        assert _extract_reply_text(data) == ""

    def test_empty_list_output(self):
        data = {"output": []}
        assert _extract_reply_text(data) == ""

    def test_list_with_dict_but_no_content(self):
        data = {"output": [{"other_key": "value"}]}
        # Falls through to join since content key is missing
        assert _extract_reply_text(data) == "{'other_key': 'value'}"


# ---------------------------------------------------------------------------
# MatchaClient
# ---------------------------------------------------------------------------


class TestMatchaClient:
    @patch("app.gateway.matcha_client.requests.post")
    def test_complete_returns_gateway_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"output": "Test response"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = MatchaClient(
            url="https://example.com/api",
            api_key="test-key",
            mission_id="12345",
            api_key_header="X-API-Key",
        )
        result = client.complete(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            system_prompt=None,
            user_content="What is 2+2?",
        )

        assert isinstance(result, GatewayResponse)
        assert result.content == "Test response"
        assert result.model == "gpt-4o-mini"
        assert result.latency_ms > 0

    @patch("app.gateway.matcha_client.requests.post")
    def test_system_prompt_prepended(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"output": "response"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = MatchaClient(
            url="https://example.com/api",
            api_key="key",
            mission_id="123",
            api_key_header="X-API-Key",
        )
        client.complete(
            model="test",
            temperature=0.5,
            max_tokens=512,
            system_prompt="You are a helper.",
            user_content="Hello",
        )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert "You are a helper." in payload["input"]
        assert "Hello" in payload["input"]

    @patch("app.gateway.matcha_client.requests.post")
    def test_headers_include_api_key(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"output": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = MatchaClient(
            url="https://example.com/api",
            api_key="my-secret-key",
            mission_id="123",
            api_key_header="X-Custom-Key",
        )
        client.complete(
            model="test",
            temperature=0.5,
            max_tokens=512,
            system_prompt=None,
            user_content="test",
        )

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["X-Custom-Key"] == "my-secret-key"
        assert headers["Content-Type"] == "application/json"

    @patch("app.gateway.matcha_client.requests.post")
    def test_mission_id_in_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"output": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = MatchaClient(
            url="https://example.com/api",
            api_key="key",
            mission_id="99999",
            api_key_header="X-API-Key",
        )
        client.complete(
            model="test",
            temperature=0.5,
            max_tokens=512,
            system_prompt=None,
            user_content="test",
        )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["mission_id"] == "99999"

    @patch("app.gateway.matcha_client.requests.post")
    def test_nested_response_extraction(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "output": [{"content": [{"text": "Deep response"}]}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = MatchaClient(
            url="https://example.com/api",
            api_key="key",
            mission_id="123",
            api_key_header="X-API-Key",
        )
        result = client.complete(
            model="test",
            temperature=0.5,
            max_tokens=512,
            system_prompt=None,
            user_content="test",
        )

        assert result.content == "Deep response"

    @patch("app.gateway.matcha_client.time.sleep")
    @patch("app.gateway.matcha_client.requests.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"output": "recovered"}
        mock_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [
            requests.exceptions.ConnectionError("fail"),
            mock_resp,
        ]

        client = MatchaClient(
            url="https://example.com/api",
            api_key="key",
            mission_id="123",
            api_key_header="X-API-Key",
            max_retries=3,
            retry_backoff=1,
        )
        result = client.complete(
            model="test",
            temperature=0.5,
            max_tokens=512,
            system_prompt=None,
            user_content="test",
        )

        assert result.content == "recovered"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch("app.gateway.matcha_client.time.sleep")
    @patch("app.gateway.matcha_client.requests.post")
    def test_raises_after_max_retries(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.exceptions.Timeout("timeout")

        client = MatchaClient(
            url="https://example.com/api",
            api_key="key",
            mission_id="123",
            api_key_header="X-API-Key",
            max_retries=2,
            retry_backoff=1,
        )

        with pytest.raises(requests.exceptions.Timeout):
            client.complete(
                model="test",
                temperature=0.5,
                max_tokens=512,
                system_prompt=None,
                user_content="test",
            )

        assert mock_post.call_count == 2

    @patch("app.gateway.matcha_client.requests.post")
    def test_raw_response_stored(self, mock_post):
        raw_data = {"output": "test", "meta": {"id": "abc"}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw_data
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = MatchaClient(
            url="https://example.com/api",
            api_key="key",
            mission_id="123",
            api_key_header="X-API-Key",
        )
        result = client.complete(
            model="test",
            temperature=0.5,
            max_tokens=512,
            system_prompt=None,
            user_content="test",
        )

        assert result.raw == raw_data
