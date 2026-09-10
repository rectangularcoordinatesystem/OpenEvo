from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import pytest

from openevo_asta_bench.openai_compat import (
    FlatToolOpenAIAPI,
    flatten_chat_completion_request,
)
from openevo_asta_bench.solver import _inspect_model_name


def test_flat_tool_schema_rewrites_only_outbound_tool_declarations() -> None:
    request = {
        "model": "cc-gpt-5.5",
        "messages": [
            {"role": "user", "content": "Call ping."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "ping", "arguments": '{"value":"OK"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "OK"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Return a test value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "ping"}},
    }
    original = deepcopy(request)

    transformed = flatten_chat_completion_request(request)

    assert request == original
    assert transformed["messages"] == original["messages"]
    assert transformed["tools"] == [
        {
            "type": "function",
            "name": "ping",
            "description": "Return a test value.",
            "parameters": original["tools"][0]["function"]["parameters"],
            "strict": True,
        }
    ]
    assert transformed["tool_choice"] == {"type": "function", "name": "ping"}


def test_flat_tool_schema_is_idempotent() -> None:
    request = {
        "tools": [
            {
                "type": "function",
                "name": "ping",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "tool_choice": "auto",
    }

    assert flatten_chat_completion_request(request) == request


def test_flat_tool_schema_rejects_conflicting_fields() -> None:
    with pytest.raises(ValueError, match="conflicting field 'name'"):
        flatten_chat_completion_request(
            {
                "tools": [
                    {
                        "type": "function",
                        "name": "outer",
                        "function": {"name": "inner", "parameters": {}},
                    }
                ]
            }
        )


def test_solver_selects_flat_provider_explicitly() -> None:
    assert (
        _inspect_model_name("openai/cc-gpt-5.5", openai_tool_schema="flat")
        == "openevo_flat_openai/cc-gpt-5.5"
    )
    assert (
        _inspect_model_name("openai/cc-gpt-5.5", openai_tool_schema="standard")
        == "openai/cc-gpt-5.5"
    )


def test_solver_rejects_flat_schema_for_non_openai_provider() -> None:
    with pytest.raises(ValueError, match="only supports OpenAI-compatible models"):
        _inspect_model_name("anthropic/claude", openai_tool_schema="flat")


def test_flat_provider_serializes_through_openai_sdk_and_accepts_chat_response() -> None:
    if FlatToolOpenAIAPI is None:
        pytest.skip("Inspect optional dependency is not installed")

    async def run_test() -> None:
        import httpx
        from inspect_ai.model import GenerateConfig
        from openai import AsyncOpenAI

        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-5.5",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "done",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(
            api_key="test-key",
            base_url="https://relay.invalid/v1",
            http_client=http_client,
        )
        provider = FlatToolOpenAIAPI(
            model_name="cc-gpt-5.5",
            api_key="test-key",
            base_url="https://relay.invalid/v1",
        )
        await provider.client.close()
        provider.client = client
        request = {
            "model": "cc-gpt-5.5",
            "messages": [
                {"role": "user", "content": "Call ping."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "ping",
                                "arguments": '{"value":"OK"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "OK"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "description": "Return a test value.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        }

        try:
            response = await provider._get_completion(request, GenerateConfig())
        finally:
            await client.close()

        assert response.choices[0].message.content == "done"
        assert captured["messages"] == request["messages"]
        assert captured["tools"] == [
            {
                "type": "function",
                "name": "ping",
                "description": "Return a test value.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    asyncio.run(run_test())
