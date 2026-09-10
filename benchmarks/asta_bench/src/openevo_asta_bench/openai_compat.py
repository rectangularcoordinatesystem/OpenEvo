from __future__ import annotations

from typing import Any


FLAT_OPENAI_PROVIDER = "openevo_flat_openai"


def flatten_chat_completion_request(request: dict[str, Any]) -> dict[str, Any]:
    """Return a copy using flat function declarations for nonstandard relays.

    Official OpenAI Chat Completions requests use a nested ``function`` object.
    Some compatible relays instead require the Responses-style function fields
    at the tool's top level while still returning a Chat Completions response.
    """

    transformed = dict(request)
    tools = request.get("tools")
    if isinstance(tools, list):
        transformed["tools"] = [
            _flatten_function_envelope(tool, location=f"tools[{index}]")
            for index, tool in enumerate(tools)
        ]

    tool_choice = request.get("tool_choice")
    if isinstance(tool_choice, dict):
        transformed["tool_choice"] = _flatten_function_envelope(
            tool_choice,
            location="tool_choice",
        )

    return transformed


def _flatten_function_envelope(value: Any, *, location: str) -> Any:
    if not isinstance(value, dict) or value.get("type") != "function":
        return value
    function = value.get("function")
    if function is None:
        return dict(value)
    if not isinstance(function, dict):
        raise ValueError(f"{location}.function must be an object")

    flattened = {key: item for key, item in value.items() if key != "function"}
    for key, item in function.items():
        if key in flattened and flattened[key] != item:
            raise ValueError(f"{location} has conflicting field {key!r}")
        flattened[key] = item
    return flattened


# Keep this package importable for local smoke tests that do not install the
# optional Inspect dependency. Importing the module from the Inspect plugin
# entry point registers the provider when Inspect is present.
try:
    from inspect_ai.model import modelapi
    from inspect_ai.model._providers.openai import OpenAIAPI
except ImportError:
    FlatToolOpenAIAPI = None  # type: ignore[misc,assignment]
else:

    @modelapi(FLAT_OPENAI_PROVIDER)
    class FlatToolOpenAIAPI(OpenAIAPI):
        """OpenAI Chat provider with an opt-in flat outbound tool schema."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("responses_api") is True:
                raise ValueError(
                    f"{FLAT_OPENAI_PROVIDER} supports Chat Completions only"
                )
            kwargs["responses_api"] = False
            super().__init__(*args, **kwargs)

        async def _get_completion(
            self,
            request: dict[str, Any],
            config: Any,
        ) -> Any:
            return await super()._get_completion(
                flatten_chat_completion_request(request),
                config,
            )


__all__ = [
    "FLAT_OPENAI_PROVIDER",
    "FlatToolOpenAIAPI",
    "flatten_chat_completion_request",
]
