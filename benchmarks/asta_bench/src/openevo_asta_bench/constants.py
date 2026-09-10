from __future__ import annotations

DEFAULT_CODEX_MODEL = "gpt-5.5"
GPT55_INSPECT_MODEL_ALIASES = frozenset({"gpt-5.5", "cc-gpt-5.5"})
DEFAULT_INSPECT_MODEL = "openai/cc-gpt-5.5"
DEFAULT_AGENT_MODEL = DEFAULT_INSPECT_MODEL
DEFAULT_REFLECTOR_PROVIDER = "codex_cli"
DEFAULT_REFLECTOR_MODEL = DEFAULT_CODEX_MODEL
DEFAULT_TOOLSET = "standard"
SOURCE = "asta_bench.inspect"
EVENT_TYPE = "openevo.session_completed"
