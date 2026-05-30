"""Composio adapter — exposes Gmail, Calendar, Slack, GitHub, … to LLMEngine.

Composio's Python SDK is the auth + execution layer. We don't drive its native
provider integrations (which assume the LLM is the one issuing function
calls). Instead, we read each Composio tool's slug and JSON schema, wrap it
in a ``StructuredTool``, and let the LLMEngine planner decide when to invoke
it. Each call goes through ``composio.tools.execute(slug, user_id, arguments)``.

Why this shape:

- LLMEngine plans a DAG of tool calls *before* invoking the LLM with tools;
  Composio's "pass tools straight to OpenAI" path doesn't apply.
- Direct ``execute()`` is documented as a supported (if not the default)
  interface. We use it deliberately.
- Auth (OAuth, API keys, refresh) stays inside Composio. We never see
  credentials.

Install::

    pip install composio

Usage::

    from LLMEngine import LLMEngine
    from LLMEngine.integrations import ComposioToolkit

    toolkit = ComposioToolkit(
        user_id="omkar@example.com",
        toolkits=["GMAIL", "GOOGLECALENDAR"],
    )
    tools = toolkit.get_tools()              # list of StructuredTool
    result = await engine.run(question=..., tools=tools)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..base import StructuredTool

logger = logging.getLogger("LLMEngine.composio")

# Type-only hints for the Composio SDK: imported lazily inside __init__ so
# users without the SDK installed don't pay for it.
_Composio = Any
_Session = Any


class ComposioToolkit:
    """Adapter that yields LLMEngine-compatible tools from a Composio session."""

    def __init__(
        self,
        user_id: str,
        toolkits: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[str]] = None,
        api_key: Optional[str] = None,
        composio: Optional[_Composio] = None,
        session: Optional[_Session] = None,
    ) -> None:
        """
        Args:
            user_id: Composio user/entity identifier. Auth is scoped to this.
            toolkits: Restrict to whole toolkits (e.g. ``["GMAIL", "GOOGLECALENDAR"]``).
            tools: Restrict to specific tool slugs (e.g. ``["GMAIL_SEND_EMAIL"]``).
                If both are given, tools take precedence.
            api_key: Composio API key. Falls back to ``COMPOSIO_API_KEY`` env var.
            composio: Pre-built ``Composio`` client (mostly for tests).
            session: Pre-built session (mostly for tests).
        """
        self.user_id = user_id
        self.toolkits = list(toolkits) if toolkits else []
        self.tool_slugs = list(tools) if tools else []
        self._composio = composio or self._build_client(api_key)
        self._session = session or self._composio.create(user_id=user_id)
        self._raw_tools: List[Dict[str, Any]] = self._fetch_tools()
        logger.info(
            "ComposioToolkit user_id=%s loaded %d tools (toolkits=%s, tools=%s)",
            user_id, len(self._raw_tools), self.toolkits, self.tool_slugs,
        )

    # ---- public API -------------------------------------------------------

    def get_tools(self) -> List[StructuredTool]:
        """Return one ``StructuredTool`` per available Composio tool."""
        return [self._wrap_tool(spec) for spec in self._raw_tools]

    def __iter__(self):
        for spec in self._raw_tools:
            yield self._wrap_tool(spec)

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _build_client(api_key: Optional[str]) -> _Composio:
        try:
            from composio import Composio  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised at runtime only
            raise ImportError(
                "Composio is not installed. Run `pip install composio` to use "
                "ComposioToolkit, or pass a pre-built `composio=` client."
            ) from exc
        if api_key:
            return Composio(api_key=api_key)
        return Composio()

    def _fetch_tools(self) -> List[Dict[str, Any]]:
        """Pull tool specs from the session in a shape we control.

        Composio's ``session.tools()`` returns provider-formatted tools
        (default: OpenAI function-calling shape). That's what we parse, since
        it is the documented path.
        """
        kwargs: Dict[str, Any] = {}
        if self.tool_slugs:
            kwargs["tools"] = list(self.tool_slugs)
        elif self.toolkits:
            kwargs["toolkits"] = list(self.toolkits)
        try:
            tools = self._session.tools(**kwargs)
        except TypeError:
            # Older SDK signatures may not accept the kwargs above; degrade gracefully.
            tools = self._session.tools()
        normalized: List[Dict[str, Any]] = []
        for t in tools:
            spec = self._normalize_tool(t)
            if spec is not None:
                normalized.append(spec)
        return normalized

    @staticmethod
    def _normalize_tool(raw: Any) -> Optional[Dict[str, Any]]:
        """Reduce a Composio tool to ``{name, description, parameters}``.

        Tries the OpenAI function-calling shape first (the current default),
        then a couple of fallbacks for robustness across SDK versions.
        """
        if isinstance(raw, dict):
            if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
                fn = raw["function"]
                return {
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters") or {},
                }
            if "name" in raw:
                return {
                    "name": raw["name"],
                    "description": raw.get("description") or "",
                    "parameters": raw.get("parameters") or raw.get("input_schema") or {},
                }
        # Pydantic / dataclass-like objects
        for attr in ("model_dump", "dict", "to_dict"):
            f = getattr(raw, attr, None)
            if callable(f):
                try:
                    return ComposioToolkit._normalize_tool(f())
                except Exception:
                    continue
        logger.warning("Could not normalize Composio tool: %r", raw)
        return None

    def _wrap_tool(self, spec: Dict[str, Any]) -> StructuredTool:
        slug = spec["name"]
        description = self._build_description(spec)
        params = spec.get("parameters") or {}
        args_schema = self._schema_to_pydantic(slug, params)
        property_order: List[str] = list((params.get("properties") or {}).keys())

        async def _execute(*args: Any, **kwargs: Any) -> str:
            # The engine passes args positionally (parsed from the planner
            # output ``slug(arg1, arg2, ...)``). Zip them onto the JSON-Schema
            # property order; explicit kwargs win.
            payload: Dict[str, Any] = {}
            for name, value in zip(property_order, args):
                payload[name] = value
            payload.update(kwargs)
            return await self._call_composio(slug, payload)

        return StructuredTool(
            name=self._tool_name(slug),
            description=description,
            args_schema=args_schema,
            func=None,
            coroutine=_execute,
        )

    async def _call_composio(self, slug: str, arguments: Dict[str, Any]) -> str:
        """Invoke ``composio.tools.execute(slug, user_id=..., arguments=...)``."""
        execute = self._composio.tools.execute  # type: ignore[attr-defined]

        def _sync_call() -> Any:
            try:
                return execute(slug, user_id=self.user_id, arguments=arguments)
            except TypeError:
                # Older positional signatures.
                return execute(slug, arguments, user_id=self.user_id)

        try:
            result = await asyncio.to_thread(_sync_call)
        except Exception as exc:  # surface a clean string so the planner can react
            logger.exception("Composio execute failed for %s", slug)
            return json.dumps({"error": str(exc), "tool": slug})

        return self._stringify_result(result)

    @staticmethod
    def _stringify_result(result: Any) -> str:
        # Composio returns dict-like objects; the planner's joiner reads strings.
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, default=str)[:8000]
        except Exception:
            return str(result)[:8000]

    @staticmethod
    def _tool_name(slug: str) -> str:
        # Make the slug palatable to a Python-flavoured planner.
        return slug.lower()

    @staticmethod
    def _build_description(spec: Dict[str, Any]) -> str:
        params = spec.get("parameters") or {}
        prop_names = ", ".join((params.get("properties") or {}).keys()) or ""
        sig = f"({prop_names})" if prop_names else "(...)"
        desc = (spec.get("description") or "").strip()
        return f"{spec['name'].lower()}{sig}:\n - {desc}\n - Backed by Composio."

    @staticmethod
    def _schema_to_pydantic(slug: str, schema: Dict[str, Any]) -> type:
        """Build a pydantic model from a JSON Schema ``parameters`` dict.

        Only the common JSON Schema primitives are handled — anything fancier
        is collapsed to ``Any``. Required fields stay required; everything
        else gets a default of ``None``.
        """
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        fields: Dict[str, Any] = {}
        for name, prop in properties.items():
            py_type = _json_type_to_python(prop.get("type"))
            description = prop.get("description") or ""
            if name in required:
                fields[name] = (py_type, Field(..., description=description))
            else:
                fields[name] = (Optional[py_type], Field(default=None, description=description))
        model_name = _safe_model_name(slug)
        if not fields:
            # Empty-args tools still need a schema with at least one optional field
            # so older pydantic doesn't choke on a model with zero fields.
            fields["_unused"] = (Optional[str], Field(default=None, description="(unused)"))
        model = create_model(model_name, __config__=_LOOSE_CONFIG, **fields)
        return model


# Pydantic v2 expects a ConfigDict here, v1 a class. Use the v2 dict form;
# v1 callers can override by passing their own ``args_schema``.
_LOOSE_CONFIG = ConfigDict(arbitrary_types_allowed=True, extra="allow")


def _json_type_to_python(t: Any) -> Any:
    # JSON Schema's "type" can be a string or a list of strings (union types).
    if isinstance(t, list):
        return Any
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(t, Any)


def _safe_model_name(slug: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", slug.title())
    return f"{cleaned or 'ComposioTool'}Args"
