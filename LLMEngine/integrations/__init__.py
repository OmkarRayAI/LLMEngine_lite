"""Optional third-party integrations.

These modules import their respective vendor SDKs lazily so that:

- the core engine has no hard dependency on Composio / Google / etc.
- a missing SDK fails fast with an actionable install hint, not deep in a
  call stack at runtime.
"""
from .composio_toolkit import ComposioToolkit  # noqa: F401

__all__ = ["ComposioToolkit"]
