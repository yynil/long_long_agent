"""Dataset-specific adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..types import NormalizedEpisode
from .nebius import adapt_nebius_openhands, adapt_nebius_swe_agent
from .open_swe import adapt_open_swe_traces
from .openthoughts import adapt_openthoughts_agent
from .orchard import adapt_orchard_swe

Adapter = Callable[[dict[str, Any], str, str], NormalizedEpisode]

ADAPTERS: dict[str, Adapter] = {
    "openthoughts_agent": adapt_openthoughts_agent,
    "open_swe_traces": adapt_open_swe_traces,
    "orchard_swe": adapt_orchard_swe,
    "nebius_swe_agent": adapt_nebius_swe_agent,
    "nebius_openhands": adapt_nebius_openhands,
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown adapter {name!r}; expected one of {sorted(ADAPTERS)}") from error
