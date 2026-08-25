"""Backend registry — maps backend type names to classes."""

from __future__ import annotations

from typing import Any, Type

from fedskill.backends.base import ExecutionBackend

_REGISTRY: dict[str, Type[ExecutionBackend]] = {}


def register_backend(name: str, cls: Type[ExecutionBackend]) -> None:
    _REGISTRY[name] = cls


def get_backend(name: str, config: dict[str, Any] | None = None) -> ExecutionBackend:
    """Instantiate a backend by its registered type name.

    Args:
        name: Backend type. The public artifact provides ``tau2bench``.
        config: Backend-specific configuration dict.

    Returns:
        An ExecutionBackend instance.

    Raises:
        ValueError: If the backend type is not registered.
    """
    # Lazy import built-in backends so they auto-register
    if not _REGISTRY:
        _import_builtins()

    if name not in _REGISTRY:
        # Try one more import in case it's a new built-in
        _import_builtins()

    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend type {name!r}. Available: {available}"
        )

    return _REGISTRY[name](name=name, config=config or {})


def _import_builtins() -> None:
    """Import all built-in backend modules to trigger their register() calls."""
    try:
        import fedskill.backends.tau2bench_backend  # noqa: F401
    except ImportError:
        pass
