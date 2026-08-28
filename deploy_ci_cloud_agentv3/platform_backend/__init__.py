"""V3 deterministic simulated AutoDrive platform backend."""

__all__ = ["build_platform_facade"]


def __getattr__(name):
    if name == "build_platform_facade":
        from .runtime import build_platform_facade
        return build_platform_facade
    raise AttributeError(name)
