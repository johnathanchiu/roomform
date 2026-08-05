"""Agent error types."""


class RoomformAgentError(Exception):
    """Base error for the roomform agent harness."""


class ModelError(RoomformAgentError):
    """A provider/model call failed."""


__all__ = ["RoomformAgentError", "ModelError"]
