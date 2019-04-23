import pathlib
from typing import Any, Dict
from implements import Interface


class StateBackendConfiguration:
    __slots__ = ('name', 'vars')

    def __init__(self) -> None:
        self.name: str = "local"
        self.vars: Dict[str, str] = {}


JsonType = Dict[str, Any]


class IConfiguration(Interface):
    def __init__(self) -> None:
        pass

    def schema(self, schema: JsonType) -> None:
        pass

    def versions(self, versions: Dict[str, Any]) -> None:
        pass

    def state_backend_cfg(self, cfg: StateBackendConfiguration) -> None:
        pass

    def build(
        self,
        *,
        model: "yapytfgen.model",  # type: ignore  # noqa
        data: JsonType,
        step_data: Any,
    ) -> None:
        pass

    def on_success(
        self,
        *,
        state: "yapytfgen.state",  # type: ignore  # noqa
    ) -> None:
        pass

    def mementos(
        self,
        *,
        state: "yapytfgen.state",  # type: ignore  # noqa
        dest: pathlib.Path,
    ) -> None:
        pass
