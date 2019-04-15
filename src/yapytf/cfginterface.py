from typing import Any, Dict
from implements import Interface


JsonType = Dict[str, Any]


class IConfiguration(Interface):
    def __init__(self) -> None:
        pass

    def schema(self, schema: JsonType) -> None:
        pass

    def versions(self, versions: Dict[str, Any]) -> None:
        pass

    def build(
        self,
        *,
        model: "yapytfgen.model",  # type: ignore  # noqa
        data: JsonType,
        step_data: Any,
    ) -> None:
        pass
