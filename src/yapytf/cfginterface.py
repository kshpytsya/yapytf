from typing import Any, Dict
from implements import Interface


class IConfiguration(Interface):
    @staticmethod
    def schema(schema: Dict[str, Any]) -> None:
        pass

    def __init__(self, data: Dict[str, Any]) -> None:
        pass

    def versions(self, versions: Dict[str, Any]) -> None:
        pass

    def build(self, model: "yapytfgen.model") -> None:
        pass
