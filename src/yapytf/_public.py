import pathlib as _pathlib
import typing as _typing

from implements import Interface as _Interface

JsonType = _typing.Dict[str, _typing.Any]
SimpleOutputResult = _typing.Dict[str, _typing.Union[str, _typing.Tuple[str, int]]]


class StateBackendConfig:
    __slots__ = ('name', 'vars')

    def __init__(self) -> None:
        self.name: str = "local"
        self.vars: _typing.Dict[str, str] = {}


class Configurator(_Interface):
    @staticmethod
    def requires() -> _typing.Set[str]:
        return set()

    @classmethod
    def schema(cls, schema: JsonType) -> None:
        for i in cls.schema_required_strings():
            schema["properties"].update({i: {"type": "string"}})
            schema["required"].append(i)

    @staticmethod
    def schema_required_strings() -> _typing.Set[str]:
        return set()

    @staticmethod
    def versions(versions: _typing.Dict[str, _typing.Any]) -> None:
        pass

    def __init__(self, data: JsonType) -> None:
        self._data = data

    @property
    def data(self) -> JsonType:
        return self._data

    def state_backend_cfg(self, cfg: StateBackendConfig) -> None:
        pass

    def populate_providers(
        self,
        *,
        model: "yapytfgen.providers_model",  # type: ignore  # noqa
    ) -> None:
        pass

    def populate(
        self,
        *,
        model: "yapytfgen.model",  # type: ignore  # noqa
        step_data: _typing.Any,
    ) -> None:
        pass

    def output(
        self,
        *,
        state: "yapytfgen.state",  # type: ignore  # noqa
        dest: _pathlib.Path,
    ) -> None:
        for path_str, value in self.simple_output(state=state).items():
            mode: _typing.Optional[int]

            if isinstance(value, tuple):
                text, mode = value
            else:
                text, mode = value, None

            path = _pathlib.PurePosixPath(path_str)

            if path.is_absolute():
                raise RuntimeError(f"simple_output returned absolute path: {path_str}")

            if ".." in path.parts:
                raise RuntimeError(f"simple_output returned path containing \"..\": {path_str}")

            file_path = dest.joinpath(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(text)

            if mode is not None:
                file_path.chmod(mode)

    def simple_output(
        self,
        *,
        state: "yapytfgen.state",  # type: ignore  # noqa
    ) -> SimpleOutputResult:
        return {}
