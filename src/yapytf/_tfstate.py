from typing import Any, Dict


def get_resources_attrs(j: Dict[str, Any]) -> Dict[str, Any]:
    ver = j["version"]
    if not ver == 4:
        raise RuntimeError(f"Unsupported version of terraform state: \"{ver}\"")

    result: Dict[str, Any] = {}

    for res in j["resources"]:
        k = result.setdefault(res["mode"], {}).setdefault(res["type"], {}).setdefault(res["name"], {})
        assert not k
        for instance in res["instances"]:
            assert instance["schema_version"] == 0
            index_key = instance.get("index_key")
            assert index_key not in k
            k[index_key] = instance["attributes"]

    return result
