from typing import Any, Dict, Tuple


def get_resources_attrs(
    j: Dict[str, Any],
    resources_schemas_versions: Dict[Tuple[str, str], int]
) -> Dict[str, Any]:
    ver = j["version"]
    if not ver == 4:
        raise RuntimeError(f"Unsupported version of terraform state: \"{ver}\"")

    result: Dict[str, Any] = {}

    for res in j["resources"]:
        res_mode = res["mode"]
        res_type = res["type"]
        res_name = res["name"]
        k = result.setdefault(res_mode, {}).setdefault(res_type, {}).setdefault(res_name, {})
        assert not k
        for instance in res["instances"]:
            index_key = instance.get("index_key")

            instance_schema_version = instance["schema_version"]
            expected_instance_schema_version = resources_schemas_versions[res_mode, res_type]
            if instance_schema_version != expected_instance_schema_version:
                raise RuntimeError(
                    "Unexpected resource schema version for {}.{}.{}{}: got {}, expected {}".format(
                        res_mode,
                        res_type,
                        res_name,
                        f"[{index_key}]" if index_key else "",
                        instance_schema_version,
                        expected_instance_schema_version
                    )
                )

            assert index_key not in k
            k[index_key] = instance["attributes"]

    return result
