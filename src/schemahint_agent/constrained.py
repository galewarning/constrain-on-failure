"""JSON Schema construction for constrained function-call generation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


_TYPE_ALIASES = {
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "str": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "Bigint": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "Set": "array",
    "Queue": "array",
    "Stack": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Any": "string",
    "any": "string",
    "String": "string",
    "Boolean": "boolean",
}


def normalize_generation_schema(value: Any) -> Any:
    """Convert BFCL function documents into portable JSON Schema."""

    if isinstance(value, list):
        return [normalize_generation_schema(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)

    original_type = value.get("type")
    normalized = {
        key: normalize_generation_schema(item)
        for key, item in value.items()
    }
    type_value = normalized.get("type")
    if type_value == "":
        normalized.pop("type")
    elif isinstance(type_value, str):
        normalized["type"] = _TYPE_ALIASES.get(type_value, type_value)
    elif isinstance(type_value, list):
        normalized["type"] = [
            _TYPE_ALIASES.get(item, item) for item in type_value
        ]

    if (
        isinstance(original_type, str)
        and original_type in {"dict", "HashMap", "Hashtable"}
        and "properties" in normalized
        and "additionalProperties" not in normalized
    ):
        normalized["additionalProperties"] = False
    return normalized


def build_function_call_schema(
    functions: Sequence[Mapping[str, Any]],
    *,
    language: str = "python",
) -> dict[str, Any]:
    """Build a closed union tying each function name to its argument schema."""

    alternatives: list[dict[str, Any]] = []
    for item in functions:
        definition = item.get("function", item)
        name = definition.get("name")
        if not isinstance(name, str):
            continue
        parameters = normalize_generation_schema(
            definition.get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        )
        if language in {"java", "javascript"}:
            original_properties = parameters.get("properties", {})
            parameters = {
                "type": "object",
                "properties": {
                    key: {
                        "type": "string",
                        **(
                            {"description": details["description"]}
                            if isinstance(details, Mapping)
                            and "description" in details
                            else {}
                        ),
                    }
                    for key, details in original_properties.items()
                },
                "required": parameters.get("required", []),
                "additionalProperties": False,
            }
        alternatives.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "arguments": parameters,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )

    if not alternatives:
        raise ValueError("At least one named function is required")
    if len(alternatives) == 1:
        return alternatives[0]
    return {"oneOf": alternatives}
