"""Deterministic validation and non-oracle hint compilation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import ValidationError


@dataclass(frozen=True)
class Hint:
    error_class: str
    path: str
    text: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    parsed_call: Mapping[str, Any] | None
    hint: Hint | None
    raw_error: str | None


def _pointer(parts: Sequence[Any]) -> str:
    if not parts:
        return "/"
    escaped = (str(part).replace("~", "~0").replace("/", "~1") for part in parts)
    return "/" + "/".join(escaped)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _function_map(functions: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    normalized: dict[str, Mapping[str, Any]] = {}
    for item in functions:
        definition = item.get("function", item)
        name = definition.get("name")
        if isinstance(name, str):
            normalized[name] = definition
    return normalized


def _is_bfcl_float(checker: Any, instance: Any) -> bool:
    del checker
    return type(instance) is float


def _is_bfcl_integer(checker: Any, instance: Any) -> bool:
    del checker
    return type(instance) is int


_BFCL_TYPE_CHECKER = (
    Draft202012Validator.TYPE_CHECKER
    .redefine("bfcl_float", _is_bfcl_float)
    .redefine("bfcl_integer", _is_bfcl_integer)
)
_BFCL_VALIDATOR = validators.extend(
    Draft202012Validator,
    type_checker=_BFCL_TYPE_CHECKER,
)


def _normalize_bfcl_schema(value: Any, *, nested_item: bool = False) -> Any:
    """Translate BFCL's Python-like labels while preserving its type semantics."""

    if isinstance(value, list):
        return [
            _normalize_bfcl_schema(item, nested_item=nested_item)
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    original_type = value.get("type")
    normalized = {
        key: _normalize_bfcl_schema(
            item,
            nested_item=nested_item or key == "items",
        )
        for key, item in value.items()
    }
    type_value = normalized.get("type")
    aliases = {
        "dict": "object",
        "list": "array",
        "tuple": "array",
        # BFCL permits Python int-to-float conversion for a scalar float
        # parameter, but its nested checker requires exact float elements.
        "float": "bfcl_float" if nested_item else "number",
        "int": "bfcl_integer",
        "bool": "boolean",
        "str": "string",
        "byte": "bfcl_integer",
        "short": "bfcl_integer",
        "long": "bfcl_integer",
        "Bigint": "bfcl_integer",
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
    if type_value == "":
        normalized.pop("type")
    elif isinstance(type_value, str):
        normalized["type"] = aliases.get(type_value, type_value)
    elif isinstance(type_value, list):
        normalized["type"] = [aliases.get(item, item) for item in type_value]
    # BFCL function documents describe a closed Python signature. Its official
    # evaluator rejects argument names outside the declared properties even
    # though the documents usually omit JSON Schema's additionalProperties.
    if (
        isinstance(original_type, str)
        and original_type in {"dict", "HashMap", "Hashtable"}
        and "properties" in normalized
        and "additionalProperties" not in normalized
    ):
        normalized["additionalProperties"] = False
    return normalized


def _normalize_language_signature(
    schema: Mapping[str, Any],
    language: str,
) -> Mapping[str, Any]:
    """Match BFCL's Java/JavaScript convention of string-encoded arguments."""

    if language not in {"java", "javascript"}:
        return schema
    properties = schema.get("properties", {})
    string_properties = {
        key: {
            "type": "string",
            **(
                {"description": details["description"]}
                if isinstance(details, Mapping) and "description" in details
                else {}
            ),
        }
        for key, details in properties.items()
    }
    normalized: dict[str, Any] = {
        "type": "object",
        "properties": string_properties,
        "additionalProperties": False,
    }
    if "required" in schema:
        normalized["required"] = schema["required"]
    return normalized


def _compile_schema_error(error: ValidationError) -> Hint:
    path = _pointer(tuple(error.absolute_path))

    if error.validator == "required":
        missing = sorted(set(error.validator_value) - set(error.instance))
        return Hint(
            "missing_required_argument",
            path,
            f"At {path}, add required key(s): {', '.join(missing)}.",
        )

    if error.validator == "additionalProperties":
        allowed = sorted(error.schema.get("properties", {}))
        observed = sorted(set(error.instance) - set(allowed))
        return Hint(
            "unexpected_argument",
            path,
            f"At {path}, remove unexpected key(s): {', '.join(observed)}. "
            f"Permitted key(s): {', '.join(allowed)}.",
        )

    if error.validator == "type":
        required = error.validator_value
        display_names = {
            "bfcl_float": "float",
            "bfcl_integer": "integer",
        }
        if isinstance(required, list):
            required_text = ", ".join(
                display_names.get(item, item) for item in required
            )
        else:
            required_text = display_names.get(str(required), str(required))
        return Hint(
            "type_mismatch",
            path,
            f"At {path}, observed {_type_name(error.instance)}; expected {required_text}.",
        )

    if error.validator == "enum":
        allowed_values = ", ".join(json.dumps(value, ensure_ascii=False) for value in error.validator_value)
        return Hint(
            "enum_mismatch",
            path,
            f"At {path}, value {json.dumps(error.instance, ensure_ascii=False)} is invalid; "
            f"allowed value(s): {allowed_values}.",
        )

    return Hint(
        "nested_schema_failure",
        path,
        f"At {path}, satisfy the local schema constraint: {error.validator}.",
    )


def validate_function_call(
    candidate: str | Mapping[str, Any],
    functions: Sequence[Mapping[str, Any]],
    *,
    language: str = "python",
) -> ValidationResult:
    """Validate one call and return at most one deterministic typed hint.

    The function only reads the candidate and supplied schemas. It never receives
    or derives a benchmark reference answer.
    """

    if isinstance(candidate, str):
        try:
            parsed: Any = json.loads(candidate)
        except json.JSONDecodeError as exc:
            hint = Hint(
                "invalid_json",
                "/",
                f"Invalid JSON at line {exc.lineno}, column {exc.colno}; "
                'return one object shaped as {"name": "...", "arguments": {...}}.',
            )
            return ValidationResult(False, None, hint, str(exc))
    else:
        parsed = dict(candidate)

    if not isinstance(parsed, dict):
        hint = Hint(
            "invalid_json",
            "/",
            'Return one JSON object shaped as {"name": "...", "arguments": {...}}.',
        )
        return ValidationResult(False, None, hint, "Top-level value is not an object.")

    available = _function_map(functions)
    name = parsed.get("name")
    if name not in available:
        observed = json.dumps(name, ensure_ascii=False)
        choices = ", ".join(sorted(available))
        hint = Hint(
            "unknown_function",
            "/name",
            f"Function name {observed} is unavailable; choose one of: {choices}.",
        )
        return ValidationResult(
            False,
            parsed,
            hint,
            f"Unknown function {observed}; available functions: {choices}.",
        )

    arguments = parsed.get("arguments")
    if not isinstance(arguments, dict):
        hint = Hint(
            "type_mismatch",
            "/arguments",
            f"At /arguments, observed {_type_name(arguments)}; expected object.",
        )
        return ValidationResult(
            False,
            parsed,
            hint,
            f"Arguments must be an object, not {_type_name(arguments)}.",
        )

    schema = _normalize_language_signature(
        _normalize_bfcl_schema(
            available[name].get("parameters", {"type": "object"})
        ),
        language,
    )
    errors = sorted(
        _BFCL_VALIDATOR(schema).iter_errors(arguments),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), str(error.validator)),
    )
    if errors:
        return ValidationResult(
            False,
            parsed,
            _compile_schema_error(errors[0]),
            errors[0].message,
        )

    return ValidationResult(True, parsed, None, None)
