import pytest

from schemahint_agent import validate_function_call


FUNCTIONS = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    }
]


@pytest.mark.parametrize(
    ("candidate", "error_class", "path"),
    [
        ('{"name": "get_weather",', "invalid_json", "/"),
        ({"name": "weather", "arguments": {}}, "unknown_function", "/name"),
        ({"name": "get_weather", "arguments": []}, "type_mismatch", "/arguments"),
        (
            {"name": "get_weather", "arguments": {}},
            "missing_required_argument",
            "/",
        ),
        (
            {"name": "get_weather", "arguments": {"city": 5}},
            "type_mismatch",
            "/city",
        ),
        (
            {"name": "get_weather", "arguments": {"city": "Boston", "days": 2}},
            "unexpected_argument",
            "/",
        ),
        (
            {
                "name": "get_weather",
                "arguments": {"city": "Boston", "units": "kelvin"},
            },
            "enum_mismatch",
            "/units",
        ),
    ],
)
def test_compiles_expected_typed_hint(candidate, error_class, path):
    result = validate_function_call(candidate, FUNCTIONS)

    assert not result.valid
    assert result.hint is not None
    assert result.hint.error_class == error_class
    assert result.hint.path == path


def test_accepts_valid_call_without_hint():
    candidate = {
        "name": "get_weather",
        "arguments": {"city": "Boston", "units": "celsius"},
    }

    result = validate_function_call(candidate, FUNCTIONS)

    assert result.valid
    assert result.parsed_call == candidate
    assert result.hint is None
    assert result.raw_error is None


def test_accepts_bfcl_openai_style_function_wrapper():
    wrapped = [{"type": "function", "function": FUNCTIONS[0]}]

    result = validate_function_call(
        {"name": "get_weather", "arguments": {"city": "Boston"}},
        wrapped,
    )

    assert result.valid


def test_accepts_bfcl_python_type_aliases():
    functions = [
        {
            "name": "calculate",
            "parameters": {
                "type": "dict",
                "properties": {
                    "values": {
                        "type": "list",
                        "items": {"type": "float"},
                    },
                    "metadata": {"type": "any"},
                },
                "required": ["values"],
            },
        }
    ]

    result = validate_function_call(
        {
            "name": "calculate",
            "arguments": {"values": [1.0, 2.5], "metadata": "test-metadata"},
        },
        functions,
    )

    assert result.valid


def test_rejects_integer_inside_bfcl_float_array():
    functions = [
        {
            "name": "calculate",
            "parameters": {
                "type": "dict",
                "properties": {
                    "values": {
                        "type": "list",
                        "items": {"type": "float"},
                    },
                },
                "required": ["values"],
            },
        }
    ]

    result = validate_function_call(
        {"name": "calculate", "arguments": {"values": [1, 2.5]}},
        functions,
    )

    assert not result.valid
    assert result.hint is not None
    assert result.hint.error_class == "type_mismatch"
    assert result.hint.path == "/values/0"
    assert "expected float" in result.hint.text


def test_rejects_float_for_bfcl_integer():
    functions = [
        {
            "name": "calculate",
            "parameters": {
                "type": "dict",
                "properties": {"count": {"type": "int"}},
                "required": ["count"],
            },
        }
    ]

    result = validate_function_call(
        {"name": "calculate", "arguments": {"count": 1.0}},
        functions,
    )

    assert not result.valid
    assert result.hint is not None
    assert result.hint.error_class == "type_mismatch"
    assert result.hint.path == "/count"


def test_rejects_undeclared_bfcl_function_argument():
    functions = [
        {
            "name": "calculate",
            "parameters": {
                "type": "dict",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        }
    ]

    result = validate_function_call(
        {
            "name": "calculate",
            "arguments": {"count": 1, "undeclared": "value"},
        },
        functions,
    )

    assert not result.valid
    assert result.hint is not None
    assert result.hint.error_class == "unexpected_argument"


def test_accepts_java_and_javascript_type_aliases():
    functions = [
        {
            "name": "process",
            "parameters": {
                "type": "dict",
                "properties": {
                    "name": {"type": "String"},
                    "enabled": {"type": "Boolean"},
                    "items": {"type": "ArrayList"},
                    "options": {"type": "HashMap"},
                    "count": {"type": "long"},
                },
                "required": ["name", "enabled", "items", "options", "count"],
            },
        }
    ]

    result = validate_function_call(
        {
            "name": "process",
            "arguments": {
                "name": "example",
                "enabled": True,
                "items": ["a", "b"],
                "options": {"mode": "fast"},
                "count": 2,
            },
        },
        functions,
    )

    assert result.valid


def test_treats_empty_bfcl_type_as_unspecified():
    functions = [
        {
            "name": "create",
            "parameters": {
                "type": "dict",
                "properties": {
                    "options": {
                        "type": "dict",
                        "properties": {"issuer": {"type": ""}},
                    }
                },
                "required": ["options"],
            },
        }
    ]

    result = validate_function_call(
        {
            "name": "create",
            "arguments": {"options": {"issuer": "myapp.net"}},
        },
        functions,
    )

    assert result.valid


def test_bfcl_javascript_requires_string_encoded_arguments():
    functions = [
        {
            "name": "process",
            "parameters": {
                "type": "dict",
                "properties": {
                    "enabled": {"type": "Boolean"},
                    "items": {"type": "array", "items": {"type": "String"}},
                },
                "required": ["enabled", "items"],
            },
        }
    ]

    valid = validate_function_call(
        {
            "name": "process",
            "arguments": {"enabled": "true", "items": '["one", "two"]'},
        },
        functions,
        language="javascript",
    )
    invalid = validate_function_call(
        {
            "name": "process",
            "arguments": {"enabled": True, "items": ["one", "two"]},
        },
        functions,
        language="javascript",
    )

    assert valid.valid
    assert not invalid.valid
    assert invalid.hint is not None
    assert invalid.hint.error_class == "type_mismatch"
