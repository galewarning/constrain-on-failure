from schemahint_agent import build_function_call_schema, normalize_generation_schema


def test_normalizes_bfcl_generation_schema():
    schema = {
        "type": "dict",
        "properties": {
            "values": {"type": "list", "items": {"type": "float"}},
            "metadata": {"type": "any"},
        },
        "required": ["values"],
    }

    normalized = normalize_generation_schema(schema)

    assert normalized["type"] == "object"
    assert normalized["properties"]["values"]["type"] == "array"
    assert normalized["properties"]["values"]["items"]["type"] == "number"
    assert normalized["properties"]["metadata"]["type"] == "string"
    assert normalized["additionalProperties"] is False


def test_builds_name_argument_union_for_multiple_functions():
    functions = [
        {
            "name": "get_weather",
            "parameters": {
                "type": "dict",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        {
            "name": "get_time",
            "parameters": {
                "type": "dict",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
            },
        },
    ]

    schema = build_function_call_schema(functions)

    assert len(schema["oneOf"]) == 2
    assert schema["oneOf"][0]["properties"]["name"] == {
        "const": "get_weather"
    }
    assert schema["oneOf"][1]["properties"]["arguments"]["required"] == ["zone"]


def test_rejects_empty_function_set():
    try:
        build_function_call_schema([])
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected an empty function set to be rejected")


def test_normalizes_java_and_javascript_generation_aliases():
    schema = normalize_generation_schema(
        {
            "type": "dict",
            "properties": {
                "text": {"type": "String"},
                "enabled": {"type": "Boolean"},
                "values": {"type": "Array"},
                "mapping": {"type": "HashMap"},
                "large": {"type": "Bigint"},
            },
        }
    )

    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["enabled"]["type"] == "boolean"
    assert schema["properties"]["values"]["type"] == "array"
    assert schema["properties"]["mapping"]["type"] == "object"
    assert schema["properties"]["large"]["type"] == "integer"


def test_drops_empty_bfcl_generation_type():
    schema = normalize_generation_schema({"type": "", "description": "unknown"})

    assert "type" not in schema
    assert schema["description"] == "unknown"


def test_builds_string_encoded_javascript_arguments():
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

    schema = build_function_call_schema(functions, language="javascript")
    arguments = schema["properties"]["arguments"]

    assert arguments["properties"]["enabled"]["type"] == "string"
    assert arguments["properties"]["items"]["type"] == "string"
