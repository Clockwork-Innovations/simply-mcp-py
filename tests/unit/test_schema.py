"""Unit tests for schema generation system."""

import dataclasses
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel, Field

from simply_mcp.validation.schema import (
    SchemaGenerationError,
    auto_generate_schema,
    extract_description_from_docstring,
    extract_param_descriptions_from_docstring,
    generate_schema_from_dataclass,
    generate_schema_from_function,
    generate_schema_from_pydantic,
    generate_schema_from_typeddict,
    python_type_to_json_schema_type,
)


class TestPythonTypeToJsonSchemaType:
    """Tests for python_type_to_json_schema_type function."""

    def test_basic_types(self) -> None:
        """Test conversion of basic Python types."""
        assert python_type_to_json_schema_type(int) == {"type": "integer"}
        assert python_type_to_json_schema_type(float) == {"type": "number"}
        assert python_type_to_json_schema_type(str) == {"type": "string"}
        assert python_type_to_json_schema_type(bool) == {"type": "boolean"}

    def test_none_type(self) -> None:
        """Test None type conversion."""
        assert python_type_to_json_schema_type(type(None)) == {"type": "null"}

    def test_list_type(self) -> None:
        """Test list type conversion."""
        assert python_type_to_json_schema_type(list) == {"type": "array"}
        assert python_type_to_json_schema_type(List[int]) == {
            "type": "array",
            "items": {"type": "integer"}
        }
        assert python_type_to_json_schema_type(List[str]) == {
            "type": "array",
            "items": {"type": "string"}
        }

    def test_dict_type(self) -> None:
        """Test dict type conversion."""
        assert python_type_to_json_schema_type(dict) == {"type": "object"}
        assert python_type_to_json_schema_type(Dict[str, int]) == {
            "type": "object",
            "additionalProperties": {"type": "integer"}
        }

    def test_optional_type(self) -> None:
        """Test Optional type conversion."""
        result = python_type_to_json_schema_type(Optional[str])
        assert "type" in result
        assert set(result["type"]) == {"string", "null"}

        result = python_type_to_json_schema_type(Optional[int])
        assert "type" in result
        assert set(result["type"]) == {"integer", "null"}

    def test_union_type(self) -> None:
        """Test Union type conversion."""
        result = python_type_to_json_schema_type(Union[int, str])
        assert "anyOf" in result
        assert len(result["anyOf"]) == 2

    def test_any_type(self) -> None:
        """Test Any type conversion."""
        result = python_type_to_json_schema_type(Any)
        # Any type returns empty schema (no constraints)
        assert result == {} or isinstance(result, dict)

    def test_nested_list(self) -> None:
        """Test nested list conversion."""
        result = python_type_to_json_schema_type(List[List[int]])
        assert result["type"] == "array"
        assert result["items"]["type"] == "array"
        assert result["items"]["items"]["type"] == "integer"

    def test_tuple_type(self) -> None:
        """Test tuple type conversion."""
        # Empty tuple
        assert python_type_to_json_schema_type(tuple) == {"type": "array"}

        # Tuple with specific types
        from typing import Tuple
        result = python_type_to_json_schema_type(Tuple[int, str, bool])
        assert result["type"] == "array"
        assert isinstance(result["items"], list)
        assert len(result["items"]) == 3


class TestExtractDescriptionFromDocstring:
    """Tests for extract_description_from_docstring function."""

    def test_simple_docstring(self) -> None:
        """Test extraction from simple docstring."""
        def func() -> None:
            """This is a simple description."""
            pass

        result = extract_description_from_docstring(func)
        assert result == "This is a simple description."

    def test_google_style_docstring(self) -> None:
        """Test extraction from Google-style docstring."""
        def func(a: int, b: int) -> int:
            """Add two numbers together.

            Args:
                a: First number
                b: Second number

            Returns:
                Sum of a and b
            """
            return a + b

        result = extract_description_from_docstring(func)
        assert result == "Add two numbers together."

    def test_numpy_style_docstring(self) -> None:
        """Test extraction from NumPy-style docstring."""
        def func(a: int, b: int) -> int:
            """Add two numbers together.

            Parameters
            ----------
            a : int
                First number
            b : int
                Second number

            Returns
            -------
            int
                Sum of a and b
            """
            return a + b

        result = extract_description_from_docstring(func)
        assert result == "Add two numbers together."

    def test_multiline_description(self) -> None:
        """Test extraction with multiline description."""
        def func() -> None:
            """This is a multiline description.
            It spans multiple lines.

            Args:
                None
            """
            pass

        result = extract_description_from_docstring(func)
        assert "multiline description" in result

    def test_no_docstring(self) -> None:
        """Test with no docstring."""
        def func() -> None:
            pass

        result = extract_description_from_docstring(func)
        assert result is None

    def test_empty_docstring(self) -> None:
        """Test with empty docstring."""
        def func() -> None:
            """"""
            pass

        result = extract_description_from_docstring(func)
        assert result is None or result == ""


class TestExtractParamDescriptionsFromDocstring:
    """Tests for extract_param_descriptions_from_docstring function."""

    def test_google_style_params(self) -> None:
        """Test extraction from Google-style docstring."""
        def func(name: str, age: int, active: bool) -> None:
            """Test function.

            Args:
                name: Person's name
                age: Person's age
                active: Whether person is active
            """
            pass

        result = extract_param_descriptions_from_docstring(func)
        assert result["name"] == "Person's name"
        assert result["age"] == "Person's age"
        assert result["active"] == "Whether person is active"

    def test_google_style_with_types(self) -> None:
        """Test extraction with type annotations in docstring."""
        def func(count: int) -> None:
            """Test function.

            Args:
                count (int): Number of items
            """
            pass

        result = extract_param_descriptions_from_docstring(func)
        assert result["count"] == "Number of items"

    def test_numpy_style_params(self) -> None:
        """Test extraction from NumPy-style docstring."""
        def func(x: float, y: float) -> float:
            """Calculate distance.

            Parameters
            ----------
            x : float
                X coordinate
            y : float
                Y coordinate

            Returns
            -------
            float
                Distance from origin
            """
            return (x**2 + y**2) ** 0.5

        result = extract_param_descriptions_from_docstring(func)
        assert result["x"] == "X coordinate"
        assert result["y"] == "Y coordinate"

    def test_no_params_section(self) -> None:
        """Test with no params section."""
        def func(a: int) -> int:
            """Simple function."""
            return a

        result = extract_param_descriptions_from_docstring(func)
        assert result == {}


class TestGenerateSchemaFromFunction:
    """Tests for generate_schema_from_function function."""

    def test_simple_function(self) -> None:
        """Test schema generation from simple function."""
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        schema = generate_schema_from_function(add)
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["b"]["type"] == "integer"
        assert set(schema["required"]) == {"a", "b"}

    def test_function_with_defaults(self) -> None:
        """Test function with default values."""
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet a person."""
            return f"{greeting}, {name}"

        schema = generate_schema_from_function(greet)
        assert "name" in schema["required"]
        assert "greeting" not in schema["required"]
        assert schema["properties"]["greeting"]["default"] == "Hello"

    def test_function_with_optional(self) -> None:
        """Test function with Optional parameters."""
        def process(data: str, config: Optional[Dict[str, Any]] = None) -> None:
            """Process data."""
            pass

        schema = generate_schema_from_function(process)
        assert "data" in schema["required"]
        assert "config" not in schema["required"]

    def test_function_with_docstring_descriptions(self) -> None:
        """Test extraction of parameter descriptions from docstring."""
        def calculate(x: int, y: int) -> int:
            """Calculate something.

            Args:
                x: First operand
                y: Second operand
            """
            return x + y

        schema = generate_schema_from_function(calculate)
        assert schema["properties"]["x"]["description"] == "First operand"
        assert schema["properties"]["y"]["description"] == "Second operand"

    def test_function_with_mixed_types(self) -> None:
        """Test function with various type hints."""
        def mixed(
            text: str,
            count: int,
            ratio: float,
            active: bool,
            tags: List[str],
            metadata: Dict[str, Any]
        ) -> None:
            """Function with mixed types."""
            pass

        schema = generate_schema_from_function(mixed)
        assert schema["properties"]["text"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["ratio"]["type"] == "number"
        assert schema["properties"]["active"]["type"] == "boolean"
        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["metadata"]["type"] == "object"

    def test_function_skips_self_and_cls(self) -> None:
        """Test that self and cls parameters are skipped."""
        class MyClass:
            def method(self, value: int) -> None:
                """Instance method."""
                pass

            @classmethod
            def class_method(cls, value: int) -> None:
                """Class method."""
                pass

        schema = generate_schema_from_function(MyClass.method)
        assert "self" not in schema["properties"]
        assert "value" in schema["properties"]

        schema = generate_schema_from_function(MyClass.class_method)
        assert "cls" not in schema["properties"]
        assert "value" in schema["properties"]

    def test_function_without_type_hints(self) -> None:
        """Test function without type hints."""
        def no_hints(a, b):  # type: ignore[no-untyped-def]
            """Function without hints."""
            return a + b

        schema = generate_schema_from_function(no_hints)
        assert schema["type"] == "object"
        # Should still generate properties, but without type info
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]


class TestGenerateSchemaFromPydantic:
    """Tests for generate_schema_from_pydantic function."""

    def test_simple_pydantic_model(self) -> None:
        """Test schema generation from simple Pydantic model."""
        class User(BaseModel):
            name: str
            age: int

        schema = generate_schema_from_pydantic(User)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert set(schema["required"]) == {"name", "age"}

    def test_pydantic_with_field_descriptions(self) -> None:
        """Test Pydantic model with Field descriptions."""
        class Product(BaseModel):
            name: str = Field(description="Product name")
            price: float = Field(description="Product price", gt=0)
            in_stock: bool = Field(default=True, description="Availability")

        schema = generate_schema_from_pydantic(Product)
        assert schema["properties"]["name"]["description"] == "Product name"
        assert schema["properties"]["price"]["description"] == "Product price"
        assert schema["properties"]["price"]["exclusiveMinimum"] == 0
        assert schema["properties"]["in_stock"]["default"] is True

    def test_pydantic_with_validators(self) -> None:
        """Test Pydantic model with validators."""
        class Config(BaseModel):
            port: int = Field(ge=1, le=65535, description="Port number")
            host: str = Field(min_length=1, max_length=255, description="Host address")
            timeout: float = Field(default=30.0, ge=0)

        schema = generate_schema_from_pydantic(Config)
        assert schema["properties"]["port"]["minimum"] == 1
        assert schema["properties"]["port"]["maximum"] == 65535
        assert schema["properties"]["host"]["minLength"] == 1
        assert schema["properties"]["host"]["maxLength"] == 255
        assert schema["properties"]["timeout"]["minimum"] == 0

    def test_pydantic_with_optional_fields(self) -> None:
        """Test Pydantic model with optional fields."""
        class Settings(BaseModel):
            api_key: str
            api_secret: Optional[str] = None
            timeout: Optional[int] = None

        schema = generate_schema_from_pydantic(Settings)
        assert "api_key" in schema["required"]
        assert "api_secret" not in schema.get("required", [])
        assert "timeout" not in schema.get("required", [])

    def test_pydantic_nested_models(self) -> None:
        """Test Pydantic model with nested models."""
        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        schema = generate_schema_from_pydantic(Person)
        # Nested model should be referenced
        assert "address" in schema["properties"]
        # Pydantic may use $ref for nested models
        assert "$ref" in schema["properties"]["address"] or "type" in schema["properties"]["address"]


class TestGenerateSchemaFromDataclass:
    """Tests for generate_schema_from_dataclass function."""

    def test_simple_dataclass(self) -> None:
        """Test schema generation from simple dataclass."""
        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        schema = generate_schema_from_dataclass(Point)
        assert schema["type"] == "object"
        assert schema["properties"]["x"]["type"] == "integer"
        assert schema["properties"]["y"]["type"] == "integer"
        assert set(schema["required"]) == {"x", "y"}

    def test_dataclass_with_defaults(self) -> None:
        """Test dataclass with default values."""
        @dataclasses.dataclass
        class Config:
            host: str = "localhost"
            port: int = 8080
            debug: bool = False

        schema = generate_schema_from_dataclass(Config)
        assert "required" not in schema or len(schema["required"]) == 0
        assert schema["properties"]["host"]["default"] == "localhost"
        assert schema["properties"]["port"]["default"] == 8080
        assert schema["properties"]["debug"]["default"] is False

    def test_dataclass_with_optional(self) -> None:
        """Test dataclass with Optional fields."""
        @dataclasses.dataclass
        class User:
            name: str
            email: Optional[str] = None

        schema = generate_schema_from_dataclass(User)
        assert "name" in schema["required"]
        assert "email" not in schema["required"]

    def test_dataclass_mixed_defaults(self) -> None:
        """Test dataclass with mixed required and optional fields."""
        @dataclasses.dataclass
        class Task:
            title: str
            description: str
            priority: int = 0
            completed: bool = False

        schema = generate_schema_from_dataclass(Task)
        assert set(schema["required"]) == {"title", "description"}
        assert schema["properties"]["priority"]["default"] == 0
        assert schema["properties"]["completed"]["default"] is False

    def test_non_dataclass_raises_error(self) -> None:
        """Test that non-dataclass raises error."""
        class NotADataclass:
            x: int
            y: int

        with pytest.raises(SchemaGenerationError):
            generate_schema_from_dataclass(NotADataclass)


class TestGenerateSchemaFromTypedDict:
    """Tests for generate_schema_from_typeddict function."""

    def test_simple_typeddict(self) -> None:
        """Test schema generation from simple TypedDict."""
        from typing import TypedDict

        class UserDict(TypedDict):
            name: str
            age: int

        schema = generate_schema_from_typeddict(UserDict)
        assert schema["type"] == "object"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"

    def test_typeddict_with_total_false(self) -> None:
        """Test TypedDict with total=False."""
        from typing import TypedDict

        class PartialDict(TypedDict, total=False):
            name: str
            age: int

        schema = generate_schema_from_typeddict(PartialDict)
        assert schema["type"] == "object"
        # With total=False, no fields are required
        assert "required" not in schema or len(schema["required"]) == 0

    def test_non_typeddict_raises_error(self) -> None:
        """Test that non-TypedDict raises error."""
        class NotATypedDict:
            name: str
            age: int

        with pytest.raises(SchemaGenerationError):
            generate_schema_from_typeddict(NotATypedDict)


class TestAutoGenerateSchema:
    """Tests for auto_generate_schema function."""

    def test_auto_detect_function(self) -> None:
        """Test auto-detection of function."""
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        schema = auto_generate_schema(add)
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]

    def test_auto_detect_pydantic(self) -> None:
        """Test auto-detection of Pydantic model."""
        class User(BaseModel):
            name: str
            age: int

        schema = auto_generate_schema(User)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]

    def test_auto_detect_dataclass(self) -> None:
        """Test auto-detection of dataclass."""
        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        schema = auto_generate_schema(Point)
        assert schema["type"] == "object"
        assert "x" in schema["properties"]

    def test_auto_detect_typeddict(self) -> None:
        """Test auto-detection of TypedDict."""
        from typing import TypedDict

        class ConfigDict(TypedDict):
            host: str
            port: int

        schema = auto_generate_schema(ConfigDict)
        assert schema["type"] == "object"
        assert "host" in schema["properties"]

    def test_unsupported_type_raises_error(self) -> None:
        """Test that unsupported type raises error."""
        class UnsupportedType:
            pass

        with pytest.raises(SchemaGenerationError) as exc_info:
            auto_generate_schema(UnsupportedType)

        assert "Unsupported source type" in str(exc_info.value)

    def test_auto_detect_lambda(self) -> None:
        """Test auto-detection of lambda function."""
        func = lambda x, y: x + y  # noqa: E731

        # Lambda should be detected as callable
        schema = auto_generate_schema(func)
        assert schema["type"] == "object"


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_nested_optional_types(self) -> None:
        """Test handling of nested Optional types."""
        def process(data: Optional[List[Optional[str]]]) -> None:
            """Process optional list of optional strings."""
            pass

        schema = generate_schema_from_function(process)
        assert "data" in schema["properties"]

    def test_union_with_multiple_types(self) -> None:
        """Test Union with multiple non-None types."""
        def handle(value: Union[int, str, bool]) -> None:
            """Handle multiple types."""
            pass

        schema = generate_schema_from_function(handle)
        assert "value" in schema["properties"]
        assert "anyOf" in schema["properties"]["value"]

    def test_complex_nested_structure(self) -> None:
        """Test complex nested data structure."""
        def analyze(data: Dict[str, List[Dict[str, Any]]]) -> None:
            """Analyze nested data."""
            pass

        schema = generate_schema_from_function(analyze)
        assert schema["properties"]["data"]["type"] == "object"

    def test_function_with_all_features(self) -> None:
        """Test function with all supported features."""
        def comprehensive(
            required_str: str,
            required_int: int,
            optional_float: float = 1.5,
            optional_bool: Optional[bool] = None,
            list_param: List[str] = None,  # type: ignore[assignment]
            dict_param: Dict[str, int] = None,  # type: ignore[assignment]
        ) -> None:
            """Comprehensive function.

            Args:
                required_str: A required string
                required_int: A required integer
                optional_float: An optional float with default
                optional_bool: An optional boolean
                list_param: A list of strings
                dict_param: A dictionary
            """
            pass

        schema = generate_schema_from_function(comprehensive)
        assert set(schema["required"]) == {"required_str", "required_int"}
        assert schema["properties"]["optional_float"]["default"] == 1.5
        assert "required_str" in schema["properties"]
        assert schema["properties"]["required_str"]["description"] == "A required string"

    def test_pydantic_with_complex_validation(self) -> None:
        """Test Pydantic model with complex validation rules."""
        class ComplexModel(BaseModel):
            email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
            age: int = Field(ge=0, le=150)
            tags: List[str] = Field(default_factory=list, min_length=0, max_length=10)
            score: float = Field(ge=0.0, le=100.0)

        schema = generate_schema_from_pydantic(ComplexModel)
        assert "pattern" in schema["properties"]["email"]
        assert schema["properties"]["age"]["minimum"] == 0
        assert schema["properties"]["age"]["maximum"] == 150
        assert schema["properties"]["score"]["minimum"] == 0.0
        assert schema["properties"]["score"]["maximum"] == 100.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_function_with_no_parameters(self) -> None:
        """Test function with no parameters."""
        def no_params() -> str:
            """Return hello."""
            return "hello"

        schema = generate_schema_from_function(no_params)
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert "required" not in schema or len(schema["required"]) == 0

    def test_function_with_varargs(self) -> None:
        """Test function with *args and **kwargs."""
        def with_varargs(a: int, *args: int, **kwargs: str) -> None:
            """Function with varargs."""
            pass

        schema = generate_schema_from_function(with_varargs)
        # Should only include 'a', not *args or **kwargs
        assert "a" in schema["properties"]
        assert "args" not in schema["properties"]
        assert "kwargs" not in schema["properties"]

    def test_empty_pydantic_model(self) -> None:
        """Test empty Pydantic model."""
        class EmptyModel(BaseModel):
            pass

        schema = generate_schema_from_pydantic(EmptyModel)
        assert schema["type"] == "object"
        assert schema["properties"] == {}

    def test_pydantic_model_with_any(self) -> None:
        """Test Pydantic model with Any type."""
        class FlexibleModel(BaseModel):
            data: Any

        schema = generate_schema_from_pydantic(FlexibleModel)
        assert "data" in schema["properties"]

    def test_function_with_none_default(self) -> None:
        """Test function with None as default value."""
        def with_none(value: Optional[str] = None) -> None:
            """Function with None default."""
            pass

        schema = generate_schema_from_function(with_none)
        assert "value" not in schema.get("required", [])
        # Default of None might not be included in schema
        assert "value" in schema["properties"]

    def test_literal_type(self) -> None:
        """Test Literal type conversion."""
        from typing import Literal

        def with_literal(mode: Literal["read", "write", "append"]) -> None:
            """Function with literal type."""
            pass

        schema = generate_schema_from_function(with_literal)
        assert "mode" in schema["properties"]
        assert "enum" in schema["properties"]["mode"]
        assert set(schema["properties"]["mode"]["enum"]) == {"read", "write", "append"}

    def test_empty_docstring_description(self) -> None:
        """Test function with only whitespace in docstring."""
        def func() -> None:
            """   """
            pass

        result = extract_description_from_docstring(func)
        assert result is None or result == ""

    def test_dataclass_with_default_factory(self) -> None:
        """Test dataclass with default_factory."""
        from dataclasses import field

        @dataclasses.dataclass
        class Config:
            items: List[str] = field(default_factory=list)
            counts: Dict[str, int] = field(default_factory=dict)

        schema = generate_schema_from_dataclass(Config)
        # Fields with default_factory are not required
        assert "items" not in schema.get("required", [])
        assert "counts" not in schema.get("required", [])
