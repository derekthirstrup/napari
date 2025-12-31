"""
Pydantic V2 compatibility module for napari.

This module provides the Pydantic V2 API for napari, with some compatibility
shims for code that was written using Pydantic V1 patterns.

Migration from V1 to V2:
- BaseModel: use directly from pydantic
- BaseSettings: now in pydantic-settings package
- validator -> field_validator (with @classmethod, mode='before'/'after')
- root_validator -> model_validator (mode='before'/'after')
- class Config -> model_config = ConfigDict(...)
- __fields__ -> model_fields
- .dict() -> .model_dump()
- .json() -> .model_dump_json()
- .parse_obj() -> .model_validate()
- parse_obj_as() -> TypeAdapter(T).validate_python()
"""

from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar

# Core Pydantic V2 imports
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    conlist,
    constr,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

# BaseSettings is now in a separate package
from pydantic_settings import BaseSettings, SettingsConfigDict

# Color type moved to pydantic-extra-types
try:
    from pydantic_extra_types.color import Color
except ImportError:
    # Fallback if pydantic-extra-types not installed
    Color = None  # type: ignore[misc, assignment]

# Type variable for generic models
T = TypeVar('T')


# For sequence_like functionality
def sequence_like(v: Any) -> bool:
    """
    Determine whether a value is sequence-like (list, tuple, set, or frozenset), excluding strings and bytes.
    
    Returns:
        True if the value is a list, tuple, set, or frozenset, False otherwise.
    """
    return isinstance(v, (list, tuple, set, frozenset))


# ROOT_KEY equivalent (used in some validation contexts)
ROOT_KEY = '__root__'

# ---- Compatibility aliases for migration ----

# These are deprecated and will be removed. Use the V2 equivalents.
validator = field_validator  # deprecated: use @field_validator
root_validator = model_validator  # deprecated: use @model_validator


def parse_obj_as(type_: type[T], obj: Any) -> T:
    """
    Validate and convert `obj` to the specified target type.
    
    Parameters:
        type_ (type[T]): The target type to parse the input into.
        obj (Any): The value to be validated and converted.
    
    Returns:
        T: The parsed value as the specified type.
    
    Deprecated:
        Use `TypeAdapter(type_).validate_python(obj)` directly.
    """
    return TypeAdapter(type_).validate_python(obj)


# ---- Removed V1 APIs with stubs/alternatives ----


# ModelMetaclass is no longer used in V2. Instead, use:
# - __pydantic_complete__ class method for post-model-creation hooks
# - model_post_init for per-instance initialization
# We provide a stub that just returns the original metaclass
class ModelMetaclass(type):
    """Stub for V1 ModelMetaclass. Not needed in V2.

    In Pydantic V2, use __pydantic_complete__ or model_post_init instead.
    This is provided only for compatibility during migration.
    """


# ModelField is replaced by FieldInfo in V2
# Provide alias for compatibility
ModelField = FieldInfo


# ClassAttribute stub - not needed in V2
class ClassAttribute:
    """Stub for V1 ClassAttribute. Use standard class attributes in V2."""

    def __init__(self, name: str, value: Any) -> None:
        """
        Create an error-like object holding a name and its associated value.
        
        Parameters:
            name (str): Identifier or code for the error/context.
            value (Any): The value associated with the name.
        """
        self.name = name
        self.value = value


# GenericModel is no longer needed - just use Generic with BaseModel
GenericModel = BaseModel  # Just use BaseModel with Generic[T]


# SHAPE_LIST constant - used for checking field shapes in V1
# In V2, check annotation directly instead
SHAPE_LIST = 'list'


# ---- Error handling compatibility ----


# V2 uses different error handling
class ErrorWrapper:
    """Stub for V1 ErrorWrapper. Use ValidationError directly in V2."""

    def __init__(self, exc: Exception, loc: tuple) -> None:
        """
        Initialize the ErrorWrapper with an underlying exception and its location.
        
        Parameters:
            exc (Exception): The original exception being wrapped.
            loc (tuple): A tuple describing the location/path in the model where the error occurred (e.g., field names or indices).
        """
        self.exc = exc
        self.loc = loc


def display_errors(errors: list[Any]) -> str:
    """
    Format a list of validation errors into a human-readable multiline string.
    
    Accepts Pydantic v1-style error dicts (mapping with 'loc' and 'msg') or v2-style ErrorDetails-like objects/typed dicts with 'loc' and 'msg' attributes/keys. Each input error is rendered as a single line "  <location>: <message>" where location parts are joined by dots.
    
    Parameters:
        errors (list[Any]): Iterable of error entries (dict-like or object-like) with `loc` and `msg`.
    
    Returns:
        str: Multiline string with one formatted error per line; lines are joined with '\n'.
    """
    lines = []
    for error in errors:
        # Handle both dict and ErrorDetails (TypedDict) formats
        if hasattr(error, 'get'):
            loc = '.'.join(str(loc_part) for loc_part in error.get('loc', ()))
            msg = error.get('msg', '')
        else:
            loc = '.'.join(
                str(loc_part) for loc_part in getattr(error, 'loc', ())
            )
            msg = getattr(error, 'msg', '')
        lines.append(f'  {loc}: {msg}')
    return '\n'.join(lines)


# ---- Settings compatibility ----


class SettingsError(ValueError):
    """Error raised when settings validation fails."""


# Settings source types (simplified for V2)
if TYPE_CHECKING:
    from pydantic_settings import (
        EnvSettingsSource,
        PydanticBaseSettingsSource,
    )

    SettingsSourceCallable = PydanticBaseSettingsSource
else:
    EnvSettingsSource = Any
    SettingsSourceCallable = Any


# ---- Extra enum for forbid/allow/ignore ----
class Extra:
    """V1-style Extra enum. Use ConfigDict(extra='...') in V2 instead."""

    allow = 'allow'
    forbid = 'forbid'
    ignore = 'ignore'


# ---- Stub modules for compatibility ----


class _ErrorsModule:
    """Stub for pydantic.v1.errors module."""

    class PydanticValueError(ValueError):
        """Base class for pydantic value errors."""

        code = 'value_error'
        msg_template = 'value error'

        def __init__(self, **ctx: Any) -> None:
            """
            Initialize the error with context and format its message.
            
            Stores the provided context on `self.ctx` and initializes the base exception message by formatting the instance's `msg_template` using the given context keys and values.
            
            Parameters:
                **ctx: Mapping of placeholder names to values used to format `msg_template`.
            """
            self.ctx = ctx
            super().__init__(self.msg_template.format(**ctx))

    class PydanticTypeError(TypeError):
        """Base class for pydantic type errors."""

        code = 'type_error'
        msg_template = 'type error'

        def __init__(self, **ctx: Any) -> None:
            """
            Initialize the error with context and format its message.
            
            Stores the provided context on `self.ctx` and initializes the base exception message by formatting the instance's `msg_template` using the given context keys and values.
            
            Parameters:
                **ctx: Mapping of placeholder names to values used to format `msg_template`.
            """
            self.ctx = ctx
            super().__init__(self.msg_template.format(**ctx))


errors = _ErrorsModule()


class _TypesModule:
    """Stub for pydantic.v1.types module."""

    class ConstrainedInt(int):
        """V1-style constrained int. Use Annotated[int, Field(...)] in V2."""

        strict: bool = False
        gt: int | None = None
        ge: int | None = None
        lt: int | None = None
        le: int | None = None
        multiple_of: int | None = None

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            handler,
        ):
            """
            Provide a pydantic-core schema for this constrained integer type.
            
            Parameters:
                cls: The constrained type class implementing `_validate`.
                source_type (Any): The original Python type being processed by the schema.
                handler: A pydantic-core schema handler callable (used by pydantic to resolve nested schemas).
            
            Returns:
                core_schema: A pydantic-core schema that runs `cls._validate` before applying the integer schema.
            """
            from pydantic_core import core_schema as cs

            return cs.no_info_before_validator_function(
                cls._validate,
                cs.int_schema(),
            )

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema_, handler):
            """
            Augments the JSON Schema produced for this constrained numeric type with applicable numeric constraint keywords.
            
            Parameters:
                core_schema_ (Any): The pydantic core schema for the type; passed to the handler to produce the base JSON Schema.
                handler (Callable[[Any], dict]): Function that converts `core_schema_` to a JSON Schema dict.
            
            Returns:
                dict: The JSON Schema dict for the class, with any of the following keys added when defined on the class: `exclusiveMinimum`, `minimum`, `exclusiveMaximum`, `maximum`, `multipleOf`.
            """
            json_schema = handler(core_schema_)
            if cls.gt is not None:
                json_schema['exclusiveMinimum'] = cls.gt
            if cls.ge is not None:
                json_schema['minimum'] = cls.ge
            if cls.lt is not None:
                json_schema['exclusiveMaximum'] = cls.lt
            if cls.le is not None:
                json_schema['maximum'] = cls.le
            if cls.multiple_of is not None:
                json_schema['multipleOf'] = cls.multiple_of
            return json_schema

        @classmethod
        def _validate(cls, v: Any) -> int:
            """
            Validate and coerce a numeric input to an integer while enforcing the class's numeric constraints.
            
            Parameters:
                v (Any): Value to validate and coerce to an integer.
            
            Returns:
                int: The coerced integer value that satisfies the class constraints.
            
            Raises:
                TypeError: If `v` is not an int or float.
                ValueError: If the coerced integer violates any configured bounds (`gt`, `ge`, `lt`, `le`) or `multiple_of`.
            """
            if not isinstance(v, (int, float)):
                raise TypeError('integer required')
            v = int(v)
            if cls.gt is not None and v <= cls.gt:
                raise ValueError(f'must be greater than {cls.gt}')
            if cls.ge is not None and v < cls.ge:
                raise ValueError(f'must be greater than or equal to {cls.ge}')
            if cls.lt is not None and v >= cls.lt:
                raise ValueError(f'must be less than {cls.lt}')
            if cls.le is not None and v > cls.le:
                raise ValueError(f'must be less than or equal to {cls.le}')
            if cls.multiple_of is not None and v % cls.multiple_of != 0:
                raise ValueError(f'must be a multiple of {cls.multiple_of}')
            return v

    class ConstrainedFloat(float):
        """V1-style constrained float. Use Annotated[float, Field(...)] in V2."""

        strict: bool = False
        gt: float | None = None
        ge: float | None = None
        lt: float | None = None
        le: float | None = None
        multiple_of: float | None = None

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            handler,
        ):
            """
            Provide a pydantic-core schema for this constrained-float type that runs the class `_validate` pre-validator and then enforces float validation.
            
            Parameters:
                cls: The constrained type class providing `_validate`.
                source_type: The original Python type being adapted (unused by this schema).
                handler: Schema generation handler callable provided by pydantic (unused by this schema).
            
            Returns:
                A pydantic-core schema that invokes `cls._validate` before applying float validation.
            """
            from pydantic_core import core_schema as cs

            return cs.no_info_before_validator_function(
                cls._validate,
                cs.float_schema(),
            )

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema_, handler):
            """
            Augments the JSON Schema produced for this constrained numeric type with applicable numeric constraint keywords.
            
            Parameters:
                core_schema_ (Any): The pydantic core schema for the type; passed to the handler to produce the base JSON Schema.
                handler (Callable[[Any], dict]): Function that converts `core_schema_` to a JSON Schema dict.
            
            Returns:
                dict: The JSON Schema dict for the class, with any of the following keys added when defined on the class: `exclusiveMinimum`, `minimum`, `exclusiveMaximum`, `maximum`, `multipleOf`.
            """
            json_schema = handler(core_schema_)
            if cls.gt is not None:
                json_schema['exclusiveMinimum'] = cls.gt
            if cls.ge is not None:
                json_schema['minimum'] = cls.ge
            if cls.lt is not None:
                json_schema['exclusiveMaximum'] = cls.lt
            if cls.le is not None:
                json_schema['maximum'] = cls.le
            if cls.multiple_of is not None:
                json_schema['multipleOf'] = cls.multiple_of
            return json_schema

        @classmethod
        def _validate(cls, v: Any) -> float:
            """
            Validate a value and coerce it to a float while enforcing optional comparison bounds on the class.
            
            Parameters:
                cls: Class-like object providing optional numeric attributes `gt`, `ge`, `lt`, and `le` used as bounds.
                v (Any): Value to validate and convert.
            
            Returns:
                float: The validated value converted to a Python float.
            
            Raises:
                TypeError: If `v` is not an int or float.
                ValueError: If `v` violates any of the configured bounds (`gt`, `ge`, `lt`, `le`).
            """
            if not isinstance(v, (int, float)):
                raise TypeError('float required')
            v = float(v)
            if cls.gt is not None and v <= cls.gt:
                raise ValueError(f'must be greater than {cls.gt}')
            if cls.ge is not None and v < cls.ge:
                raise ValueError(f'must be greater than or equal to {cls.ge}')
            if cls.lt is not None and v >= cls.lt:
                raise ValueError(f'must be less than {cls.lt}')
            if cls.le is not None and v > cls.le:
                raise ValueError(f'must be less than or equal to {cls.le}')
            return v


types = _TypesModule()


class _UtilsModule:
    """Stub for pydantic.v1.utils module."""

    ROOT_KEY = '__root__'

    @staticmethod
    def sequence_like(v: Any) -> bool:
        """
        Determine whether a value is a sequence-like container (list, tuple, set, or frozenset), excluding strings and bytes.
        
        Parameters:
            v: Value to test for sequence-like type.
        
        Returns:
            True if `v` is an instance of `list`, `tuple`, `set`, or `frozenset`, False otherwise.
        """
        return isinstance(v, (list, tuple, set, frozenset))


utils = _UtilsModule()


class _MainModule:
    """Stub for pydantic.v1.main module."""

    ModelMetaclass = ModelMetaclass


main = _MainModule()


# color module compatibility
class _ColorModule:
    """Stub for pydantic.v1.color module."""

    Color = Color


color = _ColorModule()


__all__ = (
    'ROOT_KEY',
    'SHAPE_LIST',
    'Annotated',
    'BaseModel',
    'BaseSettings',
    'ClassAttribute',
    'Color',
    'ConfigDict',
    'EnvSettingsSource',
    'ErrorWrapper',
    'Extra',
    'Field',
    'FieldInfo',
    'FilePath',
    'Generic',
    'GenericModel',
    'GetCoreSchemaHandler',
    'GetJsonSchemaHandler',
    'JsonSchemaValue',
    'ModelField',
    'ModelMetaclass',
    'PrivateAttr',
    'SettingsConfigDict',
    'SettingsError',
    'SettingsSourceCallable',
    'TypeAdapter',
    'ValidationError',
    'color',
    'conlist',
    'constr',
    'core_schema',
    'display_errors',
    'errors',
    'field_serializer',
    'field_validator',
    'main',
    'model_serializer',
    'model_validator',
    'parse_obj_as',
    'root_validator',
    'sequence_like',
    'types',
    'utils',
    'validator',
)