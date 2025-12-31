import re
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, SupportsInt

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from napari.utils.theme import available_themes, is_theme_available
from napari.utils.translations import _load_language, get_language_packs, trans


class Theme(str):
    """
    Custom theme type to dynamically load all installed themes.
    """

    # https://docs.pydantic.dev/latest/concepts/types/#custom-types

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """
        Provide a Pydantic core schema that enforces string input and then delegates validation to the class's `validate` method.
        
        Returns:
            CoreSchema: A schema that first validates the value as a string and then calls `cls.validate` to produce the normalized/validated result.
        """
        return core_schema.no_info_before_validator_function(
            cls.validate,
            core_schema.str_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema_: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # TODO: Provide a way to handle keys so we can display human readable
        # option in the preferences dropdown
        """
        Populate the JSON Schema for the Theme type with the current list of available themes.
        
        Parameters:
            core_schema_ (CoreSchema): The core Pydantic schema for the Theme type.
            handler (GetJsonSchemaHandler): Callable that converts the core schema into a JSON Schema value.
        
        Returns:
            JsonSchemaValue: The JSON Schema produced by `handler(core_schema_)` with its `enum` field set to the list returned by `available_themes()`.
        """
        json_schema = handler(core_schema_)
        json_schema['enum'] = available_themes()
        return json_schema

    @classmethod
    def validate(cls, v):
        """
        Validate and normalize a theme name to a lowercase available theme.
        
        Parameters:
            v (str): Candidate theme name.
        
        Returns:
            theme (str): Normalized lowercase theme name that is available.
        
        Raises:
            ValueError: If `v` is not a string or is not one of the available themes.
        """
        if not isinstance(v, str):
            raise ValueError(trans._('must be a string', deferred=True))  # noqa: TRY004

        value = v.lower()
        if not is_theme_available(value):
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {themes}',
                    deferred=True,
                    value=value,
                    themes=', '.join(available_themes()),
                )
            )

        return value


class Language(str):
    """
    Custom theme type to dynamically load all installed language packs.
    """

    # https://docs.pydantic.dev/latest/concepts/types/#custom-types

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """
        Provide a Pydantic core schema that enforces string input and then delegates validation to the class's `validate` method.
        
        Returns:
            CoreSchema: A schema that first validates the value as a string and then calls `cls.validate` to produce the normalized/validated result.
        """
        return core_schema.no_info_before_validator_function(
            cls.validate,
            core_schema.str_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema_: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # TODO: Provide a way to handle keys so we can display human readable
        # option in the preferences dropdown
        """
        Populate the JSON Schema with the available language pack identifiers.
        
        Augments the schema produced by the provided handler by adding an "enum"
        entry containing the available language pack keys from the current language.
        
        Returns:
            JsonSchemaValue: The JSON schema dictionary with an 'enum' key set to the list of language pack identifiers.
        """
        language_packs = list(get_language_packs(_load_language()).keys())
        json_schema = handler(core_schema_)
        json_schema['enum'] = language_packs
        return json_schema

    @classmethod
    def validate(cls, v):
        """
        Validate that a value is an available language pack.
        
        Checks that `v` is a string and that it matches one of the installed language packs; returns the validated value unchanged.
        
        Parameters:
            v: The candidate language pack identifier to validate.
        
        Returns:
            The input `v` when it is a valid, installed language pack.
        
        Raises:
            ValueError: If `v` is not a string or if it is not one of the available language packs.
        """
        if not isinstance(v, str):
            raise ValueError(trans._('must be a string', deferred=True))  # noqa: TRY004

        language_packs = list(get_language_packs(_load_language()).keys())
        if v not in language_packs:
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {language_packs}.',
                    deferred=True,
                    value=v,
                    language_packs=', '.join(language_packs),
                )
            )

        return v


@total_ordering
@dataclass
class Version:
    """A semver compatible version class.

    mostly vendored from python-semver (BSD-3):
    https://github.com/python-semver/python-semver/
    """

    major: SupportsInt
    minor: SupportsInt = 0
    patch: SupportsInt = 0
    prerelease: bytes | str | int | None = None
    build: bytes | str | int | None = None

    _SEMVER_PATTERN = re.compile(
        r"""
            ^
            (?P<major>0|[1-9]\d*)
            \.
            (?P<minor>0|[1-9]\d*)
            \.
            (?P<patch>0|[1-9]\d*)
            (?:-(?P<prerelease>
                (?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)
                (?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*
            ))?
            (?:\+(?P<build>
                [0-9a-zA-Z-]+
                (?:\.[0-9a-zA-Z-]+)*
            ))?
            $
        """,
        re.VERBOSE,
    )

    @classmethod
    def parse(cls, version: bytes | str) -> 'Version':
        """Convert string or bytes into Version object."""
        if isinstance(version, bytes):
            version = version.decode('UTF-8')
        match = cls._SEMVER_PATTERN.match(version)
        if match is None:
            raise ValueError(
                trans._(
                    '{version} is not valid SemVer string',
                    deferred=True,
                    version=version,
                )
            )
        matched_version_parts: dict[str, Any] = match.groupdict()
        return cls(**matched_version_parts)

    # NOTE: we're only comparing the numeric parts for now.
    # ALSO: the rest of the comparators come  from functools.total_ordering
    def __eq__(self, other) -> bool:
        try:
            return self.to_tuple()[:3] == self._from_obj(other).to_tuple()[:3]
        except TypeError:
            return NotImplemented

    def __lt__(self, other) -> bool:
        try:
            return self.to_tuple()[:3] < self._from_obj(other).to_tuple()[:3]
        except TypeError:
            return NotImplemented

    @classmethod
    def _from_obj(cls, other):
        if isinstance(other, str | bytes):
            other = Version.parse(other)
        elif isinstance(other, dict):
            other = Version(**other)
        elif isinstance(other, tuple | list):
            other = Version(*other)
        elif not isinstance(other, Version):
            raise TypeError(
                trans._(
                    'Expected str, bytes, dict, tuple, list, or {cls} instance, but got {other_type}',
                    deferred=True,
                    cls=cls,
                    other_type=type(other),
                )
            )
        return other

    def to_tuple(self) -> tuple[int, int, int, str | None, str | None]:
        """Return version as tuple (first three are int, last two Opt[str])."""
        return (
            int(self.major),
            int(self.minor),
            int(self.patch),
            str(self.prerelease) if self.prerelease is not None else None,
            str(self.build) if self.build is not None else None,
        )

    def __iter__(self):
        yield from self.to_tuple()

    def __str__(self) -> str:
        """
        Return the canonical semantic version string, including prerelease and build metadata when present.
        
        Returns:
            The version as a string in the form "major.minor.patch", with prerelease and build appended if set (for example: "1.2.3", "1.2.3-rc.1+build.1").
        """
        v = f'{self.major}.{self.minor}.{self.patch}'
        if self.prerelease:  # pragma: no cover
            v += str(self.prerelease)
        if self.build:  # pragma: no cover
            v += str(self.build)
        return v

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """
        Provide a Pydantic core schema that runs the class's `validate` method before applying any further schema validation.
        
        Parameters:
            source_type (Any): The original Python type being validated (unused by this implementation).
            handler (GetCoreSchemaHandler): Pydantic handler for building core schemas (unused by this implementation).
        
        Returns:
            CoreSchema: A core schema that invokes `cls.validate` as a pre-validation step and then accepts any value for downstream validation.
        """
        return core_schema.no_info_before_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """
        Provide a JSON Schema that describes a semantic version string with at least three numeric components.
        
        Returns:
            dict: JSON Schema object with "type" set to "string" and "pattern" enforcing a leading `major.minor.patch` numeric sequence.
        """
        return {'type': 'string', 'pattern': r'^\d+\.\d+\.\d+'}

    @classmethod
    def validate(cls, v):
        """
        Validate and normalize a value into a Version instance.
        
        Parameters:
            v: Input value representing a version (e.g., str, bytes, dict, tuple, list, or Version).
        
        Returns:
            Version: A normalized Version instance corresponding to the provided value.
        
        Raises:
            TypeError: If the input type is not supported for conversion to a Version.
        """
        return cls._from_obj(v)

    def _json_encode(self):
        return str(self)