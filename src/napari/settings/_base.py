from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from warnings import warn

from pydantic import ConfigDict

from napari._pydantic_compat import (
    PrivateAttr,
    ValidationError,
    display_errors,
)
from napari.settings._yaml import PydanticYamlMixin
from napari.utils.events import EmitterGroup, EventedModel
from napari.utils.misc import deep_update
from napari.utils.translations import trans

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet
    from typing import Union

    from napari.utils.events import Event

    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictStrAny = dict[str, Any]
    MappingIntStrAny = Mapping[IntStr, Any]

Dict = dict  # rename, because EventedSettings has method dict


def _exclude_defaults_evented(
    obj: EventedModel,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Exclude public model fields whose current values equal their defaults from the given serialized data.
    
    This function compares only public `model_fields` of an EventedModel to avoid pydantic v2's exclude_defaults issues with private attributes. It handles fields whose default is provided via a `default_factory` (by calling the factory), and recurses into nested EventedModel fields—including a nested field only if it contains any non-default values.
    
    Parameters:
        obj (EventedModel): The model instance to compare against.
        data (dict[str, Any]): Serialized representation of `obj` (e.g., model_dump output).
    
    Returns:
        dict[str, Any]: A dictionary containing only fields from `data` whose values differ from the model's defaults.
    """
    from pydantic.fields import PydanticUndefined

    result = {}
    for field_name, field_info in type(obj).model_fields.items():
        if field_name not in data:
            continue
        current_value = getattr(obj, field_name)
        default_value = field_info.default

        # If default is PydanticUndefined, check for default_factory
        if default_value is PydanticUndefined:
            if field_info.default_factory is not None:
                # Create a default instance to compare against
                # default_factory is a no-arg callable in Pydantic V2
                factory = field_info.default_factory
                default_value = factory()  # type: ignore[call-arg]
            else:
                # No default available, include the field
                result[field_name] = data[field_name]
                continue

        # For nested EventedModels, use their custom __eq__ which compares only fields
        if isinstance(current_value, EventedModel):
            if current_value == default_value:
                # Skip - equals default
                continue
            # Recurse for nested models
            nested_data = _exclude_defaults_evented(
                current_value, data[field_name]
            )
            if nested_data:  # Only include if there's non-default data
                result[field_name] = nested_data
        else:
            # For non-EventedModel fields, compare values directly
            if current_value != default_value:
                result[field_name] = data[field_name]

    return result


class SettingsError(ValueError):
    """Error raised when settings validation fails."""


class EventedSettings(EventedModel):
    """A variant of EventedModel designed for settings.

    Pydantic's BaseSettings model will attempt to determine the values of any
    fields not passed as keyword arguments by reading from the environment.

    Note: In Pydantic V2, BaseSettings is in a separate package (pydantic-settings).
    We inherit from EventedModel and add settings-like behavior.
    """

    # Pydantic V2 configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='ignore',
    )

    def __init__(self, **values: Any) -> None:
        """
        Initialize the EventedSettings instance, register a top-level 'changed' event, and connect nested sub-models for event propagation.
        
        Parameters:
            values: Initial field values to populate the model.
        """
        super().__init__(**values)
        self.events.add(changed=None)
        self._connect(self)

    @staticmethod
    def _warn_restart(*_: Event) -> None:
        warn(
            trans._(
                'Restart required for this change to take effect.',
                deferred=True,
            )
        )

    def _connect(self, model: EventedModel, prefix: str = '') -> None:
        """
        Connect event emitters of a nested EventedModel to this instance so sub-field changes are re-emitted with a dotted path and restart-required fields trigger a restart warning.
        
        Parameters:
            model (EventedModel): The sub-model whose evented fields will be connected.
            prefix (str): String prefix to prepend to emitted field paths (should end with '.' when non-empty).
        """
        # Use type(model).model_fields to avoid deprecation warning in V2.11+
        for name, field_info in type(model).model_fields.items():
            attr = getattr(model, name)
            if isinstance(getattr(attr, 'events', None), EmitterGroup):
                path = f'{prefix}{name}'
                attr.events.connect(partial(self._on_sub_event, field=path))
                self._connect(attr, f'{path}.')

            # Check for requires_restart in json_schema_extra
            extra = field_info.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get('requires_restart'):
                emitter = getattr(model.events, name)
                emitter.connect(self._warn_restart)

    def _on_sub_event(self, event: Event, field=None):
        """emit the field.attr name and new value"""
        if field:
            field += '.'
        value = getattr(event, 'value', None)
        self.events.changed(key=f'{field}{event._type}', value=value)


_NOT_SET = object()


class EventedConfigFileSettings(EventedSettings, PydanticYamlMixin):
    """This adds config read/write and yaml support to EventedSettings.

    If your settings class *only* needs to read variables from the environment,
    such as environment variables (but not a config file), then subclass from
    EventedSettings.
    """

    _config_path: Path | None = PrivateAttr(default=None)
    _save_on_change: bool = PrivateAttr(default=True)
    # this dict stores the data that came specifically from the config file.
    # it's populated in `config_file_settings_source` and
    # used in `_remove_env_settings`
    _config_file_settings: dict = PrivateAttr(default_factory=dict)
    # Store env settings for later removal
    _env_settings_cache: dict = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='ignore',
    )
    # Settings-specific config (not part of Pydantic V2 ConfigDict)
    # Use ClassVar to prevent Pydantic from treating this as a PrivateAttr
    _env_prefix: ClassVar[str] = 'NAPARI_'

    # provide config_path=None to prevent reading from disk.
    def __init__(self, config_path=_NOT_SET, **values: Any) -> None:
        """
        Create an instance of EventedConfigFileSettings, optionally loading and merging values from a config file and environment.
        
        Parameters:
            config_path (Path | str | _NOT_SET): Path to a configuration file to load. If `_NOT_SET`, no config file is read and `None` is used for the instance's config path.
            **values: Any: Explicit values passed to the constructor which take precedence over values from the config file but are overridden by environment-provided values.
        
        Behavior:
            - If `config_path` is provided, reads file settings via `config_file_settings_source`, deep-copies and preserves the original file data in `self._config_file_settings`, then merges explicit `values` on top of the file data (explicit values override file values).
            - Loads environment settings via `self._load_env_settings()` and merges them last so environment variables override both explicit and file-provided values.
            - Calls the parent initializer with the merged values.
            - Stores the resolved config path in `self._config_path` and caches raw environment values in `self._env_settings_cache`.
        
        Side effects:
            - May read and validate a config file when `config_path` is provided.
            - Sets private attributes `_config_path`, `_env_settings_cache`, and `_config_file_settings`.
        """
        import copy as copy_module

        # Determine the config path to use
        # Use provided config_path if given, otherwise None
        _cfg = config_path if config_path is not _NOT_SET else None

        # Load settings from config file if path exists
        original_file_settings = {}
        if _cfg is not None:
            file_settings = config_file_settings_source(self.__class__, _cfg)
            # Store original file settings before merging
            original_file_settings = copy_module.deepcopy(file_settings)
            # Merge with provided values (provided values take precedence)
            file_settings.update(values)
            values = file_settings

        # Load environment variables (returns parsed values for model, raw for cache)
        env_parsed, env_raw = self._load_env_settings()
        # Merge env settings (they take precedence over file settings)
        # Use deep_update with copy=True to avoid modifying the original dicts
        # (this is important when validation creates temporary instances)
        values = copy_module.deepcopy(values)
        deep_update(values, env_parsed, copy=False)

        super().__init__(**values)
        # Set private attributes after super().__init__()
        self._config_path = _cfg
        self._env_settings_cache = env_raw.copy()
        # Store the original file settings for later reference
        self._config_file_settings = original_file_settings

    def _load_env_settings(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Load settings from environment variables and return both parsed values for model initialization and raw strings for caching.
        
        Supports flat names (e.g., NAPARI_FIELD), nested names (e.g., NAPARI_SECTION_FIELD), and custom environment names declared via a field's `json_schema_extra` `env` entry. Values are JSON-decoded when possible; common boolean string forms ("true"/"false", "1"/"0", "yes"/"no") are mapped to booleans for custom env entries. The environment prefix defaults to the class attribute `_env_prefix` (uppercased) or "NAPARI_".
        
        Returns:
            tuple[dict, dict]: A pair (parsed_values, raw_values) where `parsed_values` is a dict suitable for model initialization (with nested dicts for nested fields) and `raw_values` contains the original environment strings used to produce `parsed_values`.
        """
        import json

        parsed: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        env_prefix = getattr(type(self), '_env_prefix', 'NAPARI_').upper()

        env_vars: Mapping[str, str | None] = {
            k.upper(): v for k, v in os.environ.items()
        }

        # Check for direct field mappings (flat access)
        # Use type(self).model_fields to avoid deprecation warning in V2.11+
        for field_name, field_info in type(self).model_fields.items():
            env_name = f'{env_prefix}{field_name.upper()}'
            if env_name in env_vars:
                val = env_vars[env_name]
                if val is not None:
                    raw[field_name] = val
                    # Try to parse as JSON for complex types
                    try:
                        parsed[field_name] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        parsed[field_name] = val

            # Check for nested field access (e.g., NAPARI_APPEARANCE_THEME)
            nested_prefix = f'{env_prefix}{field_name.upper()}_'
            for env_name, env_val in env_vars.items():
                if env_name.startswith(nested_prefix) and env_val is not None:
                    nested_path = env_name[len(nested_prefix) :].lower()
                    if field_name not in parsed:
                        parsed[field_name] = {}
                        raw[field_name] = {}
                    elif not isinstance(parsed[field_name], dict):
                        continue  # Already set to non-dict value
                    raw[field_name][nested_path] = env_val
                    # Try to parse nested value as JSON
                    try:
                        parsed[field_name][nested_path] = json.loads(env_val)
                    except (json.JSONDecodeError, TypeError):
                        parsed[field_name][nested_path] = env_val

            # Check for custom env names in nested model fields (json_schema_extra)
            annotation = field_info.annotation
            if annotation is not None:
                try:
                    if hasattr(annotation, 'model_fields'):
                        for (
                            nested_name,
                            nested_info,
                        ) in annotation.model_fields.items():
                            extra = nested_info.json_schema_extra or {}
                            if isinstance(extra, dict) and 'env' in extra:
                                custom_env = extra['env'].upper()
                                if custom_env in env_vars:
                                    val = env_vars[custom_env]
                                    if val is None:
                                        continue
                                    if field_name not in parsed:
                                        parsed[field_name] = {}
                                        raw[field_name] = {}
                                    elif not isinstance(
                                        parsed[field_name], dict
                                    ):
                                        continue
                                    raw[field_name][nested_name] = val
                                    # Try to parse as JSON, but handle booleans specially
                                    if val.lower() in ('true', '1', 'yes'):
                                        parsed[field_name][nested_name] = True
                                    elif val.lower() in ('false', '0', 'no'):
                                        parsed[field_name][nested_name] = False
                                    else:
                                        try:
                                            parsed[field_name][nested_name] = (
                                                json.loads(val)
                                            )
                                        except (
                                            json.JSONDecodeError,
                                            TypeError,
                                        ):
                                            parsed[field_name][nested_name] = (
                                                val
                                            )
                except (TypeError, AttributeError):
                    pass

        return parsed, raw

    def _maybe_save(self):
        """
        Persist settings to the configured config file when automatic saving is enabled.
        
        If the instance has `_save_on_change` enabled and a `config_path` is set, this triggers saving the current settings to that path; otherwise no action is taken.
        """
        if self._save_on_change and self.config_path:
            self.save()

    def _on_sub_event(self, event, field=None):
        super()._on_sub_event(event, field)
        self._maybe_save()

    @property
    def config_path(self):
        """
        Get the filesystem path used for loading and saving configuration.
        
        Returns:
            Path | None: The resolved path to the configuration file, or `None` if no config path is set.
        """
        return self._config_path

    def model_dump(  # type: ignore[override]
        self,
        *,
        mode: str = 'python',
        include: AbstractSetIntStr | MappingIntStrAny | None = None,
        exclude: AbstractSetIntStr | MappingIntStrAny | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_env: bool = False,
        **kwargs: Any,
    ) -> DictStrAny:
        """
        Produce a dictionary representation of the model.
        
        If `exclude_defaults` is True, fields whose values equal their defaults (including nested EventedModel fields) are removed from the output. If `exclude_env` is True, values that originated from environment variables are removed. Other parameters (include, exclude, by_alias, exclude_unset, exclude_none, mode) control selection and formatting consistent with Pydantic's `model_dump`.
        Returns:
            dict: The model represented as a dictionary.
        """
        # Don't pass exclude_defaults to super() - we handle it ourselves
        # because Pydantic V2's exclude_defaults doesn't work correctly for
        # EventedModel (it compares private attrs which are always different)
        data = super().model_dump(
            mode=mode,
            include=include,  # type: ignore[arg-type]
            exclude=exclude,  # type: ignore[arg-type]
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=False,  # Handle below
            exclude_none=exclude_none,
            **kwargs,
        )
        if exclude_defaults:
            data = _exclude_defaults_evented(self, data)
        if exclude_env:
            self._remove_env_settings(data)
        return data

    # Backwards compatibility alias
    def dict(  # type: ignore[override]
        self,
        *,
        include: AbstractSetIntStr | MappingIntStrAny | None = None,
        exclude: AbstractSetIntStr | MappingIntStrAny | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_env: bool = False,
    ) -> DictStrAny:
        """
        Provide a backward-compatible dict representation of the model; deprecated — use `model_dump`.
        
        This preserves the same include/exclude and filtering options accepted here for compatibility with older callers. It is kept only for backward compatibility and may be removed in a future release.
        
        Returns:
            A dictionary mapping field names (or aliases when requested) to their serialized values.
        """
        return self.model_dump(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_env=exclude_env,
        )

    def _save_dict(self, **dict_kwargs: Any) -> DictStrAny:
        """
        Produce the minimal dictionary of settings to persist to disk.
        
        By default this excludes fields equal to their model defaults, excludes values
        sourced from environment variables, and removes empty dictionaries. Additional
        options are forwarded to `model_dump` (e.g., `exclude_defaults` and
        `exclude_env`) and can be used to override the defaults.
        
        Parameters:
            dict_kwargs: Keyword arguments forwarded to `model_dump`.
        
        Returns:
            A dict representing the settings to write to disk with defaults,
            environment-provided values, and empty dicts removed.
        """
        dict_kwargs.setdefault('exclude_defaults', True)
        dict_kwargs.setdefault('exclude_env', True)
        data = self.model_dump(**dict_kwargs)
        _remove_empty_dicts(data)
        return data

    def save(self, path: str | Path | None = None, **dict_kwargs):
        """Save current settings to path.

        By default, this will exclude settings values that match the default
        value, and will exclude values that were provided by environment
        variables.  (see `_save_dict` method.)
        """
        path = path or self.config_path
        if not path:
            raise ValueError(
                trans._(
                    'No path provided in config or save argument.',
                    deferred=True,
                )
            )

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(exist_ok=True, parents=True)
        self._dump(str(path), self._save_dict(**dict_kwargs))

    def _dump(self, path: str, data: Dict) -> None:
        """
        Write settings data to a file using a serializer selected by the file extension.
        
        Parameters:
            path (str): Filesystem path to write. The serializer is chosen from the path suffix.
            data (Dict): Mapping of settings to serialize and write.
        
        Raises:
            NotImplementedError: If the path extension is not `.yaml`, `.yml`, or `.json`.
        """
        if str(path).endswith(('.yaml', '.yml')):
            _data = self._yaml_dump(data)
        elif str(path).endswith('.json'):
            import json

            _data = json.dumps(data, default=str)
        else:
            raise NotImplementedError(
                trans._(
                    'Can only currently dump to `.json` or `.yaml`, not {path!r}',
                    deferred=True,
                    path=path,
                )
            )
        with open(path, 'w') as target:
            target.write(_data)

    def env_settings(self) -> Dict[str, Any]:
        """
        Retrieve the cached environment-provided settings.
        
        Returns:
            A mapping of setting keys (flat or nested paths) to the raw values that were read from environment variables and applied during initialization.
        """
        return self._env_settings_cache

    def _remove_env_settings(self, data):
        """
        Remove entries from `data` that were supplied via environment variables.
        
        Modifies `data` in place by deleting or restoring keys that match the cached
        environment-provided settings for this instance; when a removed key has a
        saved default in the configuration file, that default is restored instead of
        being left absent.
        
        Parameters:
            data (dict): Mutable mapping representing settings to persist; updated in place.
        """
        env_data = self.env_settings()
        if env_data:
            _restore_config_data(
                data, env_data, getattr(self, '_config_file_settings', {})
            )


# Utility functions


def config_file_settings_source(
    settings_cls: type,
    config_path: Path | str | None,
) -> dict[str, Any]:
    """
    Read and validate configuration data from the given file path(s) for initializing an EventedConfigFileSettings subclass.
    
    This function loads YAML (.yaml, .yml) or JSON (.json) files at the provided path, merges their mappings, and validates the merged data against the provided settings class by instantiating a temporary settings_cls(config_path=None, **data). If validation errors occur, invalid keys are removed and a backup of the original file is attempted; if the settings class enables `strict_config_check` in its `model_config`, the ValidationError is re-raised instead of removing keys. Nonexistent or unsupported paths return an empty dict.
    
    Parameters:
        settings_cls (type): The settings class to validate the loaded data against.
        config_path (Path | str | None): Path to the config file to read; if None or the file does not exist, no data is loaded.
    
    Returns:
        dict: A mapping of validated configuration values suitable for passing into the settings class (may be empty).
    
    Raises:
        ValidationError: Re-raised when validation fails and `settings_cls.model_config['strict_config_check']` is truthy.
    """
    if not config_path:
        return {}

    sources: list[str] = []
    if config_path:
        sources.append(str(config_path))

    if not sources:
        return {}

    data: dict = {}
    for path in sources:
        if not path:
            continue
        path_ = Path(path).expanduser().resolve()

        # if the requested config path does not exist, move on to the next
        if not path_.is_file():
            continue

        # get loader for yaml/json
        if str(path).endswith(('.yaml', '.yml')):
            load = __import__('yaml').safe_load
        elif str(path).endswith('.json'):
            load = __import__('json').load
        else:
            warn(
                trans._(
                    'Unrecognized file extension for config_path: {path}',
                    path=path,
                )
            )
            continue

        try:
            # try to parse the config file into a dict
            new_data = load(path_.read_text()) or {}
        except Exception as err:  # noqa: BLE001
            _logger.warning(
                trans._(
                    'The content of the napari settings file could not be read\n\nThe default settings will be used and the content of the file will be replaced the next time settings are changed.\n\nError:\n{err}',
                    deferred=True,
                    err=err,
                )
            )
            continue
        assert isinstance(new_data, dict), path_.read_text()
        deep_update(data, new_data, copy=False)

    try:
        # validate the data by creating a temporary instance
        settings_cls(config_path=None, **data)
    except ValidationError as err:
        # Check if strict mode is enabled - if so, re-raise the error
        if hasattr(settings_cls, 'model_config'):
            strict_check = settings_cls.model_config.get(
                'strict_config_check', False
            )
            if strict_check:
                raise

        # if errors occur, we still want to boot, so we just remove bad keys
        errors = err.errors()
        msg = trans._(
            'Validation errors in config file(s).\nThe following fields have been reset to the default value:\n\n{errors}\n',
            deferred=True,
            errors=display_errors(errors),
        )
        with contextlib.suppress(Exception):
            # we're about to nuke some settings, so just in case... try backup
            backup_path = path_.parent / f'{path_.stem}.BAK{path_.suffix}'
            backup_path.write_text(path_.read_text())

        _logger.warning(msg)
        try:
            _remove_bad_keys(data, [e.get('loc', ()) for e in errors])
        except KeyError:
            _logger.warning(
                trans._(
                    'Failed to remove validation errors from config file. Using defaults.'
                )
            )
            data = {}

    return data


def _remove_bad_keys(data: dict, keys: list[tuple[int | str, ...]]):
    """
    Remove specified nested keys from a dictionary in place.
    
    Parameters:
        data (dict): Mapping to modify; entries will be deleted directly.
        keys (list[tuple[int | str, ...]]): List of key paths to remove. Each key path is a tuple of path components;
            traversal descends dictionaries following the components. An empty tuple is ignored.
            If an integer component is encountered during traversal, it is treated as an index indicator and stops
            further descent so the final deletion happens at the last resolved mapping level.
    """
    for key in keys:
        if not key:
            continue
        d = data
        while True:
            base, *key = key  # type: ignore[assignment]
            if not key:
                break
            # since no pydantic fields will be integers, integers usually
            # mean we're indexing into a typed list. So remove the base key
            if isinstance(key[0], int):
                break
            d = d[base]
        del d[base]


def _restore_config_data(dct: dict, delete: dict, defaults: dict) -> dict:
    """
    Restore or remove keys in a configuration mapping using a "delete" specification and fallback defaults.
    
    Parameters:
        dct (dict): Target dictionary representing current configuration; will be modified in place.
        delete (dict): Mapping describing keys to remove or restore. For keys with dict values, the function applies the same logic recursively to nested mappings.
        defaults (dict): Mapping of default values used to restore keys when available.
    
    Returns:
        dict: The modified `dct` after restoration and deletions.
    """
    for k, v in delete.items():
        # recurse
        if isinstance(v, dict):
            dflt = defaults.get(k, {})
            if not isinstance(dflt, dict):
                dflt = {}
            # Only recurse if the key exists in dct
            if k in dct:
                _restore_config_data(dct[k], v, dflt)
            elif dflt:
                # Key was excluded (e.g., by exclude_defaults), restore from defaults
                dct[k] = dflt
        # restore from defaults if present, or just delete the key
        elif k in defaults:
            dct[k] = defaults[k]
        elif k in dct:
            del dct[k]

    return dct


def _remove_empty_dicts(dct: dict, recurse=True) -> dict:
    """Remove all (nested) keys with empty dict values from `dct`"""
    for k, v in list(dct.items()):
        if isinstance(v, Mapping) and recurse:
            _remove_empty_dicts(dct[k])
        if v == {}:
            del dct[k]
    return dct