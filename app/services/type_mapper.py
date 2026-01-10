"""TypeMapper - Handles conversion from Pydantic schemas to BAML TypeBuilder."""

from __future__ import annotations

from enum import Enum
from types import UnionType
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

if TYPE_CHECKING:
    from app.services.prompt import ToolInfo
    from baml_client.type_builder import TypeBuilder


class TypeMapper:
    """Handles conversion from Pydantic schemas to BAML TypeBuilder.

    Prevents class and enum name collisions by namespacing with tool names.
    Tracks added types locally to avoid inappropriate intimacy with BAML internals.
    """

    def __init__(self):
        # Cache TypeBuilder instances by tool configuration
        # Key: tuple of (tool_name, schema_name, classification) tuples
        # Note: This caching assumes tools are static and never change after startup.
        # Hot reloading or dynamic tool registration will cause stale cache issues.
        self._tb_cache: dict[tuple[tuple[str, str, str], ...], TypeBuilder] = {}

    def build_type_builder(self, tool_infos: list[ToolInfo]) -> TypeBuilder:
        # Create cache key from tool configuration
        cache_key = tuple(
            (tool_info.name, tool_info.schema.__name__, tool_info.classification)
            for tool_info in sorted(tool_infos, key=lambda ti: ti.name)
        )

        # Return cached TypeBuilder if available
        if cache_key in self._tb_cache:
            return self._tb_cache[cache_key]

        # Build new TypeBuilder (lazy load heavy import)
        from baml_client.type_builder import TypeBuilder

        tb = TypeBuilder()

        for tool_info in tool_infos:
            # Each tool gets its own isolated type tracking to enforce strict namespacing
            # This prevents shared types from leaking across tool boundaries
            added_types: dict[type, Any] = {}
            # Let the recursive mapper handle the model creation
            # This ensures the tool's main schema is namespaced and cached
            self._map_pydantic_to_baml(tb, tool_info.schema, added_types, tool_info.name)

            # Add rationale to external tools
            if tool_info.classification == "EXTERNAL":
                cls_builder = added_types[tool_info.schema]
                cls_builder.add_property("rationale", tb.string()).description(
                    "User-facing explanation: Why is this action necessary?"
                )

        # Cache for reuse
        self._tb_cache[cache_key] = tb
        return tb

    def _map_pydantic_to_baml(
        self, tb: TypeBuilder, field_type: type, added_types: dict[type, Any], namespace: str
    ) -> Any:
        """Map Pydantic types to BAML types, tracking added types to prevent collisions."""
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Literal handling
        if origin is Literal:
            if all(isinstance(v, str) for v in args):
                return tb.string()
            return tb.union(
                [self._map_pydantic_to_baml(tb, type(v), added_types, namespace) for v in args]
            )

        # Union / Optional
        if origin in (Union, UnionType):
            non_none = [t for t in args if t is not type(None)]
            if len(args) == 2 and len(non_none) == 1:
                return self._map_pydantic_to_baml(
                    tb, non_none[0], added_types, namespace
                ).optional()
            return tb.union(
                [self._map_pydantic_to_baml(tb, t, added_types, namespace) for t in args]
            )

        # List
        if origin is list:
            inner_type = args[0] if args else str
            return self._map_pydantic_to_baml(tb, inner_type, added_types, namespace).list()

        # Dict
        if origin is dict:
            key_type, value_type = args if len(args) >= 2 else (str, str)
            if key_type is not str and not (
                isinstance(key_type, type) and issubclass(key_type, Enum)
            ):
                raise ValueError(f"BAML maps only allow 'str' or Enum as keys, got {key_type}")
            return tb.map(
                self._map_pydantic_to_baml(tb, key_type, added_types, namespace),
                self._map_pydantic_to_baml(tb, value_type, added_types, namespace),
            )

        # Enum - identity-based tracking prevents duplicate add_enum calls
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            if field_type in added_types:
                return added_types[field_type].type()

            enum_builder = tb.add_enum(f"{namespace}_{field_type.__name__}")
            for member in field_type:
                enum_builder.add_value(member.name)

            added_types[field_type] = enum_builder
            return enum_builder.type()

        # Nested Pydantic Model - identity-based tracking prevents name collisions
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            if field_type in added_types:
                # Reuse existing class (identity-based lookup)
                return added_types[field_type].type()

            # Global uniqueness within the TypeBuilder instance using namespace
            cls = tb.add_class(f"{namespace}_{field_type.__name__}")
            added_types[field_type] = cls

            # Recursively map nested fields with same namespace
            for name, info in field_type.model_fields.items():
                baml_type = self._map_pydantic_to_baml(tb, info.annotation, added_types, namespace)
                prop = cls.add_property(name, baml_type)
                if info.description:
                    prop.description(info.description)

            return cls.type()

        # Primitives
        primitive_map = {str: tb.string(), int: tb.int(), float: tb.float(), bool: tb.bool()}
        if field_type in primitive_map:
            return primitive_map[field_type]

        raise ValueError(f"Unsupported type for BAML mapping: {field_type}")
