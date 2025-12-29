"""
Settings Manager - The single source of truth for plugin configuration.

Merges three tiers: Defaults (Pydantic) < Database < Environment Variables.
Returns validated settings and locked field names.
"""

import os
import re
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import select

from app.core.database import PluginSetting, get_session
from app.core.logger import logger


class SettingsManager:
    """
    Manages plugin settings with 3-tier priority: ENV > Database > Defaults.

    Environment variables lock fields (immutable, shown as read-only in UI).
    Database stores user-configured values (mutable via API).
    Pydantic defaults as fallback.
    """

    @staticmethod
    def _slugify_plugin_name(name: str) -> str:
        """
        Convert plugin name to deterministic ENV prefix slug.

        Matches PluginLoader._slugify_plugin_name() for consistency.
        """
        slug = re.sub(r"[^a-zA-Z0-9]", "_", name)
        slug = slug.upper()
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug

    @staticmethod
    def _get_env_prefix(plugin_slug: str) -> str:
        """Get environment variable prefix for a plugin."""
        return f"MMCP_PLUGIN_{plugin_slug}_"

    async def get_plugin_settings(
        self, plugin_slug: str, model_cls: type[BaseModel]
    ) -> tuple[BaseModel | None, set[str], ValidationError | None]:
        """
        Get plugin settings with 3-tier merging: Defaults < Database < Environment.

        Args:
            plugin_slug: Plugin slug (e.g., "TMDB_LOOKUP_METADATA")
            model_cls: Pydantic model class for settings

        Returns:
            Tuple of (validated BaseModel instance or None, set of locked field names, ValidationError or None)
            If validation fails, returns (None, locked_fields, validation_error) with error details.
        """
        env_prefix = self._get_env_prefix(plugin_slug)
        env_prefix_lower = env_prefix.lower()

        # Step 1: Load defaults from model_cls
        defaults: dict[str, Any] = {}
        for field_name, field_info in model_cls.model_fields.items():
            if field_info.default is not ...:
                defaults[field_name] = field_info.default
            elif field_info.default_factory is not ...:
                defaults[field_name] = field_info.default_factory()

        # Step 2: Query database and overlay on defaults
        db_values: dict[str, Any] = {}
        try:
            async with get_session() as session:
                stmt = select(PluginSetting).where(PluginSetting.plugin_slug == plugin_slug)
                result = await session.execute(stmt)
                db_rows = result.scalars().all()

                for row in db_rows:
                    # Handle nested keys (e.g., "retry_policy__max_retries")
                    key_parts = row.key.split("__")
                    if len(key_parts) == 1:
                        db_values[row.key] = row.value
                    else:
                        # Nested field
                        nested_key = key_parts[0]
                        nested_field = key_parts[1]
                        if nested_key not in db_values:
                            db_values[nested_key] = {}
                        db_values[nested_key][nested_field] = row.value
        except Exception as e:
            logger.warning(f"Failed to load database settings for {plugin_slug}: {e}")

        # Step 3: Check environment variables and overlay (highest priority)
        env_values: dict[str, Any] = {}
        locked_fields: set[str] = set()

        for env_key, env_value in os.environ.items():
            env_key_lower = env_key.lower()
            if env_key_lower.startswith(env_prefix_lower):
                # Strip prefix to get field name
                field_name_with_nesting = env_key_lower[len(env_prefix_lower) :]

                # Handle nested configuration: double underscore (__) maps to nested model
                if "__" in field_name_with_nesting:
                    parts = field_name_with_nesting.split("__", 1)
                    nested_key = parts[0]
                    nested_field = parts[1]
                    if nested_key not in env_values:
                        env_values[nested_key] = {}
                    env_values[nested_key][nested_field] = env_value
                    # Track both parts as locked
                    locked_fields.add(nested_key)
                else:
                    env_values[field_name_with_nesting] = env_value
                    locked_fields.add(field_name_with_nesting)

        # Step 4: Merge: defaults < db < env
        merged_data = {**defaults, **db_values, **env_values}

        # Step 5: Validate with Pydantic
        try:
            validated = model_cls.model_validate(merged_data)
            return validated, locked_fields, None
        except ValidationError as e:
            # Log warning and return None if validation fails
            # This can happen if schema changed and DB has invalid data
            logger.warning(
                f"Settings validation failed for {plugin_slug}: {e.errors()}. "
                f"Using defaults/env only (ignoring invalid DB values)."
            )
            # Try again with only defaults + env (skip DB)
            try:
                merged_data_no_db = {**defaults, **env_values}
                validated = model_cls.model_validate(merged_data_no_db)
                return validated, locked_fields, None
            except ValidationError as final_e:
                # Even defaults + env failed, return None with the final validation error
                return None, locked_fields, final_e

    async def save_plugin_setting(self, plugin_slug: str, key: str, value: str) -> None:
        """
        Save a plugin setting to the database (upsert).

        Uses SQLAlchemy merge() for atomic upsert operations to handle concurrent updates safely.

        Args:
            plugin_slug: Plugin slug
            key: Setting key (can be nested with "__" delimiter)
            value: Setting value (as string)
        """
        async with get_session() as session:
            # Create the setting object
            setting = PluginSetting(plugin_slug=plugin_slug, key=key, value=value)

            # Use merge() for atomic upsert - handles concurrent updates safely
            # This is equivalent to INSERT OR REPLACE but with ORM benefits
            session.merge(setting)

            await session.commit()
            logger.debug(f"Saved setting {plugin_slug}.{key} = {value}")

    async def delete_plugin_setting(self, plugin_slug: str, key: str) -> None:
        """
        Delete a plugin setting from the database.

        Args:
            plugin_slug: Plugin slug
            key: Setting key to delete
        """
        async with get_session() as session:
            stmt = select(PluginSetting).where(
                PluginSetting.plugin_slug == plugin_slug, PluginSetting.key == key
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                await session.delete(existing)
                await session.commit()
                logger.debug(f"Deleted setting {plugin_slug}.{key}")
