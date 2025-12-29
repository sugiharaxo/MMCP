"""
Settings API routes for plugin configuration management.

Provides endpoints for listing, updating, and revealing plugin settings.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.auth import verify_admin
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader
from app.core.settings_manager import SettingsManager

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


class PluginSettingSchema(BaseModel):
    """Schema for a single plugin setting field."""

    name: str = Field(description="Field name")
    type: str = Field(description="Field type (e.g., 'string', 'integer')")
    description: str | None = Field(None, description="Field description")
    default: Any = Field(None, description="Default value if any")
    required: bool = Field(description="Whether field is required")
    locked: bool = Field(description="Whether field is locked by environment variable")


class PluginSettingsInfo(BaseModel):
    """Information about a plugin's settings schema."""

    slug: str = Field(description="Plugin slug")
    name: str = Field(description="Plugin name")
    version: str = Field(description="Plugin version")
    description: str = Field(description="Plugin description")
    is_available: bool = Field(description="Whether plugin is currently available")
    settings_schema: list[PluginSettingSchema] = Field(description="Settings schema fields")
    current_values: dict[str, Any] = Field(description="Current setting values (secrets masked)")


class UpdateSettingsRequest(BaseModel):
    """Request body for updating plugin settings."""

    settings: dict[str, Any] = Field(description="Settings to update (key-value pairs)")


class RevealSecretRequest(BaseModel):
    """Request body for revealing a secret field."""

    field: str = Field(description="Field name to reveal")


class RevealSecretResponse(BaseModel):
    """Response for secret reveal."""

    field: str = Field(description="Field name")
    value: str = Field(description="Revealed secret value")


def get_settings_manager() -> SettingsManager:
    """Dependency to get SettingsManager instance."""
    return SettingsManager()


def get_plugin_loader(request: Request) -> PluginLoader:
    """Dependency to get PluginLoader instance."""
    return request.app.state.loader


@router.get("/plugins", response_model=dict[str, PluginSettingsInfo])
async def list_plugins(
    loader: PluginLoader = Depends(get_plugin_loader),
    settings_manager: SettingsManager = Depends(get_settings_manager),
    _: bool = Depends(verify_admin),
) -> dict[str, PluginSettingsInfo]:
    """
    List all plugins with their settings schemas and current values.

    Returns plugins from both active and standby states.
    Secrets are masked in the response.
    """
    plugins_info: dict[str, PluginSettingsInfo] = {}

    # Combine active and standby tools
    all_tools = {**loader.tools, **loader.standby_tools}

    for tool_name, tool in all_tools.items():
        plugin_slug = loader._slugify_plugin_name(tool_name)
        locked_fields = loader._locked_fields.get(tool_name, set())

        # Get settings schema if plugin has one
        settings_schema_fields: list[PluginSettingSchema] = []
        current_values: dict[str, Any] = {}

        if hasattr(tool, "settings_model") and tool.settings_model is not None:
            settings_model = tool.settings_model
            model_schema = settings_model.model_json_schema()

            # Get current settings (merged from defaults/db/env)
            current_settings, _, _ = await settings_manager.get_plugin_settings(
                plugin_slug, settings_model
            )

            # Build schema fields
            required = model_schema.get("required", [])

            for field_name, field_info in settings_model.model_fields.items():
                field_type = "string"  # Default
                if hasattr(field_info.annotation, "__name__"):
                    field_type = field_info.annotation.__name__.lower()

                # Check if it's a SecretStr
                is_secret = "SecretStr" in str(field_info.annotation)

                settings_schema_fields.append(
                    PluginSettingSchema(
                        name=field_name,
                        type=field_type,
                        description=field_info.description,
                        default=field_info.default if field_info.default is not ... else None,
                        required=field_name in required,
                        locked=field_name in locked_fields,
                    )
                )

                # Add current value (mask secrets)
                if current_settings:
                    value = getattr(current_settings, field_name, None)
                    if is_secret and value is not None:
                        # Mask secret values
                        if hasattr(value, "get_secret_value"):
                            current_values[field_name] = "**********"
                        else:
                            current_values[field_name] = "**********"
                    else:
                        current_values[field_name] = value

        plugins_info[plugin_slug.lower()] = PluginSettingsInfo(
            slug=plugin_slug.lower(),
            name=tool_name,
            version=tool.version,
            description=tool.description,
            is_available=tool_name in loader.tools,
            settings_schema=settings_schema_fields,
            current_values=current_values,
        )

    return plugins_info


@router.patch("/plugins/{slug}", status_code=status.HTTP_200_OK)
async def update_plugin_settings(
    slug: str,
    request: UpdateSettingsRequest,
    loader: PluginLoader = Depends(get_plugin_loader),
    settings_manager: SettingsManager = Depends(get_settings_manager),
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """
    Update plugin settings.

    Validates settings against plugin's schema and saves to database.
    Returns 409 Conflict if trying to update a locked field (set via environment variable).
    """
    # Find plugin by slug
    plugin_slug_upper = slug.upper()
    tool = None
    tool_name = None

    # Search in both active and standby tools
    for name, t in {**loader.tools, **loader.standby_tools}.items():
        if loader._slugify_plugin_name(name) == plugin_slug_upper:
            tool = t
            tool_name = name
            break

    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin with slug '{slug}' not found",
        )

    # Check if plugin has settings model
    if not hasattr(tool, "settings_model") or tool.settings_model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plugin '{tool_name}' does not have configurable settings",
        )

    locked_fields = loader._locked_fields.get(tool_name, set())

    # Check for locked fields
    for field_name in request.settings:
        if field_name in locked_fields:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Field '{field_name}' is locked by environment variable and cannot be updated",
            )

    # Save each setting to database
    for key, value in request.settings.items():
        # Convert value to string for storage
        value_str = str(value)

        # Prevent submission of masked secret values (backend safeguard)
        if value_str == "**********":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Field '{key}' appears to contain a masked secret value. "
                f"Please provide the actual value or use the reveal endpoint to see the current value.",
            )

        await settings_manager.save_plugin_setting(plugin_slug_upper, key, value_str)

    logger.info(f"Updated settings for plugin '{tool_name}': {list(request.settings.keys())}")

    # Reload plugin settings (this will trigger a reload on next request)
    # For now, just return success - full reload would require restart or hot reload
    return {
        "success": True,
        "message": f"Settings updated for plugin '{tool_name}'. Restart required for changes to take effect.",
        "updated_fields": list(request.settings.keys()),
    }


@router.post("/plugins/{slug}/reveal", response_model=RevealSecretResponse)
async def reveal_secret(
    slug: str,
    request: RevealSecretRequest,
    loader: PluginLoader = Depends(get_plugin_loader),
    settings_manager: SettingsManager = Depends(get_settings_manager),
    _: bool = Depends(verify_admin),
) -> RevealSecretResponse:
    """
    Reveal a secret field value.

    Uses POST to avoid logging secrets in URL or access logs.
    """
    # Find plugin by slug
    plugin_slug_upper = slug.upper()
    tool = None
    tool_name = None

    # Search in both active and standby tools
    for name, t in {**loader.tools, **loader.standby_tools}.items():
        if loader._slugify_plugin_name(name) == plugin_slug_upper:
            tool = t
            tool_name = name
            break

    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin with slug '{slug}' not found",
        )

    # Check if plugin has settings model
    if not hasattr(tool, "settings_model") or tool.settings_model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plugin '{tool_name}' does not have configurable settings",
        )

    settings_model = tool.settings_model

    # Get current settings
    current_settings, _, _ = await settings_manager.get_plugin_settings(
        plugin_slug_upper, settings_model
    )

    if not current_settings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Settings not found for plugin '{tool_name}'",
        )

    # Get the field value
    if not hasattr(current_settings, request.field):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field '{request.field}' not found in plugin settings",
        )

    field_value = getattr(current_settings, request.field)

    # Handle SecretStr
    if hasattr(field_value, "get_secret_value"):
        revealed_value = field_value.get_secret_value()
    else:
        revealed_value = str(field_value)

    logger.info(f"Secret field '{request.field}' revealed for plugin '{tool_name}'")

    return RevealSecretResponse(field=request.field, value=revealed_value)
