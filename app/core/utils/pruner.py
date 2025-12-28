"""Context Pruner - Safety Fuse for context truncation."""

import json
from typing import Any

from app.core.config import Settings
from app.core.logger import logger


class ContextPruner:
    """Context truncation utility with structural pruning."""

    @staticmethod
    def _prune_dict(
        data: Any, max_chars: int, current_size: int = 0, settings: Settings | None = None
    ) -> tuple[Any, int]:
        """
        Recursively prune dictionary/list to fit within character limit.

        Uses structural pruning instead of fragile JSON string slicing to prevent
        JSONDecodeError on complex nested structures.

        Args:
            data: The data structure to prune (dict, list, or primitive).
            max_chars: Maximum character count allowed.
            current_size: Current character count (for tracking).
            settings: Settings object with truncation configuration.

        Returns:
            Tuple of (pruned_data, estimated_size).
        """
        if settings is None:
            from app.core.config import settings as global_settings

            settings = global_settings

        # Base case: primitive types
        if isinstance(data, (str, int, float, bool, type(None))):
            str_repr = str(data)
            max_string_len = settings.context_max_string_length
            if len(str_repr) > max_string_len:
                return str_repr[:max_string_len] + "...", current_size + max_string_len + 3
            return data, current_size + len(str_repr)

        # List: limit to configured number of items
        if isinstance(data, list):
            pruned_list = []
            size = current_size + 2  # Account for brackets
            max_list_items = settings.context_max_list_items
            truncated_marker = "... (truncated)"
            truncated_size = len(truncated_marker)
            for idx, item in enumerate(data):
                if idx >= max_list_items:
                    pruned_list.append(truncated_marker)
                    size += truncated_size
                    break
                # Process item to get its size estimate
                pruned_item, new_size = ContextPruner._prune_dict(item, max_chars, size, settings)
                # Only add item if it doesn't exceed the limit
                if new_size > max_chars:
                    # Size would exceed limit, add truncation marker instead
                    pruned_list.append(truncated_marker)
                    size += truncated_size
                    break
                # Item fits, add it and update size
                pruned_list.append(pruned_item)
                size = new_size
            return pruned_list, size

        # Dict: recursively prune values
        if isinstance(data, dict):
            pruned_dict = {}
            size = current_size + 2  # Account for braces
            for key, value in data.items():
                key_str = str(key)
                size += len(key_str) + 3  # Key + quotes + colon
                if size > max_chars:
                    pruned_dict["... (truncated)"] = True
                    break
                pruned_value, size = ContextPruner._prune_dict(value, max_chars, size, settings)
                pruned_dict[key] = pruned_value
                if size > max_chars:
                    break
            return pruned_dict, size

        # Fallback: convert to string
        str_repr = str(data)
        max_string_len = settings.context_max_string_length
        if len(str_repr) > max_string_len:
            truncated_suffix = "..."
            return str_repr[
                :max_string_len
            ] + truncated_suffix, current_size + max_string_len + len(truncated_suffix)
        return data, current_size + len(str_repr)

    @staticmethod
    def truncate_provider_data(
        data: dict[str, Any], provider_key: str, settings: Settings | None = None
    ) -> dict[str, Any]:
        """
        Truncate provider data to prevent context bloat using the Safety Fuse approach.

        High-ceiling check: Allow bursts up to 10k chars, but apply structural pruning
        if exceeded. This prevents JSON corruption while allowing legitimate large responses.

        Args:
            data: The provider data dictionary.
            provider_key: The provider key for logging.
            settings: Settings object with truncation configuration.

        Returns:
            Truncated data dictionary (pruned recursively if over limit).
        """
        if settings is None:
            from app.core.config import settings as global_settings

            settings = global_settings

        max_chars = settings.context_max_chars_per_provider

        # Quick size check first
        json_str = json.dumps(data, default=str)
        if len(json_str) <= max_chars:
            return data

        # Apply structural pruning to bring it down to safe size
        pruned_data, _ = ContextPruner._prune_dict(data, max_chars, settings=settings)
        pruned_json = json.dumps(pruned_data, default=str)

        # Actionable warning that helps developers/users fix the issue
        logger.warning(
            f"Context for plugin '{provider_key}' was truncated from {len(json_str)} to "
            f"{len(pruned_json)} chars. "
            f"Increase MMCP_CONTEXT_MAX_CHARS_PER_PROVIDER or ask the plugin creator to reduce noise."
        )
        return pruned_data
