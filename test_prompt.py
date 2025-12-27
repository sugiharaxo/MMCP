#!/usr/bin/env python3
"""
Test script to check system prompt generation without full dependencies.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


# Mock the required modules to avoid import errors
class MockTool:
    def __init__(self, name, description, input_schema=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockSchema:
    def __init__(self, fields):
        self.__fields__ = fields

    def __fields__(self):
        return self.__fields__


class MockField:
    def __init__(self, name, required, description):
        self.name = name
        self.required = required
        self.description = description

    def is_required(self):
        return self.required


# Create mock TMDb input schema
tmdb_fields = {
    "title": MockField("title", True, "The movie or TV show title"),
    "year": MockField("year", False, "Release year (optional but recommended)"),
    "type": MockField("type", False, "Type of media: 'movie' or 'tv'"),
}

tmdb_schema = MockSchema(tmdb_fields)

# Mock tools dict
mock_tools = {
    "tmdb_lookup_metadata": MockTool(
        name="tmdb_lookup_metadata",
        description="Finds movie/TV show metadata (ID, year, overview, poster) using The Movie Database (TMDb) API.",
        input_schema=tmdb_schema,
    )
}


# Mock loader
class MockLoader:
    def __init__(self):
        self.tools = mock_tools


# Test the system prompt generation
def test_system_prompt():
    loader = MockLoader()

    # Copy the logic from _get_system_prompt
    if not loader:
        tool_desc = "No tools are currently available."
    else:
        tool_descriptions = []
        for name, tool in loader.tools.items():
            desc = f"- {name}: {tool.description}"

            # Add input schema information to help LLM understand required arguments
            if hasattr(tool, "input_schema"):
                schema = tool.input_schema
                if hasattr(schema, "__fields__"):
                    # Pydantic model - extract field descriptions
                    fields = []
                    for field_name, field_info in schema.__fields__.items():
                        required = "required" if field_info.is_required() else "optional"
                        desc_text = field_info.description or field_name
                        fields.append(f"    - {field_name} ({required}): {desc_text}")
                    if fields:
                        desc += "\n  Arguments:\n" + "\n".join(fields)

            tool_descriptions.append(desc)

        tool_desc = "\n".join(tool_descriptions)

    system_prompt = f"""You are MMCP (Modular Media Control Plane), an intelligent media assistant.
You help users manage their media library, search for metadata, and handle downloads.

Available Tools:
{tool_desc}

Guidelines:
1. Use tools when you need specific information or actions.
2. Always provide ALL required arguments when calling a tool.
3. If a tool fails due to missing arguments, check the tool description above and try again with the correct arguments.
4. If a tool fails, explain why to the user.
5. Be concise and helpful.
6. Always provide mode='final' with an answer when you have enough information to answer the user.
"""

    print("=== GENERATED SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== REPR ===")
    print(repr(system_prompt))


if __name__ == "__main__":
    test_system_prompt()
