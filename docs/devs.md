# Developer Guide

This section is for DEVS
If you just want to run MMCP, get the fluff outta here

If you want to contribute, keep in mind our core philosophies:

- Lightweight
- Cross-Platform
- Modular
- Ease of use

If you want to build with an AI, inject [these](llm-instructions.md) instructions

to install dev deps use uv `sync --extra dev`
then do `baml-cli generate` cause its in gitignore for som reason

tests r `uv run pytest`

[Creating a plugin](creating-a-plugin.md)
