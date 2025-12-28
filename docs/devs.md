# Developer Guide

This section is for contributors and anyone extending MMCP.
If you just want to run MMCP, get the fluff outta here

If you want to contribute, keep in mind our core philosophies:

- Lightweight
- Cross-Platform
- Modular

If you want to build with an AI, inject [these](llm-instructions.md) instructions

## Roadmap

- [ ] Make it actually useful
- [ ] Braindead easy installation, clean uninstall (no leftover dogshit)
- [ ] More tools
- [ ] UI that isn't slopgarbage
- [ ] Agent loop robustness + context engine
- [ ] Dynamic settings / frontend config
- [ ] Decoupled Plugin Memory
- [ ] Global Agent Memory
- [ ] Dual context API, on every user message "pull" (already implemented), and an event bus to allow plugins to receive messages: "push"
- [ ] Config API, when we have many plugins and want to register settings and whatnot. will also allow to set settings as sensitive which will filter their values in the system.
