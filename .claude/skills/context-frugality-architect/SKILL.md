---
name: context-frugality-architect
description: Token conservation, Repomix codebase packing, ast-grep structural search, prompt caching prefix preservation, and subagent context insulation. Use when exploring large codebases or when context window exceeds 50,000 tokens.
allowed-tools: [Bash, Read, Grep]
---
# Context Frugality & Token Conservation

## Inviolable Rules
1. Never view full files over 150 lines without explicit line slices (start/end line ranges).
2. Search structurally using ast-grep or skeletal symbol queries rather than recursive text grep.
3. Generate token-budgeted repo XML maps using `npx -y repomix <target-dir> --style xml`.
4. Isolate broad repository reconnaissance in read-only subagents so the orchestrator context remains lean.
