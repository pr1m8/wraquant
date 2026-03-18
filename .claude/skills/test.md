---
name: test
description: Run tests for wraquant modules
---

Run tests for the specified module or full suite.

Usage: /test [module]
Examples: /test vol, /test regimes, /test (full suite)

Steps:
1. If module specified: run `pdm run pytest tests/{module}/ -x -q --tb=short`
2. If no module: run `pdm run pytest tests/ --tb=no -q`
3. Report: passed count, failed count, skip count
4. If failures: show failure details and suggest fixes
