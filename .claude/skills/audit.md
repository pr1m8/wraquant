---
name: audit
description: Audit wraquant codebase for consistency issues
---

Run a comprehensive audit checking:

1. **Duplicate functions** — Search for same function names across modules
2. **Missing exports** — Compare defined public functions vs `__all__` in each module
3. **Parameter naming** — Check for `period` vs `window` vs `lookback` inconsistencies
4. **Docstring format** — Verify Google/Napoleon style across all public functions
5. **Import patterns** — Check for circular deps, missing `@requires_extra`
6. **Integration gaps** — Find modules that should reference each other but don't

Report findings as a structured list with file:line references and severity (CRITICAL/HIGH/MEDIUM/LOW).
