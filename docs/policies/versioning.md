# Versioning And Migration Policy

SKTK follows semantic versioning for all public APIs exported through `sktk` and documented modules.

## Versioning Rules

- `MAJOR`: breaking API changes or behavior changes requiring code updates.
- `MINOR`: backward-compatible features and additive APIs.
- `PATCH`: backward-compatible fixes and internal improvements.

## Compatibility Guarantees

- Public interfaces in `src/sktk/__init__.py` are considered stable within a major version.
- Deprecated symbols remain available for at least one minor release before removal.
- Breaking changes require release notes with concrete before/after migration guidance.

## Migration Checklist For Breaking Changes

1. Add deprecation warnings in the previous minor release.
2. Document replacement APIs and example diffs.
3. Add tests covering both legacy and replacement paths during transition.
4. Remove deprecated paths only in the next major release.

## Runtime Support

- Supported Python versions are declared in `pyproject.toml` (`>=3.11`).
- CI must validate all supported Python versions before release.
