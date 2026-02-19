# Compatibility Matrix

## Runtime

| Component | Supported |
|-----------|-----------|
| Python | 3.11, 3.12, 3.13 |
| Package Manager | `pip` (PEP 517 editable install) |
| OS (CI) | Ubuntu latest |

## Core Dependencies

| Dependency | Policy |
|------------|--------|
| `semantic-kernel` | pinned as declared in `pyproject.toml` |
| `pydantic` | v2+ |
| `opentelemetry-*` | >=1.20 |

## Optional Integrations

| Integration | Notes |
|-------------|-------|
| Redis session backend | install with `.[redis]` |
| FAISS retrieval backend | install with `.[rag-faiss]` |
| HNSW retrieval backend | install with `.[rag-hnsw]` |

## CI Validation

- Unit tests run on Python 3.11, 3.12, and 3.13.
- Lint/type/security checks run on Python 3.12.
- Example smoke tests run on Python 3.12.
- Benchmark SLO gate runs on Python 3.12.
