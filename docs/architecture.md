# FirstWatch implementation

The executable MVP implements the account lifecycle and the external-model integration boundary. Actual trained artifacts and notebook preprocessing are pending delivery. `DEMO_MODE=true` explicitly enables deterministic heuristic signals; `false` requires a verified external bundle. Missing external models return `not_ready` and never fall back to synthetic scores.

## Runtime and persistence

Next.js serves the command center and proxies REST/SSE to FastAPI. The API calls the three isolated runners over HTTP, persists each invocation and feature snapshot, applies configurable fusion, and updates accounts and alerts. PostgreSQL is the Compose database; SQLite is available for local unit tests. Each replay reset creates a new run, preserving prior audit records. Restart restores the latest run paused. The API runs **one worker** because its playback coordinator uses an in-process lock. Scaling needs a distributed event coordinator and database concurrency controls.

The browser consumes `/api/state` and named `state` events from `/api/scenarios/{id}/stream`. It never fabricates fallback scores. Reconnection obtains a complete state snapshot, so missed transient events do not leave stale account data. Source events retain event time, observed time and a stable sequence number. Scores use 0–1 at HTTP/storage boundaries and 0–100 only for display.

## Public interfaces

See `/docs` on port 8000 for generated OpenAPI schemas. The implemented controls match the README. Additional integration endpoints:

- `POST /api/applications`: `{application_id, account_id, alias, event_time, raw_payload, labels?}`.
- `POST /api/events`: `{event_id, account_id, event_time, type, amount, currency?, counterparty_id?, description?, raw_payload?, labels?}`.
- `POST /api/accounts/{id}/rescreen`: retry a pending admission after repairing its runner.
- `PATCH /api/alerts/{id}`: `{status: "escalated" | "dismissed", notes}`. Notes are required and a reviewed alert cannot silently be overwritten.

Identifiers are idempotency keys scoped to a run. Reusing an identifier with different content returns 409; out-of-order events return 409. Admission-rejected and pending accounts cannot transact. Event amount must be nonnegative and timestamps must include a timezone. One account has one currency; conversion is the responsibility of an explicit client adapter.

## Risk decisions

Admission and monitoring have separate thresholds. Changing policy reevaluates monitoring alerts and leaves past admission decisions intact. Crossing uses `>=`. One unresolved alert exists per account; a new case requires a new below-to-above crossing after the earlier case is dismissed. Escalation is a human review record, not a funds freeze. Lowering a score does not erase an open case.

Fusion uses `exp(-event_count/tau)` for admission weight, with quality-weighted available transaction/sequence scores. No available behavior signals means current behavior risk is unavailable, not zero. ECDF anomaly normalization produces a relative rank, not a calibrated fraud probability. Do not claim probability or expected monetary savings without external validation.

## Boundaries

Flexible JSONB accommodates raw source fields; each model bundle owns exact feature engineering. Label fields are isolated from inference. No real identity join exists between BAF and the multi-pattern transaction dataset: any cross-dataset account mapping must be labelled synthetic. Existing BAF EDA uses a random split, so month 7 is not automatically unseen. Candidate fixture provenance records this explicitly. `/api/models/status` distinguishes service health, bundle status and readiness for external scoring.

This is a local hackathon service. Docker binds public ports to loopback. Bank deployment still requires authentication/RBAC, transport security, tenant isolation, retention, model governance and load testing. Analysts' decisions are saved for later supervised review; no automatic online retraining is implemented or claimed.
