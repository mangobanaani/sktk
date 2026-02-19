# Rollback Playbook

## Preconditions

- Current release tag and previous stable tag are known.
- Database or stateful migration impact has been assessed.
- Incident commander approves rollback.

## Application Rollback

1. Identify target stable release tag (`vX.Y.Z`).
2. Redeploy artifacts from that tag.
3. Verify health checks and readiness probes.
4. Monitor error rate and p95 latency for at least 10 minutes.

## Configuration Rollback

1. Revert provider routing policy to previous known-good config.
2. Revert guardrail policy changes if they introduced false positives.
3. Revert quota/rate-limit changes that caused denial spikes.

## Data/Migration Considerations

- Never run destructive down-migrations during active incident response unless explicitly validated.
- If schema changed incompatibly, prefer forward-fix over emergency down-migration.

## Verification Checklist

1. Error rate below alert threshold.
2. Provider fallback rate normalized.
3. No sustained queue backlog.
4. SLO gate benchmark trend is back to baseline.
