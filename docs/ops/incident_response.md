# Incident Response

## Severity Levels

- `SEV-1`: user-visible outage or data loss risk.
- `SEV-2`: major degradation with partial functionality.
- `SEV-3`: minor degradation or non-critical defect.

## First 15 Minutes

1. Confirm impact and assign severity.
2. Create incident channel/ticket with timestamp and owner.
3. Freeze deploys to affected environment.
4. Capture initial telemetry:
- error rate
- latency p95/p99
- provider fallback rate
- rate-limit/guardrail deny spikes

## Containment

1. Route traffic to known-stable provider policy (e.g. fallback-first).
2. Reduce concurrency or traffic shaping if saturation is detected.
3. Disable newly introduced features behind flags where possible.

## Recovery

1. Apply fix or rollback using [`rollback_playbook.md`](rollback_playbook.md).
2. Verify SLO recovery for at least two consecutive windows.
3. Re-enable deploys after incident commander sign-off.

## Postmortem Checklist

1. Timeline with exact UTC timestamps.
2. Root cause and contributing factors.
3. Detection and alerting gaps.
4. Corrective actions with owners and due dates.
