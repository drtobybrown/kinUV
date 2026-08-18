---
id: DEC-OPS-AUTH
status: accepted
generation: 2
date: 2026-08-18
owner: ops
scope: 066-11 naming + cert at launch; campaign cron deferred
---
# CANFAR authentication and session naming

**Question:** How to authenticate 066-11, and how to name Skaha sessions?

## 066-11 (in scope)

One headless job, not a 452-day campaign.

1. Immediately before `canfar create`, run `cadc-get-cert` (CADC default lifetime is now ~30 days; `--days-valid 10` is also fine for a single job).
2. Confirm expiry: `openssl x509 -enddate -noout -in ~/.ssl/cadcproxy.pem`
3. `canfar create` (not `launch`). The session inherits the cert at creation; it does not need renewal while running.

**Session name** (Skaha ≤ 63 chars). Do **not** slice `KILOGAS066[:8]` → `KILOGAS0` (collides with 007):

`kinuv-KGAS066-{git_sha[:6]}-{map|nuts}`

Example: `kinuv-KGAS066-a1b2c3-map`

## Survey campaign (deferred)

Unattended `cadc-get-cert` in cron still needs a password or OIDC token. That is **not solved**. Do not implement a renewal cron for 066. When the 452-galaxy dispatcher exists, reopen this id: dispatcher checks expiry before each batch, pauses new submits if renewal fails, does not kill running sessions.
