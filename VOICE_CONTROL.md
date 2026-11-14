# Voice-controlled continuous inspection

Enable with `main.py --continuous --voice`. Real motion still requires explicit
`--confirm-motion`, idle teleop/arm, empty-gripper checks and all existing motion
limits. Initially stopped; fake audio never opens a microphone or loads Whisper.

## Current utterances

- **robot start** (or **start robot**): discard pending work and inspect afresh.
- **robot stop** (or **stop robot**): finish the current pick/delivery, then remain
  stopped. Only a fresh start resumes; neither a new policy nor scene change can
  override this explicit stop.
- **new policy <criteria> over**: activate/log a new policy version and reset a
  running loop automatically. No separate start confirmation is needed. Discard
  old targets/verdicts, return home and rescan under the new policy.
- **robot hello** / **hello robot**: taught left/right wave twice.
- **dance**: taught left/up/right/down once. See `GESTURES.md`.

Stop and gestures still require a fresh start afterward. Policies received while
initially stopped or explicitly stopped are updated but do not start motion. A
policy received during a gesture cancels remaining waypoints at the next waypoint
boundary; picking stays stopped until a fresh start. Faults and capacity cannot be
cleared by any voice command. Buffered starts cannot cancel an acknowledged stop
or resume picking after a gesture.

## Policy reset timing

Commands are consumed before/after each VLM request. A currently in-flight request
is allowed to finish; its result is discarded on reset, and no remaining old-policy
requests are issued. The GPU service is not restarted, and requests never overlap.
A reset cannot interrupt a held-object operation: that pick/delivery completes first.
Speech recognition and an in-flight request/move can delay acknowledgment. **Voice
stop is not an emergency stop**; use machine Stop / physical E-stop as needed.

The policy is an immutable per-inspection snapshot, inserted into the request's
`question` for `mainVLM`. A custom policy replaces default defect criteria and
**bypasses both the default color-only/blue-tape rejection rule and Anomalib
fast-pass**: every candidate is decided by Qwen under that policy. Invalid/uncertain
model output remains HOLD. Without a custom policy, configured default routing is
unchanged. Policy text cannot issue robot commands or change the JSON contract.

## Verify the exact policy sent to Qwen

- `[VOICE]` / `logs/voice/<session>.jsonl`: `policy_activated`, text, version,
  `automatic_restart` and paused state.
- `logs/wrist_inspection/inspect_*/vlm_results.json`: `inspection_policy` and
  **`prompt`**, the effective policy prompt used to construct the Qwen question.
  Per-object records include policy/version and whether Qwen was called.
- An aborted batch writes `inspection_reset.json` with `status: discarded`; its
  partial decisions must never reach picking. A new inspection directory identifies
  the fresh rescan.

Policies persist across batches **only within the same process**. Restarting the
script restores the configured default policy and stopped state. No organizer
Viam configuration or prompt is persisted/modified.

Speech settings are under `runtime_events.producers.audio`. The producer uses a
bounded queue, sequence/freshness validation, and its own connection/thread.
Producer failures and queue overflow fault the next controller checkpoint. There
is no speaker authentication or guarantee against ASR errors; verify the logged
rule. Diagnostics are private/gitignored. Live behavior requires supervised
validation; fake tests are not proof of microphone/NPU or physical-path safety.
