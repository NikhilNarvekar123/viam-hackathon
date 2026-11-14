# Voice-controlled continuous inspection

Opt-in with `main.py --continuous --voice`. Default batch and continuous modes
remain unchanged when voice control is disabled. This integration is fake/mock
validated only; do not interpret it as physical validation or deployment.

## Commands

The process initially waits for audio readiness and a command, without automatic
home/pick movement. **start** begins picking; hello/dance request only their taught
gesture and leave picking stopped. Model initialization is allowed before start.

| Utterance | Controller behavior |
|---|---|
| `hello` | After any active delivery completes, replay the taught left/right wave twice, then home; remain stopped until a fresh start. |
| `start` | Confirm any pending policy; discard all pending targets/verdicts; start a fresh home/capture/inspection epoch. Also resets a currently running epoch. |
| `stop please` / `please stop` | Finish the active attempt and delivery/return, including the picker's normal return-to-scan; then stay stopped and listening. No extra home/open or next pick. |
| `new policy <criteria>` | Log the exact proposed rule and pause at the safe checkpoint. Nothing activates until a subsequent start. |
| `dance` | After any active delivery completes, replay taught left/up/right/down once, then home; remain stopped until a fresh start. |

Wave and dance use the teammate's saved switch poses in `config/gestures.yaml`;
see `GESTURES.md`. Real execution still requires motion authorization and verified
empty-gripper/idle checks. Saved endpoints do not validate transition clearance.
There is no text-to-speech acknowledgment: read the `[VOICE]` console records.

Speak one complete command at a time. The stop phrase requires **stop please** or
**please stop** (optionally prefixed by robot); bare stop is ignored. Optional
`robot` / `please` prefixes are accepted for other commands. Negation, conversational keyword matches, multiple-command utterances,
empty/oversized policies, and fuzzy verb forms are not actionable. For example,
`do not start`, `stop then start`, and `I started yesterday` are ignored. This
parser cannot detect every speech-recognition error; verify the logged transcript.

After a stop, gesture, or policy acknowledgment, say **start again** (the literal
utterance `start`, not the phrase "start again"). A start buffered before that
acknowledgment is deliberately ignored. For gestures, wait until completion; a start
spoken during a gesture cannot resume picking afterward. Sequence numbers suppress replay; freshness
uses the oldest acquisition included in the speech window, not delayed inference
completion. Non-stop commands expire after `command_max_age_s` (default 30 seconds).
A stop never expires merely because delivery/inference took a long time.

## Safety and lifecycle

- Voice control requires continuous mode and the existing explicit
  `--confirm-motion` authorization for real hardware. Voice cannot grant that
  authorization, clear capacity, clear faults, override clearance, or command poses.
- **Voice stop is NOT an emergency stop.** Whisper's rolling window, transcription,
  an in-flight inference batch, and a running SDK/picker operation can delay the
  checkpoint. Use machine Stop / physical E-stop for immediate stopping needs.
- During gestures, stop/new policy cancels remaining waypoints after the current
  waypoint completes; no automatic home/release occurs on interruption or failure.
  A new policy remains pending for start confirmation. See `GESTURES.md`.
- During delivery, commands only queue. On successful completion, the controller
  consumes them **before** additional home/open motion. If the delivery fails,
  failure/holding-state handling wins: no automatic recovery, release, or restart.
- While stopped, scene-change/reset events do not resume movement. Audio remains
  alive; only a fresh start resumes. Capacity or latched faults terminate the run
  rather than being cleared by start. Physical placement counters survive resets.
- Audio uses its own connection/event loop in the existing producer thread. Capture
  waits are bounded, cancellation closes the connection, and producer failure or
  queue overflow faults the controller's next checkpoint. A blocked native model
  compile/inference cannot be forcibly killed by Python; runtime shutdown reports
  a producer that exceeds its join timeout.
- Do not start a second controller, change organizer configuration, install packages,
  or run real motion simply to test voice. None of those actions is part of this change.

## Confirmed policies and Qwen

Policies are process-local, not persistent Viam component configuration. The
confirmed policy is sent to `mainVLM` in each request's `question`, using
`custom_policy_prompt` in `config/vlm_service.yaml`. It **replaces** the original
clean-bare-metal criteria rather than appending contradictory criteria. The model
receives the same isolated crop; observation/boolean-defect parsing and authoritative
ACCEPT/REJECT/HOLD routing remain unchanged. Uncertain/invalid output is HOLD.

A policy remains pending until start. Start increments its version, discards old
geometry/verdicts, and captures/reinspects. Each inspection gets a private settings
snapshot: a policy arriving mid-inference cannot change half the batch. Results
from that obsolete inspection cannot be picked after its next checkpoint.

**While a custom policy is active, Stage 1 fast-pass is disabled.** All candidates
reach Qwen, including scores below 0.60, because visually normal parts could violate
a new rule. Anomalib still runs on CPU. Qwen calls remain sequential; the GPU service
is not restarted. Before any custom policy, the normal 0.60 fast-pass remains.
Restarting the process restores the configured default policy and stopped state.

Private logs:
- `logs/voice/<session>.jsonl`: readiness, ignored commands, pending/active policy,
  version, starts/stops, and gesture requests/completions/interruption.
- Per-inspection VLM and batch reports: effective policy/version and verdicts.

## Configuration and testing

`--voice` enables `voice_control`, the runtime, and its audio producer in memory;
it does not edit configuration files. `voice_control.enabled: true` is also
supported, but only with continuous mode. Explicit audio settings live under
`runtime_events.producers.audio`: microphone resource, device, optional model path,
window/RMS/cooldown and I/O timeout. Model resolution retains the audio teammate's
base-then-tiny convention; verify the installed path before deployment. An unavailable
requested NPU faults rather than silently changing device placement.

Outside voice mode, an explicitly enabled audio producer remains diagnostic-only
(`NOTICE`). Typed `audio_ready` and `voice_command` events are enabled by voice mode;
only the controller consumes them. Generic notices and other producers cannot start
or stop the robot.

```bash
python -m pytest tests/test_voice_control.py tests/test_audio_commands.py \
  tests/test_continuous_voice.py tests/test_continuous_pick.py -q
python main.py --fake --n 1
```

The fake audio hook deliberately does not open a microphone or load Whisper.
`--continuous --voice --fake` therefore waits without real listening; tests inject
readiness and bounded command envelopes into `EventRuntime`. This is intentional,
not proof of a working microphone/NPU. Before any authorized live trial, verify
model/device/microphone readiness without motion, review machine-local edits, and
keep an accessible physical stop.

## Merged placement compatibility

Continuous validation now permits expected disappearance for a successfully
placed GOOD **or REJECT** part. Placement counters must advance after successful
delivery; full-area put-backs and reject-return do not qualify. Home-image validation
still verifies the actual disappearance and rejects unrelated changes. This is
conservative scene validation, not object tracking or a personnel-safety system.
