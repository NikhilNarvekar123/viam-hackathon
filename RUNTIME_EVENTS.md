# Optional background-event scaffold

This is a small extension to the supplied-ROI batch pipeline, not a replacement
controller. `main.py` still owns orchestration; the original picker still owns
its grasp/refinement/delivery logic. No speech model, camera change detector,
secondary scene-matching algorithm, new device placement, or safety-stop system
is implemented here. This runtime scaffold was deployed from source `9fd2589`
and fake-tested on the NUC; physical event-enabled behavior is not validated.
Listeners remain disabled. The separate CPU overview comparison/replay primitive
is offline-only, not a live producer; see `monitors/OVERVIEW_DETECTION.md`.

## Flow

```
audio hook --------\
                    > bounded thread-safe event queue
 overview hook ----/                  |
                                      v
home/capture -> Anomalib/VLM -> order remaining targets -> CHECKPOINT
                                      ^                    |
                                      |                    v
                            update remaining work <- pick + deliver/return
```

- Checkpoints occur before the first target and after every completed attempt,
  including HOLD/refusals and the final object.
- Ordering is recalculated from remaining stored heights; this is not new depth
  localization or another model pass. Each row is attempted/skipped once per batch.
- No event is consumed between grasp and delivery/return. The active target must
  stay put. A reset cannot interrupt a held-object operation or open the gripper.
- `reset_batch` and `scene_changed` request a fresh home/capture/inspection cycle.
  Pending boxes/verdicts are abandoned, not carried across captures. Multiple
  queued reset events collapse into one reset at that checkpoint.
- GOOD placement counters survive a reset. Reset does **not** clear capacity,
  motion authorization, operator pauses, or physical fault/holding state.
- Delivery failures propagate through existing cleanup; there is no automatic
  reset/home/release recovery after a failed operation.
- A request already queued before a new cycle is covered by that cycle's fresh
  capture. A request arriving during inspection is consumed before picking.
- Existing `--n` limits still bound batch attempts, including reset attempts.
  A reset at the configured limit is reported but cannot start another cycle.

## Configuration and compatibility

`config/config.yaml` contains `runtime_events`. Everything is **disabled by
default**. Existing runs retain the old behavior. This scaffold is supported only
for the main pipeline with `pick_target_source: pipeline`, not the legacy sort,
independent picker, offline image, or inspection-only paths.

When `runtime_events.enabled: true`, the configured producer hooks run in separate
threads. Enabling a hook currently starts an **inert wait**, not a real listener:

```yaml
runtime_events:
  enabled: true
  queue_capacity: 64
  idle_poll_s: 0.1
  join_timeout_s: 2.0
  reset_events: [reset_batch, scene_changed]
  producers:
    audio: {enabled: true}
    overview: {enabled: true}
```

With events enabled, an exhausted batch in an unpaused fake run idles until a
reset event instead of automatically retrying returned rejects or blocked items.
An exhausted batch is **not proof of an empty workstation**. Listeners stay alive
while waiting. A finite `--n` run still exits at its limit.

Default batch-mode live multi-cycle picking **still requires the existing operator
pause**. A reset request is not permission to skip ENTER or `--confirm-motion`.
The explicitly selected `--continuous` mode has different orchestration: no ENTER
pauses or cycle limits, safe home-view validation after completed attempts, and
idle monitoring when pending work is exhausted. It retains motion authorization,
capacity, fault and holding-state gates. See `CONTINUOUS_PICKING.md`. The event
kinds and producer interface below remain unchanged. At that pause,
ENTER starts a fresh cycle, Q/EOF exits, and the producer threads remain alive.
Making validated voice commands authorize later live cycles is deliberately a
separate change, not part of this scaffold. Capacity and faults may end a run
regardless of whether the workstation is clear.

## Producer contract / audio handoff

Implement `run(context)` in `monitors/audio.py` or `monitors/overview.py`:

- `context.publish(EventKind.RESET_BATCH, detail)` submits a validated reset request.
- `context.publish(EventKind.SCENE_CHANGED, detail)` reports a confirmed unexpected
  scene change. Camera adapters must distinguish robot movement/occlusion and
  expected pick/removal activity; raw frame differences are insufficient.
- `context.publish(EventKind.NOTICE, detail)` is informational, with no motion action.
- `context.stop` is a `threading.Event`; honour it and use bounded input waits.
- `context.settings` is this producer's config section (add resources/devices there).
- `context.fake` must prevent real microphone/camera/model access in fake mode.
- Own and close your resources in `finally`. Do not share main-loop Viam SDK
  clients across threads. Producers must never command the arm/gripper, call
  inspection directly, modify pending targets, or consume the event queue.
- Validate/debounce commands and scene-change events in the adapter. Reject unknown
  commands; do not publish raw/interim speech as an actionable instruction.

The runtime injects the event source name and a monotonic timestamp. This is an
**in-process** interface; a separate ASR process needs an explicit IPC adapter.
No ASR/NPU implementation or model downloads are included. Extra command semantics,
source confidence/expiry policy and identity tracking remain future work.

For tests or another embedding application, supply a caller-owned `EventRuntime`
to `main.run_overview_loop(..., event_runtime=runtime)`. Start/close its producer
threads using its context manager. A supplied runtime is not closed by `main`.

## Failure and latency boundaries

Queue publication never blocks the producer. Overflow or producer failure latches
a fault; the next controller checkpoint refuses further picks rather than silently
losing an important event. Faults are not reset by draining the queue.

There is **no immediate STOP event** here. A long inference or SDK operation can
delay the next checkpoint. Producer threads can enqueue in the meantime, but
Python/native workload scheduling still affects responsiveness. This is not a
human-presence detector, collision interlock, voice E-stop, or arbitrary thread
interruption mechanism. Use existing machine/physical stopping facilities.

Shutdown signals both producers and joins with a configured timeout. A stuck
thread is reported; Python cannot forcibly stop it. Producers must honour the
contract. Batch reports record consumed checkpoint events and remaining IDs on
reset. No motion code or model implementation was moved into background threads.

## Tests

```bash
python -m pytest tests/test_runtime_events.py tests/test_main_runtime_events.py tests/test_external_box_pick.py -q
python main.py --fake --n 1
```

Tests inject events before picks, during a fake delivery, after HOLD/final attempts,
and on delivery failure; verify reset recaptures/reinspects, idle behavior, retained
operator gates, overflow faults, producer failure, and thread shutdown. These are
fake/mock checks, not physical validation.
