# TipScroll

TipScroll is a Windows MVP that turns one angled index fingertip into a vertical
rate controller. Hold the finger still for 0.2 seconds to establish a neutral
zone, move above it to scroll toward the top, move below it to scroll toward the
bottom, and move farther to scroll faster.

Only the fingertip Y coordinate controls scrolling. MediaPipe index-finger joint
angles are used only to decide whether the finger is extended, curled, or
ambiguous. Any loss, stale result, ambiguous pose, or tracking jump immediately
clears all pending scroll output.

Clearing the output and discarding the anchor are separate steps. The first is a
safety requirement and always happens at once. The second is not: once nothing
is being emitted, dropping the anchor buys no safety, and for a rate controller
it is expensive, because a new 0.2-second dwell is the whole cost of re-entry.
So a transient fault parks the session in a hold that emits nothing but keeps
the anchor, and only a fault outlasting `hold_grace_ms` clears it. A deliberate
curl and a tracking jump skip the hold and stop outright.

The gate uses separate thresholds for the naturally angled pose: PIP must reach
150 degrees and DIP 140 degrees to arm; curling below 140/120 degrees stops the
session. Values between the start and stop thresholds remain fail-safe.

An ambiguous pose stops an active session immediately, but during the arming
dwell nothing is being emitted yet, so a brief dropout only pauses the dwell:
the deadline moves out by however long the pose was untrusted, up to
`arm_pose_grace_ms`. Landmark noise produces one-frame dropouts often enough
that tearing the dwell down for each one made arming fail roughly two times in
three. A deliberate curl still ends the dwell with no grace.

The fingertip is smoothed by a One Euro filter rather than a fixed-alpha EMA.
Its cutoff rises with the smoothed speed of the fingertip, so a resting hand is
filtered hard enough to hide landmark jitter while a deliberate move keeps
almost no lag, and its alpha is derived from the elapsed time so the amount of
smoothing does not change with the camera frame rate. `smoothing_beta` in
`AppConfig` is expressed in hertz per normalized-height unit per second; set
`smoothing_enabled=False` for a raw, unfiltered signal.

The dwell is filtered by the same instance, which then carries into the active
session unchanged. Measuring stillness on the raw landmark conflates hand drift
with landmark noise, and in recorded sessions noise was the larger half of it,
so `arm_max_y_span` now bounds drift alone and means the same thing in any
lighting.

## Setup

```powershell
cd C:\Users\inkay\Desktop\TipScroll
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\download_model.py
```

The pinned model is downloaded from the official MediaPipe model bucket to
`models\hand_landmarker.task`.

## Run safely first

Start with diagnostics and no injected wheel events:

```powershell
.\.venv\Scripts\python.exe -m tipscroll --no-scroll --debug
```

When the state transitions, zones, and direction look correct, enable scrolling:

```powershell
.\.venv\Scripts\python.exe -m tipscroll --debug
```

Normal mode without the camera diagnostics window:

```powershell
.\.venv\Scripts\python.exe -m tipscroll
```

Use `--camera 1` (or another index) to select a different camera. Press
`Ctrl+Alt+Esc` at any time to clear scrolling and exit.

Scroll output is emitted in wheel units, where 120 units is one notch, so rates
below one notch per second still move continuously instead of stalling until a
whole notch is owed. Applications that ignore sub-notch deltas need
`--legacy-wheel`, which restores whole-notch output.

## Interaction

1. Extend one index finger at a natural iPad-like angle.
2. Keep the fingertip within the middle 15%-85% of the camera image and hold it
   still for 0.2 seconds.
3. Move the fingertip upward or downward. Holding away from the anchor continues
   scrolling; distance controls speed.
4. Return to the neutral zone to stop.
5. Curl the index finger or remove it from view to end the current anchor. Extend
   it and hold still for 0.2 seconds to establish a new one.

The first version intentionally does not reject an open palm, recognize multiple
hands, normalize by palm size, scroll horizontally, click, or add inertia.

## Status indicator

- Hidden: idle or outside the activation zone.
- Amber ring: arming for 0.2 seconds.
- Blue dot: active and stopped.
- Green dot and tail: scrolling; tail direction and length show rate.
- Hollow blue ring: holding; output stopped but the anchor is kept.
- Brown dot: safe stop; a new 0.2-second dwell is required.

## Development

Run the unit tests without opening a camera or injecting input:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Each application run writes a session CSV under `logs\`. The debug window shows
the fingertip, anchor, hysteresis bands, fast bands, joint angles, result age,
state, stop reason, desired/committed rate, and the camera format the driver
actually granted alongside the measured capture and result rates.

The camera requests MJPG before the frame size. Left at the driver default,
most webcams negotiate an uncompressed format whose bandwidth caps 720p near
10fps, which silently triples input latency. The negotiated format is printed
at startup and a warning is emitted when the driver does not grant it; check
that line first whenever tracking feels laggy.

Use [`MANUAL_TESTS.md`](MANUAL_TESTS.md) for the angled-finger and application
pressure-test checklist before enabling scrolling for regular use.

