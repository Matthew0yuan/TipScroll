# TipScroll manual pressure tests

Run the first pass with no input injection:

```powershell
.\.venv\Scripts\python.exe -m tipscroll --no-scroll --debug
```

Record failures with the generated session CSV filename and the test number.

## Camera negotiation

- [ ] The startup line reports MJPG at the requested size; no format warning appears.
- [ ] The debug overlay shows capture and result rates near the requested 30fps.
- [ ] Repeat in a dim room; note both rates, since a dropped capture rate raises
      latency and landmark jitter more than any threshold change can offset.

## Recognition and arming

- [ ] Hold an extended index finger at 20°, 35°, 50°, and 65°; each pose arms in
      roughly 0.2 seconds when stable.
- [ ] Repeat from the left and right sides of the image.
- [ ] Hold outside the middle 15%-85% activation band; the state remains idle.
- [ ] Move while arming; the 0.2-second timer restarts rather than anchoring mid-motion.
- [ ] Hold a steady pose that occasionally flickers below the gate; the amber
      ring pauses and resumes rather than resetting to zero.
- [ ] Hold a partly bent finger continuously; the dwell gives up after the grace
      period instead of arming.
- [ ] Curl slowly and quickly; both immediately clear the active anchor.
- [ ] Hold a partly bent finger between the thresholds; it never produces a rate.

## Rate control

- [ ] Hold in the neutral zone for 60 seconds; desired and committed rates remain zero.
- [ ] Move slightly above/below the start boundary; direction is correct.
- [ ] Increase distance in steps; rate increases monotonically and caps at 18 notches/s.
- [ ] Return through the hysteresis band; scrolling stops without boundary chatter.
- [ ] Move quickly but continuously into the fast band; speed alone does not trigger a stop.
- [ ] Hold in a rate band; rate remains steady without requiring repeated swipes.
- [ ] Hold just past the start boundary; the page creeps smoothly rather than
      jumping one notch every few seconds.
- [ ] Hold the fingertip still for 30 seconds; the smoothed offset stays inside
      the neutral band without visible jitter on the debug overlay.

## Failure safety

- [ ] Remove the finger abruptly; committed rate becomes zero immediately.
- [ ] Cover the fingertip for one and several frames; no stale trajectory is replayed.
- [ ] Disconnect or disable the camera; output reaches zero after the 80ms freshness limit.
- [ ] Move the fingertip farther than 25% of the frame in one result; track-jump stop occurs.
- [ ] Press Ctrl+Alt+Esc; the process exits and no wheel output remains.
- [ ] Re-extend after every stop; a fresh stable 0.2-second dwell is required.

## Application pass

After the no-scroll checks pass, repeat rate and failure tests with scrolling enabled in:

- [ ] Chrome on a long page.
- [ ] A PDF reader.
- [ ] A text editor or document reader.

If any of these ignores slow scrolling entirely, it is discarding sub-notch
wheel deltas; re-run it with `--legacy-wheel` and record which application
needed it.

Verify that finger-up moves toward the page top and finger-down moves toward the page
bottom. Run continuously for five minutes and confirm there is no stuck scrolling,
uncommanded re-arming, or thread/window freeze.

