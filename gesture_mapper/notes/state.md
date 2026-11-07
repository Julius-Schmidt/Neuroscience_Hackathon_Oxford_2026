# Current State

Last updated: 2026-08-09

## Status

GUI shipped: `python3 gui.py` — onboarding wizard, Home (live preview + Start/Stop
control), Gestures (card grid + guided recording, replaces manage_gestures.py for
humans), Mappings (per-context editor → `data/mappings.json`, applies live), Settings
(sensitivity/pinch sliders). Mapper is data-driven now; hardcoded tables remain as
factory defaults/fallback.

Verified: 5 suites green (115 checks); offscreen smoke of every page/dialog; screenshot
review of all screens; preview↔control camera handover + restart, 2 cycles, OS calls
stubbed. **Not yet verified live:** Start Control driving real apps from the GUI with
hands in frame, and recording a brand-new gesture end-to-end through the dialog.

## Next step

Live run-through of the GUI as a user would: onboard → record a new gesture → bind it →
Start Control → desktop + Minecraft spot-check. Then the video.

## Blocked on

Nothing.

Running the app, tuning, permissions and their gotchas: `README.md`. Parked work:
`notes/deferred.md`.
