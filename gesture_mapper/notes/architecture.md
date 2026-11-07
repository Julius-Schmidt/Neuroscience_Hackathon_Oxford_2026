# Architecture

## Modules

### 1. cv/ — Gesture recognition
Input: camera frame stream.
Output: classified gesture + confidence, on a stable interface (e.g. `Gesture(name, confidence, timestamp)`).
Internals (model choice, preprocessing, frame sampling strategy) are private to this module —
nothing outside cv/ should depend on how the classification happens, only on the output shape.
Requires macOS camera permission (System Settings → Privacy & Security → Camera) — grant this
on whichever machine is running dev builds.

### 2. mapping-engine/ — Context-aware command mapping
Input: Gesture + current OS/app context (active app, "mode").
Output: Command (an enum/struct describing what to do, e.g. `OpenSpotlight`, `OpenAccessibilityKeyboard`).
This is where customisable gesture→command mappings and context rules live. Should be
testable with fake Gesture/context inputs — no camera or OS calls in here.

### 3. os-integration/ — Executes commands on macOS
Input: Command.
Output: side effect (Spotlight opened, Accessibility Keyboard invoked, etc.)
Owns all macOS API calls (Accessibility API / CGEvent for simulated input, NSWorkspace for app/window
context, etc.). Nothing else in the codebase should call macOS APIs directly.
Requires Accessibility permission grant (System Settings → Privacy & Security → Accessibility) for
the app/binary — same caveat as camera, re-grant on the dev machine in use.

### 4. ui/ — Quick-start panel & config
Surfaces gesture config, live gesture status, quick-access panel. Talks to mapping-engine
for current mappings, not directly to cv/ or os-integration/.

## Why this split

Each module can be developed, tested, and reasoned about independently — a session working
on mapping-engine never needs cv/ loaded into context, and vice versa. This is the actual
mechanism (not just discipline) that keeps token usage per session bounded as the project grows.

## Data flow

```
camera → [cv/] → Gesture → [mapping-engine/] → Command → [os-integration/] → OS action
                                    ^
                                    |
                                  [ui/] (reads/writes mappings, shows status)
```

## MVP slice (build this first, end to end, before generalizing)

One gesture (e.g. closed fist) → hardcoded "desktop" context → one command (open Spotlight search).
Nothing configurable yet. Just prove the pipe works.
