# Neuratect — Project Context

Gesture-controlled command mapper. macOS + Windows. CV (and Myo / LSL via src/sources) in, OS actions out.

## Architecture (detail: notes/architecture.md)

1. `src/cv/` — camera in, gesture classification out
2. `src/mapping-engine/` — (gesture, active context) in, command out
3. `src/os-integration/` — command in, macOS action executed
4. `src/ui/` — quick-start panel, gesture config, live status

Load one module's CLAUDE.md, not its neighbours'. Module gotchas live there, not here.

## Session start

Read `notes/state.md`. Nothing else.

## Session end

Invoke the `session-end` skill — it owns the notes protocol.

## Working rules

- Verify by running, not by reading code back. Grep before Read; read line ranges, not
  files. Dispatch read-heavy "where does X happen" to a subagent.
- Narrowest vertical slice first. End sessions at a green test suite.
  `pytest -q` (or `python3 tests/<name>.py`) — no camera, no Minecraft, every suite prints PASS/FAIL.
- Slice order and its rationale: `notes/build-plan.md`. Read once at project start, not per
  session.

## Non-obvious constraints

Keep this file under ~350 tokens. A new item almost always belongs in a module CLAUDE.md,
`notes/deferred.md`, or a skill instead.

- Camera and Accessibility permissions are granted by hand (System Settings → Privacy &
  Security) and must be re-granted per machine. Both fail *silently* — no error, just
  nothing happens.
- Windows goes through src/os-integration/quartz_compat.py (Quartz-shaped Win32). Keep new OS
  calls Quartz-shaped or add them there, never call Win32 from other modules.
- General mapping = src/mapping-engine/rules.py + src/sources/ + ui/page_blocks.py (data/rules.json).
