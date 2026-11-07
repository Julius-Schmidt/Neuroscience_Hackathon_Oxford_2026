# Deferred

Deliberately not built. One line on why. Job: stop these re-entering conversation.

- **Scroll** — no pose spent on it yet. Leading design: left-hand modifier (left fist held +
  right hand vertical) so it can't collide with the pointer, which uses every right-hand
  pose. Left pinches are bound in Minecraft but free on the desktop, which is where scroll
  matters. Decide after desktop mileage.
- **Pose/pinch clash, real fix** — poses and pinches on one hand interfere by construction
  (see src/cv/CLAUDE.md). Avoided for now by unmapping index_pointer/peace. Candidates, in
  order of preference: normalize each pinch by that finger's own MCP→tip length instead of
  hand_scale, so a curled finger and a pinching finger stop looking alike; gate a pinch on
  the non-pinching fingers' state; or one classifier over the whole landmark set instead of
  two signals that can disagree.
- **Virtual HID device** — an order of magnitude more effort than the UI work (signed
  DriverKit extension, HID report descriptors, IOKit outside Python). Its main motivation,
  per-app GUI-state tracking, was retired by reading CGCursorIsVisible. Remaining value is
  fullscreen camera-look feel only.
- **Dedicated gesture keyboard** — far future. Accessibility Keyboard driven by the desktop
  pointer is the cheap version; user will drive that test.
- **Chains, scenes, natural-language authoring, dual-hand relational, per-gesture
  confidence sliders** — post-slice-6 (build-plan.md §4/§5). NL authoring in particular
  resolves at *authoring* time, never runtime; no model call downstream of the detector.
- **Cross-platform anything** — macOS MVP first.
- **src/cv/run_test.py** — broken before this work (imports `_is_fist`, which no longer
  exists in detector.py). Dev tool, not on any live path. Fix or delete when next in there.
- **The three flat files in `.claude/skills/`** (gesture-cv-pipeline, macos-spotlight-invoke,
  macos-accessibility-keyboard) have no YAML frontmatter, so they never load as skills — they
  are inert stubs. Two are already superseded by module CLAUDE.mds (Spotlight's Cmd+Space
  method, the MediaPipe choice). Convert to `<name>/SKILL.md` or delete; they cost nothing
  where they are.
- **`data/settings.json` is tracked**, so every calibrate.py run that saves a value shows up
  as a change to commit. Kept tracked because the tuned numbers are real findings worth having
  in history. Alternative if it gets annoying: commit a defaults file and gitignore the live
  one.
- **A real lockfile** (uv/poetry, pinning transitive deps with hashes) over the pinned
  requirements.txt. Not worth new tooling for a solo macOS project yet.
