# Neuratect — Build Plan (token-efficient)

Purpose: get from the current flat macOS codebase to the redesigned UX without burning
context on re-derivation, file re-reading, or run-describe-iterate loops.

Read this once at project start. Do **not** read it every session — sessions read
`notes/state.md` and their own module CLAUDE.md only.

---

## 0. The five levers

Everything below is an application of these. If a decision doesn't serve one of them, it's
not a token decision, it's a taste decision — make it fast and move on.

1. **Freeze contracts in text, once.** Interfaces between modules are written as plain-text
   schemas and never re-derived. A session that needs a neighbour reads a 15-line contract,
   not a 400-line file.
2. **Replay, don't run.** Live-camera iteration is the single biggest token sink: run it,
   describe what happened in prose, guess, retry. Recorded landmark fixtures turn that into
   a pass/fail line.
3. **Verify by exit code, not by prose.** Claude should never paste output for you to read.
   Tests print `PASS`/`FAIL` and a diff only on failure.
4. **Subagents for anything read-heavy.** Inventory, audit, review, "where is X handled" —
   dispatch it. The subagent reads 3,000 lines; main context receives 15.
5. **Defer every configurable axis.** Each "user chooses" in the spec is a settings screen,
   a persistence path, a migration, and a test matrix. Opinionated default now, choice later.

---

## 1. Session 0 — Reconcile the existing code (do this first, once)

The repo has flat modules; the notes describe a four-module architecture. These must be
reconciled before any feature session, or every future session pays the confusion tax.

**Do not read the files into main context.** Dispatch a subagent:

> Inventory this repo. For each Python file: one line on what it owns, its public
> functions/classes with signatures only, and which other project modules it imports.
> Also report: why Pylance can't resolve the imports in run.py (packaging, sys.path,
> missing `__init__.py`, or working-directory assumption), and whether any macOS API
> calls (Quartz, CGEvent, Accessibility) appear outside a single module.
> Return under 60 lines. Do not paste file contents.

From that summary, produce two artefacts and stop the session:

- `notes/contracts.md` — the frozen interfaces (see §2)
- `notes/inventory.md` — file → module-boundary mapping, and a short list of boundary
  violations to fix later (not now)

**Fix the import resolution before anything else.** Pick one: proper package with
`__init__.py` and `pip install -e .`, or an explicit `src/` layout with the path set once.
Do not let sessions work around it with `sys.path` hacks — that bug will resurface in every
future session and cost tokens each time.

---

## 2. The contracts (write once, never re-derive)

`notes/contracts.md` is the only cross-module file any session may read. Keep it under
~120 lines total. Each contract is a shape, not an implementation.

Four boundaries to specify:

- **Frame source → detector.** What a frame is, what resolution/colour space, who owns the
  camera handle.
- **Detector → mapper.** The recognition event: gesture id, confidence, timestamp, hand(s),
  and whether it's an onset, a hold tick, or a release. *Get the onset/hold/release
  distinction into the contract now* — retrofitting it later touches every module.
- **Mapper → executor.** The command: a tagged union of the concrete action types. Never a
  string to be parsed downstream.
- **Anything → context provider.** Current app/window identity, as a plain value.

Rules that make the contract earn its keep:

- Executor accepts only concrete commands. **Natural language is resolved at authoring
  time, never at runtime** (see §5 critique). No LLM call is permitted downstream of the
  detector.
- All macOS API calls live behind the executor and the context provider. Nothing else
  imports Quartz.
- The mapper is pure: gesture event + context in, command out. No I/O, no camera, no OS.
  This is what makes it testable without fixtures at all.

Once written, contracts are frozen for the duration of a slice. Changing one is a
deliberate act that gets a line in `notes/decisions.md`.

---

## 3. Fixtures — the highest-leverage thing in this plan

Build this before any feature work. It's a day that pays for itself within a week.

**Recorder.** A small script that captures raw landmark streams to JSON with a label:
`fixtures/fist_clean.json`, `fixtures/fist_partial_occlusion.json`,
`fixtures/swipe_left_fast.json`, `fixtures/idle_talking_hands.json`.

**Replayer.** Feeds a fixture through the detector as if it were live.

What this buys:

- Detector tuning becomes `pytest` over a fixture corpus, not "sit in front of the camera
  and describe what you saw." That loop is the difference between a 200-token verification
  and a 4,000-token one, repeated dozens of times.
- Regressions are caught automatically. Without fixtures, "did my threshold change break
  the other four gestures?" is answered by manually redoing all five gestures on camera and
  narrating the result.
- Sessions become machine-independent — the friend's MacBook isn't a dependency for
  detector work.

**Record the negative corpus too**, and record it early: hands moving while talking,
reaching for the mouse, scratching, adjusting glasses, typing. These are what the false-
positive gate (§5) is tuned against, and they're the fixtures you'll wish you had.

Target: ~15 positive fixtures, ~10 negative, before slice 2. Recording all of them is one
sitting.

---

## 4. Build order — vertical slices with hard session boundaries

Each slice is one session, ends at a green test suite and an overwritten `state.md`,
then `/clear`. Do not carry a session across a slice boundary.

**Slice 1 — the pipe.**
One hardcoded gesture → hardcoded context → one command, end to end. No config, no
persistence, no UI. Proves permissions (Camera, Accessibility), proves the contracts hold,
proves the packaging is sane. Success = the gesture opens Spotlight, and the same fixture
replayed through the same pipe with the executor mocked produces the same command.

**Slice 2 — detection quality with no new features.**
Fixture corpus green. Tune only. Add the **engagement gate** here (§5) — it is not a
feature, it is the thing that makes everything downstream trustworthy. Do not add gestures
in this slice; add confidence in the ones you have.

**Slice 3 — persistence and the mapper.**
Gestures and mappings load from disk. Mapper handles context lookup with a fallback chain.
Still no UI, still no authoring. Testable entirely without a camera.

**Slice 4 — authoring, minimum viable.**
Record a gesture, name it, pick a command from a list, save. **Not** the full authoring
spec: no icon picker, no preview clip, no critical/optional feature marking, no chains.
One opinionated path through. This is where you find out whether the recording flow is
even right, and it's cheaper to find out with one path than with four.

**Slice 5 — the visible surface.**
Config mode with live prediction, confidence, matched command, context indicator, and the
isolation toggle. This is the first slice where visual iteration is unavoidable, so batch
it: build the whole screen, screenshot once, critique once, revise once. Do not iterate a
screenshot per widget.

**Slice 6 — live mode and the ambient signal.**
Tray presence, silent operation, pause meta-gesture, and the minimal always-on feedback
that §5 argues you need.

Everything else — chains, scenes, natural-language authoring, dual-hand relational,
continuous cursor control, critical/optional feature marking, per-gesture confidence
sliders — is **post-slice-6**. Listed in `notes/deferred.md` so it stops re-entering
conversations.

---

## 5. UX critique — fold these in before building

My review of the current design. Several of these are also token decisions, because the
cheapest code is code you didn't write.

**"User chooses" is specified in three places and it shouldn't be.** How a gesture is
defined, how it's labelled, and how many reps register it were all answered "user chooses."
That's three settings surfaces standing between a non-technical user and their first
working gesture — which contradicts the progressive-complexity principle stated elsewhere
in the same spec. Pick a default path (demonstrate live, custom name, progressive
accuracy), ship it, and put the alternatives behind an advanced panel if anyone asks.
Every configurable axis is UI + persistence + migration + a test matrix.

**There is no false-positive story, and this is the one that kills gesture products.**
Nothing in the spec prevents firing when you scratch your face, gesture while talking, or
reach for the mouse. In an assistive context this is worse than a miss — an unwanted
command is disorienting and may be hard to undo. You need an engagement gate before any
of the expressive features matter: a hand-in-zone requirement, a wake pose, or a dwell
threshold. Build it in slice 2, tune it against the negative fixture corpus.

**Confidence-based preview has an undefined interaction model.** Fires above threshold,
previews below — but what dismisses the preview? A timeout, a confirm gesture, moving your
hand? Holding a preview state is a *motor task*, and for users with fine motor challenges
it may be harder than the gesture that triggered it. Consider inverting: never preview,
just make the threshold conservative and make undo excellent. Undo is a keystroke; preview
is a state machine.

**The modifier system fights the target user.** "Deliberately effortful" gesture chains to
reach global commands is exactly wrong ergonomics for someone with limited fine motor
control — you've made the safety mechanism the hardest thing in the product. The safety
goal is real, but context-scoping already provides most of it. Cheaper alternative: keep
global commands few, and gate them with dwell (hold the pose 800ms) rather than with
chained complexity. Dwell scales down for impaired users; chains don't.

**Live mode's silence is a failure mode, not a feature.** Tray-icon-only means a user
performs a gesture and nothing happens, with no way to distinguish: not detected /
detected but unmapped in this context / fired but the app ignored it. Those three need
different responses from the user, and silence gives them nothing. A minimal ambient signal
— a brief menu-bar glyph change on detection — preserves the invisible aesthetic while
making failure legible.

**Natural language must be authoring-time only.** "Skip the song" resolving at runtime
means a model call in the hot path: latency, cost, non-determinism, and offline failure in
a product whose whole value is immediacy. Resolve NL to a concrete command once, at
creation, show the user what it resolved to, store the concrete command. This should be an
explicit line in `notes/decisions.md` because it will otherwise get relitigated.

**Chains before reliability is the wrong order.** Arbitrary-length chains with per-chain
timing tolerance is the most complex thing in the spec and it's built on a detector whose
single-gesture reliability isn't yet established. Defer until slice 6+ — and note that if
the engagement gate and dwell work well, you may find you don't want chains at all.

---

## 6. Subagent review checkpoints (Claude Code)

Use subagents wherever the work is read-heavy and the output is a judgement. The subagent
pays the reading cost; main context receives a verdict.

**After each slice**, dispatch a reviewer with a scoped brief. Good briefs are narrow and
demand a bounded response:

> Review `src/detector/` against `notes/contracts.md`. Report only: contract violations,
> functions doing work that belongs in another module, and any macOS API call outside
> the executor. Max 20 lines. No code, no praise, no summary of what the module does.

**Before slice 5**, dispatch a UX reviewer against the plan rather than the code:

> Read `notes/ux-spec.md`. You are a first-time non-technical user with limited fine motor
> control. Walk through creating your first gesture and list every point where you would
> not know what to do next, or where the required physical action is harder than the task
> it accomplishes. Max 15 findings, ranked.

**Anti-pattern to avoid:** dispatching a subagent to *write* something substantial. The
output comes back into main context in full, so you pay for it anyway plus the dispatch.
Subagents are for reading a lot and saying a little.

---

## 7. Session hygiene

- `notes/state.md` — overwritten every session, never appended. Current status, next step,
  blocked-on. If it exceeds ~20 lines something is wrong. Overflow has a destination, so
  hitting the limit is a filing job, not a deletion job: gotchas → the module's `CLAUDE.md`,
  parked ideas → `deferred.md`, why-X-over-Y → `decisions.md`, how-to-run → `README.md`,
  finished work → git. The `session-end` skill has the full protocol.
- `notes/decisions.md` — appended, reasoning only ("why X over Y"), never activity.
- `notes/deferred.md` — the parking lot. Its job is to stop deferred features from
  re-entering conversation.
- Module-level `CLAUDE.md` — gotchas specific to that module, loaded only when it's in
  context. Let these grow; they're free to other sessions.
- `/clear` at every slice boundary. `/compact` only mid-slice when genuinely mid-task.

**Reading policy for sessions:** grep before read; read line ranges, not files; never read
a file to "get oriented." If a session finds itself reading three files to answer one
question, that's a signal the contract in §2 is missing something — fix the contract rather
than repeating the reads next time.

---

## 8. Verification protocol

Claude should never ask you to read output. Every check is a command whose result is a
line.

- `pytest` over the fixture corpus for detector work
- Pure unit tests for the mapper (no fixtures needed — it's a function)
- A single smoke script for the executor that asserts the OS side effect where it can, and
  is skipped in CI where it can't
- Golden files for anything with a stable expected output, so a change shows as a diff
  rather than as narrated prose

Failures print the diff. Successes print one word.
