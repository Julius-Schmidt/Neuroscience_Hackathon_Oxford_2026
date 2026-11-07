# Decisions Log

Format: Date — Decision — Reasoning. Append only; do not delete old entries even if superseded
(mark superseded entries instead, so Claude sees the history of why something changed).

---

**2026-08-01 — macOS-only MVP, no cross-platform abstraction yet**
Reasoning: cross-platform abstraction adds complexity before the core pipeline is proven.
Add abstraction only once porting is an actual near-term goal, not preemptively.
Superseded: originally scoped Windows-only; switched to macOS because the primary dev laptop's
camera stopped working, so development is happening on a borrowed Mac instead.

**2026-08-01 — Four-module hard boundary (cv / mapping-engine / os-integration / ui)**
Reasoning: past attempts stalled from context breakdown + token ballooning. Hard boundaries
with stable interfaces mean a session can work on one module without loading the others,
which is the actual fix (not just "being disciplined").

**2026-08-01 — Build narrowest E2E slice first (one gesture, hardcoded context, one command)**
Reasoning: past attempts built scaffolding for all modules before anything ran end-to-end.
An early working loop gives a concrete regression check as complexity/generality is added.

**2026-08-02 — Template matching over trained model for gesture recognition**
Reasoning: MediaPipe 1.0.0 GestureRecognizer requires a downloaded .task model with bundled
gesture labels. Template matching on normalized landmarks is zero-download, user-trainable,
and good enough for 6 distinct gestures at this scale.

**2026-08-02 — Joystick uses fist gesture as activation, not always-on**
Reasoning: always-on joystick caused constant recalibration drift and accidental WASD presses.
Fist as explicit activation gives a clear mental model: open hand = neutral, fist = driving.

**2026-08-02 — Right-fist mouse works everywhere, left-fist WASD Minecraft-only**
Reasoning: mouse is universally useful (Spotlight, general nav). WASD outside Minecraft
would type characters into any focused text field — scoped to Minecraft context only.

**2026-08-02 — Accessibility Keyboard dropped for now**
Reasoning: it's mouse/dwell driven, not arrow-key navigable. Right-fist mouse now makes it
usable without special keyboard context handling. Revisit if dedicated typing mode needed.

**2026-08-02 — Mouse clamped to Minecraft window bounds via CGWindowList**
Reasoning: unclamped synthetic mouse events caused cursor to escape Minecraft, click desktop,
minimise all windows. Clamping to window bounds (lookup by "java" owner name) fixes this.
Outside Minecraft the mouse is unclamped — intentional for general navigation use.

**2026-08-03 — Joystick anchor changed from wrist (lm0) to middle finger MCP (lm9)**
Reasoning: MCP is more stable under rotation and more central to the hand's "intent" plane.
Wrist position shifts significantly with forearm rotation even when hand pose is unchanged.

**2026-08-03 — Right stick uses infinite-surface auto-calibrate (fist entry = new neutral)**
Reasoning: fixed neutral (shaaa) required awkward recalibration mid-game. Auto-calibrate on
fist entry lets user reposition freely, like picking up and replacing a mouse. Velocity mode
was tried and felt better than fixed joystick but jitter and sensitivity were hard to tune;
position-with-auto-calibrate gave the trackpad feel with less noise sensitivity.

**2026-08-03 — Gesture library cached at process start; requires restart after re-recording**
Reasoning: load_library() was being called every recognize() call (~60/sec), causing disk I/O
that contributed to game lag. Cache is process-lifetime; acceptable since gestures are static
during a play session. Side effect: manage_gestures.py changes don't hot-reload into run.py.

**2026-08-03 — Deferred: virtual HID device (IOHIDUserDevice) as the general fix for
mouse-mode dependence, instead of app-specific GUI-state tracking**
Reasoning: CGEventPost synthetic mouse events only partially participate in macOS's real
HID/cursor-association machinery (CGAssociateMouseAndMouseCursorPosition), which is why
Minecraft's raw-look-vs-GUI-cursor switching doesn't naturally work for injected events —
concretely surfaced when pinning the cursor to the window center for sensitivity broke
inventory item dragging. A true virtual HID device (what Karabiner-Elements uses for keyboard
remapping, via Karabiner-VirtualHIDDevice) would be indistinguishable from real hardware at
the input-stack level, fixing this generally for any app rather than needing per-app GUI-state
tracking (which is what we implemented instead, for now — see run.py's _gui_open). Deferred
because it's a materially bigger lift than our current Python/CGEventPost approach: needs a
hand-built HID report descriptor, IOKit calls outside our current PyObjC/Quartz stack, and
likely an entitlement or signed DriverKit extension on Apple Silicon. Revisit as a dedicated
side-quest, not folded into the MVP.

**2026-08-04 — Context detection reads topmost on-screen window's owner, not
NSWorkspace.frontmostApplication()**
Reasoning: NSWorkspace never reported Minecraft as frontmost even when its window visibly had
focus — traced to Minecraft's launcher spawning the actual game window without properly
handing off "frontmost app" status (a quirk of how it's packaged, not something we control).
CGWindowListCopyWindowInfo's z-ordered list gives the same signal Spotlight-overlay detection
already used, extended to the general case: first layer-0 window's owner = the real active
app. Verified live (polled context while manually switching focus) before trusting it.

**2026-08-04 — Fist/click-gesture anchor persists across releases; only explicit open-palm
clears it**
Reasoning: earlier version re-anchored the joystick on every fist entry ("infinite-surface"
idea, 2026-08-03 entry above) — reintroduced drift/inversion-perception bugs because the
anchor silently moved every grip. Persisting until an explicit "let go" (open_palm) matches
how a real stick's center never moves; auto-recalibrating per-grip was solving a problem
(awkward fixed-neutral recalibration) that a persistent anchor already solves without the
side effect.

**2026-08-04 — Click gestures (index_pointer/peace) are held commands with a drag-arm-delay,
not tap commands**
Reasoning: crafting needs both quick discrete clicks and real click-and-drag (spreading items
across slots). Making them held (mouse down while gesture active) with cursor movement gated
behind a delay (1.5s, raised from 0.3s after live testing) gives both for free: a quick
gesture is an ordinary click, holding past the delay lets the same gesture drag. Rejected a
separate "sticky toggle" design (index_pointer toggles the button on/off) tried first — more
complex and unnecessary once the delay-gated approach covers both cases naturally.

**2026-08-04 — GUI cursor uses a depth-normalized delta model, not the joystick/velocity model**
Reasoning: rate/velocity control (real analog stick or our joystick) fundamentally can't give
clean precision pointing — you must return to an exact center to stop, and hand tremor in
free air (no desk to rest on, unlike a mouse) makes that centering imprecise. A mouse can do
both coarse aiming and fine pointing because it's a relative-displacement device: the user
chooses the physical movement scale, the mapping stays linear. Adopted the same shape for the
GUI pointer: move the cursor by the delta between consecutive hand positions (stops instantly
when the hand stops), divided by the live wrist-to-middle-finger-MCP distance (Gesture.
hand_scale) to cancel out perspective/depth dependence, times one sensitivity gain constant.
Design pattern (delta × gain, not absolute position) taken from inspecting github.com/
jorisparet/hamoco's cursor controller rather than reinvented from scratch. First version
(fixed camera-zone → absolute window position) was tried and discarded — reported unstable
and depth-dependent, exactly the problems this design targets. Not yet verified as actually
fixed — see state.md.

**2026-08-04 — calibrate.py: live-typed sensitivity value, not a target-acquisition wizard**
Reasoning: considered building an interactive calibration UI (show a target, measure
overshoot, auto-adjust) but the user's actual ask was simpler — type a number while playing
normally and see the effect immediately, no separate mini-game. Persisted to
data/settings.json so a calibration session's result actually carries into normal run.py use.
Needs line-buffered stdin, which conflicts with run.py's cbreak-mode single-key 'q'-to-quit —
calibrate.py drops that listener entirely and treats 'q' + Enter as a quit line instead.

**2026-08-07 — GUI clicks are distance-based pinches (thumb–index = left, thumb–middle =
right), not pose-template gestures; template clicks (index_pointer/peace) are gameplay-only**
Reasoning: forming index_pointer/peace translates landmark 9 (the tracked anchor) and
pitches the hand — shrinking hand_scale and spiking the 1/scale gain — at exactly the moment
of clicking, so every click dragged the cursor off its own target (the 21:08 crafting-error
pattern in the 2026-08-04 log: index_pointer→fist cycles every ~0.5s, repeated misses). A
fingertip pinch barely moves the anchor (same reason Vision Pro clicks with a pinch), and
measuring it as a continuous distance ratio with hysteresis (engage <0.40, release >0.60 of
hand_scale, settings.json-overridable) avoids pose-classification flicker entirely. Backed by
two safety nets: clicks land at the cursor position 0.12s in the past (position trail rewind
— aiming finished before the pinch started), and the pointer freezes 0.15s around each click
edge. Replaces DRAG_ARM_DELAY in GUI mode (drag = just move while pinched); open_palm is the
clutch ("mouse lifted"), consistent with its existing "let go" meaning for the joystick.

**2026-08-07 — GUI pointer filtering: One Euro + speed-adaptive gain + isotropic coords,
replacing fixed EMA + constant gain**
Reasoning: a fixed EMA can't resolve smoothing-vs-lag — at the working gain (~10
window-widths per frame-width), 1px of landmark noise at 320px survived α=0.5 as ~30px of
cursor jitter, making slot-precision impossible; any α low enough to kill it would lag fast
moves badly. One Euro (min_cutoff 0.5, beta 5.0 — chosen by synthetic noise sweep: ~3px
hover jitter on 1080p at ~0.7 frames lag) smooths hard at rest and passes fast moves
through. Speed-adaptive gain (1 + 0.15×hand-speed, capped 3×) mirrors OS pointer accel: base
gain does precision, accel does reach — one linear gain forced choosing. Isotropy fix:
MediaPipe normalizes x by width and y by height, so deltas/hand_scale mixed units and
sensitivity varied ~33% with hand orientation and axis; all GUI-pointer math now runs in
frame-height units (x × aspect), applied to the window in uniform pixels. GUI_SENSITIVITY
semantics therefore changed: window-*heights* (was widths) per hand-width, before accel.

**2026-08-07 (later) — Pinch thresholds set from live-session measurements, ratio EMA'd
before thresholding; "pinch" treated as fist-equivalent for both joysticks**
Reasoning: first live run (logs from 22:19 session, preserved in git-visible numbers here)
showed deliberate pinch reads 0.03–0.25, relaxed hand/fist reads 0.42–0.53, raw jitter ±0.1.
The guessed thresholds (0.40/0.60) put the *relaxed hand inside the hysteresis band* — a
pinch engaged but relaxing never released it (left button stuck 30+s = the "can't put the
axe down" session). New band 0.26/0.40 sits inside the measured 0.25–0.42 gap, so relaxing
the hand IS the release; ±0.1 jitter is wider than any viable band, hence EMA (α=0.5) on the
ratio first — costs one frame of click latency, prevents spike clicks/drops. Separately, the
new pinch template classification-flickers with fist (visually close), which stuttered WASD
and camera look — both joysticks now accept "pinch" as fist rather than trying to make the
templates more separable (flicker is inherent to two similar closed-hand poses).

**2026-08-08 — All synthetic cursor positions confined to an inset "content rect";
gameplay clicks center the cursor first**
Reasoning: kCGWindowBounds includes the title bar, and macOS resize zones extend several px
*inside* every window edge. The velocity clamp pinned the cursor at edge±1px, so a held
click gesture with the cursor riding the bottom edge posted mouse-down on the resize handle
— the subsequent look movement dragged the window to zero height and java froze mid
live-resize (the "game freezes then resizes to y=0 when breaking blocks" reports; the log's
minecraft→default context flapping was the collapsed window losing topmost status). Fix:
every synthetic position (velocity clamp, absolute mapping, warps, in-bounds snaps) goes
through _content_bounds() — raw bounds inset 30px top / 12px other edges. Gameplay clicks
additionally snap to window center first: the game aims with the crosshair, so cursor
position is free, and center is immune even to stale cached bounds. Also: absolute mode now
only posts when the target moved — the unconditional 60 Hz warp pinned the cursor against
real trackpad input ("mouse trapped, alt-tab to retrieve").

**2026-08-08 — GUI sensitivity re-derived from slot size (1.5 → 0.6), camera 320×240 → 640×480**
Reasoning: an inventory slot is ~4% of window height; at sensitivity S it corresponds to
0.04/S hand-widths of hand travel. At S=1.5 with 320px capture that was ~1 camera pixel per
slot — precision physically impossible, and the live log showed clicks slammed into window
corners from overshoot ((0,1), (1,0) repeatedly). S=0.6 + 640px capture puts a slot at
several camera px; the clutch covers the lost reach; accel cap trimmed 3.0→2.5. The 1.5 in
data/settings.json was never user-chosen (calibrate saved the untuned default) — reset to
0.6. MediaPipe resizes internally so inference cost is unchanged; watch detection cadence
for capture overhead and revert the resolution if it drops.

**2026-08-08 — _gui_open read from CGCursorIsVisible (ground truth), not keypress counting**
Reasoning: Minecraft hides the OS cursor while gameplay-captured and shows it whenever any
screen is open — cursor visibility IS the GUI state. Counting our own Inventory/Escape
keypresses desynced on anything we didn't send (death screens, real keyboard, misrecognized
shaaa — observed live 2026-08-07). CGCursorIsVisible is deprecated since 10.9 but present
and verified correct on this machine; old keypress-counting implementation is in git history
as the fallback if a macOS update removes it. This also retires the per-app GUI-state
tracking motivation for the virtual HID side-quest (HID still relevant for fullscreen feel).

**2026-08-08 — Desktop pointer: GuiPointer promoted to the general mouse in "default"
context; supersedes 2026-08-02 "right-fist velocity mouse works everywhere"**
Reasoning: the delta+pinch+clutch pointer proven in Minecraft GUIs is strictly better at
precision pointing than the velocity fist-stick (that was the whole redesign rationale), so
the desktop gets it too: same code, coordinate space switched to full main-display bounds
(no insets — menu bar/Dock are legitimate targets, unlike window chrome). Hand-role split
formalized: right hand = pointer (any pose tracks, pinches click, open palm = clutch), left
hand = commands — which forced OpenSpotlight from ("either", index_pointer) to left-only,
since pointing right-handed must not open Spotlight. Scroll deliberately NOT included yet;
leading design is a left-hand modifier (left fist + right-hand vertical movement) so it
can't collide with pointer poses — decide after the pointer has desktop mileage.

**2026-08-08 — Pinch is the click everywhere: gameplay clicks moved off pose templates,
DRAG_ARM_DELAY retired; Spotlight moved onto the absolute pointer**
Reasoning: extends the 2026-08-04 GUI-mode decision to gameplay, retiring its "template
clicks are gameplay-only" carve-out. The pose-click objections (forming index_pointer/peace
moves landmark 9 and shrinks hand_scale mid-click; classifier flicker, which the 1.5s
drag-arm delay only papered over) apply just as much to gameplay — and pinch is strictly
better there, since a thumb pinch can be held while a fist keeps driving camera look, so
mine-while-turning is one motion instead of a mode. index_pointer/peace left deliberately
unmapped on the right hand rather than reassigned. Pinch hysteresis extracted from GuiPointer
into PinchClicker (own instance per mode, reset on switch, so a pinch can't leak across).
Spotlight was falling through to the velocity joystick — reachable only via left-hand arrow
keys while the right hand rate-controlled a mouse over a search panel; it's an ordinary
pointable UI, so `_pointer_mode` now gives every non-Minecraft context the pointer in screen
coordinates, and gameplay camera look is the only velocity consumer left (aiming a camera
genuinely is a rate control). Spotlight and desktop share the "desktop" mode name so moving
between them doesn't re-seed the cursor. Known collision left standing: right open_palm →
Space in the spotlight table is also the pointer's clutch.

**2026-08-08 — Pinch bindings became mapper table data; left hand pinches for Jump/Crouch;
sprint reads stick travel past full deflection**
Reasoning: needed because left-hand pinches must mean different things in different states
(Crouch has to stay live while a GUI screen is open — that IS inventory shift-click — while
Jump must not, since space can activate a focused button in the pause menu). Hardcoding a
second pair of commands in run.py would have buried that; instead PinchClicker reports which
fingers are pinched and run.py resolves through map_gesture with "pinch_index"/"pinch_middle"
pseudo-gesture keys, so every pinch binding is visible in the same tables as the poses. The
resolution is re-done every frame rather than latched at the click edge: a context change
mid-pinch then just re-resolves and the held-key reconciliation swaps keys, where latching
would leave a key held with nothing to release it. Left clicker is non-exclusive (jump while
crouched is ordinary play) unlike the right's mouse buttons. Sprint deliberately is NOT a
gesture: it reads forward stick offset past MAX_RANGE, where WASD output has already
saturated and the remaining travel was otherwise wasted — "shove the stick against its
stop". Held, not latched, for the same desync reason as the cursor-visibility fix; MC keeps
sprinting while W is held, so it feels like the toggle anyway.
Also: pinch thresholds halved (0.26/0.40 → 0.13/0.20) because clicks stayed held after the
fingers separated — release at 0.40 was above what a partly-opened hand reads. Cost, measured
on the 2026-08-07 drag log: release at 0.20 sits inside the range a real pinch reads, so that
logged drag now breaks into 2 clicks instead of 1 (test_pinch_replay prints this). Retune from
live ratios, not guesses, if long drags start dropping.

**2026-08-08 — index_pointer/peace disconnected on both hands; hotbar moved to ring/pinky
pinches; pose/pinch clash logged as an open issue rather than fixed**
Reasoning: poses and pinches on the same hand interfere by construction — a pose is defined
by where the fingers are and a pinch is measured from where the fingers are, so forming
index_pointer/peace moves the thumb relative to the very fingertips the pinch ratios measure
(pointing read as a pinch and vice versa, seen live on the left hand where the hotbar poses
sat next to the Jump/Crouch pinches). Chose avoidance over a fix for now: both poses unmapped
everywhere, hotbar next/prev rebound to the left ring/pinky pinches (the only free pinch
surfaces, since index/middle are Jump/Crouch), and the measurement redesign written up in
src/cv/CLAUDE.md instead of attempted — the leading candidate is normalizing each pinch by
that finger's own MCP→tip length rather than hand_scale, so a curled finger and a pinching
finger stop looking alike. Consequences: the detector now emits four pinch ratios,
PinchClicker takes a finger tuple (left hand watches all four, non-exclusive; right stays
index/middle exclusive because two mouse buttons at once is not a thing), and pinches can
carry tap-mode bindings, not just held ones — the hotbar repeats on its cadence while
pinched, same as a held pose did. A held pinch also now keeps the WASD stick live regardless
of what the pose classifier says, because a ring pinch doesn't necessarily read as "fist" and
dropping the walk on every slot change would be the old stutter by another route.
Also: hotbar cycling joins Jump/Sprint in _GUI_BLOCKED_COMMANDS. Number keys are not inert in
a container — they swap the hovered slot into that hotbar slot, so a stray cycle silently
rearranges your inventory. Crouch, Inventory and Escape stay live with a screen open (shift-
click needs the first; the other two are how you close the screen).

**2026-08-08 — Cursor motion posts drag events while a button is held (moved events only
when nothing is down); gameplay camera-look deliberately exempt**
Reasoning: live symptom was that outside Minecraft a window being dragged only jumped to the
cursor on release, and desktop files wouldn't drag at all. Cause is not our pointer math —
it's that AppKit drives drag sessions off mouseDragged:, and we posted kCGEventMouseMoved
unconditionally. A moved event with the button down still moves the hardware cursor, which is
exactly why it looked like it worked: the cursor went where you pointed while the drag
session never received a single update. Fix is one helper (executor.post_cursor_move) that
picks Left/RightMouseDragged from a published `_mouse_button` state; that state is derived
from _held_commands *after* reconciliation, not from the pinch state, so it can never claim a
button is down before the mouse-down event exists (the click-rewind warp on the release edge
does need the drag type, since the button is still down at that point). Minecraft gameplay
camera-look stays on moved events on purpose: GLFW reads the delta fields and treats both
identically, that path was confirmed working live, and no AppKit drag session exists there —
not worth disturbing. Minecraft GUI screens go through the absolute pointer, so they get drag
events too, which should improve dragging items across crafting slots.

**2026-08-08 — Always-loaded context is a budget: session-end protocol moved to a skill,
state.md's 20-line rule enforced rather than relaxed**
Reasoning: root CLAUDE.md had grown to 587 words (~760 tokens) against the ~500-token budget
its own last line sets, and state.md to 76 lines against build-plan §7's ~20. Chose to enforce
both limits rather than raise them, because the limits are tripwires whose diagnostic value
depends on being uncomfortable: applied to state.md, the overflow immediately identified the
misfiled content (a "Gotchas" section of stable operational facts — stray clicks, log
overwriting, quit keys — sitting in a file that gets overwritten every session), and the fact
that they had nowhere better to live because no README existed. Relaxing the limit would have
hidden that. Content moved by lifetime, not by topic: instructions needed at one moment →
`.claude/skills/session-end/SKILL.md` (a real skill with frontmatter — note the three
pre-existing flat .md files in .claude/skills/ have none, so they never load at all);
deliberately-not-built → notes/deferred.md; how-to-run → README.md; per-session status →
state.md. Also dropped the gesture-vocabulary summary from state.md: mapper.py is the source
of truth and a second copy read every session guarantees drift. build-plan §7 amended to name
the overflow destinations, since a cap without destinations invites deleting facts to hit the
number instead of filing them.

**2026-08-09 — GUI shares one GestureDetector between preview and control (handover),
rather than each owning a camera**
Reasoning: two VideoCaptures on one device in-process is undefined on AVFoundation, and
closing/reopening per mode-switch costs ~1s of black preview. Chose: one detector for the
process; a preview pump thread parks (idle-Event handshake) when run.main() borrows the
detector, and the UI renders from detector.last_frame/last_hands/last_gestures stashed by
whichever thread is pumping — so the preview works identically in both modes. Required
detector.close() to release only the capture (landmarker lives for the process) and
run.main() to take stop_event/detector params. Evidence: handover test ran two full
start→stop→restart cycles with frames confirmed flowing in every phase.
