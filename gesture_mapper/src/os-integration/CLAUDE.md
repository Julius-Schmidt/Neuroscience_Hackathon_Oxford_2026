# os-integration — module context

## Non-obvious constraints

- Minecraft camera look: CGEventPost with kCGMouseEventDeltaX/Y fields works but is imperfect.
  GLFW reads NSEvent.deltaX/deltaY which may be zero on synthetic events depending on macOS
  version. Full fix requires a Fabric mod exposing a local socket, or DriverKit virtual HID.
- Hotbar slot state (_hotbar_slot) is module-level in executor.py — resets to 1 on restart.
  If the in-game slot drifts out of sync, re-press the number key manually to resync.
- CGEventPost requires Accessibility permission for the running terminal/app. Without it,
  keystrokes and mouse events are silently dropped on macOS 10.14+.
- Cursor motion while a button is held must post kCGEventLeft/RightMouseDragged, not
  kCGEventMouseMoved. A moved event still moves the cursor, so it looks fine — but AppKit
  drives drags off mouseDragged:, so no drag session sees it: dragged windows snap over only
  on mouse-up and Finder never starts a file drag (live, 2026-08-08). Use
  `post_cursor_move(x, y, dx, dy, button)`. Exception: gameplay camera-look stays on moved
  events — GLFW reads the deltas and treats both alike.
- Synthetic mouse events inherit live modifier flags on this macOS (measured), so a held
  shift makes our click a shift-click. `_mouse_event` stamps them explicitly anyway.
- Spotlight is opened via Cmd+Space CGEvent (not osascript) to avoid Finder window side effects.
  Earlier attempt with `tell Finder to activate` brought two Finder windows to front.
