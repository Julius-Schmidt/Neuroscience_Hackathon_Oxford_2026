# ui — module context

## Non-obvious constraints

- **This PySide6 lacks a TIFF decoder**, so NSImage.TIFFRepresentation() → QImage silently
  yields a null image. SF Symbols must round-trip through PNG via NSBitmapImageRep
  (theme._render_symbol). Symptom if regressed: every icon falls back to text glyphs.
- **polish_macos_window segfaults off-cocoa.** objc_object(winId()) is only an NSView on the
  cocoa platform; under QT_QPA_PLATFORM=offscreen it's a hard segfault that no try/except
  catches. The platformName() guard in theme.py is load-bearing — keep it.
- **Camera ownership:** exactly one thread pumps detector.next_gestures() at a time. Preview
  = Engine's pump thread; control = run.main() (which borrows the same detector). All
  rendering reads detector.last_frame/last_hands/last_gestures — take the reference, never
  mutate. Never create a second GestureDetector in this module.
- The preview is mirrored (selfie view): draw overlay landmarks at x' = 1 − lm.x. Gesture
  hand labels ("left"/"right") are already correct for the mirrored view — don't flip them.
- After gesture add/delete: recognizer.reload_library() (process-lifetime cache) **and**
  handicon.reload() (pixmap cache), or the old library keeps recognizing/rendering.
  Engine.save_gesture/delete_gesture do both — go through them.
- Qt widgets/paints on the main thread only. Engine signals cross from worker threads;
  connect them normally (queued) and touch widgets in the slot.
