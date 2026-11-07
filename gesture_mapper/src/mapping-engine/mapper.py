"""
Context-aware gesture → command mapping.

Each table entry: (hand, gesture_name) → Command

Two kinds of gesture name appear as keys:
  pose names    — "fist", "index_pointer", ... from the CV classifier
  "pinch_index" / "pinch_middle" — not poses but the thumb-distance click
                  signal (run.py PinchClicker). Kept in the same tables so a
                  pinch is bound like anything else, per hand and per context.

Bindings load from data/mappings.json when it exists (the GUI's mapping editor
writes it); otherwise the built-in defaults below apply. The JSON is flat rows
so the editor never touches Python:

  {"contexts": {"default": [{"hand": "left", "gesture": "index_pointer",
                             "command": "OpenSpotlight"}, ...], ...}}

Commands are resolved by name through commands.ALL_COMMANDS — an unknown name
is skipped (logged), never a crash, so a hand-edited file can't take the app
down.
"""

import sys, os, json, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'os-integration'))

from commands import (
    Command,
    ALL_COMMANDS,
    OpenSpotlight,
    ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
    Enter, Delete, Tab, Space,
    HoldBreak, HoldPlace,
    Jump, Inventory, Escape, HotbarPrev, HotbarNext,
    Crouch,
)
# NB Sprint is not here on purpose: it's driven by how far the WASD stick is
# pushed past full deflection (run.py), not by a gesture, so it has no
# (hand, gesture) key to bind to.

log = logging.getLogger("neuratect")

MAPPINGS_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'mappings.json')

# (context, hand, gesture) → Command. These are the factory defaults — the
# live tables come from data/mappings.json when present (see _load()).
_DEFAULT_TABLES: dict[str, dict[tuple[str, str], Command]] = {

    "default": {
        # Right hand is the desktop pointer (run.py GuiPointer): any pose
        # moves the cursor, pinches click, open palm is the clutch — so no
        # right-hand command mappings here; pointing right-handed must NOT
        # open Spotlight. Left hand keeps the command role, same split as
        # Minecraft (right = mouse, left = keys).
        ("left", "index_pointer"):  OpenSpotlight,
        ("right", "pinch_index"):   HoldBreak,   # left mouse button
        ("right", "pinch_middle"):  HoldPlace,   # right mouse button
    },

    "spotlight": {
        # Left hand — nav
        ("left",  "fist"):          ArrowUp,
        ("left",  "open_palm"):     ArrowDown,
        ("left",  "peace"):         ArrowLeft,
        ("left",  "index_pointer"): ArrowRight,
        # Right hand — actions. NB the right hand is also the pointer here
        # (Spotlight is a normal pointable UI — see run.py's ptr_mode), so
        # these poses move the cursor as well as firing; open_palm in
        # particular is the pointer's clutch, and typing Space while
        # repositioning the hand is a live collision worth revisiting.
        ("right", "thumbs_up"):     Enter,
        ("right", "fist"):          Delete,
        ("right", "peace"):         Tab,
        ("right", "open_palm"):     Space,
        # ...and it clicks, same as anywhere else the pointer runs
        ("right", "pinch_index"):   HoldBreak,
        ("right", "pinch_middle"):  HoldPlace,
    },

    "minecraft": {
        # Left hand — fist = joystick (handled in run.py), others = discrete.
        # shaaa was freed up when it stopped being the joystick calibrator;
        # left-hand only so it doesn't collide with the right hand's clicking.
        ("left",  "thumbs_up"):     Inventory,
        ("left",  "shaaa"):         Escape,
        # index_pointer and peace are UNMAPPED on purpose (2026-08-08): forming
        # either pose moves the thumb relative to the very fingertips the pinch
        # ratios measure, so they fought the pinch clicks living on the same
        # hand — pointing read as a middle pinch, etc. The hotbar bindings they
        # used to hold moved onto the ring/pinky pinches below. See the
        # pose/pinch clash note in src/cv/CLAUDE.md before re-using them.
        #
        # Left pinches ride along with the WASD stick — a held pinch keeps the
        # stick live (run.py), so walking never drops when you pinch. Jump and
        # the hotbar cycles are suppressed while a GUI screen is open (see
        # run.py _GUI_BLOCKED_COMMANDS); Crouch deliberately is not — held
        # shift + a right-hand pinch click is inventory shift-click.
        ("left",  "pinch_index"):   Jump,
        ("left",  "pinch_middle"):  Crouch,
        ("left",  "pinch_ring"):    HotbarNext,
        ("left",  "pinch_pinky"):   HotbarPrev,
        # Right hand — fist = camera-look joystick (handled in run.py). Mouse
        # buttons are NOT poses: HoldBreak/HoldPlace come from the thumb pinch
        # (run.py PinchClicker) in gameplay exactly as they do on GUI screens,
        # so a pinch can be held while a fist keeps looking around.
        # index_pointer and peace are deliberately unmapped — free for
        # whatever they're worth once there's mileage on pinch clicking.
        ("right", "pinch_index"):   HoldBreak,
        ("right", "pinch_middle"):  HoldPlace,
        ("right", "thumbs_up"):     Jump,
    },
}


def _tables_from_json(raw: dict) -> dict[str, dict[tuple[str, str], Command]]:
    tables: dict[str, dict[tuple[str, str], Command]] = {}
    for context, rows in raw.get("contexts", {}).items():
        table: dict[tuple[str, str], Command] = {}
        for row in rows:
            cmd = ALL_COMMANDS.get(row.get("command", ""))
            if cmd is None:
                log.info(f"[mapper] unknown command {row.get('command')!r} "
                         f"in {MAPPINGS_PATH} — skipped")
                continue
            table[(row["hand"], row["gesture"])] = cmd
        tables[context] = table
    # A file that lost a whole context (or "default" itself) shouldn't strand
    # map_gesture's fallback — defaults fill any missing context.
    for context, table in _DEFAULT_TABLES.items():
        tables.setdefault(context, dict(table))
    return tables


def _load() -> dict[str, dict[tuple[str, str], Command]]:
    try:
        with open(MAPPINGS_PATH) as f:
            return _tables_from_json(json.load(f))
    except FileNotFoundError:
        return {ctx: dict(t) for ctx, t in _DEFAULT_TABLES.items()}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.info(f"[mapper] bad mappings.json ({e}) — using defaults")
        return {ctx: dict(t) for ctx, t in _DEFAULT_TABLES.items()}


_TABLES = _load()


def reload() -> None:
    """Re-read data/mappings.json (the GUI calls this after saving)."""
    global _TABLES
    _TABLES = _load()


def tables() -> dict[str, dict[tuple[str, str], Command]]:
    """Live tables, read-only by convention — edit via save_tables()."""
    return _TABLES


def default_tables() -> dict[str, dict[tuple[str, str], Command]]:
    """Factory defaults (for the GUI's 'restore defaults')."""
    return {ctx: dict(t) for ctx, t in _DEFAULT_TABLES.items()}


def save_tables(tables_in: dict[str, dict[tuple[str, str], Command]]) -> None:
    """Persist to data/mappings.json and make them live."""
    raw = {
        "contexts": {
            context: [
                {"hand": hand, "gesture": gesture, "command": cmd.name}
                for (hand, gesture), cmd in table.items()
            ]
            for context, table in tables_in.items()
        }
    }
    os.makedirs(os.path.dirname(MAPPINGS_PATH), exist_ok=True)
    with open(MAPPINGS_PATH, "w") as f:
        json.dump(raw, f, indent=2)
    reload()


def map_gesture(gesture_name: str, hand: str, context: str) -> Command | None:
    """
    Returns Command for (context, hand, gesture) or None if no mapping.
    Tries exact hand match first, then "either".
    """
    table = _TABLES.get(context, _TABLES["default"])
    return (
        table.get((hand, gesture_name)) or
        table.get(("either", gesture_name))
    )
