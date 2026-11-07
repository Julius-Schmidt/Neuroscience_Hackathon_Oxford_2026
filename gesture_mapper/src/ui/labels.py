"""
Human names for everything the engine calls by identifier: commands,
gestures, contexts, hands — plus the "built-in" (non-remappable) behaviors
per context, so onboarding and the mapping editor can be honest about what
lives in run.py rather than in the tables.
"""

import theme

# command name → (label, detail)
COMMANDS: dict[str, tuple[str, str]] = {
    "OpenSpotlight": ("Open Spotlight", "⌘ Space"),
    "ArrowUp":       ("Arrow Up", "↑ key"),
    "ArrowDown":     ("Arrow Down", "↓ key"),
    "ArrowLeft":     ("Arrow Left", "← key"),
    "ArrowRight":    ("Arrow Right", "→ key"),
    "Enter":         ("Return", "confirm / open"),
    "Delete":        ("Delete", "⌫ key"),
    "Tab":           ("Tab", "next field"),
    "Space":         ("Space", "space bar"),
    "HoldW":         ("Hold W", "walk forward"),
    "HoldA":         ("Hold A", "walk left"),
    "HoldS":         ("Hold S", "walk back"),
    "HoldD":         ("Hold D", "walk right"),
    "HoldBreak":     ("Left Mouse Button", "click · mine · drag"),
    "HoldPlace":     ("Right Mouse Button", "right-click · place"),
    "Jump":          ("Jump", "space, held"),
    "Crouch":        ("Crouch / Sneak", "shift, held"),
    "Inventory":     ("Inventory", "E key"),
    "Escape":        ("Escape", "pause / back"),
    "HotbarPrev":    ("Hotbar ◀", "previous slot"),
    "HotbarNext":    ("Hotbar ▶", "next slot"),
}

GESTURES: dict[str, str] = {
    "fist":          "Fist",
    "open_palm":     "Open Palm",
    "peace":         "Peace",
    "index_pointer": "Point",
    "thumbs_up":     "Thumbs Up",
    "shaaa":         "Shaaa",
    "pinch":         "Pinch",
    "pinch_index":   "Index Pinch",
    "pinch_middle":  "Middle Pinch",
    "pinch_ring":    "Ring Pinch",
    "pinch_pinky":   "Pinky Pinch",
}

PINCH_GESTURES = ["pinch_index", "pinch_middle", "pinch_ring", "pinch_pinky"]

CONTEXTS: dict[str, tuple[str, str]] = {   # id → (label, SF symbol)
    "default":   ("Desktop", "menubar.rectangle"),
    "spotlight": ("Spotlight", "magnifyingglass"),
    "minecraft": ("Minecraft", "cube.fill"),
}
CONTEXT_ORDER = ["default", "spotlight", "minecraft"]

HANDS: dict[str, tuple[str, str]] = {      # id → (label, color)
    "left":   ("Left hand", theme.HAND_LEFT_COLOR),
    "right":  ("Right hand", theme.HAND_RIGHT_COLOR),
    "either": ("Either hand", theme.TEXT_DIM),
}

# Behaviors implemented directly in run.py (joysticks, the pointer, clutch) —
# real parts of the control scheme that are not rows in the mapping tables.
# context → list of (hand, title, detail, SF symbol)
BUILTINS: dict[str, list[tuple[str, str, str, str]]] = {
    "default": [
        ("right", "Move the cursor", "any pose tracks — your hand is the mouse",
         "cursorarrow.motionlines"),
        ("right", "Open palm = clutch", "lift the “mouse” to reposition without moving the cursor",
         "hand.raised.fill"),
    ],
    "spotlight": [
        ("right", "Pointer keeps working", "Spotlight is a normal clickable surface",
         "cursorarrow.motionlines"),
    ],
    "minecraft": [
        ("left", "Fist = walk stick", "hold a fist and steer — push far forward to sprint",
         "gamecontroller.fill"),
        ("right", "Fist = camera look", "hold a fist and steer the view",
         "arrow.up.left.and.arrow.down.right"),
        ("right", "Open palm", "releases the look stick",
         "hand.raised.fill"),
    ],
}


def command_label(name: str) -> str:
    return COMMANDS.get(name, (name, ""))[0]


def command_detail(name: str) -> str:
    return COMMANDS.get(name, ("", ""))[1]


def gesture_label(name: str) -> str:
    return GESTURES.get(name) or name.replace("_", " ").title()


def context_label(ctx: str) -> str:
    return CONTEXTS.get(ctx, (ctx.title(), "questionmark.circle"))[0]


def context_symbol(ctx: str) -> str:
    return CONTEXTS.get(ctx, (ctx.title(), "questionmark.circle"))[1]
