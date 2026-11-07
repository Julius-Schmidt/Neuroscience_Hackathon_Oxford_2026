"""
Command definitions.

mode:
  "tap"  — fire once, cooldown applies
  "held" — key/button held while gesture is active, released when gesture stops
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    mode: str = "tap"  # "tap" | "held"

    def __repr__(self):
        return f"Command({self.name}, {self.mode})"


# ── Spotlight ────────────────────────────────────────────────────────────────
OpenSpotlight            = Command("OpenSpotlight")
ArrowUp                  = Command("ArrowUp")
ArrowDown                = Command("ArrowDown")
ArrowLeft                = Command("ArrowLeft")
ArrowRight               = Command("ArrowRight")
Enter                    = Command("Enter")
Delete                   = Command("Delete")
Tab                      = Command("Tab")
Space                    = Command("Space")

# ── Minecraft ────────────────────────────────────────────────────────────────
HoldW        = Command("HoldW",       mode="held")
HoldA        = Command("HoldA",       mode="held")
HoldS        = Command("HoldS",       mode="held")
HoldD        = Command("HoldD",       mode="held")
HoldBreak    = Command("HoldBreak",   mode="held")  # left mouse button
HoldPlace    = Command("HoldPlace",   mode="held")  # right mouse button
Jump         = Command("Jump", mode="held")
Crouch       = Command("Crouch",      mode="held")  # left shift — also the shift in
                                                    # shift-click (inventory quick-move),
                                                    # which is why it stays live while a
                                                    # GUI screen is open
Sprint       = Command("Sprint",      mode="held")  # left ctrl
Inventory    = Command("Inventory")
Escape       = Command("Escape")       # pause menu — reaches Options/Settings
HotbarPrev   = Command("HotbarPrev")   # presses slot - 1 key (1-9)
HotbarNext   = Command("HotbarNext")   # presses slot + 1 key (1-9)

# ── Joystick ─────────────────────────────────────────────────────────────────
# Not fired as commands — handled directly in run.py via joystick module

# ── Registry ─────────────────────────────────────────────────────────────────
# name → Command, for anything that resolves commands from data (the mapping
# config, the UI's command picker). Sprint is excluded: it has no (hand,
# gesture) binding — it's driven by stick deflection in run.py.
ALL_COMMANDS: dict[str, Command] = {
    c.name: c for c in (
        OpenSpotlight,
        ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
        Enter, Delete, Tab, Space,
        HoldW, HoldA, HoldS, HoldD,
        HoldBreak, HoldPlace, Jump, Crouch,
        Inventory, Escape, HotbarPrev, HotbarNext,
    )
}


def by_name(name: str) -> Command | None:
    return ALL_COMMANDS.get(name)
