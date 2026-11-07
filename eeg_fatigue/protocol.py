"""The session as a list of steps, shared by the live session and the simulator.

Step = {"marker": str | None, "cue": str, "dur": float, "kind": "squeeze" | "relax" | "wait"}

Markers (on LSL stream FatigueMarkers and in the saved file):
    session_start, session_end
    block_start:<name>:<kind>   kind = sanity | baseline | fatigue | recovery
    block_end:<name>
    cue_squeeze, cue_relax
    rest_start, rest_end
    eyes_open, eyes_closed      (sanity check only)
"""

from __future__ import annotations

import numpy as np

from . import config as C


def _block(name, kind, n, rng):
    steps = [{"marker": f"block_start:{name}:{kind}", "cue": "relax, get ready", "dur": C.LEAD_IN_S, "kind": "relax"}]
    for _ in range(n):
        steps.append({"marker": "cue_squeeze", "cue": "SQUEEZE AS HARD AS YOU CAN", "dur": C.SQUEEZE_S,
                      "kind": "squeeze"})
        steps.append({"marker": "cue_relax", "cue": "relax", "kind": "relax",
                      "dur": C.RELAX_S + rng.uniform(-C.RELAX_JITTER_S, C.RELAX_JITTER_S)})
    steps.append({"marker": f"block_end:{name}", "cue": "", "dur": 0.0, "kind": "wait"})
    return steps


def session_steps(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    steps = [{"marker": "session_start", "cue": "", "dur": 0.0, "kind": "wait"}]
    steps += _block("B1", "baseline", C.SQUEEZES_PER_BLOCK, rng)
    for i in range(C.FATIGUE_BLOCKS):
        steps.append({"marker": None, "cue": "keep going - next block", "dur": C.BETWEEN_BLOCKS_S, "kind": "wait"})
        steps += _block(f"F{i + 1}", "fatigue", C.SQUEEZES_PER_BLOCK, rng)
    steps.append({"marker": "rest_start", "cue": "REST - arm relaxed, eyes open", "dur": C.REST_S, "kind": "wait"})
    steps.append({"marker": "rest_end", "cue": "", "dur": 0.0, "kind": "wait"})
    for i in range(C.RECOVERY_BLOCKS):
        steps += _block(f"R{i + 1}", "recovery", C.SQUEEZES_PER_BLOCK, rng)
    steps.append({"marker": "session_end", "cue": "", "dur": 0.0, "kind": "wait"})
    return steps


def sanity_steps(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    steps = [{"marker": "session_start", "cue": "", "dur": 0.0, "kind": "wait"},
             {"marker": "eyes_open", "cue": "EYES OPEN, look at one point, relax", "dur": C.SANITY_EYES_S, "kind": "wait"},
             {"marker": "eyes_closed", "cue": "CLOSE YOUR EYES, relax", "dur": C.SANITY_EYES_S, "kind": "wait"},
             {"marker": "eyes_open", "cue": "open your eyes", "dur": 3.0, "kind": "wait"}]
    steps += _block("S1", "sanity", C.SANITY_SQUEEZES, rng)
    steps.append({"marker": "session_end", "cue": "", "dur": 0.0, "kind": "wait"})
    return steps


def duration(steps) -> float:
    return sum(s["dur"] for s in steps)
