"""The two choices every script shares: how you click and how you move the cursor.

    --click fist    squeeze a fist                                        [default]
    --click index   lift the index finger (experimental, not reliable: only with the flag)
    --move  table   forearm on the table: turn = left/right, tilt the hand = up/down  [default]
    --move  air     point the forearm in 3D (yaw / pitch)
    --move  slide   slide the hand on the table (accelerometer, experimental: can drift)

Each click gesture has its own recordings and its own model, so both can be trained for
the same user and you pick one when you run:

    data/<user>/<side>_<ts>.npz           fist recordings     models/<user>/fist_<side>.joblib
    data/<user>/index_<side>_<ts>.npz     index recordings    models/<user>/index_<side>.joblib
"""

from . import config as C

CLICKS = ("fist", "index")
DEFAULT_CLICK = "fist"    # never "index": that one is experimental and opt-in via --click index
MOVES = ("table", "air", "slide", "game")

# words for messages / UI
WORD = {"fist": "fist", "index": "index lift"}      # "caught 5/8 fists" / "index lifts"
CUE = {"fist": "SQUEEZE", "index": "LIFT"}          # big red cue while recording


def add_style_args(ap, move=True):
    ap.add_argument("--click", choices=CLICKS, default=DEFAULT_CLICK,
                    help="click gesture: fist = squeeze (default), "
                         "index = lift the index finger (experimental, not reliable)")
    if move:
        ap.add_argument("--move", choices=MOVES, default=C.MOVE,
                        help=f"cursor: table = turn + tilt on the table, air = point the forearm, "
                             f"slide = slide on the table (default {C.MOVE})")
        ap.add_argument("--axes", choices=("dominant", "x", "y", "both"), default=C.AXES,
                        help="which cursor axes move: dominant = only the one you move more right now, "
                             f"x / y = only that one, both = free (default {C.AXES})")


def recording_prefix(click: str, side: str) -> str:
    """File name start of a recording; fist keeps the original names so old data still loads."""
    return f"{side}_" if click == "fist" else f"{click}_{side}_"
