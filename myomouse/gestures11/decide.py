"""From classifier output to what the live readout shows. Biased towards "neutral" (= no gesture).

Every decision (one per STEP samples) goes through three filters, from the myoelectric
control literature:

1. amplitude gate:  EMG about as quiet as your recorded rest -> neutral, whatever the classifier
                    says (onset detection: rest level + GATE_K x its spread)
2. rejection:       the classifier's best guess is below REJECT probability -> neutral
                    (confidence-based rejection; docs/REFERENCES.md [4], [5])
3. switching:       a gesture must win N_ON decisions in a row to be shown; N_OFF neutral
                    decisions in a row are enough to drop back (asymmetric, like the click debouncer)

Neutral and rest both count as "no gesture" and are shown as "neutral".
"""

import numpy as np

from . import IDLE_KEYS, N_OFF, N_ON, REJECT

NONE = "neutral"


def gate_from_rest(envelopes):
    """Gate level from rest-window envelopes: median + GATE_K x robust spread. None if too little data."""
    from . import GATE_K
    env = np.asarray(envelopes, float)
    if len(env) < 20:
        return None
    med = float(np.median(env))
    spread = 1.4826 * float(np.median(np.abs(env - med)))  # robust standard deviation
    return med + GATE_K * max(spread, 0.02)


def raw_decision(proba, env, classes, gate=None, reject=REJECT):
    """One decision before switching: a gesture key or NONE."""
    if gate is not None and env < gate:
        return NONE
    i = int(np.argmax(proba))
    if classes[i] in IDLE_KEYS or proba[i] < reject:
        return NONE
    return classes[i]


class Switcher:
    """Asymmetric switching: N_ON in a row to enter/change a gesture, N_OFF to drop back to NONE."""

    def __init__(self, n_on=N_ON, n_off=N_OFF):
        self.n_on, self.n_off = n_on, n_off
        self.current, self.candidate, self.count = NONE, None, 0

    def update(self, decision):
        """Returns the new shown gesture when it changes, else None."""
        if decision == self.current:
            self.candidate, self.count = None, 0
            return None
        if decision == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = decision, 1
        if self.count >= (self.n_off if decision == NONE else self.n_on):
            self.current, self.candidate, self.count = decision, None, 0
            return decision
        return None


def simulate(proba, env, holds, classes, gate=None, reject=REJECT, n_on=N_ON, n_off=N_OFF):
    """Run the live pipeline over windows in time order; switching restarts at every new hold.
    Returns the shown key per window."""
    out, sw, prev = [], None, None
    for p, e, h in zip(proba, env, holds):
        if h != prev:
            sw, prev = Switcher(n_on, n_off), h
        sw.update(raw_decision(p, e, classes, gate, reject))
        out.append(sw.current)
    return np.array(out, dtype=object)
