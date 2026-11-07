"""eeg_switch: replayed-EEG simulator + eyes-closed safety switch, all over LSL.

    python -m eeg_switch.download            fetch PhysioNet runs 1-2 (eyes open / closed)
    python -m eeg_switch.simulator           publish replayed EEG as a Unicorn-like LSL stream
    python -m eeg_switch.run_detector        read the EEG stream, publish intent events
    python -m eeg_switch.evaluate            held-out metrics for the slides
    python -m eeg_switch.listen              print intent events (debugging)

Swap simulator <-> real g.tec Unicorn by changing SOURCE in config.py.
"""
