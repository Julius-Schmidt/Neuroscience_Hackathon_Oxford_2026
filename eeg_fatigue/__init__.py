"""eeg_fatigue: brain-muscle divergence during a fatiguing squeeze task (Level 1).

    python -m eeg_fatigue.session --participant P01 --sanity     3 min check: alpha + squeeze response
    python -m eeg_fatigue.session --participant P01              full session (~15 min), autosaves
    python -m eeg_fatigue.analyse data/fatigue/P01_*.npz         ratio per block, figure, verdict
    python -m eeg_fatigue.simulate --out data/fatigue/SIM.npz    offline simulated recording
    python -m eeg_fatigue.simulate --lsl                         live fake headset + arm (dry run)

Ratio per block = EEG effort (C3 beta drop, or motor potential) / EMG squeeze peak,
both normalised to block 1. Rises with fatigue, drops after rest = divergence.
"""
