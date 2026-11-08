# MS stress test: final numbers (Sun 10:45, supersedes earlier chart notes and the "12%" claim)

**Framing for the video:** we couldn't test on people with MS in 24 hours, so we recreated the three things MS does to a muscle signal (fatigue, weakness, tremor) on Dylan's real recording and measured where the decoder breaks. One healthy user; weakness and tremor are simulated. Next step is real patients.

Charts: `docs/charts/1_fatigue.png`, `2_weakness.png`, `3_tremor.png` (regenerate with `python -m devtool.slide_charts data/devtool/sessions/20261108_022630_dylan_fatigue`). Use these instead of the four older charts.

**Fatigue (real recording).** Squeeze strength fell to about half over 6 minutes of squeezing every 2.5 s. The decoder caught 168 of 168 cued squeezes, with 9 false clicks. Frequency-based fatigue markers barely moved: the Myo samples at 200 Hz so it only sees up to 100 Hz, where most of the fatigue shift is out of view. Signal strength is the usable fatigue marker on this band.

**Weakness (simulated: muscle signal shrunk, real sensor noise kept).** The standard decoder works down to about half strength and fails at a third (40% caught). Rescaling the input to the user's current strength (prototype, offline only, not yet in the live mouse) keeps 95% caught at a quarter strength. A simple threshold recalibrated at the weaker strength holds 97% at 18%, but with 2-3 false clicks a minute. Training with `--augment` helped only modestly (at half strength, today's model goes from 21% to 54% caught).

**Tremor (simulated 4-6 Hz bursts).** With the current 30 ms hold, false clicks reach 77 a minute when the tremor is half the size of a squeeze. A 200 ms hold keeps them near zero up to that size; each click lands about 170 ms later. At 75% tremor it catches 63% and false clicks rise to 5 a minute. Weakness and tremor combined (tremor fixed while the squeeze shrinks): false clicks stayed at or below 0.3 a minute in every combination tested; the failure is missed clicks, not random clicks.

**Corrections.** "Reliable at 12% strength" was the recalibrated threshold detector, not the live decoder: don't claim it for the live system. The 200 ms tremor hold is a tested setting, not the current default. Fatigue adaptation did not help on the real fatigue recording (the adaptive threshold was worst: 149/168).
