# References

The methods in this repo are standard ones from the myoelectric-control and EMG-fatigue
literature. Each entry says where it is used. Numbers in square brackets are cited from the
READMEs and code comments.

## EMG pattern recognition (click and gesture classifiers)

**[1]** Hudgins B, Parker P, Scott RN. A new strategy for multifunction myoelectric control.
*IEEE Transactions on Biomedical Engineering* 40(1):82–94, 1993. doi:10.1109/10.204774
→ The time-domain feature set (MAV, waveform length, zero crossings, slope sign changes) in
[myomouse/features.py](../myomouse/features.py), used by the fist click and by `gestures11`.

**[2]** Englehart K, Hudgins B. A robust, real-time control scheme for multifunction myoelectric
control. *IEEE Transactions on Biomedical Engineering* 50(7):848–854, 2003. doi:10.1109/TBME.2003.813539
→ Overlapping analysis windows with frequent decisions, an LDA classifier and majority-style
smoothing of decisions ([myomouse/detector.py](../myomouse/detector.py),
[myomouse/gestures11/decide.py](../myomouse/gestures11/decide.py)).

**[3]** Scheme E, Englehart K. Electromyogram pattern recognition for control of powered
upper-limb prostheses: state of the art and challenges for clinical use. *Journal of
Rehabilitation Research and Development* 48(6):643–659, 2011. doi:10.1682/JRRD.2010.09.0177
→ Background for the design choices and their known problems (electrode shift between sessions,
arm-position effect, need for rest/"no motion" classes). This is why we record "relax while
moving" data and report new-session accuracy.

**[4]** Scheme EJ, Hudgins BS, Englehart KB. Confidence-based rejection for improved pattern
recognition myoelectric control. *IEEE Transactions on Biomedical Engineering*
60(6):1563–1570, 2013. doi:10.1109/TBME.2013.2238939
→ `REJECT` in [myomouse/gestures11/\_\_init\_\_.py](../myomouse/gestures11/__init__.py): low-confidence
decisions are shown as neutral.

**[5]** Robertson JW, Englehart KB, Scheme EJ. Effects of confidence-based rejection on usability
and error in pattern recognition-based myoelectric control. *IEEE Journal of Biomedical and
Health Informatics* 23(5):2002–2008, 2019. doi:10.1109/JBHI.2018.2878907
→ The 0.6–0.75 range quoted for the rejection threshold.

## Myo armband

**[6]** Pizzolato S, Tagliapietra L, Cognolato M, Reggiani M, Müller H, Atzori M. Comparison of
six electromyography acquisition setups on hand movement classification tasks. *PLoS ONE*
12(10):e0186132, 2017. doi:10.1371/journal.pone.0186132
→ Evidence that the low-cost Myo (8 channels, 200 Hz) is usable for hand-movement classification.

**[7]** Côté-Allard U, Fall CL, Drouin A, Campeau-Lecours A, Gosselin C, Glette K, Laviolette F,
Gosselin B. Deep learning for electromyographic hand gesture signal classification using
transfer learning. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*
27(4):760–771, 2019. doi:10.1109/TNSRE.2019.2896269 (preprint arXiv:1801.07756)
→ Myo gesture datasets and per-user vs. cross-user accuracy; possible next step beyond LDA.

**[8]** Atzori M, Gijsberts A, Castellini C, et al. Electromyography data for non-invasive
naturally-controlled robotic hand prostheses. *Scientific Data* 1:140053, 2014.
doi:10.1038/sdata.2014.53
→ NinaPro, the standard public EMG hand-movement database (grips and wrist movements).

**[9]** Thalmic Labs. *Myo Bluetooth protocol* (`myohw.h`).
https://github.com/thalmiclabs/myo-bluetooth
→ BLE service/characteristic UUIDs and commands in [myomouse/device.py](../myomouse/device.py)
and [devtool/ble.py](../devtool/ble.py).

## Fatigue

**[10]** Dimitrov GV, Arabadzhiev TI, Mileva KN, Bowtell JL, Crichton N, Dimitrova NA. Muscle
fatigue during dynamic contractions assessed by new spectral indices. *Medicine & Science in
Sports & Exercise* 38(11):1971–1979, 2006. doi:10.1249/01.mss.0000233794.31659.6d
→ `dimitrov` (FInsm5) in [devtool/fatigue_features.py](../devtool/fatigue_features.py).

**[11]** Allison GT, Fujiwara T. The relationship between EMG median frequency and low frequency
band amplitude changes at different levels of muscle capacity. *Clinical Biomechanics*
17(6):464–469, 2002. doi:10.1016/S0268-0033(02)00033-5
→ `low_frac` (share of power in 15–45 Hz) in [devtool/fatigue_features.py](../devtool/fatigue_features.py).

**[12]** Leeb R, Sagha H, Chavarriaga R, Millán JdR. A hybrid brain–computer interface based on
the fusion of electroencephalographic and electromyographic activities. *Journal of Neural
Engineering* 8(2):025011, 2011. doi:10.1088/1741-2560/8/2/025011
→ Simulated fatigue-induced EMG changes in an assistive EMG interface; motivates letting the
strength tracker in [devtool/fatigue.py](../devtool/fatigue.py) move both ways.

**[13]** Kalman RE. A new approach to linear filtering and prediction problems. *Journal of Basic
Engineering* 82(1):35–45, 1960. doi:10.1115/1.3662552
→ The 1-D Kalman filter that tracks squeeze strength ([devtool/fatigue.py](../devtool/fatigue.py),
[myomouse/adapt.py](../myomouse/adapt.py), `--rescale`).

## Multiple sclerosis (tremor simulation)

**[14]** Makhoul K, Ahdab R, Riachi N, Chalah MA, Ayache SS. Tremor in multiple sclerosis: an
overview and future perspectives. *Brain Sciences* 10(10):722, 2020. doi:10.3390/brainsci10100722
→ MS tremor is mostly postural/intention tremor at about 2.5–7 Hz; the simulated 4–6 Hz bursts
in [devtool/tremor_sim.py](../devtool/tremor_sim.py) sit inside that range.

## Computer vision (camera labels and Neuratect)

**[15]** Zhang F, Bazarevsky V, Vakunov A, Tkachenka A, Sung G, Chang C-L, Grundmann M. MediaPipe
Hands: on-device real-time hand tracking. arXiv:2006.10214, 2020.
→ The 21-landmark hand tracker used for camera labels in `myomouse` and as the whole input of
`gesture_mapper`. Model file: [assets/hand_landmarker.task](../assets/hand_landmarker.task)
(Google, Apache License 2.0).

**[16]** Lugaresi C, Tang J, Nash H, et al. MediaPipe: a framework for building perception
pipelines. arXiv:1906.08172, 2019.

## Still to fill in

- The "10 gestures from the EMG-gesture literature" that `gestures11` started from: the code
  doesn't say which paper or dataset. Add it here.
