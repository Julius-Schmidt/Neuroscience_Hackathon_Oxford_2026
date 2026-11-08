"""Input sources for the rule engine. Each turns some device into rules.Event objects.

    camera   hand poses from the webcam (the existing CV pipeline)
    myo      Myo armbands: our trained click (LDA) or the band's built-in 5 poses, plus
             arm pointing as a continuous value
    lsl      any Lab Streaming Layer marker stream: EEG/BCI decoders (c-VEP, SSVEP, P300,
             motor imagery), PsychoPy, OpenViBE, BCI2000... each marker = one event
    test     events injected by hand (editor "try it" buttons, tests)

A source is anything with start(emit) / stop() / status. emit(Event) is thread-safe.
New devices = one small class here; the editor and rules don't change.
"""
