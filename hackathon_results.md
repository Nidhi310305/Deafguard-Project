# DeafGuard Prototype Results

## Benchmark evaluation
- Training files: 179
- Validation files: 56
- Final-test files: 56
- Filename overlap: none detected
- Final-test accuracy: 92.9%
- Validation-selected candidate threshold: 0.50
- Critical-event candidate recall: 100% on the 56-file final-test sample

## Alert-engine verification
- Three consecutive qualifying predictions produced FINAL_ALERT.
- other_unknown produced NO_ALERT.
- Repeated alerts were suppressed during the 5-second cooldown.

## Controlled live siren demonstration
- HP/Windows voice-processing features were disabled.
- External phone siren was captured through the laptop microphone.
- The system recognized emergency_siren and generated one HIGH alert after repeated confirmations.
- Duplicate siren alerts were suppressed during cooldown.

## Mixed-background test
- Room and road ambience produced isolated candidate predictions.
- No FINAL_ALERT was produced because predictions were not consistent for three consecutive windows.

## Limitations
- External microphone recordings are still a small device-level evaluation set.
- Pulsed sirens may be confused with vehicle horns.
- Mixed acoustic scenes can produce overconfident class probabilities.
- The prototype is not a safety-certified or production-ready alert system.
