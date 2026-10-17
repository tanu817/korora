# Live transcription notes

- Entry point: `python .\live_transcription.py`.
- Whisper uses `faster-whisper` with an externally downloaded model identifier.
  Set `WHISPER_MODEL` to change it; the default is `small`.
- The default Whisper device is CPU. NVIDIA imports are only needed if the
  configuration is changed to CUDA.
- Voice commands:
  - `feed me`, `i want food`, or `give me food` starts feeding.
  - `stop`, `i am done`, or `it is enough` stops feeding.
- Feeding is routed through `feeder.FluxHands` and the instructions in
  `feeder/config.py`.
- `FEEDER_DRY_RUN` defaults to `true`. Keep it enabled until the robot,
  cameras, calibration, policy, and emergency stop are verified.
- Safe integration test:

  ```powershell
  python .\live_transcription.py --test
  ```

- Useful settings: `FEEDER_BITES`, `FLUX_POLICY_PATH`, `FLUX_USE_PEFT`,
  `FLUX_ROBOT_PORT`, `FLUX_SCENE_CAMERA`, and `FLUX_WRIST_CAMERA`.
- `stop` terminates the active FLUX rollout subprocess. The sequence returns
  to standby after completion.
- The current harness does not yet use `Eyes` to verify step success or retry
  failed physical actions.
