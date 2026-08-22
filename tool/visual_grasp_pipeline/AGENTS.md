# Visual grasp pipeline guardrails

These constraints are mandatory for every agent editing this directory.

## Frozen target-selection contract

Do not change the existing selection chain unless the user explicitly asks to
redesign target selection:

1. `detect_all_track` runs YOLO and assigns stable IDs through `StableTracker`.
2. The operator selects one item from the UI and `on_add` records `name#id`.
3. `parse_sequence` preserves that class name and instance ID.
4. `find_sequence_target` must reselect the exact same class and stable ID.
5. If that exact instance is absent, fail the step. Never fall back to another
   instance, the largest box, highest confidence, nearest box, or another class.

Do not modify `detection.py`, `tracking.py`, `find_sequence_target`, `on_add`,
or the sequence format as part of grasp-execution work. Execution consumes the
already selected target; it does not choose a target.

## Controller connection contract

Reuse one persistent controller between visual TCP readback and grasp
execution. Do not close it and immediately reconnect to the single-client
6001/7000 slots. Keep `heartbeat_s=0.0`: field testing showed background
`0x7266` interleaving with MOVL `0x4502`, after which the teach pendant reports
errors for both commands.

## Current pose-selection contract

The user currently wants direct UCS1 TCP XYZ execution with the orientation
fixed to the reset-point orientation. Do not introduce or substitute automatic
orientation selection. A future orientation planner must be optional and must
not alter the frozen target selection above.
