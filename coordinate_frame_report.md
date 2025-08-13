# Coordinate Frame Analysis Report

## Executive Summary
The 56.5mm error in CartesianActionMode is caused by a **coordinate frame mismatch** when the base moves. The IK solver always assumes the pelvis is at a fixed position `[0, 0, 0.98]`, but the actual robot's pelvis starts at `[0, 0, 1.0]` and can move during operation via cumulative delta control.

## Testing Methodology

### Scripts Created and Used
1. **`debug_base_movement.py`** - Definitively determined base control mode
2. **`analyze_coordinate_system.py`** - Analyzed IK solver usage patterns
3. **`debug_coordinate_frames.py`** - Tested Cartesian action frames
4. **Source code analysis** - Read `bigym/action_modes.py` and `bigym/robots/floating_base.py`

## Detailed Analysis

### 1. Base Movement Control Mode (FULLY TESTED AND UNDERSTOOD)

**Test Script**: `debug_base_movement.py`  
**Test Method**: Applied controlled base actions and measured resulting pelvis positions

With `floating_base=True` and `absolute=True`:
- **Arm joints**: Use ABSOLUTE position control 
  - Code evidence: `action_modes.py:242` → `actuator.ctrl = action[i]`
- **Base control**: Uses CUMULATIVE DELTA control
  - Code evidence: `floating_base.py:118` → `bound_actuator.ctrl += ctrl`
  - The comment in `action_modes.py:183` confirms: "Joints of the floating_base are always controlled in delta position mode"

**Key findings from testing**:
- Base actions are **world-frame deltas** that accumulate over time
- Each action adds to the previous control value: `ctrl += action`
- Position deltas limited to `(-0.01, 0.01)` per step
- Rotation deltas limited to `(-0.05, 0.05)` per step
- When applying action `[0.1, 0, 0]`, the robot only moved ~5mm because of these limits

### 2. Cartesian Action Frame (TESTED)

**Test Script**: `debug_coordinate_frames.py` and `analyze_coordinate_system.py`  
**Test Method**: Set specific world targets and measured resulting end-effector positions

- **End-effector targets**: Absolute positions in WORLD frame
- **Test result**: When target was set to `[0.4185, 0.21353, 1.0867]`, the system attempted to reach that exact world coordinate
- **Confirmation**: Targets are world coordinates, not relative to robot base

### 3. IK Solver Frame Analysis (CODE INSPECTION)

**Files Analyzed**: 
- `bigym/cartesian_action_mode.py` (line 186)
- `vr/ik/h1_upper_body_ik.py` (lines 173-174)

#### Current Implementation Problem:
```python
# CartesianActionMode.step() line 186:
pelvis_pose = Pose(np.array([0.0, 0.0, 0.98]), Quaternion(w=1, x=0, y=0, z=0))
```

**THE PROBLEM**: Always passes a FIXED pelvis position regardless of actual robot position!

#### What H1UpperBodyIK does:
```python
# h1_upper_body_ik.py lines 173-174:
self._physics.bind(self._pelvis).pos = pelvis_pose.position
self._physics.bind(self._pelvis).quat = pelvis_pose.orientation.elements
```
- Takes the pelvis_pose parameter
- Sets it directly in its internal physics model
- Solves IK assuming pelvis is at that exact position

### 4. Root Cause Analysis (DEMONSTRATED)

**Test Scenario** (from `analyze_coordinate_system.py`):
1. Robot pelvis starts at `[0, 0, 1.0]`
2. Base moves via cumulative deltas to `[0.1, 0.05, 1.0]`
3. IK solver still thinks pelvis is at `[0.0, 0.0, 0.98]`
4. This mismatch causes incorrect IK solutions
5. Measured error: ~56.5mm

## The Fix

### Correct Implementation
Replace the fixed pelvis position with the actual current position:

```python
def step(self, action: np.ndarray):
    # ... parse action components ...
    
    # Get ACTUAL current pelvis pose (not fixed!)
    actual_pelvis_pos = self._robot.pelvis.get_position()
    actual_pelvis_quat = Quaternion(self._robot.pelvis.get_quaternion())
    
    # Adjust Z to match solver's internal model (0.98 vs 1.0)
    solver_pelvis_pos = actual_pelvis_pos.copy()
    solver_pelvis_pos[2] = 0.98  # Solver expects 0.98m height
    
    pelvis_pose = Pose(solver_pelvis_pos, actual_pelvis_quat)
    
    # Now IK solver knows actual pelvis position
    ik_solution = self._ik_solver.solve(
        pelvis_pose=pelvis_pose,  # Use actual position!
        qpos_arm_left=qpos_arm_left,
        qpos_arm_right=qpos_arm_right,
        target_pose_left=Pose(left_pos, left_quat),
        target_pose_right=Pose(right_pos, right_quat),
    )
```

### Why This Fix Works

1. **Base moves via cumulative deltas** → Pelvis position changes continuously
2. **IK solver needs actual pelvis position** → To solve in the correct reference frame
3. **Targets are in world frame** → IK must know robot's world position to reach world targets
4. **Fix provides actual position** → IK can now solve correctly relative to actual robot pose

## Expected Impact

- Error reduction from 56.5mm to <10mm
- Maintains accuracy even with significant base movement
- Properly handles mobile robot scenarios in demonstrations

## Validation Method

To validate the fix:
1. Run `tests/test_cartesian_vs_joint_execution.py`
2. Check that average error drops from 56.5mm to <10mm
3. Verify demos with base movement show consistent accuracy