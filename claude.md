# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

- Use conda environment `bygym` for all script execution
- Python path must be set: `PYTHONPATH=/Users/dian/Documents/projects/bigym` or `PYTHONPATH=.`

## Commands

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_cartesian_action_mode.py

# Run specific test
python -m pytest tests/test_cartesian_action_mode.py::TestDemoConversion::test_conversion_accuracy -v

# Run tests excluding slow tests
python -m pytest tests/ -m "not slow"

# Run with verbose output
python -m pytest tests/ -v -s
```

### Running Scripts
```bash
# Always use PYTHONPATH
PYTHONPATH=/Users/dian/Documents/projects/bigym python scripts/script_name.py

# Common scripts
python scripts/replay_cartesian_demo.py  # Replay cartesian demonstrations
python scripts/convert_demos_to_cartesian.py --max-demos 3 --output-dir cartesian_demos  # Convert demos
python examples/replay_demo.py  # Replay joint demonstrations
```

## Code Architecture

### Core Components

**Action Modes** (`bigym/action_modes.py`, `bigym/cartesian_action_mode.py`, `bigym/cartesian_action_mode_direct.py`):
- `JointPositionActionMode`: Direct joint position control with optional floating base
- `TorqueActionMode`: Torque-based control
- `CartesianActionMode`: Cartesian end-effector control using PD controllers
- `CartesianActionModeDirect`: Direct cartesian control bypassing PD controllers, sets qpos directly

**IK Solvers** (multiple implementations):
- `vr/ik/h1_upper_body_ik.py`: Original physics-based IK solver using MuJoCo simulation (40-80 iterations)
- `bigym/ik/mink_h1_ik.py`: Optimization-based IK using mink library
- `vr/ik/clean_h1_upper_body_ik.py`: SciPy-based optimization IK solver

**Robot Configuration**:
- H1 robot with bi-manual manipulation capabilities
- Floating base with configurable DOF (typically 3 DOF: X, Y, RZ)
- 10 arm actuators (5 per arm) for upper body control
- Wrist sites accessed via `robot._wrist_sites[HandSide.LEFT/RIGHT]`

**Environment Structure**:
- Base class: `BiGymEnv` in `bigym/bigym_env.py`
- 40 task environments in `bigym/envs/`
- MuJoCo XML models in `bigym/envs/xmls/`
- Control frequency typically 50Hz with 10 sub-steps (500Hz physics)

### Key Implementation Details

**Direct Mode Physics Drift Issue**:
- Direct mode sets both qpos and ctrl to same value
- Motors have P-gain of 300 but no position error correction when ctrl=qpos
- Gravity causes drift (~0.002 rad after 10 steps on joint 3/elbow)
- Solution: Add gravity compensation using `qfrc_bias`

**Demo System**:
- Demonstrations stored in `demonstrations/` folder
- Joint demos use `JointPositionActionMode` with floating_base=True, absolute=True
- Cartesian demos are 23D (18 EE + 3 base + 2 gripper) with 3 DOF floating base
- Demo conversion maintains timestep alignment for accurate replay

**Physics Sub-stepping**:
- Environment runs `sub_steps_count` physics steps per action (typically 10)
- Only first sub-step calls `action_mode.step(action)`
- Additional steps call `_mojo.step()` causing physics drift

## Development Workflow

### Script Organization
- Put temporary debug scripts in `tmp/` folder
- Move validated/important scripts from `tmp/` to `scripts/`
- Always run and verify debug scripts before finalizing

### Debugging Approach
1. Create focused debug script in `tmp/`
2. Test with known seed for reproducibility (e.g., seed=42 or seed=3873497653)
3. Compare methods side-by-side in same script
4. Verify results match across different execution methods

### Task Management
- Always maintain a todo list for complex debugging tasks
- Analyze the problem and create a plan before executing
- Update todo list when discovering new issues or completing tasks
- Track progress explicitly through the debugging process

## Common Issues and Solutions

**Import Errors**:
- Mink IK may not be available - use try/except for imports
- Use correct imports: `from bigym.const import HandSide`, not from submodules

**State Synchronization**:
- IK solvers may modify environment state during initialization
- Always sync environment states when comparing different solvers
- Use `env.robot._mojo.physics.data.qpos.copy()` for state transfer

**FK Method Differences**:
- Mink IK solutions may only work in their originating physics environment
- Always test FK in both source and target environments
- Check for qpos mismatches before applying IK solutions